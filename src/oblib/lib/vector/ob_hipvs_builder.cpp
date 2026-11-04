/*
 * Copyright (c) 2026 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "lib/vector/ob_hipvs_builder.h"

#include <limits>
#include <new>
#include <stdexcept>

#include "lib/ob_errno.h"

#ifdef __linux__
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <vector>

extern char** environ;
#endif

namespace oceanbase
{
namespace common
{
namespace obvsag
{

#ifdef __linux__
namespace
{

class HipvsFd
{
 public:
  explicit HipvsFd(int fd = -1) : fd_(fd) {}
  ~HipvsFd()
  {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }
  int get() const { return fd_; }
  HipvsFd(const HipvsFd&) = delete;
  HipvsFd& operator=(const HipvsFd&) = delete;

 private:
  int fd_;
};

class HipvsTempDir
{
 public:
  HipvsTempDir() : fd_(-1) {}
  ~HipvsTempDir()
  {
    if (fd_ >= 0) {
      // A killed helper cannot unlink its randomly named partial serialization.
      // fdopendir owns fd_; unlinkat stays within this build's private directory
      // and removes symlinks themselves rather than following their targets.
      DIR* directory = ::fdopendir(fd_);
      if (directory != nullptr) {
        while (true) {
          errno = 0;
          struct dirent* entry = ::readdir(directory);
          if (entry == nullptr) {
            if (errno == EINTR) {
              continue;
            }
            break;
          }
          if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) {
            continue;
          }
          int status;
          do {
            status = ::unlinkat(fd_, entry->d_name, 0);
          } while (status != 0 && errno == EINTR);
        }
        ::closedir(directory);
      } else {
        ::close(fd_);
      }
    }
    if (!path_.empty()) {
      ::rmdir(path_.c_str());
    }
  }
  bool create(const char* base)
  {
    std::string pattern = std::string(base) + "/seekdb-hipvs-XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    if (::mkdtemp(writable.data()) == nullptr) {
      return false;
    }
    // Move an already allocated string: cleanup remains active if later allocation fails.
    std::copy(writable.begin(), writable.end() - 1, pattern.begin());
    path_.swap(pattern);
    fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    return fd_ >= 0;
  }
  int fd() const { return fd_; }
  std::string file(const char* name) const { return path_ + "/" + name; }
  HipvsTempDir(const HipvsTempDir&) = delete;
  HipvsTempDir& operator=(const HipvsTempDir&) = delete;

 private:
  std::string path_;
  int fd_;
};

class HipvsSpawnActions
{
 public:
  HipvsSpawnActions() : status_(::posix_spawn_file_actions_init(&actions_)) {}
  ~HipvsSpawnActions()
  {
    if (status_ == 0) {
      ::posix_spawn_file_actions_destroy(&actions_);
    }
  }
  posix_spawn_file_actions_t actions_;
  int status_;
};

class HipvsSpawnAttributes
{
 public:
  HipvsSpawnAttributes() : status_(::posix_spawnattr_init(&attributes_)) {}
  ~HipvsSpawnAttributes()
  {
    if (status_ == 0) {
      ::posix_spawnattr_destroy(&attributes_);
    }
  }
  posix_spawnattr_t attributes_;
  int status_;
};

bool hipvs_size_product(size_t a, size_t b, size_t& result)
{
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
    return false;
  }
  result = a * b;
  return true;
}

std::string hipvs_system_error(const char* operation, int code)
{
  return std::string(operation) + ": " + std::strerror(code);
}

bool hipvs_integer_setting(const char* name, int default_value, int maximum, int& value,
                           std::string& error)
{
  const char* setting = std::getenv(name);
  value = default_value;
  if (setting == nullptr) {
    return true;
  }
  value = 0;
  bool valid = *setting != '\0';
  for (const char* cursor = setting; valid && *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9' || value > (maximum - (*cursor - '0')) / 10) {
      valid = false;
    } else {
      value = value * 10 + (*cursor - '0');
    }
  }
  if (!valid || value < 1 || value > maximum) {
    error = std::string(name) + " must be an integer in 1.." + std::to_string(maximum);
    return false;
  }
  return true;
}

bool hipvs_write_input(int directory, const char* name, const void* data, size_t bytes,
                       std::string& error)
{
  HipvsFd fd(::openat(directory, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
  if (fd.get() < 0) {
    error = hipvs_system_error("create helper input", errno);
    return false;
  }
  const char* cursor = static_cast<const char*>(data);
  while (bytes > 0) {
    ssize_t count = ::write(fd.get(), cursor, std::min(bytes, static_cast<size_t>(SSIZE_MAX)));
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      error = hipvs_system_error("write helper input", count == 0 ? EIO : errno);
      return false;
    }
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

std::string hipvs_read_stderr(int directory, size_t limit)
{
  HipvsFd fd(::openat(directory, "error.log", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
  struct stat info;
  if (fd.get() < 0 || ::fstat(fd.get(), &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_size <= 0) {
    return std::string();
  }
  const size_t length =
      static_cast<size_t>(std::min<off_t>(info.st_size, static_cast<off_t>(limit)));
  std::string result(length, '\0');
  size_t done = 0;
  while (done < length) {
    ssize_t count = ::pread(fd.get(), &result[done], length - done,
                            info.st_size - static_cast<off_t>(length) + static_cast<off_t>(done));
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      break;
    }
    done += static_cast<size_t>(count);
  }
  result.resize(done);
  return result;
}

int hipvs_build_impl(const float* vectors, const int64_t* ids, int64_t rows, int dim, int m,
                     int ef_construction, std::string& serialized, std::string& error,
                     std::string& diagnostics)
{
  if (vectors == nullptr || ids == nullptr || rows <= 0 || dim <= 0 || m <= 0 ||
      ef_construction <= 0) {
    error = "hipVS builder requires vectors, IDs, and positive rows/dim/m/ef-construction";
    return OB_INVALID_ARGUMENT;
  }
  size_t elements = 0;
  size_t vector_bytes = 0;
  size_t id_bytes = 0;
  if (static_cast<uint64_t>(rows) > std::numeric_limits<size_t>::max() ||
      !hipvs_size_product(static_cast<size_t>(rows), static_cast<size_t>(dim), elements) ||
      !hipvs_size_product(elements, sizeof(float), vector_bytes) ||
      !hipvs_size_product(static_cast<size_t>(rows), sizeof(int64_t), id_bytes)) {
    error = "hipVS input size overflow";
    return OB_SIZE_OVERFLOW;
  }
  const char* helper_env = std::getenv("SEEKDB_HIPVS_BUILDER");
  if (helper_env == nullptr || helper_env[0] != '/') {
    error = "SEEKDB_HIPVS_BUILDER must name an absolute executable path";
    return OB_INVALID_ARGUMENT;
  }
  const std::string helper(helper_env);
  struct stat helper_info;
  if (::stat(helper.c_str(), &helper_info) != 0 || !S_ISREG(helper_info.st_mode) ||
      ::access(helper.c_str(), X_OK) != 0) {
    error = "SEEKDB_HIPVS_BUILDER must name an executable regular file";
    return OB_INVALID_ARGUMENT;
  }
  int threads = 0;
  int timeout_seconds = 0;
  if (!hipvs_integer_setting("SEEKDB_HIPVS_THREADS", 1, 256, threads, error) ||
      !hipvs_integer_setting("SEEKDB_HIPVS_TIMEOUT_SECONDS", 600, 86400, timeout_seconds, error)) {
    return OB_INVALID_ARGUMENT;
  }
  const char* tmp_base = std::getenv("SEEKDB_HIPVS_TMPDIR");
  if (tmp_base == nullptr) {
    tmp_base = "/tmp";
  }
  if (tmp_base[0] != '/') {
    error = "SEEKDB_HIPVS_TMPDIR must name an absolute directory";
    return OB_INVALID_ARGUMENT;
  }
  HipvsTempDir temporary;
  if (!temporary.create(tmp_base)) {
    error = hipvs_system_error("create hipVS temporary directory", errno);
    return OB_IO_ERROR;
  }
  if (!hipvs_write_input(temporary.fd(), "vectors.f32", vectors, vector_bytes, error) ||
      !hipvs_write_input(temporary.fd(), "ids.i64", ids, id_bytes, error)) {
    return OB_IO_ERROR;
  }
  std::vector<std::string> arguments = {helper,
                                        "--input-f32",
                                        temporary.file("vectors.f32"),
                                        "--input-ids",
                                        temporary.file("ids.i64"),
                                        "--output",
                                        temporary.file("index.hnsw"),
                                        "--rows",
                                        std::to_string(rows),
                                        "--dim",
                                        std::to_string(dim),
                                        "--m",
                                        std::to_string(m),
                                        "--ef-construction",
                                        std::to_string(ef_construction),
                                        "--threads",
                                        std::to_string(threads)};
  std::vector<char*> argv;
  for (std::string& argument : arguments) {
    argv.push_back(&argument[0]);
  }
  argv.push_back(nullptr);
  const std::string stderr_path = temporary.file("error.log");
  HipvsSpawnActions actions;
  HipvsSpawnAttributes attributes;
  int status = actions.status_;
  if (status == 0) {
    status = attributes.status_;
  }
  // Own the helper's process group so timeout also terminates its descendants.
  if (status == 0) {
    status = ::posix_spawnattr_setpgroup(&attributes.attributes_, 0);
  }
  if (status == 0) {
    status = ::posix_spawnattr_setflags(&attributes.attributes_, POSIX_SPAWN_SETPGROUP);
  }
  if (status == 0) {
    status = ::posix_spawn_file_actions_addopen(&actions.actions_, STDIN_FILENO, "/dev/null",
                                                O_RDONLY, 0);
  }
  if (status == 0) {
    status = ::posix_spawn_file_actions_addopen(&actions.actions_, STDOUT_FILENO, "/dev/null",
                                                O_WRONLY, 0);
  }
  if (status == 0) {
    status =
        ::posix_spawn_file_actions_addopen(&actions.actions_, STDERR_FILENO, stderr_path.c_str(),
                                           O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  }
  pid_t child = -1;
  if (status == 0) {
    status = ::posix_spawn(&child, helper.c_str(), &actions.actions_, &attributes.attributes_,
                           argv.data(), environ);
  }
  if (status != 0) {
    error = hipvs_system_error("spawn hipVS builder", status);
    return OB_ERR_SYS;
  }
  int child_status = 0;
  pid_t waited = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
  while (waited == 0) {
    waited = ::waitpid(child, &child_status, WNOHANG);
    if (waited < 0 && errno == EINTR) {
      waited = 0;
    }
    if (waited != 0) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      // SIGKILL makes the timeout independent of the helper's signal handlers.
      ::kill(-child, SIGKILL);
      ::kill(child, SIGKILL);
      do {
        waited = ::waitpid(child, &child_status, 0);
      } while (waited < 0 && errno == EINTR);
      error = "hipVS builder exceeded timeout of " + std::to_string(timeout_seconds) + " seconds";
      std::string tail = hipvs_read_stderr(temporary.fd(), 4096);
      if (!tail.empty()) {
        error += ": " + tail;
      }
      return OB_TIMEOUT;
    }
    const struct timespec pause = {0, 50000000};
    ::nanosleep(&pause, nullptr);
  }
  if (waited < 0) {
    error = hipvs_system_error("wait for hipVS builder", errno);
    return OB_ERR_SYS;
  }
  if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    error = WIFEXITED(child_status)
                ? "hipVS builder exited with status " + std::to_string(WEXITSTATUS(child_status))
                : "hipVS builder terminated by signal " + std::to_string(WTERMSIG(child_status));
    std::string tail = hipvs_read_stderr(temporary.fd(), 4096);
    if (!tail.empty()) {
      error += ": " + tail;
    }
    return OB_ERR_UNEXPECTED;
  }
  diagnostics = hipvs_read_stderr(temporary.fd(), 16384);
  HipvsFd output(
      ::openat(temporary.fd(), "index.hnsw", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
  struct stat info;
  struct stat directory_info;
  if (output.get() < 0 || ::fstat(output.get(), &info) != 0 || !S_ISREG(info.st_mode) ||
      ::fstat(temporary.fd(), &directory_info) != 0 || info.st_nlink != 1 ||
      info.st_uid != directory_info.st_uid || info.st_size < 96) {
    error = "hipVS builder output must be a regular index file in its temporary directory";
    return OB_IO_ERROR;
  }
  // Permit generous graph overhead while rejecting unrelated or unbounded output.
  size_t per_row = 0;
  size_t graph_bytes = 0;
  size_t max_bytes = std::numeric_limits<size_t>::max();
  if (hipvs_size_product(static_cast<size_t>(m), 64, per_row) &&
      per_row <= std::numeric_limits<size_t>::max() - 4096 &&
      hipvs_size_product(static_cast<size_t>(rows), per_row + 4096, graph_bytes) &&
      graph_bytes <= std::numeric_limits<size_t>::max() - 1048576 &&
      vector_bytes <= std::numeric_limits<size_t>::max() - graph_bytes - 1048576) {
    max_bytes = vector_bytes + graph_bytes + 1048576;
  }
  if (static_cast<uint64_t>(info.st_size) > max_bytes ||
      static_cast<uint64_t>(info.st_size) > serialized.max_size()) {
    error = "hipVS builder output size exceeds the index size limit";
    return OB_SIZE_OVERFLOW;
  }
  std::string result(static_cast<size_t>(info.st_size), '\0');
  size_t done = 0;
  while (done < result.size()) {
    ssize_t count = ::read(output.get(), &result[done],
                           std::min(result.size() - done, static_cast<size_t>(SSIZE_MAX)));
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      error = hipvs_system_error("read hipVS builder output", count == 0 ? EIO : errno);
      return OB_IO_ERROR;
    }
    done += static_cast<size_t>(count);
  }
  serialized.swap(result);
  return OB_SUCCESS;
}

}  // namespace
#endif

int build_hipvs_hnsw(const float* vectors, const int64_t* ids, int64_t rows, int dim, int m,
                     int ef_construction, std::string& serialized, std::string& error,
                     std::string& diagnostics)
{
  serialized.clear();
  error.clear();
  diagnostics.clear();
#ifdef __linux__
  try {
    return hipvs_build_impl(vectors, ids, rows, dim, m, ef_construction, serialized, error,
                            diagnostics);
  } catch (const std::bad_alloc&) {
    error = "hipVS bridge allocation failed";
    return OB_ALLOCATE_MEMORY_FAILED;
  } catch (const std::length_error&) {
    error = "hipVS bridge size overflow";
    return OB_SIZE_OVERFLOW;
  }
#else
  (void)vectors;
  (void)ids;
  (void)rows;
  (void)dim;
  (void)m;
  (void)ef_construction;
  error = "hipVS external builder requires Linux";
  return OB_NOT_SUPPORTED;
#endif
}

}  // namespace obvsag
}  // namespace common
}  // namespace oceanbase
