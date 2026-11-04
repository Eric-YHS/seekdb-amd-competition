/*
 * Copyright (c) 2025 OceanBase.
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
#include "lib/ob_errno.h"
#include "lib/vector/ob_hipvs_builder.h"
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <signal.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace oceanbase::common;
using oceanbase::common::obvsag::build_hipvs_hnsw;
int count = 0;
void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

int main(int argc, char **argv) {
  try {
    check(argc == 2, "fixture root required");
    fs::path root(argv[1]);
    fs::path helper = root / "helper with spaces.py";
    fs::path temporary = root / "temporary with spaces";
    fs::create_directories(temporary);
    setenv("SEEKDB_HIPVS_BUILDER", helper.c_str(), 1);
    setenv("SEEKDB_HIPVS_TMPDIR", temporary.c_str(), 1);
    unsetenv("SEEKDB_HIPVS_THREADS");
    unsetenv("SEEKDB_HIPVS_TIMEOUT_SECONDS");
    float vectors[] = {1, 2, 3, 4};
    int64_t ids[] = {100, 200};
    std::string serialized, error, diagnostics;
    auto invoke = [&](const std::string &name, int expected, int64_t rows = 2,
                      int dim = 2, const float *data = nullptr,
                      bool null_data = false) {
      const float *chosen =
          null_data ? nullptr : (data == nullptr ? vectors : data);
      int ret = build_hipvs_hnsw(chosen, ids, rows, dim, 16, 100, serialized,
                                 error, diagnostics);
      check(ret == expected,
            name + " returned " + std::to_string(ret) + ": " + error);
      check(fs::is_empty(temporary), name + " leaked temporary files");
      if (ret != OB_SUCCESS) {
        check(serialized.empty(), name + " retained serialized data on error");
        check(!error.empty(), name + " lost error");
      }
      ++count;
      std::cout << "PASS " << name << '\n';
    };
    invoke("success paths with spaces and default threads", OB_SUCCESS);
    check(serialized == std::string(128, 'H') && error.empty(),
          "success bytes/error mismatch");
    check(diagnostics.find("cpu-only") != std::string::npos,
          "missing success stderr diagnostics");
    setenv("SEEKDB_HIPVS_THREADS", "256", 1);
    setenv("SEEKDB_TEST_EXPECT_THREADS", "256", 1);
    invoke("threads upper boundary", OB_SUCCESS);
    unsetenv("SEEKDB_HIPVS_THREADS");
    unsetenv("SEEKDB_TEST_EXPECT_THREADS");
    for (const char *mode : {"nonzero", "signal", "symlink", "fifo",
                             "truncated", "oversized", "missing_output"}) {
      setenv("SEEKDB_TEST_MODE", mode, 1);
      const int expected =
          std::string(mode) == "nonzero" || std::string(mode) == "signal"
              ? OB_ERR_UNEXPECTED
          : std::string(mode) == "oversized" ? OB_SIZE_OVERFLOW
                                             : OB_IO_ERROR;
      invoke(mode, expected);
      if (std::string(mode) == "nonzero") {
        check(error.find("17") != std::string::npos &&
                  error.find("KEEP_STDERR_TAIL") != std::string::npos,
              "nonzero exit lost status/stderr");
        check(error.find("DISCARD_PREFIX") == std::string::npos &&
                  error.size() < 4300,
              "stderr tail not bounded");
      }
    }
    unsetenv("SEEKDB_TEST_MODE");
    setenv("SEEKDB_HIPVS_BUILDER", (root / "does not exist").c_str(), 1);
    invoke("missing helper", OB_INVALID_ARGUMENT);
    setenv("SEEKDB_HIPVS_BUILDER", "relative helper", 1);
    invoke("relative helper", OB_INVALID_ARGUMENT);
    setenv("SEEKDB_HIPVS_BUILDER", helper.c_str(), 1);
    for (const char *invalid : {"", "0", "257", "-1", "+1", " 1", "1x",
                                "99999999999999999999999999999"}) {
      setenv("SEEKDB_HIPVS_THREADS", invalid, 1);
      invoke(std::string("invalid threads '") + invalid + "'",
             OB_INVALID_ARGUMENT);
    }
    unsetenv("SEEKDB_HIPVS_THREADS");
    for (const char *invalid :
         {"", "0", "86401", "-1", "+1", "1.5", "99999999999999999999999999"}) {
      setenv("SEEKDB_HIPVS_TIMEOUT_SECONDS", invalid, 1);
      invoke(std::string("invalid timeout '") + invalid + "'",
             OB_INVALID_ARGUMENT);
    }
    unsetenv("SEEKDB_HIPVS_TIMEOUT_SECONDS");
    invoke("null vectors", OB_INVALID_ARGUMENT, 2, 2, nullptr, true);
    invoke("zero rows", OB_INVALID_ARGUMENT, 0);
    invoke("zero dimension", OB_INVALID_ARGUMENT, 2, 0);
    invoke("input size overflow", OB_SIZE_OVERFLOW,
           std::numeric_limits<int64_t>::max(), 2147483647);
    setenv("SEEKDB_HIPVS_TIMEOUT_SECONDS", "86400", 1);
    invoke("timeout upper boundary", OB_SUCCESS);
    setenv("SEEKDB_HIPVS_TIMEOUT_SECONDS", "1", 1);
    setenv("SEEKDB_TEST_MODE", "timeout", 1);
    fs::path pid_file = root / "timeout.pid";
    setenv("SEEKDB_TEST_PIDFILE", pid_file.c_str(), 1);
    auto started = std::chrono::steady_clock::now();
    invoke("timeout kills and reaps helper", OB_TIMEOUT);
    double elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - started)
                         .count();
    check(elapsed >= .9 && elapsed < 5,
          "timeout duration wrong: " + std::to_string(elapsed));
    check(error.find("TIMEOUT_STDERR") != std::string::npos,
          "timeout lost stderr");
    int pid = 0;
    std::ifstream(pid_file) >> pid;
    check(pid > 0 && kill(pid, 0) == -1 && errno == ESRCH,
          "timeout child is still live or unreaped");
    int status = 0;
    check(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD,
          "bridge left an unreaped child");
    std::cout << "TIMEOUT_ELAPSED_SECONDS " << elapsed << '\n';
    std::cout << "ALL_PASS " << count << '\n';
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "FAIL " << exception.what() << '\n';
    return 1;
  }
}
