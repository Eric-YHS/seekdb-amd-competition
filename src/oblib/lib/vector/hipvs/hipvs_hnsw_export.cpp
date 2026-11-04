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
// hipVS CAGRA -> standard hnswlib stream with external signed int64 labels.
// Keep this executable isolated from VSAG: both libraries export hnswlib symbols.
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
using Clock = std::chrono::steady_clock;
static void ck(cuvsError_t s)
{
  if (s != CUVS_SUCCESS) {
    const char* message = cuvsGetLastErrorText();
    throw std::runtime_error(message == nullptr ? "hipVS operation failed" : message);
  }
}
static void hk(hipError_t s)
{
  if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
}
static double elapsed(Clock::time_point t)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
struct State {
  cuvsResources_t r{};
  cuvsCagraIndexParams_t cp{};
  cuvsCagraIndex_t ci{};
  cuvsHnswIndexParams_t hp{};
  cuvsHnswIndex_t hi{};
  void* device{};
  std::string tmp;
  void cleanup()
  {
    if (hi) {
      auto p = hi;
      hi = nullptr;
      ck(cuvsHnswIndexDestroy(p));
    }
    if (hp) {
      auto p = hp;
      hp = nullptr;
      ck(cuvsHnswIndexParamsDestroy(p));
    }
    if (ci) {
      auto p = ci;
      ci = nullptr;
      ck(cuvsCagraIndexDestroy(p));
    }
    if (cp) {
      auto p = cp;
      cp = nullptr;
      ck(cuvsCagraIndexParamsDestroy(p));
    }
    if (device) {
      auto p = device;
      device = nullptr;
      hk(hipFree(p));
    }
    if (r) {
      auto p = r;
      r = 0;
      ck(cuvsResourcesDestroy(p));
    }
  }
  ~State()
  {
    if (hi) cuvsHnswIndexDestroy(hi);
    if (hp) cuvsHnswIndexParamsDestroy(hp);
    if (ci) cuvsCagraIndexDestroy(ci);
    if (cp) cuvsCagraIndexParamsDestroy(cp);
    if (device) {
      auto ignored = hipFree(device);
      (void)ignored;
    }
    if (r) cuvsResourcesDestroy(r);
    if (!tmp.empty()) unlink(tmp.c_str());
  }
};
static uint64_t integer(const std::string& s)
{
  size_t p = 0;
  if (s.empty() || s[0] == '-') throw std::runtime_error("expected unsigned integer");
  auto n = std::stoull(s, &p);
  if (p != s.size()) throw std::runtime_error("invalid integer");
  return n;
}
static size_t product(size_t a, size_t b)
{
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::runtime_error("input size overflow");
  return a * b;
}
static void read_exact(const std::string& p, void* dst, size_t n)
{
  if (std::filesystem::file_size(p) != n)
    throw std::runtime_error("input file size mismatch: " + p);
  std::ifstream f(p, std::ios::binary);
  if (!f || !f.read((char*)dst, n)) throw std::runtime_error("input read failed: " + p);
}
template <class T>
static T pod(const uint8_t* b, size_t size, size_t off)
{
  if (off > size || sizeof(T) > size - off) throw std::runtime_error("truncated hnsw stream");
  T v;
  std::memcpy(&v, b + off, sizeof(v));
  return v;
}
struct Layout {
  uint64_t count, stride, maxlevel, entrypoint, maxm, maxm0;
  size_t bytes;
};
static Layout remap(const std::string& path, const std::vector<int64_t>& ids, size_t dim, size_t m)
{
  int fd = open(path.c_str(), O_RDWR);
  if (fd < 0) throw std::runtime_error("cannot open serialized file");
  struct stat st{};
  if (fstat(fd, &st) || st.st_size < 96) {
    close(fd);
    throw std::runtime_error("invalid serialized file");
  }
  size_t size = st.st_size;
  void* mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mem == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("mmap failed");
  }
  auto* b = (uint8_t*)mem;
  try {
    auto u64 = [&](size_t o) { return pod<uint64_t>(b, size, o); };
    auto u32 = [&](size_t o) { return pod<uint32_t>(b, size, o); };
    auto capacity = u64(8), n = u64(16), stride = u64(24), label = u64(32), vec = u64(40),
         maxm = u64(56), maxm0 = u64(64);
    auto level = pod<int32_t>(b, size, 48);
    auto entry = u32(52);
    if (u64(0) != 0 || n != ids.size() || capacity < n || n > UINT32_MAX || maxm != m ||
        maxm0 != 2 * m || u64(72) != m || vec != 4 + maxm0 * 4 || label != vec + dim * 4 ||
        stride != label + 8 || level < 0 || entry >= n)
      throw std::runtime_error("unsupported hnswlib layout");
    size_t end = 96 + product(n, stride);
    if (end > size) throw std::runtime_error("truncated base layer");
    std::vector<uint8_t> seen(n, 0);
    for (size_t row = 0; row < n; ++row) {
      size_t at = 96 + row * stride;
      auto degree = u32(at);
      if (degree > maxm0) throw std::runtime_error("invalid/deleted base node");
      for (size_t k = 0; k < degree; ++k)
        if (u32(at + 4 + k * 4) >= n) throw std::runtime_error("invalid base edge");
      auto old = u64(at + label);
      if (old >= n || seen[old]++) throw std::runtime_error("expected unique row labels");
      std::memcpy(b + at + label, &ids[old], 8);
    }
    size_t upper_stride = 4 + maxm * 4;
    int observed = 0;
    for (size_t row = 0; row < n; ++row) {
      auto bytes = u32(end);
      end += 4;
      if (bytes % upper_stride || bytes > size - end)
        throw std::runtime_error("invalid upper-layer length");
      int levels = bytes / upper_stride;
      observed = std::max(observed, levels);
      for (int l = 0; l < levels; ++l) {
        size_t at = end + l * upper_stride;
        auto degree = u32(at);
        if (degree > maxm) throw std::runtime_error("invalid upper degree");
        for (size_t k = 0; k < degree; ++k)
          if (u32(at + 4 + k * 4) >= n) throw std::runtime_error("invalid upper edge");
      }
      end += bytes;
    }
    if (end != size || observed != level)
      throw std::runtime_error("unexpected hnsw trailer/hierarchy");
    if (msync(mem, size, MS_SYNC) || fsync(fd))
      throw std::runtime_error("serialized file sync failed");
    munmap(mem, size);
    close(fd);
    return {n, stride, (uint64_t)level, entry, maxm, maxm0, size};
  } catch (...) {
    munmap(mem, size);
    close(fd);
    throw;
  }
}
int main(int argc, char** argv)
{
  auto total = Clock::now();
  try {
    uint16_t endian = 1;
    if (*(uint8_t*)&endian != 1 || sizeof(size_t) != 8)
      throw std::runtime_error("requires 64-bit little-endian platform");
    std::map<std::string, std::string> a;
    const std::unordered_set<std::string> allowed = {
        "--input-f32", "--input-ids",       "--output",  "--rows",     "--dim",
        "--m",         "--ef-construction", "--threads", "--hierarchy"};
    for (int i = 1; i < argc; i += 2) {
      std::string k = argv[i];
      if (!allowed.count(k) || i + 1 >= argc || a.count(k))
        throw std::runtime_error("invalid or duplicate CLI option: " + k);
      a[k] = argv[i + 1];
    }
    for (auto k : {"--input-f32", "--input-ids", "--output", "--rows", "--dim", "--m",
                   "--ef-construction", "--threads"})
      if (!a.count(k)) throw std::runtime_error(std::string("missing CLI option: ") + k);
    size_t n = integer(a["--rows"]), dim = integer(a["--dim"]), m = integer(a["--m"]),
           ef = integer(a["--ef-construction"]), threads = integer(a["--threads"]);
    std::string hierarchy = a.count("--hierarchy") ? a["--hierarchy"] : "CPU";
    if (hierarchy != "CPU" && hierarchy != "GPU")
      throw std::runtime_error("--hierarchy must be CPU or GPU");
    if (m < 2 || m > 256 || dim < 1 || dim > INT32_MAX || n > UINT32_MAX || ef < m ||
        ef > INT32_MAX || threads < 1 || threads > INT32_MAX)
      throw std::runtime_error("invalid rows/dim/m/ef-construction/threads");
    size_t intermediate = std::max<size_t>(4 * m, 64);
    if (n <= intermediate) throw std::runtime_error("NN_DESCENT requires rows > max(4*m,64)");
    auto output = std::filesystem::absolute(a["--output"]);
    if (output == std::filesystem::absolute(a["--input-f32"]) ||
        output == std::filesystem::absolute(a["--input-ids"]))
      throw std::runtime_error("output must differ from inputs");
    size_t floats = product(n, dim), bytes = product(floats, 4);
    std::map<std::string, double> timing;
    auto stage = Clock::now();
    std::vector<float> x(floats);
    std::vector<int64_t> ids(n);
    read_exact(a["--input-f32"], x.data(), bytes);
    read_exact(a["--input-ids"], ids.data(), product(n, 8));
    for (float v : x)
      if (!std::isfinite(v)) throw std::runtime_error("non-finite vector component");
    std::unordered_set<int64_t> unique;
    unique.reserve(n);
    for (auto id : ids)
      if (!unique.insert(id).second) throw std::runtime_error("duplicate external ID");
    timing["input_read_validate_ms"] = elapsed(stage);
    State s;
    stage = Clock::now();
    hk(hipSetDevice(0));
    hipDeviceProp_t prop{};
    hk(hipGetDeviceProperties(&prop, 0));
    ck(cuvsResourcesCreate(&s.r));
    ck(cuvsCagraIndexParamsCreate(&s.cp));
    s.cp->graph_degree = 2 * m;
    s.cp->intermediate_graph_degree = intermediate;
    s.cp->build_algo = NN_DESCENT;
    ck(cuvsCagraIndexCreate(&s.ci));
    ck(cuvsHnswIndexParamsCreate(&s.hp));
    s.hp->hierarchy = hierarchy == "CPU" ? CPU : GPU;
    s.hp->ef_construction = ef;
    s.hp->num_threads = threads;
    ck(cuvsHnswIndexCreate(&s.hi));
    timing["setup_ms"] = elapsed(stage);
    stage = Clock::now();
    hk(hipMalloc(&s.device, bytes));
    timing["device_alloc_ms"] = elapsed(stage);
    stage = Clock::now();
    hk(hipMemcpy(s.device, x.data(), bytes, hipMemcpyHostToDevice));
    hk(hipDeviceSynchronize());
    timing["h2d_sync_ms"] = elapsed(stage);
    int64_t shape[2] = {(int64_t)n, (int64_t)dim};
    DLManagedTensor tensor{};
    tensor.dl_tensor.data = s.device;
    // hipVS preserves the cuVS DLPack CUDA enum in its C API on ROCm.
    tensor.dl_tensor.device = {kDLCUDA, 0};
    tensor.dl_tensor.ndim = 2;
    tensor.dl_tensor.dtype = {kDLFloat, 32, 1};
    tensor.dl_tensor.shape = shape;
    stage = Clock::now();
    ck(cuvsCagraBuild(s.r, s.cp, &tensor, s.ci));
    timing["cagra_build_ms"] = elapsed(stage);
    stage = Clock::now();
    hk(hipDeviceSynchronize());
    timing["cagra_sync_ms"] = elapsed(stage);
    stage = Clock::now();
    ck(cuvsHnswFromCagra(s.r, s.hp, s.ci, s.hi));
    timing["hierarchy_ms"] = elapsed(stage);
    stage = Clock::now();
    hk(hipDeviceSynchronize());
    timing["hierarchy_sync_ms"] = elapsed(stage);
    std::string pattern = output.string() + ".tmp.XXXXXX";
    std::vector<char> tmp(pattern.begin(), pattern.end());
    tmp.push_back(0);
    int fd = mkstemp(tmp.data());
    if (fd < 0) throw std::runtime_error("cannot create output temporary file");
    close(fd);
    s.tmp = tmp.data();
    stage = Clock::now();
    ck(cuvsHnswSerialize(s.r, s.tmp.c_str(), s.hi));
    timing["serialize_ms"] = elapsed(stage);
    stage = Clock::now();
    hk(hipDeviceSynchronize());
    timing["serialize_sync_ms"] = elapsed(stage);
    stage = Clock::now();
    auto layout = remap(s.tmp, ids, dim, m);
    timing["label_remap_validate_fsync_ms"] = elapsed(stage);
    stage = Clock::now();
    s.cleanup();
    timing["cleanup_ms"] = elapsed(stage);
    stage = Clock::now();
    if (rename(s.tmp.c_str(), output.c_str()))
      throw std::runtime_error("atomic output rename failed");
    s.tmp.clear();
    timing["rename_ms"] = elapsed(stage);
    timing["total_ms"] = elapsed(total);
    std::cerr
        << std::setprecision(9)
        << "{\"status\":\"ok\",\"backend\":\"hipVS\",\"build_algo\":\"NN_DESCENT\",\"device\":\""
        << prop.name << "\",\"arch\":\"" << prop.gcnArchName << "\",\"hierarchy\":\"" << hierarchy
        << "\",\"rows\":" << n << ",\"dim\":" << dim << ",\"m\":" << m
        << ",\"graph_degree\":" << 2 * m << ",\"intermediate_graph_degree\":" << intermediate
        << ",\"ef_construction\":" << ef << ",\"threads\":" << threads
        << ",\"output_bytes\":" << layout.bytes << ",\"max_level\":" << layout.maxlevel
        << ",\"timing\":{";
    bool first = true;
    for (auto& t : timing) {
      if (!first) std::cerr << ',';
      first = false;
      std::cerr << '\"' << t.first << "\":" << t.second;
    }
    std::cerr << "}}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "hipvs_hnsw_export: " << e.what() << '\n';
    return 1;
  }
}
