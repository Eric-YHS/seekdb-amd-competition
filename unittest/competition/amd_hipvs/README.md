# hipVS HNSW 构建与性能验证

seekdb 收集完整 snapshot 的向量与 VID，在序列化和发布索引前调用 hipVS 构建 HNSW，再载入 VSAG 完成查询与持久化。GPU 构建器作为独立进程运行，隔离 hipVS 与 VSAG 的 hnswlib 符号。增量写入沿用 VSAG 路径。

当前 GPU 路径支持 FP32、L2、普通 HNSW snapshot。NN-descent 在 GPU 上构建底层图，CPU 构建上层层级。构建器输出保留原始 int64 VID，并校验图边、层级和序列化布局。

## 编译

按项目构建流程准备依赖并编译 seekdb。在安装 ROCm 和 hipVS 开发库的 AMD Linux 环境中编译构建器：

```bash
bash build.sh release --make -j8
cmake -S src/oblib/lib/vector/hipvs -B build_hipvs -DCMAKE_BUILD_TYPE=Release
cmake --build build_hipvs -j8
```

`HIPVS_ROOT` 和 `ROCM_ROOT` 默认分别为 `/opt/hipvs`、`/opt/rocm`，可通过 CMake 参数指定安装目录。

## 启用 GPU 构建

启动 seekdb 前配置：

```bash
export SEEKDB_HIPVS_BUILD=1
export SEEKDB_HIPVS_BUILDER="$(pwd)/build_hipvs/seekdb_hipvs_build"
export SEEKDB_HIPVS_THREADS=16
export SEEKDB_HIPVS_TMPDIR=/var/tmp
export SEEKDB_HIPVS_TIMEOUT_SECONDS=600
```

| 配置 | 含义 |
| --- | --- |
| `SEEKDB_HIPVS_BUILD` | `1` 启用 snapshot GPU 构建；`0` 或省略时使用原有 CPU 构建 |
| `SEEKDB_HIPVS_BUILDER` | 构建器可执行文件的绝对路径 |
| `SEEKDB_HIPVS_THREADS` | 上层层级构建线程数，范围 1–256，默认 1 |
| `SEEKDB_HIPVS_TMPDIR` | 构建输入和产物的临时目录，默认 `/tmp` |
| `SEEKDB_HIPVS_TIMEOUT_SECONDS` | 构建进程超时，范围 1–86400 秒，默认 600 |

数据库目录使用支持文件预分配的本地文件系统。AMD 参考镜像中的 `/workspace` 用于持久化源码和实验结果，数据库文件放在容器本地目录。

当向量数不超过 `max(4*M,64)` 时，seekdb 使用 CPU 构建并记录 `hipVS small snapshot CPU fallback`。GPU 性能测量要求出现真实 GPU 构建日志。大规模构建失败或超时会返回错误，并保留待构建数据供失败处理；索引完成导入后才进入发布流程。

## 端到端测量

在同一 AMD 主机上运行：

```bash
python unittest/competition/amd_hipvs/benchmark.py \
  --binary "$(pwd)/build_release/src/observer/seekdb" \
  --helper "$(pwd)/build_hipvs/seekdb_hipvs_build" \
  --output /workspace/hipvs-results/random-100k \
  --sizes 100000 --rounds 3 --queries 100 --dim 128 --seed 42 \
  --m 16 --ef-construction 200 --ef-search 1000
```

参考镜像的 10000 行数据可通过以下参数载入：

```bash
python unittest/competition/amd_hipvs/benchmark.py \
  --binary "$(pwd)/build_release/src/observer/seekdb" \
  --helper "$(pwd)/build_hipvs/seekdb_hipvs_build" \
  --output /workspace/hipvs-results/reference-10k \
  --sizes 10000 --rounds 3 --queries 100 --dim 128 \
  --base-f32 /seekdb_workspace/datasets/base.f32 \
  --queries-f32 /seekdb_workspace/datasets/query.f32
```

输入文件采用行优先、little-endian float32 原始矩阵，列数由 `--dim` 指定。程序保存输入文件 SHA256，并按本轮向量集合重新计算精确近邻真值。

程序在本地文件系统创建两个独立实例，使用 2881、2882 端口。CPU/GPU 使用相同二进制、资源配置、数据、索引参数和查询集，交替执行至少三轮全新索引构建。计时从发送 `CREATE VECTOR INDEX` 开始，到索引可执行首个近邻查询为止，覆盖扫描、传输、同步、格式转换和索引发布。数据生成、原始数据导入、实例启动和质量检查分别记录。

`results.json` 保存每轮耗时、中位数加速比、Recall@10、文档 ID 与距离检查、执行计划、GPU 构建日志、二进制 SHA256 和环境配置。程序还验证实例重启后的查询结果，以及新增、更新和删除。通过 `--min-recall` 指定质量门槛，使用比赛统一配置判断结果。

## 回归检查

进程桥测试在 Linux 上使用 CPU fixture 验证成功、错误和超时路径：

```bash
python unittest/competition/amd_hipvs/test_process_bridge.py
python unittest/competition/amd_hipvs/test_process_bridge.py --sanitizers
python unittest/competition/amd_hipvs/benchmark.py --self-check
```

构建器依赖 hipVS 的 64 位 hnswlib 序列化布局。升级 hipVS 或 VSAG 时，重新执行导入、查询、保存和恢复测试。
