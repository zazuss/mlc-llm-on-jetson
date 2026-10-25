# MLC-LLM on Jetson Orin

在 **Jetson Orin (L4T r39 / JetPack 7)** 上从源码编译 **MLC-LLM**，把 Qwen3-1.7B 跑到 **单流 53.7 tok/s** 的完整可复现记录。

> **English**: A reproducible recipe for building MLC-LLM from source on Jetson Orin (L4T r39.x / JetPack 7). No prebuilt aarch64 CUDA wheels exist for this platform, so source build is the only way. Includes the four build traps that cost us ~2 hours, and measured throughput: **53.7 tok/s single-stream** (Qwen3-1.7B q4f16_1) on an Orin Nano 8GB.

---

## 为什么必须自己编译

在 Orin + JetPack 7 上，MLC 的四条分发路线**全部不通**（2026-10 实测）：

| 途径 | 状态 |
|---|---|
| `mlc.ai` 官网 | 超时。域名解析到 `104.244.43.208` / `face:b00c` 段 —— DNS 污染 |
| PyPI `mlc-llm` | 404（`tvm` 同名包是金融库，别被骗；`apache-tvm` 有包但不含 Jetson CUDA） |
| `mlc.ai/wheels` 索引 | 41 个 wheel 里 `linux_aarch64` **一个都没有** |
| jetson-ai-lab PyPI | 只有 JP6 通道（cu126/128/129）。本机 JP7.2 对不上 |
| Docker Hub / jetson-containers | 镜像拉不动。`packages/llm/mlc` 仍在上游仓库（最后改动 2026-05-04），但只覆盖到 JP6，没有 JP7 对应 tag |
| GitHub Release | `v0.1.dev0` 零附件 |

**结论：没有退路，只能源码编译。** 编完确实有收益，但必须先看清条件 —— 见下方「速度取决于上下文长度」，那个 +32% 只在空 prompt 下成立。

---

## 实测结果（Orin Nano 8GB）

| 项目 | 数值 |
|---|---|
| 模型 | Qwen3-1.7B, `q4f16_1`（961 MB） |
| **单流 decode** | **53.7 tok/s**（三轮 53.70 / 53.72 / 53.69；独立复跑 53.66 / 53.70 / 53.71） |
| llama.cpp 同模型对照 | 40.8 tok/s → **MLC 快 32%** |
| 4 路并发 | 每流 25.1，**聚合 100.6** tok/s |
| 加载耗时 | 9.4 – 10.5 s |
| 等效带宽 | 49.6 GB/s |
| GPU 占用 | 平均 95% / 峰值 99% |
| EMC 占用 | 平均 46% / 峰值 52% |
| 整机功耗 | **6.4 W**（跑推理时） |
| GPU 频率 | 1017 – 1020 MHz |
| EMC 频率 | 3199 MHz（已 `jetson_clocks` 锁定） |

**这块板的天花板**：1.7B q4f16_1 单流 ≈ 54、4 路聚合 ≈ 100。**这些是空 prompt 下的数字**（prompt 约 20 token），真实对话带上下文会掉一个数量级，见下一节。瓶颈不是内存带宽，是 batch=1 时的反量化 ALU —— GPU 已经 95% 在算，而 EMC 只用了一半。详见 [RESULTS.md](RESULTS.md)。

---

## 速度取决于上下文长度（重要）

上面那个 53.7 是在 `请用中文写一段关于机器人的短文。`（16 个中文字，约 20 token）下测出来的，首字延迟 0.03 秒 —— KV cache 只有 2.2 MB，基本是空载跑分。

**同一份二进制、同一个模型，上下文一长就换了一副面孔：**

| 目标上下文 | 实际 prompt | prefill t/s | decode t/s | 首字耗时 |
|---|---|---|---|---|
| ~20（默认跑分 prompt） | 20 | （无意义，分母 0.03 s） | **53.7** | 0.03 s |
| 8 192 | 8 238 | **129.09** | **14.96** | 63.8 s |
| 32 768 | — | — | **OOM** | — |

**decode 从 53.7 掉到 14.96，掉了 72%。** 对照 llama.cpp 上 9B 的 Ternary-Bonsai-2 在 64K 下还有 36.10 t/s —— **小 5 倍的模型，慢 2.4 倍**。

### 为什么长上下文会崩

用 `tegrastats` 在 8K 那轮全程采样（500 ms 间隔）：

```
GR3D 平均 92.8%   峰值 99%      ← GPU 满载
EMC  平均  8.8%   峰值 28%      ← 带宽几乎没动
功耗 平均 13.1 W（空闲 6.4 W）
```

GPU 从 prefill 第 4 个采样点起就一直贴在 99%，带宽却只有 13–16%。**这不是带宽受限，是算力被低效 kernel 吃掉了**：`q4f16_1` 走的是"int4 反量化成 fp16 再 matmul"，反量化是纯 ALU 操作、不访存；TVM 生成的通用 GEMM 模板让 GPU 忙着"把数据摆成能算的样子"。

同一台机器上两种极端的对照：

| 引擎 / 模型 | 上下文 | 速度 | GPU | EMC | 瓶颈 |
|---|---|---|---|---|---|
| MLC Qwen3-1.7B | 8K | 14.96 | **99%** | **9%** | 算力（低效 kernel） |
| llama.cpp Bonsai 9B | 64K | 36.10 | — | ~77% 带宽 | 带宽（高效 kernel） |

llama.cpp 用手写 fused attention + DP4A 直接做低位宽点积，不需要显式反量化；TVM 是通用模板。短上下文主力是 matmul，TVM 还凑合（53.7）；长上下文主力变成 attention，而 attention 恰好是 TVM 最弱的部分。

### 32K 为什么直接 OOM

MLC 编译时会打印自己的内存账，数字很实在：

```
KV cache size: 0.11 MB per token          ← = 112 KB/token
Total without KV cache: 2254.16 MB        （参数 923.16 + 临时 buffer 1331.00）
```

按此推算 32K：

| 项 | 大小 |
|---|---|
| 参数 | 923 MB |
| KV cache（32768 × 112 KB） | 3 729 MB |
| 临时 buffer | 2 680 MB |
| **合计** | **7 332 MB** |
| 机器可用 | 7 485 MB（7.31 GiB） |

只剩 153 MB 余量，`CachedPagedKVCacheAuxDataManager` 一申请就死：

```
CUDADeviceAPI::AllocDataSpace → cudaMalloc → cudaErrorMemoryAllocation (2)
```

顺带一提，这个失败过程会伴随刷屏的 `NvMapMemAllocInternalTagged failed: error 12`，**那是伴生噪声，不是根因**。实测 `cudaMallocManaged` 与 `cudaHostAlloc` 都能轻松分配 4 GB，CMA 256 MB 与 LLM 无关（见 [jetson-orin-tuning](https://github.com/zazuss/jetson-orin-tuning)）。

### MLC 没有 KV cache 量化

这是长上下文不可行的根本原因，三条硬证据：

1. `search/issues?q=repo:mlc-ai/mlc-llm+kv+cache+quantiz` → **0 条**
2. `EngineConfig` 24 个字段里唯一沾边的是 `kv_state_kind`，类型 `Literal["kv_cache","rnn_state"]`，用于区分混合模型用哪种状态，**与量化无关**
3. 全库 `kv_cache_dtype` 只出现 3 次，全在 `model_preset.py` 的 `transformers.js_config` 里，值写死 `"float16"`（浏览器端兼容字段）

对比 llama.cpp：`-ctk q8_0 -ctv q8_0` 现成可用，KV 直接减半甚至更多。这就是为什么 9B 能跑 64K 而 1.7B 跑不了 32K —— 不是模型大小的问题，是 KV 每 token 成本的问题（MLC 112 KB vs Bonsai 的 32 KB，后者还是混合注意力架构）。

**权重量化 ≠ KV 量化**：权重从 fp16 的 ~3.4 GB 压到 961 MB（一直在生效），但 KV 是每 token 固定 112 KB，跟权重量化毫无关系。

---

## 平台前提

本仓库的脚本按下面这套验证过，**换成别的版本要重新确认**：

| 项 | 值 |
|---|---|
| 硬件 | Jetson Orin Nano 8GB（`nvpmodel -q` = MAXN_SUPER） |
| L4T | **R39.2.1**（`/etc/nv_tegra_release`） |
| 系统 | Ubuntu 24.04.4 LTS |
| CUDA | **13.2**（`/usr/local/cuda/bin/nvcc`） |
| GPU arch | **sm_87** |
| 内存 | 8 GB 标称，可用 **7.3 GiB**（见下方"内存账"） |
| CPU | 6 核 |
| Python | 3.12（venv） |
| mlc-llm | commit **8978ea9**（2026-10-06） |
| LLVM | 18（`/usr/bin/llvm-config-18`） |
| rustc | 1.99.1（rustup） |

### 内存账（8 GB 标称 ≠ 8 GB 可用）

```
标称 8 GB
 → present 8050304 kB = 7.68 GiB   （512 MB 被设备树 carveout 吃掉）
 → managed 7665092 kB = 7.31 GiB   （再 370 MB 给内核页表/镜像/保留区）
```

编译链接阶段很容易吃满。**先加 8 GB swap**，否则 OOM。另外别在编译的同时跑推理 —— 这台机器会直接把推理进程和网关一起杀掉。

---

## 快速开始

```bash
git clone https://github.com/<you>/mlc-llm-on-jetson.git
cd mlc-llm-on-jetson

# 0. 前置：LLVM 18 + rust 1.99 + 8GB swap + torch
./scripts/00-prereqs.sh

# 1. 拉源码（984 MB，约 160 s，含 TVM 子模块）
./scripts/01-fetch.sh

# 2. 编 TVM（637 个 target，这一步最久）
./scripts/02-build-tvm.sh

# 3. 编 MLC-LLM
./scripts/03-build-mlc.sh

# 4. Python 环境（venv + 依赖）
./scripts/04-python-env.sh

# 5. 转换模型（三步：gen_config → convert_weight → compile）
./scripts/05-convert-model.sh Qwen/Qwen3-1.7B qwen3

# 6. 跑分（空 prompt，测峰值速度）
source ./env.sh
python scripts/bench.py ~/models/qwen3-1.7b-q4f16_1-MLC

# 6b. 上下文扫描（测真实对话速度，务必 cgroup 包住）
systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
  python scripts/bench-ctx.py ~/models/qwen3-1.7b-q4f16_1-MLC \
    --ctx 4096 8192 --max-seq 16384 --gpu-mem-util 0.5

# 7. 起 OpenAI 兼容服务
./scripts/06-serve.sh ~/models/qwen3-1.7b-q4f16_1-MLC
```

每一步都能单独重跑。**每步跑完先看退出码**，别一口气全跑完再找错。

---

## 七个坑（省你两小时）

完整版见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)，这里是摘要。前四个在编译期，后三个在运行期 —— **后三个不踩，你的长上下文测试全是假的**。

### 坑 ① 别编 CUTLASS / FlashAttention

一开始没关，白等很久。**只有 MoE 模型才引用 CUTLASS**，Qwen3 系列是 dense，根本用不到。

```bash
-DUSE_CUTLASS=OFF
```

关掉后已编目标保留，进度立刻正常。

### 坑 ② rustc 1.75 太旧

`tokenizers` 要求 rustc ≥ 1.77。Ubuntu 24.04 自带的是 1.75 → 编译直接失败。

```bash
rustup install 1.99.1 && rustup default 1.99.1
rm -rf build/tokenizers/     # 换编译器后 cargo 拒绝复用旧产物，必须清
```

### 坑 ③ cmake 把 `/usr/bin/cargo` 写死成绝对路径

这是真凶。**改 `PATH` 完全无效** —— cmake 缓存里存的是绝对路径，必须换掉文件本身。

```bash
sudo ln -sf "$HOME/.rustup/toolchains/1.99.1-*/bin/cargo" /usr/bin/cargo
# 或者 unset CMAKE_CARGO / 直接 rm -rf build 重新 configure
```

### 坑 ④ TVM 的 LLVM JIT 要单独装包

缺了它，模型转换这一步直接跑不通：

```bash
pip install apache-tvm-ffi-orcjit==0.1.2
```

转换权重还需要 **torch**（我们用的是 2.x cu130）。

---

### 坑 ⑤ 不给 `model_lib`，上下文悄悄退回 8192

```python
MLCEngine(model=model_dir, device="cuda")   # ← 没有 model_lib
```

不带 `model_lib` 时，`MLCEngine` 会在临时目录里 **JIT 重编一份**（日志里出现 `tmpXXXX/lib.so`），**加载要 152.1 秒**，而且编出来的是**默认 8192 上下文**的版本。显式传入后加载只要 **9.7 秒**。

更坏的是它不报错 —— 你以为在测 32K，其实测的是 8K，然后被 `input length limit 8192` 拒掉，一头雾水。

```python
MLCEngine(model=model_dir, device="cuda", model_lib=f"{model_dir}/model_lib.so")
```

### 坑 ⑥ 给了 `model_lib` 还不够，`EngineConfig` 里还有一道 8192

这是最阴的一处。`EngineConfig.max_single_sequence_length` 和 `max_total_sequence_length` **默认都是 `None`**，C++ 引擎补上默认值 **8192**，跟你 `model_lib` 编进去的 32768 完全无关。限制取两者较小值：

```python
# python/mlc_llm/serve/engine_base.py
limit = min(max_single_sequence_length, max_total_sequence_length)
```

必须两个都显式设：

```python
from mlc_llm.serve import EngineConfig
cfg = EngineConfig(
    max_total_sequence_length=32768,
    max_single_sequence_length=32768,
)
engine = MLCEngine(model=model_dir, device="cuda",
                   model_lib=lib, engine_config=cfg)
```

顺带：`mlc-chat-config.json` 里的 `context_window_size` **只是元数据**，没人读它做限制。真正的上限由 `mlc_llm compile --overrides "context_window_size=N"` 决定，可用 `strings model_lib.so` 反查。

### 坑 ⑦ `gpu_memory_utilization` 默认 0.85 会当场分配失败

`gpu_memory_utilization` 默认 **0.85**，MLC 拿它反推 KV cache 容量。在 7.3 GiB 的板子上这个推算会超，引擎初始化阶段就 `cudaErrorMemoryAllocation`。手动降到 **0.5**，并把 `max_num_sequence` 设成你真正需要的并发数：

```python
cfg = EngineConfig(
    max_total_sequence_length=16384,
    max_single_sequence_length=16384,
    gpu_memory_utilization=0.5,
    max_num_sequence=1,
)
```

`scripts/bench-ctx.py` 里这三个坑都已经固化在代码里。

---

## 其它坑（不是编译，但会浪费时间）

- **MLC 的 Python 引擎不能多线程并发调用** —— 4 个线程同时调 `MLCEngine` 会挂死（240 秒零返回）。要并发走 HTTP 服务，不要线程池。
- **跑分输出别接 `| tail -N`** —— tail 要等进程结束才吐，MLC 退出那步本身容易卡，会被误判成"卡住了"。直接写文件。
- **`--opt cudagraph=1` 是空操作** —— 产出的 `model_lib_cg.so` 和 `model_lib.so` MD5 相同（`c6efe0a6…`），参数没进产物。别拿"开了图没变化"当性能结论。
- **`source env.sh` 不能省** —— `TVM_HOME` / `PYTHONPATH` / `LD_LIBRARY_PATH` 全靠它，漏了就是莫名其妙的 `ImportError` 或 `Library not found`。

---

## 已知限制

- **没有 KV cache 量化**。这是最要命的一条：权重压到 961 MB 了，但 KV 每 token 固定 **112 KB**（fp16，28 层 × 8 KV 头 × 128 head_dim × 2 × 2 B），`sliding_window_size: -1` 是纯全注意力，一点都省不下来。llama.cpp 有 `-ctk q8_0`，MLC 没有。改源码的落点已经摸清（`nn/kv_cache.py` 的 `dtype` 参数 → `op/attention.py` 里 `and q.dtype == "float16"` 这道硬限制 → `3rdparty/tvm/.../paged_kv_cache.cc` 的存储层），但真正卡的是 attention kernel 得懂 int4 KV，要写 fused dequant+attention，估算 **2–3 周**，且做完大概率打不过 llama.cpp 现成实现。
  - **免改代码的替代**：`--overrides "sliding_window_size=8192"`，KV 固定在 ~900 MB，长对话不爆内存，代价是丢远期记忆。
- **没有 `qwen3_vl` 支持**。MLC 实现了 `qwen / qwen2 / qwen2_moe / qwen2_5_vl / qwen3 / qwen3_moe / qwen35 / vision`，唯独缺 qwen3_vl。上游 PR **#3463**（"Add qwen3vl support"）3 月挂到现在，`mergeable_state=dirty`，代码是完整的但没合。**这块板上"带视觉"和"50 tok/s"不能兼得**：带视觉还能跑的最快是 Qwen3.5-2B 的 31.9 tok/s。
- **Qwen3.5-2B 上 MLC 反而比 llama.cpp 慢**（31.9 vs 34.7）。原因：它 24 层里 18 层是线性注意力（混合架构），MLC 对该结构只有 34 GB/s；而且视觉塔占 658 MB（38%），纯文本用不到。
- **单流 4bit 到头了**。q4f16_1 每 32 个权重打包成一个 uint32，线程一次只取 4 B 加一个 scale，1024 线程喂不满 102 GB/s 总线。想再快只能加 batch（8 路并发总吞吐 164 tok/s）或换更大的位宽。

---

## 目录结构

```
.
├── README.md            # 你正在看的
├── TROUBLESHOOTING.md   # 四个坑的完整排查过程
├── RESULTS.md           # 详细跑分、对照、瓶颈分析
├── env.example.sh       # 环境变量模板（复制成 env.sh 改路径）
├── .gitignore
└── scripts/
    ├── 00-prereqs.sh    # LLVM18 / rust / swap / torch
    ├── 01-fetch.sh      # git clone mlc-llm + TVM
    ├── 02-build-tvm.sh  # TVM 编译
    ├── 03-build-mlc.sh  # MLC 编译
    ├── 04-python-env.sh # venv + pip
    ├── 05-convert-model.sh
    ├── 06-serve.sh
    ├── bench.py         # 单流 / 多流跑分（空 prompt）
    └── bench-ctx.py     # 上下文扫描（三个 EngineConfig 坑已固化）
```

---

## 参考

- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) — CMU 的项目，**不是 NVIDIA 的**。NVIDIA 只是在 Jetson AI Lab 的教程和基准里用它。
- [Jetson AI Lab](https://jetson-ai-lab.com/) — 那份 "SmolLM2-1.7B = 64.5 tok/s" 的基准就是他们的，等效 64 GB/s，是 4bit 单流的天花板档。

## License

脚本和文档 MIT。编译出来的产物受各自上游许可证约束。
