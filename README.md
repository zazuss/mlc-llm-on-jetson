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
| Docker Hub / jetson-containers | 镜像拉不动；jetson-containers 的 mlc 包已下架（404） |
| GitHub Release | `v0.1.dev0` 零附件 |

**结论：没有退路，只能源码编译。** 好消息是编完确实值 —— 同一个 Qwen3-1.7B，MLC 比 llama.cpp 快 32%。

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

**这块板的天花板**：1.7B q4f16_1 单流 ≈ 54、4 路聚合 ≈ 100。瓶颈不是内存带宽，是 batch=1 时的反量化 ALU —— GPU 已经 95% 在算，而 EMC 只用了一半。详见 [RESULTS.md](RESULTS.md)。

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

# 6. 跑分
source ./env.sh
python scripts/bench.py ~/models/qwen3-1.7b-q4f16_1-MLC

# 7. 起 OpenAI 兼容服务
./scripts/06-serve.sh ~/models/qwen3-1.7b-q4f16_1-MLC
```

每一步都能单独重跑。**每步跑完先看退出码**，别一口气全跑完再找错。

---

## 四个坑（省你两小时）

完整版见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)，这里是摘要：

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

## 其它坑（不是编译，但会浪费时间）

- **MLC 的 Python 引擎不能多线程并发调用** —— 4 个线程同时调 `MLCEngine` 会挂死（240 秒零返回）。要并发走 HTTP 服务，不要线程池。
- **跑分输出别接 `| tail -N`** —— tail 要等进程结束才吐，MLC 退出那步本身容易卡，会被误判成"卡住了"。直接写文件。
- **`--opt cudagraph=1` 是空操作** —— 产出的 `model_lib_cg.so` 和 `model_lib.so` MD5 相同（`c6efe0a6…`），参数没进产物。别拿"开了图没变化"当性能结论。
- **`source env.sh` 不能省** —— `TVM_HOME` / `PYTHONPATH` / `LD_LIBRARY_PATH` 全靠它，漏了就是莫名其妙的 `ImportError` 或 `Library not found`。

---

## 已知限制

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
    └── bench.py         # 单流 / 多流跑分
```

---

## 参考

- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) — CMU 的项目，**不是 NVIDIA 的**。NVIDIA 只是在 Jetson AI Lab 的教程和基准里用它。
- [Jetson AI Lab](https://jetson-ai-lab.com/) — 那份 "SmolLM2-1.7B = 64.5 tok/s" 的基准就是他们的，等效 64 GB/s，是 4bit 单流的天花板档。

## License

脚本和文档 MIT。编译出来的产物受各自上游许可证约束。
