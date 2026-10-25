# 实测数据与瓶颈分析

全部在 **Jetson Orin Nano 8GB**（L4T R39.2.1 / CUDA 13.2 / sm_87）上测，条件统一：

```bash
sudo -n jetson_clocks                      # 锁最高频点
nvpmodel -q                                # MAXN_SUPER (2)
systemd-run --user --scope -p MemoryHigh=7G -p MemoryMax=7500M   # 内存笼子
```

跑分时 EMC **3199 MHz**、GPU **1017–1020 MHz**、整机功耗 **6.4 W**。

---

## 1. 主力结论：Qwen3-1.7B q4f16_1（空 prompt）

> **这一节全部是「空 prompt」跑分**：prompt 是 `请用中文写一段关于机器人的短文。`（49 字节 / 16 个汉字 / 约 20 token），KV cache 只有 2.2 MB。它测的是**峰值能力**，不是真实对话速度。真实对话请直接跳到 [第 7 节](#7-上下文长度对速度的影响这是真实使用场景)。

| 指标 | 数值 |
|---|---|
| 模型体积 | 961 MB |
| **单流 decode** | **53.7 tok/s** |
| 三轮 | 53.70 / 53.72 / 53.69 |
| 独立复跑三轮 | 53.66 / 53.70 / 53.71 |
| 加载耗时 | 9.4 – 10.5 s |
| 等效带宽 | 49.6 GB/s |
| 首字延迟 | 0.03 s（prompt 太短，prefill 无法测，算出来 37 t/s 是噪声） |
| GPU 平均 / 峰值 | 95% / 99% |
| EMC 平均 / 峰值 | 46% / 52% |

**六轮数据散布在 0.06 tok/s 内**，重复性好。

### 对照 llama.cpp

| 引擎 | 同模型单流 |
|---|---|
| MLC-LLM | **53.7 tok/s** |
| llama.cpp (Q4_K_M) | 40.8 tok/s |
| 差值 | **+32%** |

### 4 路并发（走 HTTP 服务）

| 项 | 数值 |
|---|---|
| 单流基线 | 53.4 |
| 每流 | 25.1 |
| **聚合** | **100.6** |
| 聚合倍数 | 1.88× |
| 并行期间 GPU | 98% |
| 并行期间 EMC | 掉到 20% |

一份权重摊给 4 条流，EMC 占用反而降 —— 说明带宽不是瓶颈。

**这块板 1.7B q4f16_1 的天花板 ≈ 单流 54 / 4 路聚合 100。**

---

## 2. 为什么只有 49.6 GB/s 而不是 100 GB/s

Orin Nano 的 LPDDR5 理论带宽 ~102 GB/s。只用了一半，原因三条（按贡献排序）：

### ① batch=1 的算术强度太低

每个权重每 token 只读一次，算术强度约 **4 FLOP/B**。单条流永远喂不满总线，只能靠并发请求（多 batch）去堆利用率 —— 这也是 4 路并发能把聚合推到 100 的原因。

### ② 4-bit 分组粒度太小

`q4f16_1` 每 **32 个权重**打包成一个 uint32。线程一次只取 **4 B + 一个 scale**。

要顶满 102 GB/s 需要约 **70 KB 在飞**，而 1024 线程 × 4 B 远远不够。

### ③ 反量化 ALU 占比高

移位 / 掩码 / 转 f16 / FMA 一串操作下来，**ALU 先饱和**，还没轮到带宽。

**反证**：自写的 fp16 GEMV 在同样 shape 下能跑 **99–101 GB/s** —— 说明总线本身没问题，是 4bit 反量化这条路的开销。

**8K 上下文下的进一步验证**（本文第 6.1 节）：GPU **99%** / EMC **8.8%** / 功耗 13.1 W。GPU 满载但带宽个位数，把「反量化 ALU 先饱和」这条坐实了 —— 此时瓶颈已经不是算术强度，而是 kernel 本身在浪费算力。

**天花板参照**：NVIDIA 官方在 Orin Nano 上用 MLC 跑 SmolLM2-1.7B = **64.5 tok/s**，等效 **64 GB/s**，即单流 4bit 的天花板档。这条路最多再挖 20–30%。

---

## 3. 反面案例：Qwen3.5-2B 上 MLC 反而更慢

| 引擎 | Qwen3.5-2B |
|---|---|
| MLC-LLM | **31.9 tok/s** |
| llama.cpp | **34.7 tok/s** |

MLC 输了。两个原因：

1. **视觉塔占 658 MB（38%）** —— `gen_config` 把它识别成多模态模型，纯文本对话时这些权重用不到，但加载和地址空间都吃了。
2. **24 层里 18 层是线性注意力**（混合架构）—— MLC 对该结构的 kernel 只有 **34 GB/s**。

**教训**：MLC 对稠密 Transformer 优化得好，对混合/线性注意力架构不一定。选模型前先确认 MLC 对该 `model_type` 的 kernel 质量。

---

## 4. CUDA Graph 没生效

```bash
$ md5sum model_lib.so model_lib_cg.so
c6efe0a6...  model_lib.so
c6efe0a6...  model_lib_cg.so      # 完全相同
```

`--opt cudagraph=1` **没有进产物**。所以：

- ❌ 不能拿"开了图性能没变化"当"kernel 下发开销不是瓶颈"的证据
- ✅ 真正的证据是 **GPU 95% / EMC 46%** —— GPU 一直在算

---

## 5. 视觉与速度不能兼得

MLC 实现了这些 `model_type`：

```
qwen / qwen2 / qwen2_moe / qwen2_5_vl / qwen3 / qwen3_moe / qwen35 / vision
```

**唯独没有 `qwen3_vl`。** 上游 PR [#3463](https://github.com/mlc-ai/mlc-llm/pull/3463)（"Add qwen3vl support"）3 月挂到现在，0 评论，`mergeable_state=dirty`，27 个文件的代码是完整的但没合；相关 bug #3444。

**这块板上带视觉还能跑的最快是 Qwen3.5-2B 的 31.9 tok/s。**

两条路（未做最终选择）：

1. 双模型按需切：文本 Qwen3-1.7B 53.7 / 看图切 Qwen3.5-2B 31.9
2. 单模型保视觉，接受 31.9、放弃 50

---

## 6. 上下文长度对速度的影响（这是真实使用场景）

前面的 53.7 tok/s 会误导人。同一份二进制、同一个模型，prompt 一长就崩：

| 目标 prompt | 实际 token | prefill t/s | decode t/s | 首字耗时 |
|---|---|---|---|---|
| ~20（默认跑分 prompt） | 20 | 噪声 | **53.70** | 0.03 s |
| 8 192 | 8 238 | **129.09** | **14.96** | 63.8 s |
| 32 768 | — | — | **OOM** | — |

- 8K 档跑了两轮，prefill **129.12 / 129.09**、decode **14.98 / 14.96**，复现性很好。
- **decode 从 53.7 → 14.96，掉 72%**。而对照组 llama.cpp 上的 Ternary-Bonsai-2 **9B** 在 64K 下还有 **36.10 t/s** —— 模型大 5 倍，速度快 2.4 倍。
- 交叉点在几百 token 量级。Hermes 这类系统提示词本身 2000+ token 的场景，一开口就落在十几这一档。

### 6.1 tegrastats 采样：GPU 满载，带宽几乎闲置

8K 那轮全程 `sudo tegrastats --interval 500` 采样，按 `GR3D_FREQ` / `EMC_FREQ` / `RAM` 解析：

```
活动窗口 61 个采样点（约 30 秒）
GR3D 平均 92.8%   峰值 99%
EMC  平均  8.8%   峰值 28%
CPU  平均     —
RAM  峰值 5814 MB
功耗 平均 13098 mW（空闲 6398 mW）
```

prefill 期间逐点看：从第 4 个采样点起 GPU 就贴在 **99%**，EMC 只有 **13–16%**，功耗 8.3 → 17 W。

**诊断：算力受限，而且是被低效 kernel 吃掉的算力。** GPU 忙着算，但不怎么访存 —— 典型的 ALU 饱和特征。`q4f16_1` 的路线是「int4 反量化成 fp16 → 做 fp16 matmul」，反量化是纯 ALU 操作、零访存。

同一台机器上的两个极端：

| 引擎 / 模型 | 上下文 | decode | GPU | EMC | KV / token | 瓶颈 |
|---|---|---|---|---|---|---|
| **MLC** Qwen3-1.7B | 8K | 14.96 | **99%** | **9%** | 112 KB | 算力（低效 kernel） |
| **llama.cpp** Bonsai 9B | 64K | 36.10 | — | ~77%（75.6 / 98 GB/s） | 32 KB | 带宽（已近物理极限） |

反过来算带宽：MLC 8K 时 KV 0.9 GB + 权重 0.96 GB = 1.86 GB × 15 t/s = **28 GB/s**，只用到实测 `cudaMemcpy` 上限 **98 GB/s** 的 **28%**。Bonsai 64K 是 2.1 GB × 36 = **75.6 GB/s = 77%**。**MLC 慢不是因为 KV 大（0.9 GB 比 Bonsai 的 2.1 GB 还小），是它没把带宽用起来。**

差异来源：llama.cpp 用手写 fused attention + DP4A 直接做低位宽点积，不需要显式反量化；TVM 走通用 GEMM 模板。短上下文主力是 matmul，TVM 勉强够用；长上下文主力变成 attention，那是 TVM 最弱的一环。

这也解释了 4 路并发为什么能到 100.6 —— 多条流一起把带宽堆上去，掩盖了单流 kernel 的低效。

### 6.2 32K 为什么直接 OOM

MLC 编译时自己打印内存账：

```
KV cache size: 0.11 MB per token
Total memory usage without KV cache: 2254.16 MB
  ├─ Parameters: 923.16 MB
  └─ Temporary buffer: 1331.00 MB
Total memory usage with a 4K KV cache: 2702.16 MB
```

按 112 KB/token 推 32K：

| 项 | 大小 |
|---|---|
| 参数 | 923 MB |
| KV cache（32 768 × 112 KB） | 3 729 MB |
| 临时 buffer（运行时实测，比编译期估的 1331 MB 大得多） | **2 680 MB** |
| **合计** | **7 332 MB** |
| 机器可用 | 7 485 MB（7.31 GiB） |
| **余量** | **153 MB** |

`CachedPagedKVCacheAuxDataManager` 在初始化时申请 aux data，一申请就死：

```
CachedPagedKVCacheAuxDataManager::CachedPagedKVCacheAuxDataManager(...)
  → tvm::runtime::Tensor::Empty(...)
  → cuda_device_api.cc:173 CUDADeviceAPI::AllocDataSpace(...)
  → TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&ret, nbytes));
RuntimeError: CUDA Runtime Error: cudaErrorMemoryAllocation (2): out of memory
```

**两个曾经把人带偏的现象，都不是根因：**

1. `NvMapMemAllocInternalTagged failed: error 12` / `NvMapMemHandleAlloc failed: error 12` 只是伴生噪声。当时人以为 CMA（`CmaTotal` 只有 256 MB）是瓶颈，实测 `cudaMallocManaged` 分配 **4 GB** 成功、`cudaHostAlloc` 1 GB 也成功 —— CMA 与 CUDA 分配完全无关。
2. 系统当时还显示 3801 MB 空闲 —— 那是系统级空闲，不是**单次连续分配**能力。这台机器单次 `cudaMalloc` 上限约 **4.85 GB**（5.0 / 6.5 GB 实测 OOM）。

**结论：32K 在这台机器上不是参数问题，是内存账算不平。** 64K 更是差 7 GB 以上，不用试。

### 6.3 MLC 没有 KV cache 量化

这是长上下文不可行的根本原因。三条硬证据：

| 检查 | 结果 |
|---|---|
| `search/issues?q=repo:mlc-ai/mlc-llm+kv+cache+quantiz` | **total_count 0** |
| `EngineConfig` 24 个字段 | 唯一沾边的是 `kv_state_kind`，`Literal["kv_cache","rnn_state"]`，用于混合模型区分状态类型，**与量化无关** |
| 全库 `kv_cache_dtype`（py/cc/h/cu，排除 3rdparty） | 只 3 处，全在 `model_preset.py` 的 `transformers.js_config`，值写死 `"float16"`，是浏览器端兼容字段 |

**权重量化 ≠ KV 量化**：权重是 fp16 的 ~3.4 GB 压成 961 MB（一直在生效），KV 是每 token 固定 112 KB（fp16 × 全注意力 `sliding_window_size: -1`），两者互不相干。

对照 llama.cpp 有 `-ctk q8_0 -ctv q8_0`，KV 直接砍半。**所以 9B 能跑 64K、1.7B 跑不了 32K —— 差的是 KV 每 token 成本，不是模型大小。**

#### 想自己改出 KV 量化，三处落点

| 层 | 位置 | 改什么 |
|---|---|---|
| Python API | `python/mlc_llm/nn/kv_cache.py` | `dtype: str` 参数往下传给 `mlc.create_paged_kv_cache_generic` |
| 调度 | `python/mlc_llm/op/attention.py` | `and q.dtype == "float16"` —— 这行把 flashinfer 路径锁死 fp16，**最硬的一行** |
| 存储 | `3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc` | 存储布局 / page 打包 / 读写 |

真正的难点不是这三处，而是 **attention kernel 得懂 int4 KV**：要么插一个 dequant 层（掉速），要么写 fused dequant+attention CUDA kernel（等价 llama.cpp 调了两年的 `fattn-vec`）。估计 **2–3 周**，且做完大概率还是打不过 llama.cpp 现成实现。

**免改代码的替代方案：**

```bash
mlc_llm compile . --overrides "sliding_window_size=8192" -o model_lib-slide8k.so
```

KV 固定约 **900 MB**，长对话不会爆内存，代价是只保留最近 8K token 的记忆。

**唯一没验证的口子**：日志里出现过 `flashinfer=1`，flashinfer 的 fp8 KV 在 sm_87 上能不能用没测。

---

## 7. 复现命令

```bash
source env.sh

# 单流（空 prompt，峰值）
python scripts/bench.py ~/models/qwen3-1.7b-q4f16_1-MLC

# 上下文扫描（真实对话速度，必须 cgroup 包住）
systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
  python scripts/bench-ctx.py ~/models/qwen3-1.7b-q4f16_1-MLC \
    --ctx 2048 4096 8192 --max-seq 16384 --gpu-mem-util 0.5

# 边跑边采硬件利用率（另开终端）
sudo tegrastats --interval 500

# 多流（先起服务，再并发压）
./scripts/06-serve.sh ~/models/qwen3-1.7b-q4f16_1-MLC
python scripts/bench.py --url http://127.0.0.1:8080 --concurrency 4
```

**注意**：单流跑分用进程内 `MLCEngine`；多流必须走 HTTP —— 同一个 `MLCEngine` 上开线程池会挂死。
