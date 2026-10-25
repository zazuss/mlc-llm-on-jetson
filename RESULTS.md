# 实测数据与瓶颈分析

全部在 **Jetson Orin Nano 8GB**（L4T R39.2.1 / CUDA 13.2 / sm_87）上测，条件统一：

```bash
sudo -n jetson_clocks                      # 锁最高频点
nvpmodel -q                                # MAXN_SUPER (2)
systemd-run --user --scope -p MemoryHigh=7G -p MemoryMax=7500M   # 内存笼子
```

跑分时 EMC **3199 MHz**、GPU **1017–1020 MHz**、整机功耗 **6.4 W**。

---

## 1. 主力结论：Qwen3-1.7B q4f16_1

| 指标 | 数值 |
|---|---|
| 模型体积 | 961 MB |
| **单流 decode** | **53.7 tok/s** |
| 三轮 | 53.70 / 53.72 / 53.69 |
| 独立复跑三轮 | 53.66 / 53.70 / 53.71 |
| 加载耗时 | 9.4 – 10.5 s |
| 等效带宽 | 49.6 GB/s |
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

## 6. 复现命令

```bash
source env.sh

# 单流
python scripts/bench.py ~/models/qwen3-1.7b-q4f16_1-MLC

# 多流（先起服务，再并发压）
./scripts/06-serve.sh ~/models/qwen3-1.7b-q4f16_1-MLC
python scripts/bench.py --url http://127.0.0.1:8080 --concurrency 4
```

**注意**：单流跑分用进程内 `MLCEngine`；多流必须走 HTTP —— 同一个 `MLCEngine` 上开线程池会挂死。
