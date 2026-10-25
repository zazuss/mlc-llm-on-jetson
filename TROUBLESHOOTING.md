# 排错记录

在 Jetson Orin（L4T r39.2.1 / CUDA 13.2 / sm_87）上从源码编 MLC-LLM 踩到的坑，按遇到顺序。

---

## 坑 ① 编 CUTLASS / FlashAttention 白等

**症状**：TVM 编译进度慢得离谱，8 个 target/分钟，看起来要跑一整晚。

**原因**：`USE_CUTLASS` 默认或误开时，会连带编 FlashAttention 那套模板。**只有 MoE 模型会引用 CUTLASS**，Qwen3 系列是 dense 结构，这些 kernel 编完也用不到。

**修复**：

```bash
# TVM
cmake .. \
  -DUSE_CUDA=ON \
  -DUSE_CUTLASS=OFF \
  -DUSE_CUDNN=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DUSE_LLVM=/usr/bin/llvm-config-18 \
  -DUSE_GRAPH_EXECUTOR=ON \
  -DUSE_RPC=ON \
  -DUSE_OPENMP=OFF \
  -DUSE_RANDOM=ON \
  -DUSE_SORT=ON \
  -DCMAKE_BUILD_TYPE=Release

# MLC-LLM
cmake .. \
  -DUSE_CUDA=ON \
  -DUSE_CUTLASS=OFF \
  -DUSE_CUDNN=OFF \
  -DUSE_CURAND=OFF \
  -DUSE_METAL=OFF \
  -DUSE_ROCM=OFF \
  -DUSE_LLVM=OFF \
  -DUSE_THREADS=ON \
  -DUSE_OPENMP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release
```

关掉之后**不用重头编** —— 已编好的目标文件会保留，速度立刻正常。

> 注意 MLC 侧 `USE_LLVM=OFF` 是对的：LLVM 由 TVM 那边负责，MLC 只是薄封装。

**实测**：TVM 共 **637 个 target**，5 线程，全程 0 错误。

---

## 坑 ② rustc 1.75 太旧

**症状**：编译到 `tokenizers` 时失败，报错要求更新的编译器。

**原因**：`tokenizers` 需要 **rustc ≥ 1.77**，Ubuntu 24.04 apt 装的 rustc 是 **1.75**。

**修复**：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup install 1.99.1
rustup default 1.99.1
rustc --version   # 确认 ≥ 1.77

# 关键：换编译器后 cargo 拒绝复用旧产物
rm -rf build/tokenizers/
```

**踩点**：只装新 rust 不清 `build/tokenizers/` 会继续报同样的错，看起来像没生效。

---

## 坑 ③ cmake 把 `/usr/bin/cargo` 写死成绝对路径（真凶）

**症状**：rust 明明升到 1.99.1 了，`rustc --version` 也对，但 cmake 编译 tokenizers 还是失败，还是那个"编译器太旧"的错。

**原因**：cmake 在 configure 阶段把 `CMAKE_CARGO` 解析成**绝对路径** `/usr/bin/cargo` 存进 `CMakeCache.txt`，而 `/usr/bin/cargo` 指向的是 apt 装的 1.75。

**所以 `export PATH=$HOME/.cargo/bin:$PATH` 完全无效** —— cmake 根本不看 PATH。

**修复**（换掉文件本身）：

```bash
# 确认当前指向
ls -la /usr/bin/cargo
grep CMAKE_CARGO build/CMakeCache.txt

# 换成 rustup 的
sudo ln -sf "$(rustup which cargo)" /usr/bin/cargo
/usr/bin/cargo --version    # 必须显示 1.99.1

# 保险：清掉 tokenizers 的旧产物重编
rm -rf build/tokenizers/
```

**教训**：遇到"环境变量改了但构建系统没反应"，第一件事是 `grep XXX CMakeCache.txt` 看它是不是早就把绝对路径记住了。

---

## 坑 ④ TVM 的 LLVM JIT 要单独装包

**症状**：编译全过，`import mlc_llm` 正常，但 `mlc_llm convert_weight` 或 `compile` 直接崩。

**原因**：TVM 的 LLVM JIT 后端在 Python 侧要一个独立包，`pip install -e python` 不带它。

**修复**：

```bash
pip install apache-tvm-ffi-orcjit==0.1.2
```

**版本要对齐**：我们用的是 `apache-tvm-ffi==0.1.15` + `apache-tvm-ffi-orcjit==0.1.2`。版本错配同样会崩。

另外**转换权重需要 torch**（我们装的是 2.x + cu130）—— 它是转换脚本的依赖，不装会在 `convert_weight` 阶段才报错，白等半天。

---

## 运行期三个坑（不踩，长上下文测试全是假的）

这三个坑的共同特征：**不报错，只是悄悄用 8192**。它们的调试成本远高于编译期那些，因为看起来完全正常。

### 坑 ⑤ 不给 `model_lib` → JIT 重编 + 退回 8192

**症状**：第一次跑 bench，加载阶段卡了 **152.1 秒**；之后想测 32K 被 `input length limit 8192` 拒掉，但 `mlc-chat-config.json` 里明明写着 `context_window_size: 40960`。

**原因**：`MLCEngine` 不带 `model_lib` 时，检测到没有预编译产物，就在临时目录里 **JIT 重编一份**（日志里出现 `tmpfx_fbi8p/lib.so` 这种路径），用的是默认配置，上下文 8192。

**修复**：

```python
MLCEngine(model=model_dir, device="cuda",
          model_lib=f"{model_dir}/model_lib.so")   # 加载降到 9.7 s
```

**教训**：看到 `tmpXXXXXX` 出现在日志里，就是它在偷偷重编。

### 坑 ⑥ 给了 `model_lib` 也还不够：`EngineConfig` 里还有一道 8192

**症状**：`model_lib` 已经是用 `--overrides "context_window_size=32768"` 编出来的（`strings model_lib.so` 能查到 32768），加载也正常，但请求 32K 仍然被拒：`input length limit 8192`。

**原因**：`EngineConfig.max_single_sequence_length` 和 `max_total_sequence_length` **默认都是 `None`**，C++ 引擎补上默认值 **8192**。它跟 model_lib 编进去的值毫无关系，限制取两者较小值：

```python
# python/mlc_llm/serve/engine_base.py
limit = min(max_single_sequence_length, max_total_sequence_length)
```

**修复**：两个字段都显式设。

```python
from mlc_llm.serve import EngineConfig
cfg = EngineConfig(max_single_sequence_length=32768,
                   max_total_sequence_length=32768)
MLCEngine(model=model_dir, device="cuda", model_lib=lib, engine_config=cfg)
```

**附带事实**：`mlc-chat-config.json` 里的 `context_window_size` 只是元数据，没有任何代码读它来限制长度。真正的上限由 `mlc_llm compile` 的 overrides 决定。

### 坑 ⑦ `gpu_memory_utilization` 默认 0.85 → 初始化阶段就 OOM

**症状**：上下文配对了，加载阶段直接崩：

```
RuntimeError: CUDA Runtime Error: cudaErrorMemoryAllocation (2): out of memory
```

调用栈顶是 `CachedPagedKVCacheAuxDataManager::CachedPagedKVCacheAuxDataManager` → `Tensor::Empty` → `cuda_device_api.cc:173 cudaMalloc`。**注意这不是推理中途 OOM，是引擎初始化时就 OOM。**

**原因**：`gpu_memory_utilization` 默认 **0.85**，MLC 拿它反推能开多大的 KV cache。在 7.3 GiB 的板子上这个推算会超出实际可用量。

**修复**：降到 0.5，并明确并发数。

```python
cfg = EngineConfig(max_single_sequence_length=16384,
                   max_total_sequence_length=16384,
                   gpu_memory_utilization=0.5,
                   max_num_sequence=1)
```

> 32K 即使按上面三项全设对，仍然会 OOM —— 那是内存账算不平（923 + 3729 + 2680 = 7332 MB vs 7485 MB 可用），不是参数问题。见 [RESULTS.md](RESULTS.md) 第 6.2 节。

### 顺带：`NvMap ... error 12` 会干扰诊断

OOM 过程中会刷屏：

```
NvMapMemAllocInternalTagged failed: error 12
NvMapMemHandleAlloc failed: error 12
```

**这是伴生噪声，不是根因。** 它们会让人以为是 CMA（Jetson 上著名的 256 MB 限制）不够。实测：`cudaMallocManaged` 分配 **4 GB** 成功、`cudaHostAlloc` 1 GB 成功，CMA 与 CUDA 分配路径完全无关。真正的错误永远在最后那行 `cudaErrorMemoryAllocation` 的调用栈里。

---

## 其它坑（编译之外）

### MLC 的 Python 引擎不能多线程并发调

**症状**：4 个线程同时调 `MLCEngine` → 挂死，240 秒零返回。

**原因**：`MLCEngine` 的 Python 绑定不是线程安全的，内部状态会打架。

**修复**：要并发就用 HTTP 服务多客户端，不要在同一个 engine 上开线程池。

```bash
mlc_llm serve <model_dir> \
  --model-lib <model_dir>/model_lib.so \
  --host 127.0.0.1 --port 8080
```

### 跑分输出别接 `| tail -N`

**症状**：看起来"卡住了"，实际在跑。

**原因**：`tail` 要等进程结束才吐输出；而 MLC 退出那一步本身就容易卡几秒。管道一挂，人就误判。

**修复**：结果直接写文件，跑完再读。

```bash
python scripts/bench.py <model_dir> > /tmp/bench.log 2>&1
# 另开一个终端 tail -f /tmp/bench.log
```

### `--opt cudagraph=1` 是空操作

**症状**：以为开了 CUDA Graph 优化。

**原因**：实测产出的 `model_lib_cg.so` 和 `model_lib.so` **MD5 完全相同**（`c6efe0a6…`），参数根本没进产物。

**教训**：别拿"开了图性能没变化"当"下发开销不是瓶颈"的证据。真正的证据是 **GPU 95% / EMC 46%** —— GPU 一直在算，是 kernel 自己拉不到带宽。

### 编译和推理绝对不能同时跑

**症状**：推理进程被内核杀掉，连 Hermes 网关一起崩。

**原因**：这台机器只有 **7.3 GiB** 可用内存。一次实测：同时有 2 个 `cc1plus` + 3 个 `nvcc` 在编，推理进程常驻 3343 MB，内核直接 OOM kill；因为推理进程 `oom_score_adj=200`，它第一个被选中。

**修复**：串行。编完再跑，或者别在这台机器上同时干两件重活。

```bash
# 跑推理时给个内存笼子，宁可自己被限也别拖崩别的
systemd-run --user --scope -p MemoryHigh=7G -p MemoryMax=7500M \
  python scripts/bench.py <model_dir>
```

### 内存只有 7.3 GiB 不是 8 GiB

```
标称 8 GB
 → present 8050304 kB = 7.68 GiB   （512 MB 设备树 carveout：vpr / camdbg / fsi / rce-reservation / ramoops / framebuffer）
 → managed 7665092 kB = 7.31 GiB   （再 370 MB 内核页表/镜像/保留区）
```

链接阶段建议先加 swap：

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

---

## 编译时间参考

| 阶段 | 时间 |
|---|---|
| 克隆 mlc-llm + TVM（984 MB） | ~160 s |
| TVM 编译（637 target，5 线程） | 数小时（关掉 CUTLASS 后明显缩短） |
| MLC-LLM 编译 | 比 TVM 短得多 |
| 装 LLVM 18（apt） | 几分钟（自己编 LLVM 要 ~2 小时，别干） |
| 模型转换（1.7B） | 几分钟 |
| 模型 compile（生成 model_lib.so） | 几分钟 |
