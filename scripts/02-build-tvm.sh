#!/usr/bin/env bash
# 02) 编译 TVM（这一步最久：637 个 target）
#
# 坑① 已内置：USE_CUTLASS=OFF
#   只有 MoE 模型引用 CUTLASS/FlashAttention，Qwen3 是 dense 用不到。
#   开着它白等多时。关掉后已编目标保留，不用重头来。
set -euo pipefail

MLC_HOME="${MLC_HOME:-$HOME/mlc-build}"
TVM_HOME="$MLC_HOME/mlc-llm/3rdparty/tvm"
CUDA_ARCH="${CUDA_ARCH:-87}"
JOBS="${JOBS:-5}"          # 6 核机器用 5；nvcc 吃内存，别开满
LLVM_CONFIG="${LLVM_CONFIG:-/usr/bin/llvm-config-18}"

[ -d "$TVM_HOME" ] || { echo "✗ 找不到 $TVM_HOME，先跑 01-fetch.sh" >&2; exit 1; }
command -v "$LLVM_CONFIG" >/dev/null || { echo "✗ 找不到 $LLVM_CONFIG，先跑 00-prereqs.sh" >&2; exit 1; }

export PATH="/usr/local/cuda/bin:${PATH}"

echo "=========== TVM 配置 ==========="
echo "TVM_HOME  : $TVM_HOME"
echo "LLVM      : $($LLVM_CONFIG --version)"
echo "CUDA arch : sm_${CUDA_ARCH}"
echo "并行       : $JOBS"
echo

cmake -S "$TVM_HOME" -B "$TVM_HOME/build" \
    -DUSE_CUDA=ON \
    -DUSE_CUTLASS=OFF \
    -DUSE_CUDNN=OFF \
    -DUSE_GRAPH_EXECUTOR=ON \
    -DUSE_LLVM="$LLVM_CONFIG" \
    -DUSE_OPENMP=OFF \
    -DUSE_RPC=ON \
    -DUSE_RANDOM=ON \
    -DUSE_SORT=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release

echo
echo "=========== 开始编译（637 个 target，慢慢等）==========="
echo "提示：编的同时别跑推理，7.3 GiB 内存会 OOM 并杀进程"
date
cmake --build "$TVM_HOME/build" --config Release -j"$JOBS"
date

echo
echo "=========== 产物 ==========="
ls -lh "$TVM_HOME"/build/*.so 2>/dev/null || { echo "✗ 没生成 .so" >&2; exit 1; }
echo "✓ 下一步：./scripts/03-build-mlc.sh"
