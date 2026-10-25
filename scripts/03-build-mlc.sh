#!/usr/bin/env bash
# 03) 编译 MLC-LLM（比 TVM 快得多，但坑②③都在这步）
#
# 坑② rustc < 1.77 → tokenizers 编不过
# 坑③ cmake 把 /usr/bin/cargo 写死成绝对路径，改 PATH 无效
# 两个都已在 00-prereqs.sh 处理；若仍报 rust 相关错误，先跑它。
set -euo pipefail

MLC_HOME="${MLC_HOME:-$HOME/mlc-build}"
SRC="$MLC_HOME/mlc-llm"
CUDA_ARCH="${CUDA_ARCH:-87}"
JOBS="${JOBS:-5}"

[ -d "$SRC" ] || { echo "✗ 找不到 $SRC，先跑 01-fetch.sh" >&2; exit 1; }
[ -f "$SRC/3rdparty/tvm/build/lib/libtvm.so" ] || {
    echo "⚠ TVM 还没编好，MLC 链接会失败。先跑 02-build-tvm.sh（或忽略，如果你确定）"
}

export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:${PATH}"

# 坑③ 预防：确认 cmake 会用到的是新 cargo
echo "=========== 环境检查 ==========="
echo "cargo     : $(command -v cargo) → $(cargo --version)"
echo "/usr/bin/cargo → $(/usr/bin/cargo --version 2>/dev/null || echo '不存在')"
grep -E '^CMAKE_CARGO' "$SRC/build/CMakeCache.txt" 2>/dev/null || echo "CMAKE_CARGO: 尚未缓存（首次 configure）"
echo "CUDA arch : sm_${CUDA_ARCH}"
echo

# 坑②：换编译器后 cargo 拒绝复用旧产物
if [ -d "$SRC/build/tokenizers" ]; then
    echo "--- 清理旧的 tokenizers 产物（换编译器后必须清）"
    rm -rf "$SRC/build/tokenizers"
fi

echo "=========== MLC-LLM 配置 ==========="
cmake -S "$SRC" -B "$SRC/build" \
    -DUSE_CUDA=ON \
    -DUSE_CUTLASS=OFF \
    -DUSE_CUDNN=OFF \
    -DUSE_CURAND=OFF \
    -DUSE_METAL=OFF \
    -DUSE_ROCM=OFF \
    -DUSE_LLVM=OFF \
    -DUSE_THREADS=ON \
    -DUSE_OPENMP=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release

echo
echo "=========== 开始编译 ==========="
date
cmake --build "$SRC/build" --config Release -j"$JOBS"
date

echo
echo "=========== 产物 ==========="
ls -lh "$SRC"/build/lib*.so 2>/dev/null || { echo "✗ 没生成 libmlc_llm.so" >&2; exit 1; }
echo "✓ 下一步：./scripts/04-python-env.sh"
