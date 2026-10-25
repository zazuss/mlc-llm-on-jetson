#!/usr/bin/env bash
# 00) 前置：LLVM 18 + rust >= 1.77 + swap + 坑③修复
# 需要 sudo（Jetson 默认 orin 用户免密）
set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-87}"

echo "=========== 平台检查 ==========="
if [ ! -f /etc/nv_tegra_release ]; then
    echo "✗ 这不是 Jetson：/etc/nv_tegra_release 不存在" >&2
    exit 1
fi
head -1 /etc/nv_tegra_release
# shellcheck disable=SC1091
. /etc/os-release && echo "$PRETTY_NAME"
echo "可用内存: $(free -h | awk '/^Mem:/{print $7}')"

export PATH="/usr/local/cuda/bin:${PATH}"
if ! command -v nvcc >/dev/null; then
    echo "✗ 找不到 nvcc。装 CUDA toolkit 或检查 /usr/local/cuda" >&2
    exit 1
fi
echo "nvcc: $(nvcc --version | tail -1)"
echo "目标 GPU arch: sm_${CUDA_ARCH}"
echo

echo "=========== LLVM 18 ==========="
# 自己编 LLVM 要 ~2 小时，用 apt 的省掉
if command -v llvm-config-18 >/dev/null; then
    echo "已装: llvm-config-18 $(llvm-config-18 --version)"
else
    echo "--- 安装 llvm-18-dev / libclang-18-dev"
    sudo apt-get update
    sudo apt-get install -y llvm-18-dev libclang-18-dev clang-18
    llvm-config-18 --version
fi
echo

echo "=========== rust >= 1.77 ==========="
# tokenizers 要求 rustc >= 1.77；Ubuntu 24.04 自带的 1.75 不够
RUST_OK=0
if command -v rustc >/dev/null; then
    CUR="$(rustc --version | awk '{print $2}')"
    if [ "$(printf '%s\n%s\n' "1.77" "$CUR" | sort -V | head -1)" = "1.77" ]; then
        RUST_OK=1
        echo "已满足: rustc $CUR"
    else
        echo "rustc $CUR < 1.77，升级"
    fi
fi

if [ "$RUST_OK" -eq 0 ]; then
    if ! command -v rustup >/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    rustup install 1.99.1
    rustup default 1.99.1
    rustc --version
fi
echo

echo "=========== 坑③：/usr/bin/cargo 被 cmake 写死 ==========="
# cmake 在 configure 阶段就把 CMAKE_CARGO 解析成绝对路径 /usr/bin/cargo，
# 之后改 PATH 完全无效 —— 必须换掉文件本身。
if [ -x /usr/bin/cargo ]; then
    echo "当前 /usr/bin/cargo → $(/usr/bin/cargo --version 2>/dev/null || echo '无法执行')"
fi
RUSTUP_CARGO="$(command -v cargo || true)"
if [ -n "$RUSTUP_CARGO" ] && [ "$(readlink -f /usr/bin/cargo 2>/dev/null)" != "$(readlink -f "$RUSTUP_CARGO")" ]; then
    echo "--- 把 /usr/bin/cargo 指向 $RUSTUP_CARGO"
    sudo ln -sf "$(readlink -f "$RUSTUP_CARGO")" /usr/bin/cargo
fi
echo "/usr/bin/cargo → $(/usr/bin/cargo --version 2>/dev/null || echo '缺失')"
echo

echo "=========== swap ==========="
SWAP_GB="$(free -g | awk '/^Swap:/{print $2}')"
if [ "${SWAP_GB:-0}" -lt 4 ]; then
    echo "--- swap 只有 ${SWAP_GB:-0} GB，加 8 GB（链接阶段会吃满 7.3 GiB）"
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 8G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=8192
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
    fi
    sudo swapon /swapfile 2>/dev/null || true
else
    echo "已有 ${SWAP_GB} GB swap"
fi
free -h
echo

echo "✓ 前置就绪。下一步：./scripts/01-fetch.sh"
