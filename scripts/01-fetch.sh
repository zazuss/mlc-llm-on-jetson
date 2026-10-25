#!/usr/bin/env bash
# 01) 拉源码：mlc-llm + TVM 子模块
set -euo pipefail

MLC_HOME="${MLC_HOME:-$HOME/mlc-build}"
REPO="${REPO:-https://github.com/mlc-ai/mlc-llm.git}"
# 国内直连 GitHub 常被 TLS 截断（GnuTLS recv error -110）。
# 如果是这样，用镜像下 tar 包再解压，比 clone 快得多：
#   https://ghfast.top/https://github.com/mlc-ai/mlc-llm/archive/refs/heads/main.tar.gz
# 注意 tar 包不含子模块，TVM 要单独下。

mkdir -p "$MLC_HOME"
cd "$MLC_HOME"

if [ -d mlc-llm/.git ]; then
    echo "--- 已存在，更新 + 同步子模块"
    cd mlc-llm
    git fetch --depth 1 origin
    git submodule update --init --recursive
else
    echo "--- 克隆 mlc-llm（含 TVM 子模块，约 984 MB / 160 s）"
    git clone --recursive "$REPO"
    cd mlc-llm
fi

echo
echo "=========== 版本 ==========="
git log -1 --format='mlc-llm: %h  %ad  %s' --date=short
git -C 3rdparty/tvm log -1 --format='tvm:     %h  %ad  %s' --date=short 2>/dev/null || {
    echo "✗ TVM 子模块缺失。执行：git submodule update --init --recursive" >&2
    exit 1
}
echo "体积: $(du -sh . | cut -f1)"

# 验证过的 commit（2026-10-06）
echo
echo "参考：本仓库实测跑通的 mlc-llm commit = 8978ea9"
echo "✓ 下一步：./scripts/02-build-tvm.sh"
