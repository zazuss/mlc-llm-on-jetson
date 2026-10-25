#!/usr/bin/env bash
# 04) Python 环境：venv + MLC 依赖 + 坑④
#
# 坑④：TVM 的 LLVM JIT 后端要单独装包 apache-tvm-ffi-orcjit==0.1.2，
#       缺它 convert_weight / compile 直接崩。
# 另外 convert_weight 需要 torch（不装会在转换阶段才报错，白等半天）。
set -euo pipefail

MLC_HOME="${MLC_HOME:-$HOME/mlc-build}"
SRC="$MLC_HOME/mlc-llm"
VENV="${MLC_VENV:-$HOME/mlc-venv}"
PY="${PYTHON:-python3}"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

[ -f "$SRC/python/mlc_llm/__init__.py" ] || { echo "✗ 源码不完整：$SRC" >&2; exit 1; }

echo "=========== 创建 venv ==========="
if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version

echo
echo "=========== 安装 mlc_llm（editable，不带依赖）==========="
pip install -q --upgrade pip
pip install -q -e "$SRC/python" --no-deps -i "$MIRROR" 2>&1 | tail -3

echo
echo "=========== 运行时依赖 ==========="
pip install -q -i "$MIRROR" \
    fastapi uvicorn openai prompt_toolkit requests sentencepiece shortuuid \
    tiktoken tqdm transformers datasets pandas ml_dtypes scipy psutil

echo
echo "=========== 坑④：TVM LLVM JIT ==========="
pip install -q -i "$MIRROR" "apache-tvm-ffi-orcjit==0.1.2"

echo
echo "=========== torch（convert_weight 需要）==========="
if python -c "import torch" 2>/dev/null; then
    python -c "import torch; print('torch', torch.__version__)"
else
    # Jetson 上优先用 NVIDIA 的 wheel 源；失败再退回默认
    pip install -q torch --index-url https://download.pytorch.org/whl/cu130 2>/dev/null \
        || pip install -q torch -i "$MIRROR"
    python -c "import torch; print('torch', torch.__version__)"
fi

echo
echo "=========== 验证 ==========="
# 需要 env.sh 里的 PYTHONPATH / LD_LIBRARY_PATH
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.example.sh" 2>/dev/null || true
export PYTHONPATH="$SRC/3rdparty/tvm/python:$SRC/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SRC/build/lib:$SRC/3rdparty/tvm/build/lib:${LD_LIBRARY_PATH:-}"
export TVM_LIBRARY_PATH="$SRC/3rdparty/tvm/build/lib"
export TVM_FFI_LIBRARY_PATH="$SRC/3rdparty/tvm/build/lib"

python - <<'PY'
import mlc_llm, tvm
print("mlc_llm :", mlc_llm.__file__)
print("tvm     :", tvm.__file__)
print("✓ 导入成功")
PY

mlc_llm --help | head -8
echo
echo "✓ 下一步：./scripts/05-convert-model.sh Qwen/Qwen3-1.7B qwen3"
