#!/usr/bin/env bash
# 05) 转换模型：gen_config → convert_weight → compile
#
# 用法：
#   ./scripts/05-convert-model.sh Qwen/Qwen3-1.7B qwen3
#   ./scripts/05-convert-model.sh ~/models/Qwen3-1.7B-hf qwen3 qwen3-1.7b-MLC
#
# 参数：
#   $1  HF 模型（repo id 或本地目录）
#   $2  conversation template（qwen3 / qwen3_5 / llama3 / ... 见 mlc_llm gen_config --help）
#   $3  输出目录名（默认从 $1 推导）
set -euo pipefail

HF_MODEL="${1:?用法: $0 <hf_model_or_path> <conv_template> [out_name]}"
CONV="${2:?缺少 conv template，例如 qwen3}"
NAME="${3:-$(basename "$HF_MODEL" | tr '[:upper:]' '[:lower:]')-q4f16_1-MLC}"

MLC_MODEL_DIR="${MLC_MODEL_DIR:-$HOME/models}"
OUT="$MLC_MODEL_DIR/$NAME"
QUANT="${QUANT:-q4f16_1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.example.sh" >/dev/null 2>&1 || true
export PATH="/usr/local/cuda/bin:${PATH}"

echo "=========== 转换配置 ==========="
echo "源模型   : $HF_MODEL"
echo "模板     : $CONV"
echo "量化     : $QUANT"
echo "输出     : $OUT"
echo "nvcc     : $(nvcc --version | tail -1)"
echo

mkdir -p "$OUT"

echo "=========== 1/3 gen_config ==========="
date +%H:%M:%S
mlc_llm gen_config "$HF_MODEL" \
    --quantization "$QUANT" \
    --conv-template "$CONV" \
    --output "$OUT" 2>&1 | grep -v neoverse | tail -15

python - "$OUT" <<'PY'
import json, sys, os
p = os.path.join(sys.argv[1], "mlc-chat-config.json")
c = json.load(open(p))
print("model_type =", c.get("model_type"), "| quantization =", c.get("quantization"))
PY

echo
echo "=========== 2/3 convert_weight ==========="
date +%H:%M:%S
mlc_llm convert_weight "$HF_MODEL" \
    --quantization "$QUANT" \
    --source-format huggingface-safetensor \
    -o "$OUT" 2>&1 | grep -v neoverse | tail -25
echo "转换后体积: $(du -sh "$OUT" | cut -f1)"

echo
echo "=========== 3/3 compile（生成 model_lib.so）==========="
date +%H:%M:%S
mlc_llm compile "$OUT/mlc-chat-config.json" \
    --device cuda \
    -o "$OUT/model_lib.so" 2>&1 | grep -v neoverse | tail -20

ls -lh "$OUT/model_lib.so"
date +%H:%M:%S
echo
echo "✓ 完成。跑分："
echo "   python scripts/bench.py $OUT"
