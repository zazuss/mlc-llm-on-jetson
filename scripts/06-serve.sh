#!/usr/bin/env bash
# 06) 起 OpenAI 兼容服务
#
# 用法：
#   ./scripts/06-serve.sh ~/models/qwen3-1.7b-q4f16_1-MLC
#   ./scripts/06-serve.sh ~/models/qwen3-1.7b-q4f16_1-MLC 8080
#
# 为什么要走服务：MLC 的 Python MLCEngine 不是线程安全的，
# 同一个 engine 上开线程池会挂死（240 秒零返回）。要并发就多客户端打 HTTP。
set -euo pipefail

MODEL="${1:?用法: $0 <model_dir> [port]}"
PORT="${2:-8080}"
HOST="${HOST:-127.0.0.1}"

MLC_HOME="${MLC_HOME:-$HOME/mlc-build}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.example.sh" >/dev/null 2>&1 || true
export PATH="/usr/local/cuda/bin:${PATH}"

[ -d "$MODEL" ] || { echo "✗ 模型目录不存在: $MODEL" >&2; exit 1; }

MODEL_LIB="$MODEL/model_lib.so"
if [ ! -f "$MODEL_LIB" ]; then
    echo "✗ 缺 $MODEL_LIB —— 先跑 05-convert-model.sh" >&2
    exit 1
fi

# 提示：--opt cudagraph=1 产出的 model_lib_cg.so 与 model_lib.so 内容相同（空操作），
# 所以这里直接用 model_lib.so。
echo "=========== 启动服务 ==========="
echo "模型    : $MODEL"
echo "model_lib: $MODEL_LIB"
echo "监听    : http://$HOST:$PORT"
echo "内存笼子: MemoryHigh=7G / MemoryMax=7500M（7.3 GiB 机器，别让它拖崩别的进程）"
echo

systemd-run --user --scope --quiet \
    -p MemoryHigh=7G -p MemoryMax=7500M \
    mlc_llm serve "$MODEL" \
        --model-lib "$MODEL_LIB" \
        --host "$HOST" --port "$PORT"
