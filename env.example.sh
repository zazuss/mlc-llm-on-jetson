#!/bin/bash
# 复制成 env.sh 再按你的路径改：
#   cp env.example.sh env.sh && source env.sh
#
# 所有 MLC / TVM 的路径都由这里定义。少了任何一条都会出现
# ImportError 或 "Library not found" —— 别单独 export 某一个。

# 源码根目录（01-fetch.sh 会把 mlc-llm 拉到这里）
export MLC_HOME="${MLC_HOME:-$HOME/mlc-build}"
export MLC_LLM_SOURCE="$MLC_HOME/mlc-llm"
export TVM_HOME="$MLC_LLM_SOURCE/3rdparty/tvm"

# 编译产物
export TVM_LIBRARY_PATH="$TVM_HOME/build/lib"
export TVM_FFI_LIBRARY_PATH="$TVM_HOME/build/lib"
export MLC_LIBRARY_PATH="$MLC_LLM_SOURCE/build"

# 运行时动态库 / Python 包搜索路径
export LD_LIBRARY_PATH="$MLC_LLM_SOURCE/build/lib:$TVM_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TVM_HOME/python:$MLC_LLM_SOURCE/python:${PYTHONPATH:-}"

# CUDA
export PATH="/usr/local/cuda/bin:${PATH}"

# 模型输出目录
export MLC_MODEL_DIR="${MLC_MODEL_DIR:-$HOME/models}"

# venv（04-python-env.sh 会建）
MLC_VENV="${MLC_VENV:-$HOME/mlc-venv}"
if [ -f "$MLC_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$MLC_VENV/bin/activate"
fi

# 自检
_mlc_env_check() {
    local bad=0
    [ -d "$MLC_LLM_SOURCE" ] || { echo "✗ 源码不存在: $MLC_LLM_SOURCE（跑 01-fetch.sh）" >&2; bad=1; }
    [ -d "$TVM_HOME" ] || { echo "✗ TVM 不存在: $TVM_HOME（01-fetch.sh 要 --recursive）" >&2; bad=1; }
    [ -f "$MLC_LLM_SOURCE/build/lib/libmlc_llm.so" ] || echo "⚠ libmlc_llm.so 还没编出来（跑 03-build-mlc.sh）" >&2
    command -v nvcc >/dev/null || { echo "✗ 找不到 nvcc，检查 /usr/local/cuda/bin" >&2; bad=1; }
    return $bad
}

# 想看环境是否就绪：bash -c 'source env.sh && _mlc_env_check && echo OK'
