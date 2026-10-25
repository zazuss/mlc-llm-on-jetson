#!/usr/bin/env python3
"""Context sweep for MLC-LLM: prompt length → prefill/decode throughput and memory.

Three MLC traps are baked in below. Do NOT "simplify" them away — each one cost
real debugging time and each one silently produces wrong numbers instead of an
error:

  1. model_lib must be passed explicitly. Without it MLCEngine JIT-recompiles a
     copy into a temp dir (~150 s) built with the DEFAULT context limit (8192),
     so your 32K run quietly measures 8K.

  2. EngineConfig.max_single_sequence_length / max_total_sequence_length default
     to None, and the C++ engine fills in 8192. This limit is independent of what
     the model_lib was compiled with — see cpp side of engine_base.py. Setting
     only model_lib is not enough.

  3. gpu_memory_utilization defaults to 0.85 and MLC uses it to *infer* the KV
     cache capacity. On a 7.3 GiB board that inference overshoots and the run
     dies with cudaErrorMemoryAllocation during engine init. Set it explicitly,
     and set max_num_sequence to what you actually need.

Usage:
  source env.sh
  systemd-run --user --scope -p MemoryHigh=6.5G -p MemoryMax=7G \
    python scripts/bench-ctx.py ~/models/qwen3-1.7b-q4f16_1-MLC \
      --ctx 8192 16384 --max-seq 16384 --gpu-mem-util 0.5 --gen 64
"""
import argparse
import os
import time

FILLER = (
    "The quick brown fox jumps over the lazy dog while a patient engineer "
    "measures memory bandwidth on a small embedded board. "
)

# Measured on Qwen tokenizers: chars = tokens * 5.4 gives a token count within
# ~1% of the target. The naive 4-chars-per-token assumption undershoots by a
# third on this filler and makes every "8K" run actually measure 5.5K.
CHARS_PER_TOKEN = 5.4


def make_prompt(target_tokens):
    chars = int(target_tokens * CHARS_PER_TOKEN)
    reps = chars // len(FILLER) + 1
    return (FILLER * reps)[:chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model dir (contains mlc-chat-config.json)")
    ap.add_argument("--model-lib", default=None,
                    help="prebuilt model_lib.so; defaults to <model>/model_lib.so")
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 16384],
                    help="target prompt token counts")
    ap.add_argument("--max-seq", type=int, default=16384,
                    help="engine max sequence length (default in C++ is 8192!)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.5,
                    help="fraction of GPU memory MLC may plan KV cache for")
    ap.add_argument("--gen", type=int, default=64, help="tokens to generate")
    args = ap.parse_args()

    from mlc_llm import MLCEngine
    from mlc_llm.serve import EngineConfig

    lib = args.model_lib or os.path.join(args.model, "model_lib.so")
    assert os.path.exists(lib), f"model_lib not found: {lib} (build it first)"

    t0 = time.time()
    cfg = EngineConfig(
        max_total_sequence_length=args.max_seq,
        max_single_sequence_length=args.max_seq,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_sequence=1,
    )
    engine = MLCEngine(model=args.model, device="cuda", model_lib=lib,
                       engine_config=cfg)
    print(f"load {time.time() - t0:.1f}s | "
          f"max_single=max_total={args.max_seq} | util={args.gpu_mem_util}",
          flush=True)

    print(f"\n{'target':>8} {'actual':>8} {'prefill t/s':>12} "
          f"{'decode t/s':>11} {'prefill s':>10}", flush=True)

    for target in args.ctx:
        prompt = make_prompt(target)
        t0 = time.time()
        first = None
        usage = None
        stream = engine.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.gen,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage = u
            elif chunk.choices and chunk.choices[0].delta.content and first is None:
                first = time.time() - t0
        total = time.time() - t0

        pre = usage.prompt_tokens / first if first else float("nan")
        dec = (usage.completion_tokens - 1) / (total - first) if total > first else float("nan")
        print(f"{target:>8} {usage.prompt_tokens:>8} {pre:>12.2f} "
              f"{dec:>11.2f} {first:>10.1f}", flush=True)


if __name__ == "__main__":
    main()
