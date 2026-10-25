#!/usr/bin/env python3
"""跑分：单流（进程内 MLCEngine）或多流（HTTP 服务）。

单流：
    python scripts/bench.py ~/models/qwen3-1.7b-q4f16_1-MLC

多流（先起 06-serve.sh）：
    python scripts/bench.py --url http://127.0.0.1:8080 --concurrency 4

注意：
  * 跑之前必须 source env.sh（TVM_HOME / PYTHONPATH / LD_LIBRARY_PATH 全靠它）
  * 同一个 MLCEngine 上开线程池会挂死 —— 并发只走 HTTP
  * 别把输出接 `| tail -N`，MLC 退出那步容易卡，会被误判成卡住
"""
import argparse
import os
import statistics
import sys
import threading
import time

MSG = [{"role": "user", "content": "请用中文写一段关于机器人的短文。"}]


def rate(prompt_tokens, completion_tokens, first_latency, total):
    """→ (prefill t/s, decode t/s)"""
    if not first_latency or total <= first_latency:
        return float("nan"), float("nan")
    pre = prompt_tokens / first_latency if first_latency else float("nan")
    dec = (completion_tokens - 1) / (total - first_latency)
    return pre, dec


# ---------------------------------------------------------------- 单流（进程内）

def bench_inproc(model_dir, model_lib, n_tokens, rounds, warmup):
    from mlc_llm import MLCEngine

    print(f"加载 {model_dir} …", flush=True)
    t0 = time.time()
    engine = MLCEngine(model=model_dir, device="cuda", model_lib=model_lib)
    load = time.time() - t0
    print(f"加载耗时 {load:.1f}s\n", flush=True)

    def one(n, label):
        t0 = time.time()
        first = None
        usage = None
        stream = engine.chat.completions.create(
            model=model_dir,
            messages=MSG,
            max_tokens=n,
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
        if usage is None or first is None:
            print(f"{label}: 总 {total:.2f}s（没拿到 usage/首字）", flush=True)
            return None
        pre, dec = rate(usage.prompt_tokens, usage.completion_tokens, first, total)
        print(
            f"{label}: prompt {usage.prompt_tokens} tok | 生成 {usage.completion_tokens} tok | "
            f"首字 {first:.2f}s 总 {total:.2f}s | prefill {pre:.0f} t/s | decode {dec:.2f} t/s",
            flush=True,
        )
        return dec

    if warmup > 0:
        one(warmup, "预热")

    decs = [d for d in (one(n_tokens, f"正式#{i + 1}") for i in range(rounds)) if d]
    if decs:
        print(
            f"\ndecode 中位 {statistics.median(decs):.2f} t/s"
            f"（{min(decs):.2f} – {max(decs):.2f}，n={len(decs)}）",
            flush=True,
        )
    return decs


# ---------------------------------------------------------------- 多流（HTTP）

def bench_http(url, n_tokens, concurrency, rounds):
    from openai import OpenAI

    client = OpenAI(base_url=f"{url.rstrip('/')}/v1", api_key="none")
    model = client.models.list().data[0].id
    print(f"服务 {url}  模型 {model}  并发 {concurrency}\n", flush=True)

    def worker(idx, out):
        t0 = time.time()
        first = None
        usage = None
        stream = client.chat.completions.create(
            model=model,
            messages=MSG,
            max_tokens=n_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            elif chunk.choices and chunk.choices[0].delta.content and first is None:
                first = time.time() - t0
        total = time.time() - t0
        if usage is None or first is None:
            out[idx] = None
            return
        _, dec = rate(usage.prompt_tokens, usage.completion_tokens, first, total)
        out[idx] = dec

    for r in range(rounds):
        out = [None] * concurrency
        wall0 = time.time()
        threads = [threading.Thread(target=worker, args=(i, out)) for i in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - wall0

        ok = [d for d in out if d]
        if not ok:
            print(f"第 {r + 1} 轮：全部失败", flush=True)
            continue
        per_stream = statistics.median(ok)
        aggregate = sum(ok)
        print(
            f"第 {r + 1} 轮：每流 {per_stream:.2f} t/s | 聚合 {aggregate:.1f} t/s | "
            f"墙钟 {wall:.2f}s | 成功 {len(ok)}/{concurrency}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?", help="模型目录（单流模式必填）")
    ap.add_argument("--model-lib", default=None, help="默认 <model_dir>/model_lib.so")
    ap.add_argument("-n", "--tokens", type=int, default=150)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=8, help="预热 token 数，0 关闭")
    ap.add_argument("--url", default=None, help="给了就走 HTTP 多流模式")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    if args.url:
        bench_http(args.url, args.tokens, args.concurrency, args.rounds)
        return

    if not args.model_dir:
        ap.error("单流模式需要 model_dir；多流模式用 --url")

    d = os.path.expanduser(args.model_dir)
    lib = args.model_lib or os.path.join(d, "model_lib.so")
    if not os.path.isfile(lib):
        print(f"✗ 找不到 {lib}（先跑 05-convert-model.sh）", file=sys.stderr)
        sys.exit(1)

    bench_inproc(d, lib, args.tokens, args.rounds, args.warmup)


if __name__ == "__main__":
    main()
