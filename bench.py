#!/usr/bin/env python3
"""Measure tokens/second of a running llama-server (OpenAI-compatible API).

Mirrors the two reference metrics used by llama.cpp's `llama-bench`:
  - pp = prompt processing speed (compute-bound prefill)
  - tg = token generation speed  (memory-bandwidth-bound decode)

Plus an end-to-end realistic prompt for a "feels like" number.

Usage:
    ./bench.py                          # default http://localhost:8080
    HOST=http://1.2.3.4:8080 ./bench.py
    ./bench.py --runs 3 --max-tokens 256

Requires only the Python stdlib.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass


# Prompt set. Picked so each isolates a different bottleneck:
#   1. tg_short  - tiny input, large output -> dominated by decode (tok/s out)
#   2. pp_long   - large input, tiny output -> dominated by prefill (tok/s in)
#   3. realistic - mid-sized coding ask     -> end-to-end "real" number
PROMPTS = [
    {
        "name": "tg_short  (short prompt -> long output)",
        "system": "You are a helpful assistant. Be verbose.",
        "user": (
            "Write a detailed 400-word explanation of how a CPU cache "
            "hierarchy (L1/L2/L3) works, including coherence protocols. "
            "Do not use bullet points; write flowing prose."
        ),
        "max_tokens": 512,
    },
    {
        "name": "pp_long   (long prompt -> short output)",
        "system": "You summarize text in one short sentence.",
        # ~1.5k tokens of filler so the prefill dominates.
        "user": (
            "Summarize the following text in ONE sentence:\n\n"
            + ("The quick brown fox jumps over the lazy dog. " * 250)
        ),
        "max_tokens": 32,
    },
    {
        "name": "pp_xl     (10k+ token prompt -> short output)",
        "system": "You summarize text in one short sentence.",
        # ~12k tokens of filler so a very deep prefill dominates. The phrase
        # is ~10 tokens, so 1200 repeats comfortably clears the 10k mark.
        "user": (
            "Summarize the following text in ONE sentence:\n\n"
            + ("The quick brown fox jumps over the lazy dog. " * 1200)
        ),
        "max_tokens": 32,
    },
    {
        "name": "realistic (coding question)",
        "system": "You are a senior software engineer.",
        "user": (
            "In Python, write a function `merge_intervals(intervals)` that "
            "takes a list of (start, end) tuples and returns the merged, "
            "non-overlapping intervals sorted by start. Include 3 example "
            "calls with expected output."
        ),
        "max_tokens": 384,
    }
]


@dataclass
class Result:
    name: str
    ttft_s: float            # time to first token (wall clock)
    wall_total_s: float      # total wall-clock time for the request
    prompt_n: int            # tokens in prompt (server-reported)
    predicted_n: int         # tokens generated (server-reported)
    prompt_tps: float        # server-reported prompt processing tok/s
    gen_tps: float           # server-reported generation tok/s
    wall_gen_tps: float      # generation tok/s computed from wall clock
    draft_n: int = 0         # MTP/speculative draft tokens generated
    draft_accepted: int = 0  # MTP/speculative draft tokens accepted


def stream_chat(host: str, model: str, system: str, user: str,
                max_tokens: int, timeout: int) -> Result:
    """Stream one chat completion and capture timing.

    llama-server returns a `timings` object on the final stream chunk with
    authoritative numbers. We use those, but also record wall-clock TTFT
    so the user can see request-level latency.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Required for llama-server to attach speculative/MTP draft stats
        # (draft_n, draft_n_accepted) to the final chunk's timings object.
        "timings_per_token": True,
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft = None
    timings = {}
    usage = {}
    first_token_seen = False

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # First chunk with actual delta content -> TTFT.
            if not first_token_seen:
                choices = obj.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    ttft = time.perf_counter() - t0
                    first_token_seen = True

            # llama-server attaches timings + usage on the final chunk(s).
            if "timings" in obj:
                timings = obj["timings"]
            if obj.get("usage"):
                usage = obj["usage"]

    wall_total = time.perf_counter() - t0
    if ttft is None:
        ttft = wall_total  # no streamed content -> treat as one shot

    prompt_n = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
    predicted_n = int(
        timings.get("predicted_n") or usage.get("completion_tokens") or 0
    )
    prompt_tps = float(timings.get("prompt_per_second") or 0.0)
    gen_tps = float(timings.get("predicted_per_second") or 0.0)
    draft_n = int(timings.get("draft_n") or 0)
    draft_accepted = int(timings.get("draft_n_accepted") or 0)

    # Wall-clock generation rate: tokens emitted / time after first token.
    decode_wall = max(wall_total - ttft, 1e-9)
    wall_gen_tps = predicted_n / decode_wall if predicted_n else 0.0

    return Result(
        name="",  # filled by caller
        ttft_s=ttft,
        wall_total_s=wall_total,
        prompt_n=prompt_n,
        predicted_n=predicted_n,
        prompt_tps=prompt_tps,
        gen_tps=gen_tps,
        wall_gen_tps=wall_gen_tps,
        draft_n=draft_n,
        draft_accepted=draft_accepted,
    )


def fmt_row(r: Result) -> str:
    mtp = ""
    if r.draft_n:
        mtp = (
            f"  mtp={r.draft_accepted:>4}/{r.draft_n:<4} "
            f"({r.draft_accepted / r.draft_n:.3f})"
        )
    return (
        f"  ttft={r.ttft_s*1000:7.1f} ms  "
        f"prompt={r.prompt_n:>5}t @ {r.prompt_tps:7.1f} t/s  "
        f"gen={r.predicted_n:>4}t @ {r.gen_tps:6.1f} t/s "
        f"(wall {r.wall_gen_tps:6.1f} t/s)  "
        f"total={r.wall_total_s:6.2f}s"
        f"{mtp}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("HOST", "http://localhost:8080"),
                   help="llama-server base URL (default: %(default)s)")
    p.add_argument("--model", default=os.environ.get("MODEL", "default"),
                   help="model name to send (llama-server ignores it; default: %(default)s)")
    p.add_argument("--runs", type=int, default=1,
                   help="repeat each prompt N times and average (default: 1)")
    p.add_argument("--timeout", type=int, default=600,
                   help="per-request timeout in seconds (default: 600)")
    p.add_argument("--warmup", action="store_true",
                   help="run one throwaway request first to load the model")
    p.add_argument("--loop", action="store_true",
                   help="repeat the full benchmark in a loop until Ctrl+C")
    args = p.parse_args()

    print(f"Target: {args.host}   model={args.model}   runs={args.runs}")
    print()

    if args.warmup:
        print("Warmup ...", flush=True)
        try:
            stream_chat(args.host, args.model,
                        "You are helpful.", "Say 'ready'.",
                        max_tokens=8, timeout=args.timeout)
        except Exception as e:
            print(f"  warmup failed: {e}", file=sys.stderr)

    overall_prompt_tps = []
    overall_gen_tps = []
    overall_draft_n = 0
    overall_draft_accepted = 0
    iteration = 0

    try:
        while True:
            iteration += 1
            if args.loop:
                print(f"=== Iteration {iteration} ===")

            for spec in PROMPTS:
                print(spec["name"])
                runs = []
                for i in range(args.runs):
                    try:
                        r = stream_chat(
                            args.host, args.model,
                            spec["system"], spec["user"],
                            max_tokens=spec["max_tokens"],
                            timeout=args.timeout,
                        )
                    except Exception as e:
                        print(f"  run {i+1}: ERROR {e}", file=sys.stderr)
                        continue
                    r.name = spec["name"]
                    runs.append(r)
                    print(f"  run {i+1}:" + fmt_row(r))

                if not runs:
                    continue
                if args.runs > 1:
                    avg_prompt = sum(r.prompt_tps for r in runs) / len(runs)
                    avg_gen = sum(r.gen_tps for r in runs) / len(runs)
                    print(f"  avg:   prompt {avg_prompt:7.1f} t/s   gen {avg_gen:6.1f} t/s")
                overall_prompt_tps.extend(r.prompt_tps for r in runs if r.prompt_tps)
                overall_gen_tps.extend(r.gen_tps for r in runs if r.gen_tps)
                overall_draft_n += sum(r.draft_n for r in runs)
                overall_draft_accepted += sum(r.draft_accepted for r in runs)
                print()

            if overall_prompt_tps and overall_gen_tps:
                print("Summary (across all runs):")
                print(f"  prompt processing: {max(overall_prompt_tps):7.1f} t/s peak, "
                      f"{sum(overall_prompt_tps)/len(overall_prompt_tps):7.1f} t/s mean")
                print(f"  token generation:  {max(overall_gen_tps):7.1f} t/s peak, "
                      f"{sum(overall_gen_tps)/len(overall_gen_tps):7.1f} t/s mean")
                if overall_draft_n:
                    print(f"  mtp acceptance:    {overall_draft_accepted/overall_draft_n:.5f} "
                          f"({overall_draft_accepted} accepted / {overall_draft_n} generated)")

            if not args.loop:
                break
            print()

    except KeyboardInterrupt:
        print("\nStopped.")
        if overall_prompt_tps and overall_gen_tps:
            print("\nFinal summary (across all iterations):")
            print(f"  prompt processing: {max(overall_prompt_tps):7.1f} t/s peak, "
                  f"{sum(overall_prompt_tps)/len(overall_prompt_tps):7.1f} t/s mean")
            print(f"  token generation:  {max(overall_gen_tps):7.1f} t/s peak, "
                  f"{sum(overall_gen_tps)/len(overall_gen_tps):7.1f} t/s mean")
            if overall_draft_n:
                print(f"  mtp acceptance:    {overall_draft_accepted/overall_draft_n:.5f} "
                      f"({overall_draft_accepted} accepted / {overall_draft_n} generated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
