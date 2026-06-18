#!/usr/bin/env python3
"""Quickly fill a running llama-server's context to a target token count.

Companion to bench.py. Where bench.py measures throughput on fixed prompts,
fill.py sends ONE prompt sized to ~N tokens so you can:
  - warm/populate the prompt (KV) cache to a known depth,
  - stress-test how deep the context fits before the server OOMs,
  - measure prefill (prompt-processing) speed at that depth.

Token sizing is exact: it calibrates against the server's /tokenize endpoint
(falls back to a ~tokens/word estimate if that endpoint is unavailable).

Usage:
    ./fill.py -t 1000                  # fill ~1000 prompt tokens
    ./fill.py -t 100000 --gen 16       # deep fill, generate 16 tokens too
    HOST=http://1.2.3.4:8080 ./fill.py -t 50000
    ./fill.py -t 1000,8000,32000       # sweep several depths in one run

Requires only the Python stdlib.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request


# One repeated unit of filler. ~10 tokens for most tokenizers; the exact
# per-unit count is measured at runtime via /tokenize when available.
FILLER_UNIT = "The quick brown fox jumps over the lazy dog. "


def post_json(host: str, path: str, body: dict, timeout: int):
    req = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def tokens_per_unit(host: str, timeout: int) -> float:
    """Measure how many tokens FILLER_UNIT costs, via /tokenize. Estimate on fail."""
    try:
        # Tokenize a 100x block so per-unit cost averages out boundary effects.
        block = FILLER_UNIT * 100
        out = post_json(host, "/tokenize", {"content": block}, timeout)
        toks = out.get("tokens", out) if isinstance(out, dict) else out
        n = len(toks)
        if n > 0:
            return n / 100.0
    except Exception:
        pass
    # Fallback: ~1.3 tokens/word * 9 words ≈ 10 (plus the trailing period/space).
    return 10.0


def build_prompt(target_tokens: int, tpu: float) -> str:
    repeats = max(1, math.ceil(target_tokens / tpu))
    return FILLER_UNIT * repeats


def fill_once(host: str, model: str, target: int, tpu: float,
              gen: int, timeout: int) -> dict:
    """Send one fill request. Returns a dict of measured values or raises."""
    user = (
        "Read the following text carefully; you will be asked about it later.\n\n"
        + build_prompt(target, tpu)
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
        ],
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
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
            if ttft is None:
                choices = obj.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    ttft = time.perf_counter() - t0
            if "timings" in obj:
                timings = obj["timings"]
            if obj.get("usage"):
                usage = obj["usage"]

    wall = time.perf_counter() - t0
    prompt_n = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
    predicted_n = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
    cached_n = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return {
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "cached_n": cached_n,
        "prompt_tps": float(timings.get("prompt_per_second") or 0.0),
        "gen_tps": float(timings.get("predicted_per_second") or 0.0),
        "ttft_s": ttft if ttft is not None else wall,
        "wall_s": wall,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-t", "--tokens", default="1000",
                   help="target prompt token count; comma-separated to sweep "
                        "(e.g. 1000,8000,32000). default: %(default)s")
    p.add_argument("--host", default=os.environ.get("HOST", "http://localhost:8080"),
                   help="llama-server base URL (default: %(default)s)")
    p.add_argument("--model", default=os.environ.get("MODEL", "default"),
                   help="model name to send (default: %(default)s)")
    p.add_argument("--gen", type=int, default=1,
                   help="tokens to generate after the fill (default: 1)")
    p.add_argument("--timeout", type=int, default=1200,
                   help="per-request timeout in seconds (default: 1200)")
    args = p.parse_args()

    try:
        targets = [int(x) for x in str(args.tokens).split(",") if x.strip()]
    except ValueError:
        print(f"bad --tokens value: {args.tokens!r}", file=sys.stderr)
        return 2
    if not targets:
        print("no targets given", file=sys.stderr)
        return 2

    print(f"Target: {args.host}   model={args.model}")
    tpu = tokens_per_unit(args.host, args.timeout)
    print(f"Calibration: ~{tpu:.2f} tokens per filler unit\n")

    rc = 0
    for target in targets:
        print(f"fill -> {target:>8,} tokens")
        try:
            r = fill_once(args.host, args.model, target, tpu, args.gen, args.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            print(f"  HTTP {e.code} (likely OOM / over context): {detail}\n",
                  file=sys.stderr)
            rc = 1
            continue
        except Exception as e:
            print(f"  ERROR: {e}\n", file=sys.stderr)
            rc = 1
            continue

        cached = f"  cached={r['cached_n']:>7,}t" if r["cached_n"] else ""
        print(
            f"  prompt={r['prompt_n']:>8,}t @ {r['prompt_tps']:8.1f} t/s  "
            f"ttft={r['ttft_s']:7.2f}s  "
            f"gen={r['predicted_n']:>3}t @ {r['gen_tps']:6.1f} t/s  "
            f"total={r['wall_s']:7.2f}s{cached}\n"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
