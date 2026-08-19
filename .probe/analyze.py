#!/usr/bin/env python3
"""Summarise matrix cells and Fisher-test each candidate against its baseline.

A trial is USABLE when the model actually produced the refine deliverable: it
stopped calling tools AND all four contract headings survived. "Never ended" is
the loop symptom -- the model never stopped calling tools before the cap.

Both are reported because a change that stops the loop by wrecking the answer is
not a fix, and neither is one that keeps the answer but still hangs.
"""
import json
import sys
import glob
import os
from math import comb


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c

    def p(x):
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    obs = p(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12))


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith('{'):
            out.append(json.loads(line))
    return out


def summarise(all_rows):
    if not all_rows:
        return None
    # The llama-server MTP speculative-decode path segfaults ("Aborted (core
    # dumped)" in common_speculative_draft) and the container restarts. Those
    # trials measure the CRASH, not the model, so they are void -- counting them
    # as failures would inflate whichever cell happened to be running.
    rows = [r for r in all_rows if not r['outcome'].startswith('http-error')]
    void = len(all_rows) - len(rows)
    n = len(rows)
    if n == 0:
        return {'n': 0, 'void': void, 'usable': 0, 'never': 0, 'maxCalls': 0,
                'medCalls': 0, 'maxSec': 0, 'medSec': 0, 'enoent': 0, 'reads': 0,
                'think': all_rows[0].get('think'), 'strip': all_rows[0].get('stripNoThink'),
                'thinkChars': 0}
    ended = [r for r in rows if r['outcome'] == 'answered']
    usable = [r for r in ended if r['sections'] == 4]
    never = n - len(ended)
    calls = [r['toolCalls'] for r in rows]
    secs = [r['seconds'] for r in rows]
    return {
        'n': n, 'void': void, 'usable': len(usable), 'never': never,
        'maxCalls': max(calls), 'medCalls': sorted(calls)[n // 2],
        'maxSec': max(secs), 'medSec': sorted(secs)[n // 2],
        'enoent': sum(r['enoent'] for r in rows),
        'reads': sum(r['reads'] for r in rows),
        'think': rows[0].get('think'), 'strip': rows[0].get('stripNoThink'),
        'thinkChars': sum(r.get('thinkChars', 0) for r in rows),
    }


d = sys.argv[1] if len(sys.argv) > 1 else '.'
cells = {}
for f in sorted(glob.glob(os.path.join(d, '*.jsonl'))):
    name = os.path.basename(f)[:-6]
    s = summarise(load(f))
    if s:
        cells[name] = s

hdr = (f"{'cell':<10}{'n':>4}{'void':>6}{'usable':>8}{'never':>7}{'medCall':>9}{'maxCall':>9}"
       f"{'medSec':>8}{'maxSec':>8}{'ENOENT':>8}{'think':>8}{'strip':>7}")
print(hdr)
print('-' * len(hdr))
for k, s in cells.items():
    print(f"{k:<10}{s['n']:>4}{s['void']:>6}{s['usable']:>8}{s['never']:>7}{s['medCalls']:>9}"
          f"{s['maxCalls']:>9}{s['medSec']:>8}{s['maxSec']:>8}{s['enoent']:>8}"
          f"{str(s['think']):>8}{str(s['strip']):>7}")

base = sys.argv[2] if len(sys.argv) > 2 else None
if base and base in cells:
    b = cells[base]
    print(f"\nFisher exact vs baseline {base}:")
    for k, s in cells.items():
        if k == base:
            continue
        pn = fisher(b['never'], b['n'] - b['never'], s['never'], s['n'] - s['never'])
        pu = fisher(b['usable'], b['n'] - b['usable'], s['usable'], s['n'] - s['usable'])
        print(f"  {k:<10} never-ended {b['never']}/{b['n']} -> {s['never']}/{s['n']}  p={pn:.4f}"
              f"   |  usable {b['usable']}/{b['n']} -> {s['usable']}/{s['n']}  p={pu:.4f}")
