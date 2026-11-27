"""Block averages of GRPO metrics from a training log (one dict per logged step).

    python scripts/grpo_progress.py path/to/log --block 10
"""
from __future__ import annotations

import argparse
import ast
import re

KEYS = [("rewards/correctness_reward/mean", "correct"), ("rewards/format_reward/mean", "format"),
        ("completions/clipped_ratio", "clipped"), ("completions/mean_length", "length"),
        ("frac_reward_zero_std", "no_signal"), ("kl", "kl"), ("step_time", "s/step")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0, help="step number before the log's first entry (resumed runs)")
    args = ap.parse_args()
    text = open(args.log, encoding="utf-8", errors="replace").read()
    steps = [{k: float(v) for k, v in ast.literal_eval(m).items() if k in dict(KEYS) and v is not None}
             for m in re.findall(r"\{'loss'[^}]*\}", text)]
    print(f"{'steps':>9} " + " ".join(f"{name:>9}" for _, name in KEYS))
    for b in range(0, len(steps), args.block):
        chunk = steps[b: b + args.block]
        avg = {k: sum(s.get(k, 0.0) for s in chunk) / len(chunk) for k, _ in KEYS}
        cells = [f"{avg[k]:9.4f}" if k == "kl" else f"{avg[k]:9.1f}" if k in ("completions/mean_length", "step_time")
                 else f"{avg[k]:9.2f}" for k, _ in KEYS]
        print(f"{args.offset + b + 1:>4}-{args.offset + b + len(chunk):<4} " + " ".join(cells))


if __name__ == "__main__":
    main()
