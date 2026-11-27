"""Re-judge saved eval samples with other judge models and compare with the original verdicts.

    python scripts/compare_judges.py --samples reports/debug/base.samples.jsonl \
        --models qwen3.5:9b,qwen2.5-coder:7b --n 40

For each model: seconds per verdict, unparsed replies, agreement with the verdicts in the
samples file, the score distribution, and a few disagreements to read by hand.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter

from crllm.config import add_config_args, config_from_args, load_constitution
from crllm.eval.suites import visible_reply
from crllm.judge import Judge

SUITE_FN = {"redteam": "score", "overrefusal": "refusal", "honesty": "uncertainty"}


def main() -> None:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--samples", required=True)
    ap.add_argument("--models", required=True, help="comma-separated judge model names")
    ap.add_argument("--n", type=int, default=40, help="samples per judged suite (honesty uses n/2)")
    ap.add_argument("--reasoning-effort", default="none")
    args = ap.parse_args()
    cfg = config_from_args(args)
    con = load_constitution(cfg.path("constitution"))

    rows = [json.loads(line) for line in open(args.samples, encoding="utf-8")]
    picked = []
    for suite in SUITE_FN:
        n = args.n // 2 if suite == "honesty" else args.n
        picked += [r for r in rows if r["suite"] == suite][:n]

    for model in args.models.split(","):
        jcfg = dataclasses.replace(cfg.judge, model=model, reasoning_effort=args.reasoning_effort or None)
        judge = Judge(jcfg, con)
        judge.chat('Reply with JSON only: {"ok": true}', 8)  # load the model before timing
        t0 = time.time()
        verdicts = [getattr(judge, SUITE_FN[r["suite"]])(r["question"], visible_reply(r["completion"])) for r in picked]
        dt = time.time() - t0
        print(f"\n=== {model}: {dt / len(picked):.2f} s/verdict, unparsed {sum(not v.ok for v in verdicts)}/{len(picked)}")
        for suite in SUITE_FN:
            pairs = [(r["judge"], v.value) for r, v in zip(picked, verdicts) if r["suite"] == suite and v.ok]
            agree = sum(a == b for a, b in pairs)
            line = f"  {suite:12s} agree with original {agree}/{len(pairs)}"
            if suite == "redteam":
                within1 = sum(abs(a - b) <= 1 for a, b in pairs if a is not None)
                line += f", within 1 point {within1}/{len(pairs)}, original {sorted(Counter(a for a, _ in pairs).items())}" \
                        f" -> new {sorted(Counter(b for _, b in pairs).items())}"
            else:
                line += f", original {dict(Counter(a for a, _ in pairs))} -> new {dict(Counter(b for _, b in pairs))}"
            print(line)
        shown = 0
        for r, v in zip(picked, verdicts):
            if v.ok and v.value != r["judge"] and shown < 6:
                shown += 1
                print(f"  - [{r['suite']}] {r['question'][:70]!r}\n      reply: {visible_reply(r['completion'])[:150]!r}"
                      f"\n      original={r['judge']} new={v.value}")


if __name__ == "__main__":
    main()
