"""Print the train / validation loss table from a training log or a trainer_state.json.

    python scripts/show_losses.py ckpt/stage1/checkpoint-200/trainer_state.json
    python scripts/show_losses.py path/to/training.log --steps-per-epoch 122
"""
from __future__ import annotations

import argparse
import ast
import json
import re


def from_log(text: str, steps_per_epoch: float | None) -> list[dict]:
    rows = []
    for m in re.findall(r"\{'(?:loss|eval_loss)'[^}]*\}", text):
        r = {k: float(v) for k, v in ast.literal_eval(m).items()}
        if steps_per_epoch:
            r["step"] = round(r["epoch"] * steps_per_epoch)
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--steps-per-epoch", type=float, default=None, help="for logs, to show step numbers")
    args = ap.parse_args()
    if args.path.endswith(".json"):
        rows = [r for r in json.load(open(args.path, encoding="utf-8"))["log_history"] if "loss" in r or "eval_loss" in r]
    else:
        rows = from_log(open(args.path, encoding="utf-8", errors="replace").read(), args.steps_per_epoch)
    print(f"{'step':>5} {'epoch':>6} {'train loss':>10} {'val loss':>9} {'lr':>9}")
    for r in rows:
        step = r.get("step", "")
        if "eval_loss" in r:
            print(f"{step:>5} {r['epoch']:>6.2f} {'':>10} {r['eval_loss']:>9.4f}")
        else:
            print(f"{step:>5} {r['epoch']:>6.2f} {r['loss']:>10.4f} {'':>9} {r.get('learning_rate', 0):>9.2e}")


if __name__ == "__main__":
    main()
