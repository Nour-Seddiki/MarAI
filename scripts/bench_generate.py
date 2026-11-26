"""Benchmark HF generation settings on fixed eval prompts: speed, peak memory, output agreement.

Runs every (batch_size, cache) setting on the same prompts with greedy decoding, so the
outputs should match the first setting; the 'same' column says how many do (tiny numeric
differences between batch shapes can flip a token late in a long completion).

    python scripts/bench_generate.py --config debug
    python scripts/bench_generate.py --config debug --settings 16:dynamic,16:static --prompts 16
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time

import torch

from crllm.config import add_config_args, config_from_args
from crllm.eval.suites import load_suite
from crllm.generate import HFGenerator
from crllm.modeling import load_base
from crllm.runtime import apply_runtime


def main() -> None:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", default=None, help="model path or hub id (default: cfg.model.policy)")
    ap.add_argument("--settings", default="8:dynamic,16:dynamic,32:dynamic,16:static,32:static")
    ap.add_argument("--prompts", type=int, default=32, help="prompts per setting (split across suites)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: eval.max_new_tokens")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = config_from_args(args)
    apply_runtime(cfg.runtime)

    suites = ["format", "gsm8k", "redteam", "overrefusal"]
    per = max(1, args.prompts // len(suites))
    prompts = [r["prompt"] for s in suites for r in load_suite(cfg.path("data_dir"), s, per)][: args.prompts]
    max_new = args.max_new_tokens or cfg.eval.max_new_tokens
    model, tok = load_base(cfg, args.ckpt, for_training=False, load_in_4bit=False)
    cap = cfg.runtime.cuda_memory_fraction
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{len(prompts)} prompts, max_new_tokens={max_new}, model={args.ckpt or cfg.model.policy}, "
          f"memory cap={cap * total if cap else total:.1f} GiB")

    rows, ref = [], None
    for setting in args.settings.split(","):
        bs, cache = setting.split(":")
        gen = HFGenerator(model, tok, batch_size=int(bs), cache=cache)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            out = gen.generate(prompts, max_new_tokens=max_new)
        except Exception as e:  # noqa: BLE001 - OOM, compile failures: record and move on
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:100] if str(e) else ''}"
            print(f"{setting:>12}: FAILED {msg}", flush=True)
            rows.append({"setting": setting, "error": msg})
            continue
        dt = time.time() - t0
        texts = [o[0].text for o in out]
        ntok = sum(o[0].n_tokens for o in out)
        ref = ref or texts
        row = {
            "setting": setting, "seconds": round(dt, 1), "s_per_prompt": round(dt / len(prompts), 2),
            "tokens_per_s": round(ntok / dt), "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "same_as_first": f"{sum(a == b for a, b in zip(texts, ref))}/{len(texts)}",
        }
        rows.append(row)
        print(f"{setting:>12}: {row['s_per_prompt']:>5} s/prompt  {row['tokens_per_s']:>5} tok/s  "
              f"peak {row['peak_reserved_gib']} GiB reserved / {row['peak_allocated_gib']} allocated  "
              f"same {row['same_as_first']}", flush=True)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
