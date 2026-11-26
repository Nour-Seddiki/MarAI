"""Any checkpoint -> one JSON report (+ a samples file with every completion and verdict).

    python -m crllm.eval.run_eval --config debug --ckpt base
    python -m crllm.eval.run_eval --config debug --ckpt stage1 --suites format,gsm8k
    python -m crllm.eval.run_eval --config debug --ckpt path/to/model --limit 20 --name smoke

Writes reports/<config>/<name>.json and reports/<config>/<name>.samples.jsonl. Greedy decoding,
same prompts every time. Generation for all suites runs first; the model is freed before
judging, so the policy and a local judge never share the GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from crllm.config import add_config_args, config_from_args, load_constitution, run_metadata
from crllm.eval.suites import JUDGED, SCORERS, SUITES, common_metrics, headline, load_suite
from crllm.judge import Judge
from crllm.runtime import apply_runtime

log = logging.getLogger("crllm.eval")


def resolve_ckpt(cfg, ckpt: str) -> tuple[str, str]:
    """(label, model path or hub id) for base | stage1 | stage2 | final | <path>."""
    ck = cfg.path("ckpt_dir")
    named = {"base": cfg.model.policy, "stage1": str(ck / "stage1-merged"),
             "stage2": str(ck / "stage2-merged"), "final": str(ck / "final")}
    if ckpt in named:
        path = named[ckpt]
        if ckpt != "base" and not Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist yet")
        return ckpt, path
    p = Path(ckpt)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint {ckpt!r} is neither a known name nor an existing path")
    return p.name, str(p)


def main(argv: list[str] | None = None) -> dict:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--ckpt", default="base", help="base | stage1 | stage2 | final | path")
    ap.add_argument("--suites", default=None, help="comma-separated subset (default: eval.suites)")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts per suite (smoke runs)")
    ap.add_argument("--name", default=None, help="report name (default: the checkpoint label)")
    ap.add_argument("--skip-judge", action="store_true", help="score only format and gsm8k")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per judge call otherwise

    cfg = config_from_args(args)
    applied = apply_runtime(cfg.runtime)
    label, model_path = resolve_ckpt(cfg, args.ckpt)
    name = args.name or label
    suites = args.suites.split(",") if args.suites else list(cfg.eval.suites)
    unknown = set(suites) - set(SUITES)
    if unknown:
        raise SystemExit(f"unknown suite(s): {sorted(unknown)}")

    data = {}
    for s in suites:
        limit = args.limit or (cfg.eval.gsm8k_n if s == "gsm8k" else None)
        data[s] = load_suite(cfg.path("data_dir"), s, limit)

    judge, judge_error = None, None
    if any(s in JUDGED for s in suites) and not args.skip_judge:
        judge = Judge(cfg.judge, load_constitution(cfg.path("constitution")))
        judge_error = judge.list_models_error()
        if judge_error:
            log.warning("judge unavailable, judged suites will be skipped: %s", judge_error)
            judge = None

    # ---- generate everything in one sorted pass
    from crllm.generate import make_generator

    t0 = time.time()
    gen = make_generator(cfg, model_path, batch_size=cfg.eval.batch_size)
    flat = [(s, i) for s in suites for i in range(len(data[s]))]
    comps_flat = gen.generate([data[s][i]["prompt"] for s, i in flat],
                              max_new_tokens=cfg.eval.max_new_tokens, temperature=0.0)
    gen_seconds = time.time() - t0
    del gen
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    comps = {s: [] for s in suites}
    for (s, _), c in zip(flat, comps_flat):
        comps[s].append(c[0])

    # ---- score
    report_suites, sample_rows = {}, []
    for s in suites:
        if s in JUDGED and judge is None:
            report_suites[s] = {"skipped": judge_error or "--skip-judge", **common_metrics(comps[s])}
            scored_samples = [{} for _ in data[s]]
        else:
            scored = SCORERS[s](data[s], comps[s], judge)
            report_suites[s] = {**scored.metrics, **common_metrics(comps[s])}
            scored_samples = scored.samples
        for r, c, extra in zip(data[s], comps[s], scored_samples):
            sample_rows.append({"id": r["id"], "suite": s, "question": r["question"], "answer": r["answer"],
                                "completion": c.text, "n_tokens": c.n_tokens, "finish": c.finish, **extra})

    report = {
        "name": name,
        "ckpt": args.ckpt,
        "model_path": model_path,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decoding": {"temperature": 0.0, "max_new_tokens": cfg.eval.max_new_tokens,
                     "backend": cfg.generation.backend, "batch_size": cfg.eval.batch_size,
                     "cache": cfg.generation.cache},
        "judge": None if judge is None else {"model": cfg.judge.model, "base_url": cfg.judge.base_url},
        "generation_seconds": round(gen_seconds, 1),
        "headline": {s: headline(s, m) for s, m in report_suites.items() if "skipped" not in m},
        "suites": report_suites,
        "run": run_metadata(cfg, runtime=applied),
    }
    out_dir = cfg.path("reports_dir") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / f"{name}.samples.jsonl", "w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("wrote %s", out_dir / f"{name}.json")
    for s, h in report["headline"].items():
        if h and h.get("value") is not None:
            log.info("  %-12s %.3f ± %.3f (n=%d)", s, h["value"], h["se"] or 0.0, h["n"])
    return report


if __name__ == "__main__":
    main()
