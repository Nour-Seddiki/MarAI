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


def load_completions(samples_path: Path, data: dict[str, list[dict]]):
    """Completions from an earlier samples file, in the order of `data`; every id must be present."""
    from crllm.generate import Completion

    by_id = {}
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]] = Completion(r["completion"], r["n_tokens"], r["finish"])
    missing = [r["id"] for rows in data.values() for r in rows if r["id"] not in by_id]
    if missing:
        raise SystemExit(f"{samples_path} lacks {len(missing)} prompt(s), e.g. {missing[:3]}; "
                         "re-generate or use the same --suites/--limit as the original run")
    return {s: [by_id[r["id"]] for r in rows] for s, rows in data.items()}


def load_partial(path: Path, model_path: str, max_new_tokens: int) -> dict[str, tuple]:
    """Completions checkpointed by a stopped run of the same model and settings ({} otherwise)."""
    if not path.exists():
        return {}
    done: dict[str, tuple] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line cut off by the stop
                continue
            if n == 0:
                h = r.get("_header", {})
                if h.get("model_path") != model_path or h.get("max_new_tokens") != max_new_tokens:
                    raise SystemExit(f"{path} is from a different run ({h}); pass --fresh or use another --name")
                continue
            done[r["id"]] = (r["completion"], r["n_tokens"], r["finish"])
    return done


def main(argv: list[str] | None = None) -> dict:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--ckpt", default="base", help="base | stage1 | stage2 | final | path")
    ap.add_argument("--suites", default=None, help="comma-separated subset (default: eval.suites)")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts per suite (smoke runs)")
    ap.add_argument("--name", default=None, help="report name (default: the checkpoint label)")
    ap.add_argument("--skip-judge", action="store_true", help="score only format and gsm8k")
    ap.add_argument("--fresh", action="store_true", help="ignore a <name>.partial.jsonl from a stopped run")
    ap.add_argument("--from-samples", default=None, metavar="SAMPLES_JSONL",
                    help="re-score completions from an earlier samples file instead of generating "
                         "(e.g. after changing the judge, so every stage is scored by the same one)")
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

    comps = {s: [] for s in suites}
    t0 = time.time()
    out_dir = cfg.path("reports_dir") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{name}.partial.jsonl"
    if args.from_samples:
        comps = load_completions(Path(args.from_samples), data)
        gen_seconds = None
    else:
        # ---- generate everything in one sorted pass, checkpointing each batch to <name>.partial.jsonl
        from crllm.generate import Completion, make_generator

        if args.fresh and partial.exists():
            partial.unlink()
        done = load_partial(partial, model_path, cfg.eval.max_new_tokens)
        flat = [(s, r["id"]) for s in suites for r in data[s]]
        todo = [k for k in flat if k[1] not in done]
        prompt_of = {r["id"]: r["prompt"] for s in suites for r in data[s]}
        if done:
            log.info("resuming: %d completions from %s, %d to generate", len(flat) - len(todo), partial.name, len(todo))
        if todo:
            gen = make_generator(cfg, model_path, batch_size=cfg.eval.batch_size)
            with partial.open("a", encoding="utf-8") as fp:
                if fp.tell() == 0:
                    fp.write(json.dumps({"_header": {"model_path": model_path,
                                                     "max_new_tokens": cfg.eval.max_new_tokens}}) + "\n")

                def checkpoint(idx, outs):
                    for i, o in zip(idx, outs):
                        c = o[0]
                        done[todo[i][1]] = c
                        fp.write(json.dumps({"id": todo[i][1], "completion": c.text, "n_tokens": c.n_tokens,
                                             "finish": c.finish}, ensure_ascii=False) + "\n")
                    fp.flush()

                gen.generate([prompt_of[i] for _, i in todo], max_new_tokens=cfg.eval.max_new_tokens,
                             temperature=0.0, on_batch=checkpoint)
            del gen
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        gen_seconds = round(time.time() - t0, 1)
        for s, i in flat:
            c = done[i]
            comps[s].append(c if isinstance(c, Completion) else Completion(*c))

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
        "generation_seconds": gen_seconds,
        "rescored_from": args.from_samples,
        "headline": {s: headline(s, m) for s, m in report_suites.items() if "skipped" not in m},
        "suites": report_suites,
        "run": run_metadata(cfg, runtime=applied),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / f"{name}.samples.jsonl", "w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if partial.exists():
        partial.unlink()  # the samples file now holds every completion
    log.info("wrote %s", out_dir / f"{name}.json")
    for s, h in report["headline"].items():
        if h and h.get("value") is not None:
            log.info("  %-12s %.3f ± %.3f (n=%d)", s, h["value"], h["se"] or 0.0, h["n"])
    return report


if __name__ == "__main__":
    main()
