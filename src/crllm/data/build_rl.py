"""Stage 4 prompts -> data/rl_prompts.jsonl (prompt, task, answer, tests, test_imports, source, key).

Mix (data.stage4_mix):
  math  nvidia/OpenMathInstruct-2 problems with their expected answer (half GSM8K-style,
        half MATH-style), checked by rewards.math_correct
  code  nvidia/OpenCodeInstruct tasks with their unit-test asserts; kept only if the
        reference solution passes 100% of its own asserts through our runner
  chat  Alpaca-cleaned instructions (no input field); no ground truth, format reward only

Questions in data/eval/ or in the Stage 1 SFT set are excluded (format.prompt_key), and so
are prompts longer than grpo.max_prompt_length tokens. Uses the shards Stage 1 downloaded.

    python -m crllm.data.build_rl --config debug
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pyarrow.parquet as pq

from crllm.config import add_config_args, config_from_args, write_run_json
from crllm.data.build_sft import SOURCES as SFT_SOURCES
from crllm.data.build_sft import eval_keys, targets
from crllm.data.download import fetch_shards
from crllm.eval.execute import fraction_passed
from crllm.format import build_prompt, extract_code, prompt_key

log = logging.getLogger("crllm.build_rl")


def math_items(path, rng, max_chars=600):
    t = pq.read_table(path, columns=["problem", "expected_answer", "problem_source"]).to_pydict()
    by_src: dict[str, list[int]] = {}
    for i, (q, a, s) in enumerate(zip(t["problem"], t["expected_answer"], t["problem_source"])):
        if q and a and len(q) <= max_chars and len(str(a)) <= 40:
            by_src.setdefault("gsm8k" if "gsm8k" in str(s) else "math", []).append(i)
    for v in by_src.values():
        rng.shuffle(v)
    # alternate GSM8K-style and MATH-style so both difficulty levels are present
    for g, m in zip(by_src.get("gsm8k", []), by_src.get("math", [])):
        for i, src in ((g, "omi_gsm8k"), (m, "omi_math")):
            yield {"question": t["problem"][i], "task": "math", "answer": str(t["expected_answer"][i]).strip(),
                   "tests": [], "test_imports": [], "source": src}


def code_items(path, rng, max_chars=1200):
    t = pq.read_table(path, columns=["input", "output", "unit_tests", "average_test_score"]).to_pydict()
    idx = [i for i, s in enumerate(t["average_test_score"])
           if s is not None and float(s) == 1.0 and len(t["input"][i]) <= max_chars and "```python" in t["output"][i]]
    rng.shuffle(idx)
    for i in idx:
        try:
            tests = [x.strip() for x in json.loads(t["unit_tests"][i]) if x.strip()]
        except (json.JSONDecodeError, TypeError):
            continue
        if 2 <= len(tests) <= 20:
            yield {"question": t["input"][i], "task": "code", "answer": None, "tests": tests, "test_imports": [],
                   "source": "oci", "reference": extract_code(t["output"][i])}


def chat_items(path, rng):
    t = pq.read_table(path, columns=["instruction", "input"]).to_pydict()
    idx = [i for i, (q, x) in enumerate(zip(t["instruction"], t["input"])) if not (x or "").strip() and 15 <= len(q) <= 400]
    rng.shuffle(idx)
    for i in idx:
        yield {"question": t["instruction"][i], "task": "chat", "answer": None, "tests": [], "test_imports": [],
               "source": "alpaca"}


def main(argv: list[str] | None = None):
    ap = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--n", type=int, default=None, help="override data.stage4_prompts")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = config_from_args(args)
    from transformers import AutoTokenizer

    data_dir, rng = cfg.path("data_dir"), random.Random(cfg.seed + 4)
    n = args.n or cfg.data.stage4_prompts
    want = targets(asdict(cfg.data.stage4_mix), n)
    tok = AutoTokenizer.from_pretrained(cfg.model.policy)
    banned = eval_keys(data_dir)
    sft = data_dir / "sft_think.jsonl"
    if sft.exists():
        banned |= {json.loads(line)["key"] for line in sft.open(encoding="utf-8")}
    raw = data_dir / "raw" / "stage1"
    omi = fetch_shards(*SFT_SOURCES["omi_math"][:3], raw, 1)[0]
    oci = fetch_shards(*SFT_SOURCES["oci_code"][:3], raw, 1)[0]
    alpaca = data_dir / "eval" / "raw" / "alpaca.parquet"
    if not alpaca.exists():
        raise SystemExit(f"{alpaca} missing - run scripts/download_eval.py first")
    log.info("target %s; %d banned keys (eval + Stage 1)", want, len(banned))

    seen, rejects = set(), Counter()

    def take(items, k):
        out = []
        for it in items:
            if len(out) >= k:
                break
            key = prompt_key(it["question"])
            if key in banned or key in seen:
                rejects["banned_or_dup"] += 1
                continue
            try:
                prompt = build_prompt(it["question"])
            except ValueError:
                rejects["bad_prompt"] += 1
                continue
            if len(tok(prompt, add_special_tokens=False)["input_ids"]) > cfg.grpo.max_prompt_length:
                rejects["prompt_too_long"] += 1
                continue
            seen.add(key)
            out.append({**it, "prompt": prompt, "key": key})
        return out

    rows = take(math_items(omi, rng), want.get("math", 0))
    rows += take(chat_items(alpaca, rng), want.get("chat", 0))

    # code: verify the asserts against the reference solution; keep only tasks it fully passes
    need, code_rows, pool = want.get("code", 0), [], code_items(oci, rng)
    while len(code_rows) < need:
        batch = take(pool, int((need - len(code_rows)) * 1.3) + 8)
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=6) as ex:
            ok = list(ex.map(lambda r: fraction_passed(r["reference"], r["tests"]) == 1.0, batch))
        rejects["code_reference_failed"] += ok.count(False)
        code_rows += [r for r, good in zip(batch, ok) if good][: need - len(code_rows)]
    for r in code_rows:
        r.pop("reference", None)
    rows += code_rows

    rng.shuffle(rows)
    out = data_dir / "rl_prompts.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"n": len(rows), "by_task": dict(Counter(r["task"] for r in rows)), "rejects": dict(rejects)}
    write_run_json(data_dir / "runs" / "build_rl", cfg, stats=stats)
    log.info("wrote %s: %s", out, stats)
    return out


if __name__ == "__main__":
    main()
