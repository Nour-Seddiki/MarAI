"""Stage 1 data: <think>-format SFT examples from math and code sources -> data/sft_think.jsonl.

Sources (shares set by data.stage1_mix; a source that runs short is topped up by the others):
  mot_math  open-r1/Mixture-of-Thoughts math  real DeepSeek-R1 <think> traces, kept if they fit
  mot_code  open-r1/Mixture-of-Thoughts code  same, code (only ~3.5% fit 2,048 tokens, none 1,024)
  omi_math  nvidia/OpenMathInstruct-2         short step-by-step solutions -> <think>solution</think> boxed answer
  oci_code  nvidia/OpenCodeInstruct           test-verified code (average_test_score == 1); the judge
                                              endpoint writes a short <think> plan for each one

Every example is re-emitted through format.wrap() and stored as a completion after the
<think> prefill (format.to_completion), counted with the policy tokenizer
(prompt + completion <= train.max_seq_len), de-duplicated, and dropped if its question
appears in data/eval/ (by format.prompt_key). Teacher plans are cached in
data/cache/teacher_plans.jsonl, so an interrupted build resumes where it stopped.

    python -m crllm.data.build_sft --config debug        # 2k examples, 1,024-token cap
    python -m crllm.data.build_sft --config full         # 20k examples, 2,048-token cap
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator

import pyarrow.parquet as pq

from crllm.config import add_config_args, config_from_args, write_run_json
from crllm.data.download import fetch_shards
from crllm.format import build_prompt, is_prefilled, parse, prompt_key, to_completion, wrap
from crllm.judge import Judge

log = logging.getLogger("crllm.build_sft")

SOURCES = {
    "mot_math": ("open-r1/Mixture-of-Thoughts", "math", "train", None),
    "mot_code": ("open-r1/Mixture-of-Thoughts", "code", "train", None),
    "omi_math": ("nvidia/OpenMathInstruct-2", "default", "train", 1),
    "oci_code": ("nvidia/OpenCodeInstruct", "train", "train", 1),
}

TEACHER_PROMPT = """You are writing the private reasoning a careful programmer does before answering a coding task.

Task:
<<<{task}>>>

A correct solution (for your reference only):
<<<{solution}>>>

Write the step-by-step thinking that leads to this solution, in 3 to 8 short plain-text sentences: restate what is needed, note inputs, edge cases and constraints, and choose the approach and data structures. Think as if you have not seen the solution yet: do not mention it, do not write any code, and do not use code blocks. Reply with the reasoning only."""


# ---------------------------------------------------------------- sources


def _messages_pair(messages) -> tuple[str, str] | None:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
    asst = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), None)
    return (user, asst) if user and asst else None


def read_mot(paths: list[Path], max_tokens: int, rng: random.Random) -> Iterator[dict]:
    """R1 traces that can fit: prefilter on the dataset's own num_tokens, re-emit via wrap()."""
    order = list(paths)
    rng.shuffle(order)
    for path in order:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["messages", "num_tokens"], batch_size=4096):
            d = batch.to_pydict()
            idx = [i for i, n in enumerate(d["num_tokens"]) if n is not None and n <= max_tokens]
            rng.shuffle(idx)
            for i in idx:
                pair = _messages_pair(d["messages"][i])
                if not pair:
                    continue
                p = parse(pair[1])
                if not p.well_formed:
                    continue
                try:
                    completion = wrap(p.reasoning, p.answer)
                except ValueError:
                    continue
                yield {"question": pair[0], "completion": completion}


def read_omi(path: Path, max_chars: int, rng: random.Random) -> Iterator[dict]:
    t = pq.read_table(path, columns=["problem", "generated_solution", "expected_answer"]).to_pydict()
    idx = [i for i, s in enumerate(t["generated_solution"])
           if s and len(s) + len(t["problem"][i]) <= max_chars and t["expected_answer"][i]]
    rng.shuffle(idx)
    for i in idx:
        sol, ans = t["generated_solution"][i].strip(), str(t["expected_answer"][i]).strip()
        try:
            completion = wrap(sol, f"The final answer is $\\boxed{{{ans}}}$.")
        except ValueError:
            continue
        yield {"question": t["problem"][i], "completion": completion}


def read_oci(path: Path, max_chars: int, rng: random.Random) -> Iterator[dict]:
    """Verified code candidates; the completion is filled in later by the teacher."""
    t = pq.read_table(path, columns=["id", "input", "output", "average_test_score"]).to_pydict()
    idx = [i for i, s in enumerate(t["average_test_score"])
           if s is not None and float(s) == 1.0 and "```python" in t["output"][i]
           and len(t["input"][i]) + len(t["output"][i]) <= max_chars]
    rng.shuffle(idx)
    for i in idx:
        yield {"question": t["input"][i], "solution": t["output"][i].strip(), "src_id": str(t["id"][i])}


# ---------------------------------------------------------------- teacher


_BAD_PLAN = re.compile(r"```|\bdef \w+\(|provided solution|the solution above|reference solution", re.I)


class PlanCache:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.plans: dict[str, str | None] = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                r = json.loads(line)
                self.plans[r["id"]] = r["plan"]

    def put(self, key: str, plan: str | None) -> None:
        with self.lock:
            self.plans[key] = plan
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": key, "plan": plan}, ensure_ascii=False) + "\n")


def clean_plan(text: str) -> str | None:
    text = (text or "").strip()
    if not text or len(text) > 1500 or _BAD_PLAN.search(text) or "<think>" in text or "</think>" in text:
        return None
    return text


def teacher_plans(judge: Judge, items: list[dict], cache: PlanCache, max_tokens: int) -> None:
    todo = [it for it in items if it["src_id"] not in cache.plans]
    if not todo:
        return
    t0, done = time.time(), [0]

    def one(it):
        try:
            raw = judge.chat(TEACHER_PROMPT.format(task=it["question"], solution=it["solution"]), max_tokens)
        except RuntimeError as e:
            log.warning("teacher failed for %s: %s", it["src_id"], e)
            return
        cache.put(it["src_id"], clean_plan(raw))
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(todo):
            rate = (time.time() - t0) / done[0]
            log.info("teacher plans %d/%d (%.1f s each, ~%.0f min left)", done[0], len(todo), rate,
                     rate * (len(todo) - done[0]) / 60)

    with ThreadPoolExecutor(max_workers=max(1, judge.cfg.concurrency)) as ex:
        list(ex.map(one, todo))


# ---------------------------------------------------------------- selection


def eval_keys(data_dir: Path) -> set[str]:
    keys = set()
    for p in (data_dir / "eval").glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            keys.add(json.loads(line)["key"])
    return keys


def targets(mix: dict[str, float], n: int) -> dict[str, int]:
    total = sum(mix.values())
    out = {k: int(round(n * v / total)) for k, v in mix.items() if v > 0}
    diff = n - sum(out.values())
    if out:
        out[max(out, key=out.get)] += diff
    return out


class Selector:
    """Accepts candidates that pass the token cap, eval de-dup and within-set de-dup."""

    def __init__(self, tok, max_len: int, banned: set[str]):
        self.tok, self.max_len, self.banned, self.seen = tok, max_len, banned, set()
        self.rejects = Counter()

    def accept(self, question: str, turn: str) -> dict | None:
        """`turn` is a full wrap() turn; the stored completion drops the <think> prefill."""
        key = prompt_key(question)
        if key in self.banned:
            self.rejects["in_eval"] += 1
            return None
        if key in self.seen:
            self.rejects["duplicate"] += 1
            return None
        try:
            prompt = build_prompt(question)
            completion = to_completion(turn) if is_prefilled(prompt) else turn
        except ValueError:
            self.rejects["bad_prompt"] += 1
            return None
        n = len(self.tok(prompt + completion, add_special_tokens=False)["input_ids"])
        if n > self.max_len:
            self.rejects["too_long"] += 1
            return None
        self.seen.add(key)
        return {"prompt": prompt, "completion": completion, "question": question, "key": key, "n_tokens": n}


def take(it: Iterator[dict], k: int, sel: Selector, source: str, make: Callable[[dict], str | None] | None = None,
         scan_limit: int = 200_000) -> list[dict]:
    out = []
    for scanned, c in enumerate(it):
        if len(out) >= k or scanned >= scan_limit:
            break
        completion = make(c) if make else c["completion"]
        if completion is None:
            continue
        r = sel.accept(c["question"], completion)
        if r:
            out.append({**r, "source": source})
    return out


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> Path:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--n", type=int, default=None, help="override data.stage1_examples")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = config_from_args(args)
    if cfg.runtime.keep_awake:  # the teacher pass runs for hours; the GPU work happens in the judge server
        from crllm.runtime import keep_awake

        keep_awake()
    from transformers import AutoTokenizer

    n_total = args.n or cfg.data.stage1_examples
    max_len = cfg.train.max_seq_len
    data_dir, rng = cfg.path("data_dir"), random.Random(cfg.seed)
    raw = data_dir / "raw" / "stage1"
    tok = AutoTokenizer.from_pretrained(cfg.model.policy)
    sel = Selector(tok, max_len, eval_keys(data_dir))
    log.info("%d eval prompts banned; target %d examples at <= %d tokens", len(sel.banned), n_total, max_len)

    mix = asdict(cfg.data.stage1_mix)
    want = targets(mix, n_total)
    paths = {name: fetch_shards(repo, c, s, raw, k) for name, (repo, c, s, k) in SOURCES.items() if name in want}
    chars = max_len * 3  # cheap prefilter before tokenizing (~3.5-4 chars per token)
    readers = {
        "mot_math": lambda: read_mot(paths["mot_math"], max_len, rng),
        "mot_code": lambda: read_mot(paths["mot_code"], max_len, rng),
        "omi_math": lambda: read_omi(paths["omi_math"][0], chars, rng),
        "oci_code": lambda: read_oci(paths["oci_code"][0], chars, rng),
    }

    cache = PlanCache(data_dir / "cache" / "teacher_plans.jsonl")
    judge = Judge(cfg.judge) if "oci_code" in want else None

    def oci_make(c):
        plan = cache.plans.get(c["src_id"])
        if not plan:
            return None
        try:
            return wrap(plan, c["solution"])
        except ValueError:
            return None

    picked: dict[str, list[dict]] = {}
    for name in [s for s in ("mot_math", "mot_code", "omi_math") if s in want]:
        picked[name] = take(readers[name](), want[name], sel, name)
        log.info("%s: %d/%d", name, len(picked[name]), want[name])

    # sources that ran short hand their deficit to the others (code first goes to oci_code)
    deficit = sum(want[s] - len(picked[s]) for s in picked)
    if "oci_code" in want:
        want["oci_code"] += deficit if deficit > 0 else 0
        deficit = 0
        need = want["oci_code"]
        cands = [c for c in read_oci(paths["oci_code"][0], chars, rng) if prompt_key(c["question"]) not in sel.banned]
        picked["oci_code"], pos = [], 0
        log.info("teacher: %d code plans needed (%d candidates, %d already cached)", need, len(cands),
                 sum(c["src_id"] in cache.plans for c in cands))
        while len(picked["oci_code"]) < need and pos < len(cands):
            # plan in chunks sized to what is still missing (+10%), so failures don't cost a full extra pass
            missing = need - len(picked["oci_code"])
            chunk = cands[pos: pos + max(8, int(missing * 1.1))]
            pos += len(chunk)
            teacher_plans(judge, chunk, cache, max_tokens=cfg.data.teacher_max_tokens)
            picked["oci_code"] += take(iter(chunk), missing, sel, "oci_code", make=oci_make)
        log.info("oci_code: %d/%d (%d candidates planned)", len(picked["oci_code"]), need, pos)
    if deficit > 0 and "omi_math" in want:  # no code source: top up with short math
        picked["omi_math"] += take(readers["omi_math"](), deficit, sel, "omi_math")

    rows = [r for rs in picked.values() for r in rs]
    rng.shuffle(rows)
    out = data_dir / "sft_think.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "n": len(rows), "target": n_total, "max_seq_len": max_len,
        "by_source": dict(Counter(r["source"] for r in rows)),
        "mean_tokens": sum(r["n_tokens"] for r in rows) / max(1, len(rows)),
        "rejects": dict(sel.rejects), "teacher": cfg.judge.model if judge else None,
    }
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_run_json(data_dir / "runs" / "build_sft", cfg, stats=stats)
    log.info("wrote %s: %s", out, stats)
    return out


if __name__ == "__main__":
    main()
