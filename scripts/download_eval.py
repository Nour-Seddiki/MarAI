"""One-off: download the eval sources and write data/eval/<suite>.jsonl.

Suites (fixed seed, so every machine builds the same files):
  format       200 mixed prompts: 120 Alpaca-cleaned instructions (no input) + 80 GSM8K test questions
  gsm8k        GSM8K test (all 1,319; run_eval takes the first eval.gsm8k_n)
  redteam      200 Do-Not-Answer questions (LibrAI/do-not-answer), stratified by risk area
  overrefusal  the 250 XSTest prompts labelled safe
  honesty      100 SelfAware questions marked unanswerable

Record fields: id, suite, question (raw user text), prompt (format.build_prompt), key
(format.prompt_key, used by build_*.py to keep eval prompts out of training), answer
(normalized gold, or null), meta (source-specific).

Downloads go through the Hub's parquet API with plain HTTP retries; huggingface_hub
downloads have hung on this laptop.

    python scripts/download_eval.py            # writes data/eval/*.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from crllm.config import add_config_args, config_from_args
from crllm.format import build_prompt, extract_final_answer, prompt_key

SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "alpaca": ("yahma/alpaca-cleaned", "default", "train"),
    "dna": ("LibrAI/do-not-answer", "default", "train"),
    "xstest": ("walledai/XSTest", "default", "test"),
    "xstest_alt": ("Paul/XSTest", "default", "train"),
    "selfaware": ("OkayestProgrammer/selfAware", "default", "train"),
}
API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}/0.parquet"


def fetch_parquet(name: str, raw_dir: Path, tries: int = 6) -> pd.DataFrame:
    path = raw_dir / f"{name}.parquet"
    if not path.exists():
        repo, config, split = SOURCES[name]
        url = API.format(repo=repo, config=config, split=split)
        for attempt in range(1, tries + 1):
            try:
                r = requests.get(url, timeout=(20, 120), allow_redirects=True)
                r.raise_for_status()
                pd.read_parquet(io.BytesIO(r.content))  # validate before saving
                path.write_bytes(r.content)
                print(f"  {name}: {repo} ({len(r.content) / 2**20:.1f} MB)")
                break
            except (requests.RequestException, ValueError, OSError) as e:
                print(f"  {name}: attempt {attempt} failed: {type(e).__name__}: {str(e)[:120]}")
                status = getattr(getattr(e, "response", None), "status_code", None)
                if attempt == tries or (status is not None and 400 <= status < 500 and status != 429):
                    raise
                time.sleep(3 * attempt)
    return pd.read_parquet(path)


def record(suite: str, i: int, question: str, answer: str | None = None, **meta) -> dict:
    question = question.strip()
    return {
        "id": f"{suite}-{i:04d}",
        "suite": suite,
        "question": question,
        "prompt": build_prompt(question),
        "key": prompt_key(question),
        "answer": answer,
        "meta": meta,
    }


def build_gsm8k(df: pd.DataFrame) -> list[dict]:
    out = []
    for i, row in enumerate(df.itertuples()):
        gold = extract_final_answer(row.answer)
        if gold is None:
            raise ValueError(f"no #### answer in GSM8K row {i}")
        out.append(record("gsm8k", i, row.question, gold))
    return out


def build_format(alpaca: pd.DataFrame, gsm: pd.DataFrame, rng: random.Random) -> list[dict]:
    pool = alpaca[(alpaca["input"].str.strip() == "") & (alpaca["instruction"].str.len().between(15, 400))]
    picks = [("alpaca", q) for q in pool.sample(120, random_state=rng.randrange(2**31))["instruction"]]
    picks += [("gsm8k", q) for q in gsm.sample(80, random_state=rng.randrange(2**31))["question"]]
    rng.shuffle(picks)
    return [record("format", i, q, source=src) for i, (src, q) in enumerate(picks)]


def build_redteam(dna: pd.DataFrame, rng: random.Random, n: int = 200) -> list[dict]:
    # stratify by risk area so no single harm type dominates the score
    groups = {k: g for k, g in dna.groupby("risk_area")}
    per = {k: max(1, round(n * len(g) / len(dna))) for k, g in groups.items()}
    while sum(per.values()) > n:
        per[max(per, key=per.get)] -= 1
    while sum(per.values()) < n:
        per[min(per, key=per.get)] += 1
    rows = []
    for k, g in groups.items():
        rows += list(g.sample(per[k], random_state=rng.randrange(2**31)).itertuples())
    rng.shuffle(rows)
    return [record("redteam", i, r.question, risk_area=r.risk_area, harm=r.types_of_harm, source_id=int(r.id))
            for i, r in enumerate(rows)]


def build_overrefusal(xs: pd.DataFrame) -> list[dict]:
    cols = {c.lower(): c for c in xs.columns}
    label, prompt = cols["label"], cols["prompt"]
    safe = xs[xs[label].astype(str).str.lower() == "safe"]
    if len(safe) != 250:
        print(f"  warning: expected 250 safe XSTest prompts, found {len(safe)}")
    typ = cols.get("type")
    return [record("overrefusal", i, r[prompt], type=(r[typ] if typ else None))
            for i, (_, r) in enumerate(safe.iterrows())]


def build_honesty(sa: pd.DataFrame, rng: random.Random, n: int = 100) -> list[dict]:
    un = sa[~sa["answerable"].astype(bool)]
    rows = un.sample(n, random_state=rng.randrange(2**31))
    return [record("honesty", i, r.question, source=r.source, source_id=int(r.question_id))
            for i, r in enumerate(rows.itertuples())]


def main() -> None:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    out_dir = cfg.path("data_dir") / "eval"
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)

    print("downloading sources")
    gsm = fetch_parquet("gsm8k", raw)
    alpaca = fetch_parquet("alpaca", raw)
    dna = fetch_parquet("dna", raw)
    try:
        xs = fetch_parquet("xstest", raw)
    except Exception as e:  # gated or moved: fall back to the other copy
        print(f"  xstest primary failed ({type(e).__name__}); trying {SOURCES['xstest_alt'][0]}")
        xs = fetch_parquet("xstest_alt", raw)
    sa = fetch_parquet("selfaware", raw)

    suites = {
        "format": build_format(alpaca, gsm, rng),
        "gsm8k": build_gsm8k(gsm),
        "redteam": build_redteam(dna, rng),
        "overrefusal": build_overrefusal(xs),
        "honesty": build_honesty(sa, rng),
    }
    keys: dict[str, str] = {}
    for name, rows in suites.items():
        for r in rows:  # the same question in two suites is fine; within a suite it is not
            if keys.get(r["key"]) == name:
                raise ValueError(f"duplicate question in {name}: {r['question'][:80]!r}")
            keys.setdefault(r["key"], name)
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(cfg.path('data_dir').parent)}  ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
