"""Print a per-suite summary of an eval report (+ format breakdown by prompt source).

    python scripts/summarize_report.py reports/debug/stage1.json
    python scripts/summarize_report.py reports/debug/stage1.json reports/debug/base.json   # side by side
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from crllm.eval.suites import HEADLINE
from crllm.format import parse


def fmt(h: dict | None) -> str:
    if not h or h.get("value") is None:
        return "-"
    return f"{h['value']:.3f} ± {h['se'] or 0:.3f}"


def format_by_source(report_path: Path, data_dir: Path) -> dict[str, str]:
    samples = report_path.with_name(report_path.stem + ".samples.jsonl")
    fmt_file = data_dir / "eval" / "format.jsonl"
    if not samples.exists() or not fmt_file.exists():
        return {}
    src = {json.loads(line)["id"]: json.loads(line)["meta"]["source"] for line in fmt_file.open(encoding="utf-8")}
    by = collections.defaultdict(list)
    for line in samples.open(encoding="utf-8"):
        x = json.loads(line)
        if x["suite"] == "format":
            by[src[x["id"]]].append(parse(x["completion"]).well_formed)
    return {k: f"{sum(v) / len(v):.0%} of {len(v)}" for k, v in sorted(by.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    reps = [(Path(p), json.load(open(p, encoding="utf-8"))) for p in args.reports]
    names = [r["name"] for _, r in reps]
    print(f"{'suite':12s} {'metric':18s} " + " ".join(f"{n:>16s}" for n in names))
    for suite, (key, direction) in HEADLINE.items():
        if not any(suite in r["suites"] for _, r in reps):
            continue
        vals = [fmt(r["suites"].get(suite, {}).get(key)) for _, r in reps]
        print(f"{suite:12s} {key + (' (low)' if direction == 'lower' else ''):18s} " + " ".join(f"{v:>16s}" for v in vals))
        if any(f"{key}_lenient" in r["suites"].get(suite, {}) for _, r in reps):
            vals = [fmt(r["suites"].get(suite, {}).get(f"{key}_lenient")) for _, r in reps]
            print(f"{'':12s} {'  lenient':18s} " + " ".join(f"{v:>16s}" for v in vals))
    for path, r in reps:
        print(f"\n[{r['name']}]  {r['decoding']}")
        for s, m in r["suites"].items():
            extra = {k: v for k, v in m.items() if k in ("errors", "status", "by_level", "unjudged", "skipped")}
            print(f"  {s:12s} well-formed {m['well_formed']['value']:.1%}  hit limit {m['hit_length_limit']['value']:.1%}"
                  f"  mean tokens {m['mean_tokens']:.0f}  {extra if extra else ''}")
        by = format_by_source(path, Path(args.data_dir))
        if by:
            print("  format by prompt source:", by)


if __name__ == "__main__":
    main()
