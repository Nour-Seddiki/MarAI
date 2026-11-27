"""CPU tests for the Stage 1 builder: readers on synthetic parquet, selection, teacher plan hygiene."""
import os
import random
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crllm.data.build_sft import PlanCache, Selector, clean_plan, read_mot, read_oci, read_omi, take, targets
from crllm.format import IM_END, parse, prompt_key

LOCAL_QWEN = Path(os.environ.get("CRLLM_QWEN3_DIR", Path.home() / ".cache/crllm-models/Qwen3-0.6B-Base"))


class FakeTok:
    """1 token per whitespace-separated word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def write(tmp, name, cols):
    p = Path(tmp) / name
    pq.write_table(pa.table(cols), p)
    return p


class TargetsTest(unittest.TestCase):
    def test_split_sums_to_n(self):
        t = targets({"a": 0.35, "b": 0.10, "c": 0.25, "d": 0.30}, 2000)
        self.assertEqual(t, {"a": 700, "b": 200, "c": 500, "d": 600})
        self.assertEqual(sum(targets({"a": 1, "b": 1, "c": 1}, 100).values()), 100)
        self.assertNotIn("z", targets({"a": 1.0, "z": 0.0}, 10))


class PlanTest(unittest.TestCase):
    def test_clean_plan(self):
        self.assertEqual(clean_plan("  Read the list. Track the max.  "), "Read the list. Track the max.")
        for bad in ["", "```python\nx\n```", "Use def f(x): return x", "As the provided solution shows...",
                    "<think>x</think>", "x" * 2000, None]:
            with self.subTest(bad=bad):
                self.assertIsNone(clean_plan(bad))

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            c = PlanCache(Path(d) / "plans.jsonl")
            c.put("a", "plan a")
            c.put("b", None)
            again = PlanCache(Path(d) / "plans.jsonl")
            self.assertEqual(again.plans, {"a": "plan a", "b": None})


class ReadersTest(unittest.TestCase):
    def test_mot_filters_by_num_tokens_and_rewraps(self):
        msgs = lambda q, a: [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "mot.parquet", {
                "messages": [msgs("q1", "<think>\nr1\n</think>\n\na1"), msgs("q2", "<think>\nr2\n</think>\n\na2"),
                             msgs("q3", "no tags"), msgs("q4", "<think>\nr4\n</think>\n\na4")],
                "num_tokens": [100, 5000, 100, 90],
            })
            out = list(read_mot([p], 1000, random.Random(0)))
        self.assertEqual(sorted(o["question"] for o in out), ["q1", "q4"])
        for o in out:
            self.assertTrue(parse(o["completion"]).well_formed)
            self.assertTrue(o["completion"].endswith(IM_END))
            self.assertNotIn("\n</think>", o["completion"])  # canonical: no inner padding

    def test_omi_boxes_expected_answer(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "omi.parquet", {"problem": ["What is 2+3?", "long"], "generated_solution": ["2+3=5. \\boxed{5}", "x" * 999],
                                         "expected_answer": ["5", "1"]})
            out = list(read_omi(p, 200, random.Random(0)))
        self.assertEqual(len(out), 1)
        self.assertEqual(parse(out[0]["completion"]).answer, "The final answer is $\\boxed{5}$.")

    def test_oci_keeps_only_verified_python(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "oci.parquet", {
                "id": ["a", "b", "c"], "input": ["task a", "task b", "task c"],
                "output": ["```python\ndef a(): pass\n```", "```python\ndef b(): pass\n```", "no code"],
                "average_test_score": ["1", "0.5", "1"],
            })
            out = list(read_oci(p, 1000, random.Random(0)))
        self.assertEqual([o["src_id"] for o in out], ["a"])


class SelectorTest(unittest.TestCase):
    def test_dedup_eval_ban_and_length(self):
        sel = Selector(FakeTok(), max_len=60, banned={prompt_key("Banned question?")})
        ok = sel.accept("What is 2+2?", "<think>r</think>\n4<|im_end|>")
        self.assertIsNotNone(ok)
        self.assertEqual(ok["completion"], "r</think>\n4<|im_end|>")   # stored after the prefill
        self.assertTrue(ok["prompt"].endswith("<think>"))
        self.assertIsNone(sel.accept("what is   2+2?", "<think>r</think>\n4<|im_end|>"))  # same key
        self.assertIsNone(sel.accept("banned QUESTION?", "<think>x</think>\ny"))
        self.assertIsNone(sel.accept("Long one", "<think>" + "word " * 100 + "</think>\na"))
        self.assertIsNone(sel.accept("No tag", "plain"))
        self.assertEqual(dict(sel.rejects), {"duplicate": 1, "in_eval": 1, "too_long": 1, "bad_prompt": 1})

    def test_take_stops_at_k_and_skips_unmade(self):
        sel = Selector(FakeTok(), max_len=1000, banned=set())
        cands = iter([{"question": f"q{i}", "completion": "<think>r</think>\nc"} for i in range(10)])
        got = take(cands, 3, sel, "src", make=lambda c: None if c["question"] == "q1" else c["completion"])
        self.assertEqual([g["question"] for g in got], ["q0", "q2", "q3"])
        self.assertTrue(all(g["source"] == "src" for g in got))


if __name__ == "__main__":
    unittest.main()
