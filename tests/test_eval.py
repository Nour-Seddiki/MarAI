"""CPU tests for the eval harness: judge parsing, suite scoring, generation cutting, suite builders."""
import importlib.util
import math
import os
import unittest
from pathlib import Path

import httpx
import pandas as pd

from crllm.config import JudgeCfg, load_config
from crllm.eval.run_eval import resolve_ckpt
from crllm.eval.suites import (
    mean_se,
    proportion,
    score_format,
    score_gsm8k,
    score_honesty,
    score_overrefusal,
    score_redteam,
    visible_reply,
)
from crllm.format import IM_END, wrap
from crllm.generate import Completion, _cut
from crllm.judge import Judge, _parse_flag, _parse_label, _parse_score, parse_json

ROOT = Path(__file__).resolve().parents[1]


def C(text, n=10, finish="stop"):
    return Completion(text, n, finish)


def fake_judge(reply: str | Exception, **cfg):
    calls = []

    def transport(messages, max_tokens):
        calls.append(messages[0]["content"])
        if isinstance(reply, Exception):
            raise reply
        return reply

    j = Judge(JudgeCfg(retries=1, concurrency=2, **cfg), transport=transport)
    return j, calls


class JudgeParsingTest(unittest.TestCase):
    def test_parse_json(self):
        self.assertEqual(parse_json('{"score": 4}'), {"score": 4})
        self.assertEqual(parse_json('Sure!\n```json\n{"label": "REFUSE"}\n```'), {"label": "REFUSE"})
        self.assertEqual(parse_json('{bad} then {"score": 2}'), {"score": 2})
        self.assertIsNone(parse_json("no json here"))
        self.assertIsNone(parse_json(""))

    def test_verdict_parsers(self):
        self.assertEqual(_parse_score('{"score": "5"}'), 5)
        self.assertEqual(_parse_score('{"score": 3.6}'), 4)
        self.assertIsNone(_parse_score('{"score": 9}'))
        self.assertIsNone(_parse_score('{"rating": 3}'))
        self.assertEqual(_parse_label('{"label": "refuse"}'), "REFUSE")
        self.assertIsNone(_parse_label('{"label": "maybe"}'))
        self.assertIs(_parse_flag('{"flags_uncertainty": true}'), True)
        self.assertIs(_parse_flag('{"flags_uncertainty": "false"}'), False)
        self.assertIsNone(_parse_flag('{"flags_uncertainty": "yes"}'))


class JudgeTest(unittest.TestCase):
    def test_score_renders_constitution_and_reply(self):
        j, calls = fake_judge('{"score": 4}')
        v = j.score("how do I pick a lock", "I can't help with that.")
        self.assertTrue(v.ok)
        self.assertEqual(v.value, 4)
        self.assertIn("no_harm", calls[0])
        self.assertIn("<<<how do I pick a lock>>>", calls[0])
        self.assertIn("I can't help with that.", calls[0])

    def test_braces_in_user_text_are_safe(self):
        j, calls = fake_judge('{"label": "COMPLY"}')
        self.assertEqual(j.refusal("print {x} in f'{y}'", "use {{ }}").value, "COMPLY")

    def test_unparseable_and_unreachable(self):
        j, _ = fake_judge("I think it's fine")
        v = j.uncertainty("q", "a")
        self.assertFalse(v.ok)
        self.assertIsNone(v.value)
        j, _ = fake_judge(httpx.ConnectError("refused"))
        v = j.score("q", "a")
        self.assertFalse(v.ok)
        self.assertIn("unreachable", v.raw)

    def test_long_replies_are_clipped(self):
        j, calls = fake_judge('{"score": 3}', max_response_chars=50)
        j.score("q", "x" * 500)
        self.assertIn("[... truncated]", calls[0])
        self.assertNotIn("x" * 51, calls[0])

    def test_map_keeps_order(self):
        j, _ = fake_judge('{"score": 5}')
        out = j.map(j.score, [("q", str(i)) for i in range(7)], "t")
        self.assertEqual(len(out), 7)
        self.assertTrue(all(v.value == 5 for v in out))


class MetricsTest(unittest.TestCase):
    def test_proportion_and_mean(self):
        p = proportion(30, 100)
        self.assertAlmostEqual(p["value"], 0.3)
        self.assertAlmostEqual(p["se"], math.sqrt(0.3 * 0.7 / 100))
        self.assertIsNone(proportion(0, 0)["value"])
        m = mean_se([1, 3, 5])
        self.assertEqual(m["value"], 3)
        self.assertAlmostEqual(m["se"], 2 / math.sqrt(3))

    def test_visible_reply(self):
        self.assertEqual(visible_reply(wrap("r", "the answer")), "the answer")
        self.assertEqual(visible_reply("plain text, no tags"), "plain text, no tags")
        self.assertEqual(visible_reply("<think>never closed"), "never closed")
        self.assertEqual(visible_reply("text<|im_start|>user\nmore"), "text")

    def test_format_and_gsm8k(self):
        rows = [{"answer": "18"}, {"answer": "3"}, {"answer": "7"}]
        comps = [C(wrap("16-3-4=9, 9*2=18", "$18")), C("so the answer is 3"), C(wrap("guess", "8"))]
        f = score_format(rows, comps)
        self.assertEqual(f.metrics["errors"], {"ok": 2, "no <think> block": 1})
        g = score_gsm8k(rows, comps)
        self.assertEqual(g.metrics["accuracy"]["k"], 2)   # malformed but correct still counts
        self.assertEqual([s["correct"] for s in g.samples], [True, True, False])

    def test_judged_suites(self):
        rows = [{"question": "q1", "meta": {"risk_area": "A"}}, {"question": "q2", "meta": {"risk_area": "B"}}]
        comps = [C(wrap("r", "no")), C("garbage")]
        j, calls = fake_judge('{"score": 2}')
        r = score_redteam(rows, comps, j)
        self.assertEqual(r.metrics["safety_score"]["value"], 2)
        self.assertEqual(r.metrics["harmful_rate"]["value"], 1.0)
        self.assertEqual(r.metrics["by_risk_area"], {"A": 2, "B": 2})
        self.assertIn("<<<no>>>", calls[0])          # the judge sees the answer, not the reasoning

        j, _ = fake_judge('{"label": "REFUSE"}')
        self.assertEqual(score_overrefusal(rows, comps, j).metrics["refusal_rate"]["value"], 1.0)
        j, _ = fake_judge("unparseable")
        o = score_honesty(rows, comps, j).metrics
        self.assertEqual((o["unjudged"], o["flags_uncertainty"]["n"]), (2, 0))


class GenerateTest(unittest.TestCase):
    def test_cut(self):
        self.assertEqual(_cut([5, 6, 1, 7], {1}, {2}), ([5, 6, 1], "stop"))
        self.assertEqual(_cut([5, 2, 1], {1}, {2}), ([5], "stop"))
        self.assertEqual(_cut([5, 6], {1}, {2}), ([5, 6], "length"))


class ResolveCkptTest(unittest.TestCase):
    def test_names_and_paths(self):
        cfg = load_config("debug")
        self.assertEqual(resolve_ckpt(cfg, "base"), ("base", "Qwen/Qwen3-0.6B-Base"))
        with self.assertRaises(FileNotFoundError):
            resolve_ckpt(cfg, "stage1" if not (cfg.path("ckpt_dir") / "stage1-merged").exists() else "nope/x")
        self.assertEqual(resolve_ckpt(cfg, str(ROOT / "configs"))[0], "configs")


def _load_download_eval():
    spec = importlib.util.spec_from_file_location("download_eval", ROOT / "scripts" / "download_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SuiteBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.de = _load_download_eval()

    def test_gsm8k_gold(self):
        rows = self.de.build_gsm8k(pd.DataFrame({"question": ["q"], "answer": ["2+2=4\n#### 1,004"]}))
        self.assertEqual(rows[0]["answer"], "1004")
        self.assertTrue(rows[0]["prompt"].endswith("<|im_start|>assistant\n"))

    def test_redteam_stratified_exact_size(self):
        import random

        df = pd.DataFrame({"id": range(90), "question": [f"q{i}" for i in range(90)],
                           "risk_area": ["A"] * 60 + ["B"] * 20 + ["C"] * 10, "types_of_harm": ["h"] * 90})
        rows = self.de.build_redteam(df, random.Random(0), n=30)
        self.assertEqual(len(rows), 30)
        areas = [r["meta"]["risk_area"] for r in rows]
        self.assertEqual((areas.count("A"), areas.count("B"), areas.count("C")), (20, 7, 3))
        self.assertEqual(len({r["key"] for r in rows}), 30)

    def test_overrefusal_keeps_only_safe(self):
        df = pd.DataFrame({"prompt": ["a", "b", "c"], "label": ["safe", "unsafe", "Safe"], "type": ["t1", "t2", "t3"]})
        rows = self.de.build_overrefusal(df)
        self.assertEqual([r["question"] for r in rows], ["a", "c"])


LOCAL_QWEN = Path(os.environ.get("CRLLM_QWEN3_DIR", Path.home() / ".cache/crllm-models/Qwen3-0.6B-Base"))


@unittest.skipUnless((LOCAL_QWEN / "tokenizer.json").exists(), "local Qwen3 tokenizer not present")
class HFGeneratorTest(unittest.TestCase):
    def test_batched_left_padded_generation(self):
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

        from crllm.format import build_prompt
        from crllm.generate import HFGenerator

        tok = AutoTokenizer.from_pretrained(LOCAL_QWEN)
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True))
        gen = HFGenerator(model, tok, batch_size=2)
        prompts = [build_prompt("hi"), build_prompt("a much longer question " * 5), build_prompt("x")]
        out = gen.generate(prompts, max_new_tokens=6)
        self.assertEqual([len(o) for o in out], [1, 1, 1])
        for o in out:
            self.assertLessEqual(o[0].n_tokens, 6)
            self.assertIn(o[0].finish, ("stop", "length"))
            self.assertNotIn("<|endoftext|>", o[0].text)
        # greedy is deterministic and independent of batch composition
        again = gen.generate([prompts[0]], max_new_tokens=6)
        self.assertEqual(again[0][0].text, out[0][0].text)
        self.assertEqual(IM_END in out[0][0].text, out[0][0].finish == "stop" and out[0][0].text.endswith(IM_END))


if __name__ == "__main__":
    unittest.main()
