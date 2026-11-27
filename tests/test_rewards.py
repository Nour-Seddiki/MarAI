"""Stage 4 reward functions on hand-written completions (CPU; code rewards run real subprocesses)."""
import unittest

from crllm import rewards
from crllm.eval.execute import fraction_passed
from crllm.format import build_prompt

P = build_prompt("q")   # prefilled: completions start after <think>
END, OTHER = 151645, 42  # a stop id and a non-stop id


class FormatRewardTest(unittest.TestCase):
    def setUp(self):
        rewards.STOP_IDS = {END, 151643}

    def tearDown(self):
        rewards.STOP_IDS = set()

    def test_well_formed_and_ended(self):
        # TRL decodes with special tokens skipped, so <|im_end|> is absent from the text
        self.assertEqual(rewards.format_reward([P], ["reasoning</think>\nanswer"], completion_ids=[[7, END]]), [1.0])

    def test_truncated_is_not_rewarded_even_if_text_parses(self):
        self.assertEqual(rewards.format_reward([P], ["reasoning</think>\nThe answer is"], completion_ids=[[7, OTHER]]), [0.0])

    def test_malformed(self):
        cases = ["never closed", "<think>again</think>\nanswer", "r</think>\n", "r</think>a</think>\nb"]
        self.assertEqual(rewards.format_reward([P] * 4, cases, completion_ids=[[END]] * 4), [0.0] * 4)

    def test_without_ids_falls_back_to_text(self):
        rewards.STOP_IDS = set()
        self.assertEqual(rewards.format_reward([P], ["r</think>\na"]), [1.0])


class CorrectnessRewardTest(unittest.TestCase):
    def test_math(self):
        comps = ["work</think>\nSo it is \\boxed{12}.", "work</think>\nThe answer is 0.5",
                 "work</think>\n\\boxed{13}", "\\boxed{12} but never closed"]
        out = rewards.correctness_reward([P] * 4, comps, task=["math"] * 4, answer=["12", "\\frac{1}{2}", "12", "12"])
        self.assertEqual(out, [1.0, 1.0, 0.0, 0.0])  # the last one is strict: no </think>, no credit

    def test_code_partial_credit(self):
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0", "assert add(2, 2) == 5"]
        good = "plan</think>\n```python\ndef add(a, b):\n    return a + b\n```"
        crash = "plan</think>\n```python\nraise SystemExit(3)\n```"
        inside = "plan ```python\ndef add(a, b):\n    return a + b\n``` no close"
        out = rewards.correctness_reward([P] * 3, [good, crash, inside], task=["code"] * 3,
                                         tests=[tests] * 3, test_imports=[[]] * 3)
        self.assertEqual(out, [0.75, 0.0, 0.0])

    def test_chat_has_no_ground_truth(self):
        out = rewards.correctness_reward([P, P], ["r</think>\nhi", "r</think>\n\\boxed{1}"], task=["chat", "math"],
                                         answer=[None, "1"])
        self.assertEqual(out, [None, 1.0])

    def test_reward_funcs_match_config(self):
        from crllm.config import load_config

        for name in ("debug", "full"):
            self.assertEqual(len(load_config(name).grpo.reward_weights), len(rewards.REWARD_FUNCS))


class FractionPassedTest(unittest.TestCase):
    def test_counts(self):
        code = "def sq(x):\n    return x * x"
        self.assertEqual(fraction_passed(code, ["assert sq(2) == 4", "assert sq(3) == 10"]), 0.5)
        self.assertEqual(fraction_passed(None, ["assert True"]), 0.0)
        self.assertEqual(fraction_passed(code, []), 0.0)
        self.assertEqual(fraction_passed("while True:\n    pass", ["assert True"], timeout_s=1), 0.0)
        # a test that tries to exit the process doesn't count as passing, and the rest still run
        self.assertEqual(fraction_passed(code, ["raise SystemExit(0)", "assert sq(1) == 1"]), 0.5)


if __name__ == "__main__":
    unittest.main()
