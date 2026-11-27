"""CPU-only tests for config loading and the constitution file."""
import random
import unittest

from crllm.config import (
    CONFIG_DIR,
    TEMPLATE_FIELDS,
    load_config,
    load_constitution,
    parse_override,
    run_metadata,
)


class ConfigTest(unittest.TestCase):
    def test_overlays_load_and_differ_where_the_plan_says(self):
        debug, full = load_config("debug"), load_config("full")
        self.assertEqual(debug.name, "debug")
        self.assertEqual(debug.model.policy, "Qwen/Qwen3-0.6B-Base")
        self.assertTrue(debug.model.load_in_4bit)
        self.assertEqual((debug.lora.r, debug.lora.alpha), (16, 32))
        self.assertEqual((debug.train.batch_size, debug.train.grad_accum), (1, 16))
        self.assertEqual(debug.runtime.p_core_mask, 0x0FFF)
        self.assertEqual(debug.runtime.cuda_memory_fraction, 0.8)

        self.assertEqual(full.model.policy, "Qwen/Qwen3-1.7B-Base")
        self.assertFalse(full.model.load_in_4bit)
        self.assertEqual((full.lora.r, full.lora.alpha), (32, 64))
        self.assertEqual(full.train.max_seq_len, 2048)
        self.assertEqual(full.train.optim, "adamw_torch")
        self.assertEqual((full.grpo.num_generations, full.grpo.max_completion_length), (8, 1024))
        self.assertTrue(full.grpo.use_vllm)
        self.assertFalse(full.runtime.keep_awake)

    def test_base_keys_survive_partial_overlay(self):
        full = load_config("full")
        self.assertEqual(full.train.warmup_ratio, 0.03)       # only in base.yaml
        self.assertEqual(full.lora.target_modules, "all-linear")

    def test_reward_weights_match_reward_funcs(self):
        for name in ("debug", "full"):
            cfg = load_config(name)
            self.assertEqual(len(cfg.grpo.reward_weights), 2)  # correctness, format (no Stage 3 RM yet)
            self.assertAlmostEqual(sum(cfg.grpo.reward_weights), 1.0)

    def test_cli_overrides(self):
        cfg = load_config("debug", ["train.lr=1e-4", "grpo.reward_weights=[0.6, 0.1, 0.3]", "hub.repo=me/x"])
        self.assertEqual(cfg.train.lr, 1e-4)
        self.assertEqual(cfg.grpo.reward_weights, [0.6, 0.1, 0.3])
        self.assertEqual(cfg.hub.repo, "me/x")
        self.assertEqual(parse_override("a.b.c=3"), {"a": {"b": {"c": 3}}})

    def test_typos_and_bad_types_fail(self):
        with self.assertRaises(KeyError):
            load_config("debug", ["train.learning_rate=1e-4"])
        with self.assertRaises(KeyError):
            load_config("debug", ["trian.lr=1e-4"])
        with self.assertRaises(TypeError):
            load_config("debug", ["train.batch_size=two"])
        with self.assertRaises(TypeError):
            load_config("debug", ["model.load_in_4bit=1"])
        with self.assertRaises(ValueError):
            parse_override("train.lr")

    def test_overlay_by_path(self):
        cfg = load_config(CONFIG_DIR / "full.yaml")
        self.assertEqual(cfg.name, "full")

    def test_run_metadata_is_json_ready(self):
        import json

        meta = run_metadata(load_config("debug"), stage=1)
        json.dumps(meta)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["stage"], 1)


class ConstitutionTest(unittest.TestCase):
    def setUp(self):
        self.c = load_constitution()

    def test_principles(self):
        ids = [p.id for p in self.c.principles]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertIn("no_harm", ids)
        self.assertEqual(self.c.get("honest").id, "honest")
        self.assertIn(self.c.sample(random.Random(0)), self.c.principles)
        with self.assertRaises(KeyError):
            self.c.get("nope")

    def test_every_template_formats_with_exactly_its_fields(self):
        for name, fields in TEMPLATE_FIELDS.items():
            with self.subTest(template=name):
                out = self.c.render(name, **{f: f"<{f}>" for f in fields})
                for f in fields:
                    self.assertIn(f"<{f}>", out)
                self.assertNotIn("{principle}", out)

    def test_rank_template_keeps_literal_json(self):
        out = self.c.render("rank", principle="p", prompt="q", responses="A: x")
        self.assertIn('{"ranking": ["B", "A", ...]}', out)


if __name__ == "__main__":
    unittest.main()
