"""CPU tests for format-token row handling, on a tiny random Qwen3 (no downloads).

The tiny model reproduces the Qwen3-Base situation: format-token rows and the padding
rows past the tokenizer are bit-identical (never trained), and embeddings are tied.
"""
import os
import unittest
from pathlib import Path

import torch

from crllm.config import LoraCfg
from crllm.format import IM_END, IM_START, THINK_CLOSE, THINK_OPEN
from crllm.modeling import (
    FORMAT_TOKENS,
    format_token_ids,
    init_untrained_rows,
    is_tied,
    lora_config,
    prepare_format_tokens,
    text_init_sources,
    trainable_token_spec,
    untrained_rows,
)

VOCAB_ROWS, N_TOKENS = 64, 56          # rows 56..63 are padding, like Qwen's 151669..151935
FMT = {IM_START: 50, IM_END: 51, THINK_OPEN: 52, THINK_CLOSE: 53}
SPELL = {IM_START: [1, 2, 3], IM_END: [1, 4, 3], THINK_OPEN: [5, 6, 7], THINK_CLOSE: [5, 8, 6, 7]}


class FakeTokenizer:
    """Just the tokenizer surface modeling.py uses."""

    def __len__(self):
        return N_TOKENS

    def encode(self, text, add_special_tokens=False):
        return [FMT[text]] if text in FMT else [9, 10]

    def convert_ids_to_tokens(self, tid):
        return {v: k for k, v in FMT.items()}[tid]

    def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
        return {"input_ids": list(SPELL[text]) if split_special_tokens else [FMT[text]]}


def tiny_model(tied=True, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=VOCAB_ROWS, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        tie_word_embeddings=tied,
    )
    model = Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for w in {id(m.weight): m.weight for m in (model.get_input_embeddings(), model.get_output_embeddings())}.values():
            w[50:] = w[-1].clone()  # format rows + padding: one shared untrained vector
    return model


def train_steps(model, steps=30, lr=5e-2):
    """Teach `... </think> answer <|im_end|>`: targets are the format tokens."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        body = torch.randint(11, 40, (4, 6), generator=g)
        ids = torch.cat([torch.full((4, 1), 52), body, torch.full((4, 1), 53),
                         torch.randint(11, 40, (4, 2), generator=g), torch.full((4, 1), 51)], dim=1)
        loss = model(input_ids=ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


class RowDetectionTest(unittest.TestCase):
    def test_untrained_rows_match_padding(self):
        m = tiny_model()
        mask = untrained_rows(m.get_input_embeddings().weight, N_TOKENS)
        self.assertEqual(mask.nonzero().squeeze(1).tolist(), list(range(50, 64)))

    def test_no_padding_rows_means_no_reference(self):
        w = torch.zeros(8, 4)
        self.assertFalse(untrained_rows(w, 8).any())

    def test_format_token_ids(self):
        self.assertEqual(format_token_ids(FakeTokenizer()), FMT)

        class Split(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                return [1, 2]

        with self.assertRaises(ValueError):
            format_token_ids(Split())


class InitTest(unittest.TestCase):
    def test_init_makes_rows_distinct_and_is_idempotent(self):
        m, tok = tiny_model(), FakeTokenizer()
        w = m.get_input_embeddings().weight
        before = w.detach().clone()
        changed = init_untrained_rows(m, tok, list(FMT.values()))
        self.assertEqual(sorted(changed), [50, 51, 52, 53])
        for t, tid in FMT.items():
            self.assertTrue(torch.allclose(w[tid], before[SPELL[t]].mean(0)), t)
        rows = w[[50, 51, 52, 53]]
        self.assertEqual(torch.unique(rows, dim=0).shape[0], 4)
        untouched = [i for i in range(VOCAB_ROWS) if i not in (50, 51, 52, 53)]
        self.assertTrue(torch.equal(w[untouched], before[untouched]))
        self.assertTrue(is_tied(m))
        self.assertEqual(init_untrained_rows(m, tok, list(FMT.values())), [])  # trained now -> no-op

    def test_text_init_sources(self):
        self.assertEqual(text_init_sources(FakeTokenizer(), [52]), {52: [5, 6, 7]})


class PeftTrainableTokensTest(unittest.TestCase):
    def _run(self, tied):
        from peft import get_peft_model

        m, tok = tiny_model(tied=tied), FakeTokenizer()
        spec = prepare_format_tokens(m, tok, LoraCfg(r=4, alpha=8, dropout=0.0))
        self.assertEqual(spec, list(FMT.values()) if tied else
                         {"embed_tokens": list(FMT.values()), "lm_head": list(FMT.values())})
        emb0 = m.get_input_embeddings().weight.detach().clone()
        head0 = m.get_output_embeddings().weight.detach().clone()
        pm = get_peft_model(m, lora_config(LoraCfg(r=4, alpha=8, dropout=0.0), trainable_tokens=spec))

        names = [n for n, p in pm.named_parameters() if p.requires_grad]
        self.assertTrue(any("trainable_tokens_delta" in n for n in names))
        self.assertTrue(any("lora_A" in n for n in names))
        self.assertFalse(any(n.endswith("embed_tokens.weight") or n.endswith("lm_head.weight") for n in names))

        probe = torch.tensor([[52, 20, 21, 22]])
        with torch.no_grad():
            logit0 = pm(input_ids=probe).logits[0, -1, [50, 51, 53]]
        loss = train_steps(pm)
        self.assertLess(loss, 4.0)
        with torch.no_grad():
            peft_logits = pm(input_ids=probe).logits
        fmt_logits = peft_logits[0, -1, [50, 51, 53]]
        self.assertGreater((fmt_logits - fmt_logits.mean()).abs().max().item(), 1e-3)
        self.assertFalse(torch.allclose(fmt_logits, logit0))

        merged = pm.merge_and_unload()
        emb = merged.get_input_embeddings().weight
        head = merged.get_output_embeddings().weight
        ids = list(FMT.values())
        others = [i for i in range(VOCAB_ROWS) if i not in ids]
        self.assertTrue(torch.equal(emb[others], emb0[others]))       # only format rows moved
        self.assertFalse(torch.allclose(emb[ids], emb0[ids]))
        if tied:
            self.assertTrue(is_tied(merged))
        else:
            self.assertTrue(torch.equal(head[others], head0[others]))
            self.assertFalse(torch.allclose(head[ids], head0[ids]))
        with torch.no_grad():
            self.assertTrue(torch.allclose(merged(input_ids=probe).logits, peft_logits, atol=1e-4))

    def test_tied(self):
        self._run(tied=True)

    def test_untied(self):
        self._run(tied=False)

    def test_spec_off(self):
        m = tiny_model()
        self.assertIsNone(prepare_format_tokens(m, FakeTokenizer(), LoraCfg(train_format_tokens=False)))
        self.assertEqual(trainable_token_spec(m, [1, 2]), [1, 2])


LOCAL_QWEN = Path(os.environ.get("CRLLM_QWEN3_DIR", Path.home() / ".cache/crllm-models/Qwen3-0.6B-Base"))


@unittest.skipUnless((LOCAL_QWEN / "tokenizer.json").exists(), "local Qwen3 tokenizer not present")
class RealTokenizerTest(unittest.TestCase):
    def test_spellings_use_ordinary_tokens(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(LOCAL_QWEN)
        ids = format_token_ids(tok)
        self.assertEqual(ids, {IM_START: 151644, IM_END: 151645, THINK_OPEN: 151667, THINK_CLOSE: 151668})
        src = text_init_sources(tok, list(ids.values()))
        for tid, pieces in src.items():
            self.assertTrue(all(p < 151643 for p in pieces), (tid, pieces))
            self.assertEqual(tok.decode(pieces), tok.convert_ids_to_tokens(tid))
        self.assertEqual(len({tuple(v) for v in src.values()}), len(FORMAT_TOKENS))


if __name__ == "__main__":
    unittest.main()
