"""Batched sampling from a policy checkpoint: HF generate (any OS) or vLLM (Linux).

Both backends take full ChatML prompt strings (format.build_prompt) and return the raw
completion text, cut at the first <|im_end|> (kept, so parse() sees a closed turn) or
<|endoftext|> (dropped). temperature == 0 means greedy.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from crllm.format import EOT, IM_END

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Completion:
    text: str
    n_tokens: int
    finish: str  # "stop" (hit <|im_end|> / <|endoftext|>) or "length"


class Generator(Protocol):
    def generate(
        self, prompts: Sequence[str], *, max_new_tokens: int, temperature: float = 0.0,
        top_p: float = 1.0, n: int = 1, seed: int = 0,
    ) -> list[list[Completion]]: ...


def _cut(ids: list[int], stop_keep: set[int], stop_drop: set[int]) -> tuple[list[int], str]:
    for i, t in enumerate(ids):
        if t in stop_keep:
            return ids[: i + 1], "stop"
        if t in stop_drop:
            return ids[:i], "stop"
    return ids, "length"


class HFGenerator:
    """Left-padded batched `model.generate`, longest prompts first so batches pad less."""

    def __init__(self, model, tokenizer, batch_size: int = 16, cache: str = "dynamic"):
        if cache not in ("dynamic", "static"):
            raise ValueError(f"cache must be 'dynamic' or 'static', got {cache!r}")
        self.model, self.tok, self.batch_size, self.cache = model, tokenizer, batch_size, cache
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.im_end = self.tok.convert_tokens_to_ids(IM_END)
        self.eot = self.tok.convert_tokens_to_ids(EOT)

    @torch.no_grad()
    def generate(self, prompts, *, max_new_tokens, temperature=0.0, top_p=1.0, n=1, seed=0):
        self.model.eval()
        enc = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: -len(enc[i]))
        out: list[list[Completion]] = [[] for _ in prompts]
        sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=sample,
            eos_token_id=[self.im_end, self.eot], pad_token_id=self.tok.pad_token_id,
        )
        if self.cache == "static":  # preallocated KV cache: no growth, no fragmentation
            gen_kwargs["cache_implementation"] = "static"
            # transformers auto-compiles static-cache decoding; inductor needs Triton (absent on Windows)
            gen_kwargs["disable_compile"] = True
        if sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        else:  # silence "generation flags are not valid" warnings from the model's own config
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        t0, done = time.time(), 0
        dev = next(self.model.parameters()).device
        for b in range(0, len(order), self.batch_size):
            idx = order[b: b + self.batch_size]
            width = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), width), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                ids[row, width - len(enc[i]):] = torch.tensor(enc[i])
                att[row, width - len(enc[i]):] = 1
            for k in range(n):
                if sample:
                    torch.manual_seed(seed + 1000 * k + b)
                y = self.model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), **gen_kwargs)
                for row, i in enumerate(idx):
                    new, finish = _cut(y[row, width:].tolist(), {self.im_end}, {self.eot})
                    out[i].append(Completion(self.tok.decode(new, skip_special_tokens=False), len(new), finish))
            done += len(idx)
            if torch.cuda.is_available():
                # Dynamic KV caches grow by concatenation; without this the freed blocks fragment
                # until the allocator sits at the memory cap and slows down (2.5 -> 3.2 s/prompt).
                del y
                torch.cuda.empty_cache()
            mem = f", peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB" if torch.cuda.is_available() else ""
            log.info("generated %d/%d (%.0fs%s)", done, len(prompts), time.time() - t0, mem)
        return out


class VLLMGenerator:
    """vLLM offline engine (Linux). Untested on the laptop: vLLM does not run natively on Windows."""

    def __init__(self, model_path: str, dtype: str = "bfloat16", gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 4096):
        from vllm import LLM

        self.llm = LLM(model=model_path, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len)
        tok = self.llm.get_tokenizer()
        self.im_end = tok.convert_tokens_to_ids(IM_END)
        self.eot = tok.convert_tokens_to_ids(EOT)

    def generate(self, prompts, *, max_new_tokens, temperature=0.0, top_p=1.0, n=1, seed=0):
        from vllm import SamplingParams

        sp = SamplingParams(
            n=n, temperature=temperature, top_p=top_p if temperature > 0 else 1.0, seed=seed,
            max_tokens=max_new_tokens, stop_token_ids=[self.im_end, self.eot], skip_special_tokens=False,
            include_stop_str_in_output=False,
        )
        res = self.llm.generate(list(prompts), sp)
        out = []
        for r in res:
            row = []
            for o in r.outputs:
                ids = list(o.token_ids)
                text = o.text
                if o.finish_reason == "stop" and ids and ids[-1] == self.im_end and not text.endswith(IM_END):
                    text += IM_END
                row.append(Completion(text, len(ids), "stop" if o.finish_reason == "stop" else "length"))
            out.append(row)
        return out


def make_generator(cfg, model_path: str, batch_size: int | None = None):
    """Generator for `model_path` per cfg.generation.backend. Inference always loads full precision.

    `batch_size` overrides cfg.generation.batch_size (eval passes its fixed eval.batch_size).
    """
    if cfg.generation.backend == "vllm":
        return VLLMGenerator(model_path, dtype=cfg.model.dtype)
    from crllm.modeling import load_base

    model, tok = load_base(cfg, model_path, for_training=False, load_in_4bit=False)
    return HFGenerator(model, tok, batch_size=batch_size or cfg.generation.batch_size, cache=cfg.generation.cache)
