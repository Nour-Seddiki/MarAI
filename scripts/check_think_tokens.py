"""Do <think> / </think> have trained embeddings in the base model?

The plan relies on Qwen3-*Base* already knowing these tokens (no embed_tokens training,
LoRA on linear layers only). If their rows look like the never-trained padding rows,
LoRA cannot fix them and Stage 1 will struggle to emit the tags.

Compares the L2 norm of each special token's input-embedding (and lm_head) row against
  * ordinary vocabulary rows (trained)
  * the padding rows past len(tokenizer) (never trained)
and checks that the format strings tokenize the way format.py assumes.

    python scripts/check_think_tokens.py --config debug        # Qwen3-0.6B-Base
    python scripts/check_think_tokens.py --config full         # Qwen3-1.7B-Base
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crllm.config import add_config_args, config_from_args
from crllm.format import IM_END, IM_START, THINK_CLOSE, THINK_OPEN, build_prompt, wrap


def describe(norms: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(norms.float(), torch.tensor([0.01, 0.5, 0.99]))
    return {"p01": q[0].item(), "median": q[1].item(), "p99": q[2].item()}


def main() -> None:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model", help="override cfg.model.policy")
    args = ap.parse_args()
    cfg = config_from_args(args)
    name = args.model or cfg.model.policy

    tok = AutoTokenizer.from_pretrained(name)
    report: dict = {"model": name, "tokenization": {}}

    ids = {}
    for t in (THINK_OPEN, THINK_CLOSE, IM_START, IM_END):
        enc = tok.encode(t, add_special_tokens=False)
        ids[t] = enc
        report["tokenization"][t] = enc
    single = all(len(v) == 1 for v in ids.values())

    sample = build_prompt("What is 2+2?") + wrap("2+2 is 4", "4")
    pieces = tok.convert_ids_to_tokens(tok.encode(sample, add_special_tokens=False))
    report["sample_tokens"] = pieces

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, device_map="cpu")
    emb = model.get_input_embeddings().weight.detach()
    head = model.get_output_embeddings().weight.detach()
    tied = emb.data_ptr() == head.data_ptr()
    n_tok, n_rows = len(tok), emb.shape[0]
    added = sorted(tok.get_added_vocab().values())
    first_added = added[0] if added else n_tok

    out = {}
    for label, w in (("input_embedding", emb),) + (() if tied else (("lm_head", head),)):
        norms = w.norm(dim=-1)
        # Never-trained rows keep their shared init value, so they are bit-identical to the
        # padding rows past len(tokenizer). Norm thresholds miss this (some BPE rows are untrained too).
        untrained = (w == w[-1]).all(dim=1) if n_rows > n_tok else torch.zeros(n_rows, dtype=torch.bool)
        stats = {
            "ordinary_vocab": describe(norms[:first_added]),
            "padding_rows": describe(norms[n_tok:]) if n_rows > n_tok else None,
            "untrained_rows": int(untrained.sum()),
            "untrained_added_ids": [i for i in added if untrained[i]],
            "tokens": {t: {"id": v[0], "norm": norms[v[0]].item(), "untrained": bool(untrained[v[0]])}
                       for t, v in ids.items() if len(v) == 1},
        }
        out[label] = stats
    report["tied_embeddings"] = tied
    report["norms"] = out

    print(json.dumps(report, indent=2, ensure_ascii=False))

    verdicts = []
    if not single:
        verdicts.append("FAIL: some format tags are not single tokens; format.py's assumptions break")
    for label, stats in out.items():
        bad = [t for t, s in stats["tokens"].items() if s["untrained"]]
        if bad:
            msg = f"FAIL: {', '.join(bad)} {label} rows are identical to the untrained padding rows"
            if tied:
                msg += " (tied lm_head: these tokens always get the same logit, so the model cannot tell them apart)"
            verdicts.append(msg + "; their rows must be trained (e.g. PEFT trainable_token_indices)")
    print("\n".join(verdicts) if verdicts else "OK: tags are single tokens with trained embeddings")
    raise SystemExit(1 if any(v.startswith("FAIL") for v in verdicts) else 0)


if __name__ == "__main__":
    main()
