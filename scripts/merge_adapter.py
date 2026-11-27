"""Merge a stage's LoRA (+ trained format-token rows) into a bf16 checkpoint.

    python scripts/merge_adapter.py --config debug --stage 1     # ckpt/stage1 -> ckpt/stage1-merged

The adapter is applied to a fresh bf16 copy of the stage's base (never the 4-bit one, which
can't be merged cleanly). PEFT stores the trained token rows as full values and copies them
in on merge, so the untrained rows of the fresh base are replaced exactly. The merged
directory gets the tokenizer and a run.json (config, seed, git hash, source adapter).
"""
from __future__ import annotations

import argparse
import logging

import torch

from crllm.config import add_config_args, config_from_args, write_run_json

log = logging.getLogger("crllm.merge")

STAGE_BASE = {1: None, 2: "stage1-merged", 4: "stage2-merged"}
STAGE_OUT = {1: "stage1-merged", 2: "stage2-merged", 4: "final"}


def main() -> None:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--stage", type=int, choices=sorted(STAGE_OUT), required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = config_from_args(args)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from crllm.modeling import format_token_ids, untrained_rows

    ck = cfg.path("ckpt_dir")
    adapter = ck / f"stage{args.stage}"
    base = cfg.model.policy if STAGE_BASE[args.stage] is None else str(ck / STAGE_BASE[args.stage])
    out = ck / STAGE_OUT[args.stage]
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter}")

    tok = AutoTokenizer.from_pretrained(str(adapter))
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()

    emb = model.get_input_embeddings().weight
    ids = format_token_ids(tok)
    still = untrained_rows(emb, len(tok))
    bad = [t for t, i in ids.items() if bool(still[i])]
    if bad:
        raise SystemExit(f"merged model still has untrained rows for {bad}; was the adapter trained with "
                         "lora.train_format_tokens?")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    write_run_json(out, cfg, stage=args.stage, base=base, adapter=str(adapter))
    log.info("merged %s + %s -> %s (format-token rows trained: %s)", base, adapter, out, list(ids))


if __name__ == "__main__":
    main()
