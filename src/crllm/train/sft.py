"""SFT for Stage 1 (<think> cold-start) and Stage 2 (constitutional revisions).

    python -m crllm.train.sft --config debug --stage 1
    python -m crllm.train.sft --config debug --stage 1 --resume      # continue from the last checkpoint
    python -m crllm.train.sft --config debug --stage 1 --max-steps 5 # smoke run

Stage 1 trains on data/sft_think.jsonl starting from the base model; Stage 2 trains on
data/cai_revised.jsonl starting from ckpt/stage1-merged. Loss is on the completion only
(TRL prompt/completion format). LoRA on all linear layers plus the <think>/ChatML token
rows (modeling.attach_lora). The adapter goes to ckpt/stage<N>/; scripts/merge_adapter.py
then writes ckpt/stage<N>-merged.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from crllm.config import add_config_args, config_from_args, write_run_json
from crllm.runtime import apply_runtime

log = logging.getLogger("crllm.sft")

STAGE_DATA = {1: "sft_think.jsonl", 2: "cai_revised.jsonl"}


def stage_paths(cfg, stage: int) -> tuple[str, Path, Path]:
    """(base model, data file, adapter output dir) for a stage."""
    base = cfg.model.policy if stage == 1 else str(cfg.path("ckpt_dir") / f"stage{stage - 1}-merged")
    return base, cfg.path("data_dir") / STAGE_DATA[stage], cfg.path("ckpt_dir") / f"stage{stage}"


def load_rows(path: Path, seed: int, eval_n: int) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        raise SystemExit(f"{path} missing - build it first (python -m crllm.data.build_sft)")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    rows = [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    random.Random(seed).shuffle(rows)
    return rows[eval_n:], rows[:eval_n]


def make_trainer_class():
    """SFTTrainer with the standard (unfused) loss.

    TRL >= 1.19 always computes the loss through a fused LM head that reads
    `get_output_embeddings().weight` directly. With PEFT trainable_token_indices on a tied
    model that module is a TrainableTokensWrapper: `.weight` is the BASE weight (the trained
    <think>/ChatML rows are not applied) and `.bias` doesn't exist. So the fused path either
    crashes or, worse, would never train the output side of the format tokens. The plain
    forward runs the wrapped lm_head, so the trained rows are used on both sides.
    """
    from transformers import Trainer
    from trl import SFTTrainer

    class PlainLossSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    return PlainLossSFTTrainer


def main(argv: list[str] | None = None) -> Path:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke runs)")
    ap.add_argument("--eval-n", type=int, default=50, help="held-out examples for eval loss")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = config_from_args(args)
    applied = apply_runtime(cfg.runtime)

    import torch
    from datasets import Dataset
    from trl import SFTConfig

    from crllm.format import IM_END
    from crllm.modeling import attach_lora, load_base

    base, data_path, out_dir = stage_paths(cfg, args.stage)
    train_rows, eval_rows = load_rows(data_path, cfg.seed, args.eval_n)
    log.info("stage %d: %d train / %d eval examples from %s; base %s", args.stage, len(train_rows),
             len(eval_rows), data_path, base)

    model, tok = load_base(cfg, base, for_training=True)
    model = attach_lora(model, tok, cfg)
    if cfg.train.gradient_checkpointing and not cfg.model.load_in_4bit:
        model.enable_input_require_grads()  # prepare_model_for_kbit_training does this for 4-bit

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        num_train_epochs=cfg.train.epochs,
        max_steps=args.max_steps,
        warmup_steps=cfg.train.warmup_ratio,  # transformers 5: a float in [0, 1) is a ratio of total steps
        lr_scheduler_type="cosine",
        optim=cfg.train.optim,
        bf16=True,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_rows else "no",
        eval_steps=cfg.train.save_steps,
        report_to=cfg.tracking.report_to,
        run_name=f"{cfg.tracking.project}-stage{args.stage}-{cfg.name}",
        seed=cfg.seed,
        max_length=cfg.train.max_seq_len,
        packing=False,
        completion_only_loss=True,
        eos_token=IM_END,  # completions already end with <|im_end|>; don't append <|endoftext|>
        dataloader_num_workers=0,
    )
    trainer = make_trainer_class()(
        model=model, args=sft_cfg, processing_class=tok,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
    )
    sample = trainer.train_dataset[0]  # TRL 1.19: prompt tokens are -100 in labels
    shown = tok.decode([t for t, lab in zip(sample["input_ids"], sample["labels"]) if lab != -100])
    log.info("first example, loss tokens end with: %r", shown[-60:])
    if not shown.rstrip().endswith(IM_END) or shown.count(IM_END) != 1:
        raise SystemExit(f"loss target should end with exactly one {IM_END}: {shown[-120:]!r}")

    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    write_run_json(out_dir, cfg, stage=args.stage, base=base, data=str(data_path), n_train=len(train_rows),
                   runtime=applied, final_metrics=trainer.state.log_history[-1] if trainer.state.log_history else None,
                   peak_gpu_gib=torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None)
    log.info("saved adapter to %s", out_dir)
    return out_dir


if __name__ == "__main__":
    main()
