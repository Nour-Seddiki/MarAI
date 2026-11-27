"""Stage 4: GRPO with verifiable rewards (crllm/rewards.py).

    python -m crllm.train.grpo --config debug                 # starts from ckpt/stage1-merged
    python -m crllm.train.grpo --config debug --max-steps 3   # smoke run
    python -m crllm.train.grpo --config debug --resume

Policy = a fresh LoRA on the base checkpoint (Stage 1 merged for now; the plan's Stage 2
once it exists). The reference for the KL term (beta) is the same model with the LoRA
disabled. The format-token rows are NOT trained here: they were trained in Stage 1, and
TRL's fused LM head would bypass PEFT's token-row wrapper anyway (see train/sft.py).
Rewards: [correctness, format] with cfg.grpo.reward_weights. Adapter -> ckpt/stage4/;
scripts/merge_adapter.py --stage 4 then writes ckpt/final.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from crllm.config import add_config_args, config_from_args, write_run_json
from crllm.runtime import apply_runtime

log = logging.getLogger("crllm.grpo")


def load_prompts(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} missing - build it first (python -m crllm.data.build_rl)")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return [{"prompt": r["prompt"], "task": r["task"], "answer": r["answer"],
             "tests": r.get("tests") or [], "test_imports": r.get("test_imports") or []} for r in rows]


def load_resume_adapter(model, out_dir: Path) -> str:
    """Load the LoRA weights of the last checkpoint into `model`; returns the checkpoint path.

    Trainer(resume_from_checkpoint=...) restores the optimizer, scheduler, RNG and step, but
    on this PEFT + 4-bit setup it did NOT restore the adapter: the first resumed step logged
    KL = 0, i.e. a blank LoRA. So the weights are loaded explicitly and checked here.
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers.trainer_utils import get_last_checkpoint

    ckpt = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if ckpt is None:
        raise SystemExit(f"--resume: no checkpoint-* in {out_dir}")
    state = load_file(str(Path(ckpt) / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, state)
    missing_lora = [k for k in getattr(result, "missing_keys", []) if "lora_" in k]
    if getattr(result, "unexpected_keys", None) or missing_lora:
        raise SystemExit(f"--resume: adapter keys don't match ({len(result.unexpected_keys)} unexpected, "
                         f"{len(missing_lora)} missing LoRA), e.g. {(result.unexpected_keys or missing_lora)[:2]}")
    b_norm = sum(p.detach().float().abs().sum().item() for n, p in model.named_parameters() if "lora_B" in n)
    if b_norm == 0:
        raise SystemExit("--resume: LoRA B weights are all zero after loading; the adapter is blank")
    log.info("resume: loaded %d adapter tensors from %s (sum |lora_B| = %.3f)", len(state), ckpt, b_norm)
    return ckpt


def main(argv: list[str] | None = None) -> Path:
    ap = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--base", default=None, help="checkpoint to start from (default: ckpt/stage1-merged)")
    ap.add_argument("--max-steps", type=int, default=None, help="override grpo.steps")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = config_from_args(args)
    applied = apply_runtime(cfg.runtime)

    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    from crllm import rewards
    from crllm.format import EOT, IM_END
    from crllm.modeling import load_base, lora_config

    ck = cfg.path("ckpt_dir")
    base = args.base or str(ck / "stage1-merged")
    out_dir = ck / "stage4"
    rows = load_prompts(cfg.path("data_dir") / "rl_prompts.jsonl")
    g = cfg.grpo
    if g.batch_size % g.num_generations:
        raise SystemExit(f"grpo.batch_size ({g.batch_size}) must be a multiple of num_generations ({g.num_generations})")
    log.info("stage 4: %d prompts from %s; base %s; %d steps x %d prompts x %d generations",
             len(rows), cfg.path("data_dir") / "rl_prompts.jsonl", base, args.max_steps or g.steps,
             g.batch_size // g.num_generations * g.grad_accum, g.num_generations)

    model, tok = load_base(cfg, base, for_training=True)
    lcfg = dataclasses.replace(cfg.lora, train_format_tokens=False, init_format_tokens=False)
    model = get_peft_model(model, lora_config(lcfg))
    model.print_trainable_parameters()
    if cfg.train.gradient_checkpointing and not cfg.model.load_in_4bit:
        model.enable_input_require_grads()

    stop_ids = [tok.convert_tokens_to_ids(IM_END), tok.convert_tokens_to_ids(EOT)]
    model.generation_config.eos_token_id = stop_ids  # stop at <|im_end|>, not only <|endoftext|>
    rewards.STOP_IDS = set(stop_ids)

    gcfg = GRPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=g.batch_size,
        gradient_accumulation_steps=g.grad_accum,
        num_generations=g.num_generations,
        max_completion_length=g.max_completion_length,
        temperature=g.temperature,
        top_p=cfg.generation.top_p,
        beta=g.beta,
        learning_rate=g.lr,
        max_steps=args.max_steps or g.steps,
        reward_weights=g.reward_weights,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0.05,
        logging_steps=g.logging_steps,
        save_steps=g.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        generation_kwargs={"use_cache": True},  # gradient checkpointing turns the KV cache off otherwise
        report_to=cfg.tracking.report_to,
        run_name=f"{cfg.tracking.project}-stage4-{cfg.name}",
        seed=cfg.seed,
        use_vllm=g.use_vllm,
        dataloader_num_workers=0,
    )
    if len(rewards.REWARD_FUNCS) != len(g.reward_weights):
        raise SystemExit(f"grpo.reward_weights needs {len(rewards.REWARD_FUNCS)} values "
                         f"({[f.__name__ for f in rewards.REWARD_FUNCS]})")
    resume_ckpt = load_resume_adapter(model, out_dir) if args.resume else None
    trainer = GRPOTrainer(model=model, reward_funcs=rewards.REWARD_FUNCS, args=gcfg,
                          train_dataset=Dataset.from_list(rows), processing_class=tok)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    write_run_json(out_dir, cfg, stage=4, base=base, n_prompts=len(rows), runtime=applied,
                   reward_funcs=[f.__name__ for f in rewards.REWARD_FUNCS], reward_weights=g.reward_weights,
                   final_metrics=trainer.state.log_history[-1] if trainer.state.log_history else None,
                   peak_gpu_gib=torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None)
    log.info("saved adapter to %s", out_dir)
    return out_dir


if __name__ == "__main__":
    main()
