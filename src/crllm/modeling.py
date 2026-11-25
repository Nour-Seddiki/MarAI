"""Load a base model (4-bit or bf16), make the format tokens trainable, attach LoRA.

Format tokens: in Qwen3-*Base*, <|im_start|>, <|im_end|>, <think> and </think> have
never-trained embedding rows, all bit-identical to the padding rows (see
scripts/check_think_tokens.py). The embeddings are tied, so the four tokens also share
one output row and always get the same logit. LoRA on linear layers cannot fix that, so:

  1. `init_untrained_rows` gives each untrained format token a distinct starting row:
     the mean of the trained embeddings of its plain-text spelling ("<", "think", ">").
     This must run before `get_peft_model`, which snapshots the rows it will train.
  2. `lora_config(..., trainable_token_ids=...)` trains just those rows through PEFT's
     `trainable_token_indices` (a few thousand parameters; tied lm_head follows along).
  3. merge_and_unload writes the trained rows into the merged bf16 checkpoint, so later
     stages start with trained rows, and step 1 becomes a no-op.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from crllm.format import IM_END, IM_START, THINK_CLOSE, THINK_OPEN

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from crllm.config import Config, LoraCfg

log = logging.getLogger(__name__)

FORMAT_TOKENS = (IM_START, IM_END, THINK_OPEN, THINK_CLOSE)


# ---------------------------------------------------------------- format-token rows


def format_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """Id of each format token; fails if a tag is not a single token in this vocab."""
    ids = {}
    for tok in FORMAT_TOKENS:
        enc = tokenizer.encode(tok, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"{tok!r} is not a single token in this tokenizer ({enc}); format.py assumes it is")
        ids[tok] = enc[0]
    return ids


def is_tied(model: PreTrainedModel) -> bool:
    out = model.get_output_embeddings()
    return out is not None and out.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


def untrained_rows(weight: torch.Tensor, n_real_tokens: int) -> torch.Tensor:
    """Bool mask of rows bit-identical to the padding rows past the tokenizer's length.

    Never-trained rows keep their shared init value, so they all equal the padding rows.
    Without padding rows there is no reference, and nothing is flagged.
    """
    if weight.shape[0] <= n_real_tokens:
        return torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
    return (weight == weight[-1]).all(dim=1)


def text_init_sources(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> dict[int, list[int]]:
    """For each special token id: the ordinary token ids of its plain-text spelling."""
    added = set(getattr(tokenizer, "get_added_vocab", dict)().values())

    def ordinary(pieces: list[int]) -> bool:
        return bool(pieces) and not any(p in added or p in token_ids for p in pieces)

    out = {}
    for tid in token_ids:
        text = tokenizer.convert_ids_to_tokens(tid)
        pieces = tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        if not ordinary(pieces):
            # split_special_tokens does not split non-special added tokens such as Qwen's <think>;
            # encoding "<" and "think>" separately keeps the added-token matcher from firing.
            pieces = (tokenizer(text[:1], add_special_tokens=False)["input_ids"]
                      + tokenizer(text[1:], add_special_tokens=False)["input_ids"])
        if not ordinary(pieces):
            raise ValueError(f"could not spell {text!r} with ordinary tokens (got {pieces})")
        out[tid] = pieces
    return out


@torch.no_grad()
def init_rows_from_sources(model: PreTrainedModel, sources: dict[int, list[int]]) -> None:
    """Set row t := mean(rows of sources[t]) in the input embedding, and in an untied lm_head."""
    mats = [model.get_input_embeddings().weight]
    if not is_tied(model) and model.get_output_embeddings() is not None:
        mats.append(model.get_output_embeddings().weight)
    for w in mats:
        for tid, src in sources.items():
            w[tid] = w[src].float().mean(dim=0).to(w.dtype)


def init_untrained_rows(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, token_ids: list[int]
) -> list[int]:
    """Initialize the untrained rows among `token_ids`; trained rows are left alone. Returns the ids changed."""
    mask = untrained_rows(model.get_input_embeddings().weight, len(tokenizer))
    todo = [t for t in token_ids if bool(mask[t])]
    if todo:
        init_rows_from_sources(model, text_init_sources(tokenizer, todo))
        log.info("initialized untrained format-token rows %s from their text spellings", todo)
    return todo


def trainable_token_spec(model: PreTrainedModel, token_ids: list[int]) -> list[int] | dict[str, list[int]]:
    """`trainable_token_indices` value: a list for tied models, embedding + lm_head names otherwise."""
    if is_tied(model) or model.get_output_embeddings() is None:
        return list(token_ids)
    emb_name = out_name = None
    for name, mod in model.named_modules():
        if mod is model.get_input_embeddings():
            emb_name = name.rsplit(".", 1)[-1]
        elif mod is model.get_output_embeddings():
            out_name = name.rsplit(".", 1)[-1]
    return {emb_name: list(token_ids), out_name: list(token_ids)}


# ---------------------------------------------------------------- LoRA


def lora_config(
    lcfg: LoraCfg,
    task_type: str = "CAUSAL_LM",
    trainable_tokens: list[int] | dict[str, list[int]] | None = None,
) -> LoraConfig:
    from peft import LoraConfig

    return LoraConfig(
        r=lcfg.r,
        lora_alpha=lcfg.alpha,
        lora_dropout=lcfg.dropout,
        target_modules=lcfg.target_modules,
        task_type=task_type,
        bias="none",
        trainable_token_indices=trainable_tokens or None,
    )


def prepare_format_tokens(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, lcfg: LoraCfg):
    """Init untrained format rows (if enabled) and return the trainable_token_indices spec (or None)."""
    ids = list(format_token_ids(tokenizer).values())
    if lcfg.init_format_tokens:
        init_untrained_rows(model, tokenizer, ids)
    else:
        still = untrained_rows(model.get_input_embeddings().weight, len(tokenizer))
        bad = [t for t in ids if bool(still[t])]
        if bad:
            log.warning("format tokens %s have untrained rows and init_format_tokens is off", bad)
    return trainable_token_spec(model, ids) if lcfg.train_format_tokens else None


# ---------------------------------------------------------------- loading


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_base(cfg: Config, name_or_path: str | None = None, *, for_training: bool = True):
    """(model, tokenizer) for the policy (or `name_or_path`), 4-bit NF4 or full precision per config."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = name_or_path or cfg.model.policy
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(dtype=_dtype(cfg.model.dtype), attn_implementation=cfg.model.attn_implementation)
    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_dtype(cfg.model.dtype),
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    elif torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    if for_training and cfg.model.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
        )
    elif for_training and cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if for_training and cfg.train.gradient_checkpointing:
        model.config.use_cache = False
    return model, tokenizer


def attach_lora(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, cfg: Config):
    """Format-token rows + LoRA on a freshly loaded base. Returns the PeftModel."""
    from peft import get_peft_model

    spec = prepare_format_tokens(model, tokenizer, cfg.lora)
    peft_model = get_peft_model(model, lora_config(cfg.lora, trainable_tokens=spec))
    peft_model.print_trainable_parameters()
    return peft_model
