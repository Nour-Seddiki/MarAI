# MarAI — Constitutional Reasoning LLM (`crllm`)

A small reasoning model trained in four stages. It thinks inside `<think>…</think>` before answering and is aligned to a written constitution.

| Stage | What it does | Output |
|---|---|---|
| 0. Baseline | Evaluate the untouched base model | `reports/base.json` |
| 1. Reasoning cold-start | SFT on `<think>` reasoning traces | `ckpt/stage1-merged` |
| 2. Constitutional SFT | Generate → critique → revise against a principle, then SFT on the revisions | `ckpt/stage2-merged` |
| 3. Reward model | Judge ranks K = 4 samples per prompt → preference pairs → reward model | `ckpt/rm` |
| 4. GRPO | RL with correctness, format and constitutional rewards | `ckpt/final` |

Each stage ends with a LoRA merge into a bf16 checkpoint and an eval run. The next stage starts a fresh LoRA on that checkpoint. Every stage runs first on a laptop with the **debug** config (Qwen3-0.6B-Base, 4-bit QLoRA, tiny data), and only then with the **full** config on a rented 24 GB GPU (Qwen3-1.7B-Base, bf16 LoRA).

## Status

| Milestone | | |
|---|---|---|
| M0 Skeleton | done | package, configs, config loader, laptop runtime guards |
| M1 Format module | done | `format.py` + tests, format-token check and fix (below) |
| M2 Eval harness + baseline | next | |
| M3–M6 Stages 1–4 (debug) | — | |
| M7 Full run on rented GPU | — | |
| M8 Report, GGUF export | — | |

## Finding: Qwen3-Base never trained its format tokens

The original plan assumed `<think>` / `</think>` are usable as-is in Qwen3-Base. They are not. In **Qwen3-0.6B-Base**, every added special token except `<|endoftext|>` (ids 151644–151668) has an embedding row that is bit-identical to the never-trained padding rows. That includes `<|im_start|>`, `<|im_end|>`, `<think>` and `</think>`. The embeddings are tied, so these four tokens also share one output row and **always get the same logit**: the model can't prefer `</think>` over `<|im_end|>`. LoRA on linear layers can't fix that.

The fix, in [`modeling.py`](src/crllm/modeling.py), is on by default:

1. Give each untrained format token a distinct starting row: the mean of the trained embeddings of its plain-text spelling (`<` `think` `>`).
2. Train just those rows with PEFT `trainable_token_indices`, alongside LoRA. It adds about 4k parameters, and the tied `lm_head` follows automatically.
3. Merging writes the trained rows into the stage checkpoint, so later stages start from trained rows.

A smoke test on the real model (4-bit, default lr 2e-4, 14 toy training examples, held-out prompts) gave:

| | p(`</think>`) after reasoning | p(`<|im_end|>`) after answer |
|---|---|---|
| raw base | 4.3e-06 (tied with the other tags) | 7.3e-06 |
| after 20 steps | 0.94 | 0.96 |
| after 60 steps | 0.997 | 0.997 |

This shows the mechanism works; the Stage 1 format gate (≥ 95% well-formed) is the real test. Check any new base model (e.g. Qwen3-1.7B-Base) with:

```bash
python scripts/check_think_tokens.py --config full   # exit code 1 if format-token rows are untrained
```

## Install

Install the PyTorch build that matches your GPU first (CUDA 12.8+ for RTX 50-series / Blackwell). It is deliberately not a dependency, so pip never replaces it.

```bash
pip install -e .[quant,dev]       # laptop: bitsandbytes for 4-bit
pip install -e .[vllm,track]      # rented Linux GPU: vLLM + W&B
```

## Usage

```bash
make test                          # CPU tests (on Windows: python -m unittest discover -s tests)
make think-tokens CONFIG=debug     # format-token check on the base model
make stage1 CONFIG=full            # build data -> train -> merge -> eval   (M3+)
make debug                         # every stage end to end on tiny data  (M6+)
```

Every entry point takes `--config debug|full|path.yaml` and repeatable `--set key.sub=value` overrides. Unknown keys and wrong types fail when the config loads:

```bash
python scripts/check_think_tokens.py --config debug --set model.policy=/path/to/local/model
```

## Layout

```
configs/
  base.yaml            shared defaults
  debug.yaml           laptop 8 GB overrides (0.6B, QLoRA, tiny data, runtime guards on)
  full.yaml            rented 24 GB overrides (1.7B, bf16 LoRA, vLLM)
  constitution.yaml    principles + critique / revise / rank templates
src/crllm/
  config.py            YAML -> typed dataclasses, CLI overrides, run.json metadata
  format.py            the one prompt template; <think> wrap / parse; final-answer extraction
  modeling.py          load base (4-bit or bf16), format-token rows, LoRA
  runtime.py           Windows laptop guards: keep-awake, P-core pinning, CUDA memory cap
scripts/
  check_think_tokens.py
tests/                 CPU-only (unittest, pytest-compatible)
```

Planned next, per the build plan: `generate.py`, `judge.py`, `rewards.py`, `data/build_*.py`, `train/{sft,reward_model,grpo}.py`, `eval/`, `scripts/merge_adapter.py`.

## Text format

`format.py` is the single source of truth, used by SFT data, sampling, rewards and eval:

```
prompt      <|im_start|>system\n…<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n
completion  <think>{reasoning}</think>\n{answer}<|im_end|>
```

`parse()` calls a completion well formed only if it has exactly one `<think>` block at the start, non-empty reasoning, a non-empty answer, and nothing after `<|im_end|>`. For malformed text it still returns a best-guess answer, so correctness can be scored separately from format. Gold answers are stored already passed through `extract_final_answer()` (`\boxed{}` > `####` > "answer is" > last number, normalized), so rewards compare with `==`.

Note: the default system prompt mentions `<think>...</think>`, which tokenizes to the real tag ids. Search for tags inside the completion only, never in prompt + completion.

## Laptop notes (Windows, RTX 5050 8 GB)

The debug config turns on three guards in `runtime.py`, each added after a lost run:

- **keep-awake.** Modern Standby suspends runs about 30 min after the last keyboard or mouse input.
- **P-core pinning.** Runs launched from a background shell land on E-cores and go about 5× slower.
- **`cuda_memory_fraction: 0.8`.** On WDDM the CUDA cache spills into shared RAM instead of running out of memory, which caused a 9× slowdown once.
