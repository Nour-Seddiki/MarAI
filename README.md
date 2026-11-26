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
| M2 Eval harness + baseline | done | 5 suites, local judge, baseline report (below) |
| M3 Stage 1 (debug) | next | |
| M4–M6 Stages 2–4 (debug) | — | |
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

## Evaluation

One command scores any checkpoint on fixed prompts with greedy decoding:

```bash
python scripts/download_eval.py                              # once: builds data/eval/*.jsonl
python -m crllm.eval.run_eval --config debug --ckpt base     # or stage1 | stage2 | final | <path>
```

It writes `reports/<config>/<name>.json` (metrics with standard errors) and `<name>.samples.jsonl` (every completion and verdict). Generation for all suites runs first. The model is then freed before judging, so the policy and a local judge never share the GPU. Eval always loads bf16, even in the debug config, so stages are compared without quantization noise.

| Suite | Prompts | Metric | Method | Gate |
|---|---|---|---|---|
| format | 200 mixed (120 Alpaca-cleaned + 80 GSM8K test) | % well-formed | `format.parse()` | ≥ 95% from Stage 1 on |
| gsm8k | GSM8K test (debug: first 200; full: all 1,319) | exact match | `extract_final_answer()` vs gold | never > 2 pts below the previous stage |
| redteam | 200 [Do-Not-Answer](https://huggingface.co/datasets/LibrAI/do-not-answer), stratified by risk area | mean safety 1–5 | `judge.score()` against the constitution | up after Stages 2 and 4 |
| overrefusal | 250 safe [XSTest](https://huggingface.co/datasets/Paul/XSTest) prompts | refusal rate | `judge.refusal()` | within 10 pts of base |
| honesty | 100 unanswerable [SelfAware](https://huggingface.co/datasets/OkayestProgrammer/selfAware) questions | % flagging uncertainty | `judge.uncertainty()` | up vs base |

The judge sees only the reply a user would read (the answer after `</think>`), and every judging prompt lives in `configs/constitution.yaml`. On the laptop the judge is `qwen3.5:4b` in Ollama with `reasoning_effort: none` (0.3 s per verdict; without it the model thinks for 600+ tokens). It's lenient on borderline privacy cases, so the full run should use a stronger judge. The build scripts drop any training prompt whose `key` (normalized-text hash) appears in `data/eval/`.

### Baseline: Qwen3-0.6B-Base, debug config

`reports/debug/base.json`, 2026-11-26: bf16, greedy, 768 new tokens, judge `qwen3.5:4b`. Errors are ± one standard error.

| Suite | Base | Note |
|---|---|---|
| format | **0.0%** well-formed | never emits `<think>`: those tokens are untrained in the base model |
| gsm8k (first 200) | **41.5% ± 3.5** | real step-by-step solutions, answer taken from the last number |
| redteam | **3.32 ± 0.09** / 5 | weakest area: Malicious Uses (2.56) |
| overrefusal | **15.6% ± 2.3** refused | |
| honesty | **10.0% ± 3.0** flag uncertainty | |

Without `<|im_end|>`, 23–37% of replies run to the 768-token limit, often trailing into junk symbols where the end-of-turn token should be. Generation took 46 min for 950 prompts on the RTX 5050 (batch 16); later stages stop early and should be much faster.

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
make eval-data                     # download + build the eval suites
make eval-base CONFIG=debug        # baseline report
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
  generate.py          batched sampling: HF generate (any OS), vLLM (Linux)
  judge.py             OpenAI-compatible judge client: score / refusal / uncertainty verdicts
  runtime.py           Windows laptop guards: keep-awake, P-core pinning, CUDA memory cap
  eval/
    suites.py          the five suites and their metrics
    run_eval.py        any checkpoint -> one JSON report + samples
scripts/
  check_think_tokens.py
  download_eval.py     eval sources -> data/eval/<suite>.jsonl (fixed seed)
tests/                 CPU-only (unittest, pytest-compatible)
```

Planned next, per the build plan: `rewards.py`, `data/build_*.py`, `train/{sft,reward_model,grpo}.py`, `scripts/merge_adapter.py`, and the judge's critique / revise / rank calls (M4).

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

Keep HF generation under that cap. `scripts/bench_generate.py` (32 eval prompts × 768 tokens, Qwen3-0.6B-Base, bf16):

| batch, KV cache | s / prompt | peak reserved |
|---|---|---|
| 8, dynamic | 4.13 | 2.6 GB |
| **16, dynamic (used)** | **2.53** | 4.9 GB |
| 32, dynamic | 2.26 | 6.3 GB, at the cap |
| 16, static | 3.52 | 2.9 GB |
| 32, static | 3.23 | 4.7 GB |

Decoding is launch-bound, so bigger batches win until the cap. At the cap, a long-prompt batch of 32 slowed to 9.3 s per prompt. The static cache doesn't fragment, but it's slower without `torch.compile`, and that needs Triton, which isn't available on Windows. Greedy bf16 outputs also depend on batch size (13 of 32 matched between batch 8 and 16) but repeat exactly at a fixed size. That's why `eval.batch_size` is fixed for every stage and recorded in each report.

Large `huggingface_hub` downloads have hung here. A resumable `curl -C -` loop on the `resolve/main/<file>` URL works; check the result against the `X-Linked-ETag` SHA-256.
