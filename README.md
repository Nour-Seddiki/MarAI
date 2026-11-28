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
| M2 Eval harness + baseline | done | 7 suites, local 9B judge, resumable runs (below) |
| M3 Stage 1 (debug) | done | math + code `<think>` SFT, `<think>` prefill (below) |
| M6 Stage 4 (debug) | done | GRPO on correctness + format, straight from Stage 1: format 50% → 70%, MBPP ×2 (below) |
| M4–M5 Judge + Stages 2–3 (debug) | — | constitutional SFT and reward model |
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

## Stage 1: reasoning cold-start (M3)

```bash
python -m crllm.data.build_sft --config debug      # -> data/sft_think.jsonl (2k debug / 20k full)
python -m crllm.train.sft --config debug --stage 1 # -> ckpt/stage1 (LoRA + format-token rows)
python scripts/merge_adapter.py --config debug --stage 1   # -> ckpt/stage1-merged (bf16)
```

**Data** ([`build_sft.py`](src/crllm/data/build_sft.py)): math and code, mixed by `data.stage1_mix`. Real DeepSeek-R1 traces are long, so only a few fit the token budget. A source that runs short hands its quota to the others.

| Source | Share | Debug (≤ 1,024 tokens) | What goes in `<think>` |
|---|---|---|---|
| [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) math | 35% | 676 (only 688 of 93.7k fit) | real R1 reasoning |
| Mixture-of-Thoughts code | 10% | 47 (85 of 83k fit) | real R1 reasoning |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | 25% | 500 | the step-by-step solution; answer `\boxed{}` |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), tests all passed | 30% (+ code shortfall) | 777 | a short plan written by the judge model (`qwen3.5:9b`, 3.2 s each) |

The plans are cached in `data/cache/teacher_plans.jsonl`, so rebuilds and resumed builds don't call the teacher again. Every example goes through `format.wrap()`, is counted with the policy tokenizer, de-duplicated, and dropped if its question is in the eval suites (3,326 banned keys). At 2,048 tokens (the full config) about 10k R1 math and 1.4k R1 code traces fit.

**Prefilled `<think>`.** After the first Stage 1 run, the model opened with `<think>` on 100% of chat and code prompts but only 60% of GSM8K and 70% of MATH-500 prompts. There it started "To solve…", Qwen3-Base's pretraining habit for math. Validation loss had flattened, so more epochs wouldn't fix it. As DeepSeek-R1's template does, the prompt now ends with `<think>` and the model writes only the reasoning, `</think>` and the answer.

**Trainer** ([`sft.py`](src/crllm/train/sft.py)): TRL `SFTTrainer` with loss on the completion only, QLoRA r 16 on all linear layers, plus the four format-token rows. TRL 1.19 always computes the loss through a fused LM head that reads `lm_head.weight` directly. With trainable token rows on a tied model, that weight is the *base* weight, so the trained `</think>` / `<|im_end|>` rows would never be used on the output side (or it crashes on the missing `.bias`). `sft.py` uses the standard loss instead, which runs the wrapped head.

Debug run (1,950 train / 50 held-out, 2 epochs, 244 steps, 40 min on the RTX 5050):

| step | epoch | train loss | val loss |
|---|---|---|---|
| 5 | 0.04 | 0.99 | |
| 100 | 0.82 | 0.62 | **0.606** |
| 200 | 1.64 | 0.55 | **0.584** |
| 244 | 2.00 | 0.57 | **0.584** |

Validation loss only goes down, then flattens. Epoch 2's drop in training loss (0.62 → 0.55) is mostly memorization, so 2 epochs is the most this data size supports. `python scripts/show_losses.py ckpt/stage1/checkpoint-*/trainer_state.json` prints the full table.

## Stage 4: GRPO with verifiable rewards (M6, run early)

Stage 4 runs **straight from Stage 1**, before Stages 2–3. The goal is to raise math and code accuracy and fix the `</think>` problem first. The plan's constitutional reward needs the Stage 3 reward model, so for now the rewards are:

| Reward ([`rewards.py`](src/crllm/rewards.py)) | Weight | Score |
|---|---|---|
| correctness | 0.7 | math: 1 if the final answer matches the gold (LaTeX-normalized); code: share of unit-test asserts that pass; chat: none |
| format | 0.3 | 1 if the turn is well-formed **and** actually ended on `<|im_end|>` (checked on token ids, since TRL strips special tokens from the text) |

Both rewards are **strict**: they read only what comes after `</think>`, so an unclosed thinking block earns nothing. A reply cut off at `max_completion_length` (512 in debug) also earns nothing, which pushes the model to finish its reasoning sooner.

```bash
python -m crllm.data.build_rl --config debug     # -> data/rl_prompts.jsonl
python -m crllm.train.grpo --config debug        # -> ckpt/stage4 (LoRA on stage1-merged)
python scripts/merge_adapter.py --config debug --stage 4    # -> ckpt/final
```

**Prompts** ([`build_rl.py`](src/crllm/data/build_rl.py)): 45% math (OpenMathInstruct-2, half GSM8K-style, half MATH-style), 35% code (OpenCodeInstruct with ~10 asserts each), 20% chat (Alpaca-cleaned, format reward only). Eval and Stage 1 questions are excluded. A code task is kept only if its reference solution passes all of its own asserts through our runner; 11 of 361 failed.

**Trainer** ([`grpo.py`](src/crllm/train/grpo.py)): TRL `GRPOTrainer`, a fresh LoRA, 4 generations per prompt, β = 0.02 (KL to the Stage 1 model), lr 2e-5. The format-token rows aren't trained again; Stage 1 already trained them.

## Evaluation

One command scores any checkpoint on fixed prompts with greedy decoding:

```bash
python scripts/download_eval.py                              # once: builds data/eval/*.jsonl
python -m crllm.eval.run_eval --config debug --ckpt base     # or stage1 | stage2 | final | <path>
```

It writes `reports/<config>/<name>.json` (metrics with standard errors) and `<name>.samples.jsonl` (every completion and verdict). Generation for all suites runs first. The model is then freed before judging, so the policy and a local judge never share the GPU. Eval always loads bf16, even in the debug config, so stages are compared without quantization noise. Debug runs use up to 1,024 new tokens and full runs 2,048, the same as the training length.

Generation saves every batch to `<name>.partial.jsonl`, so a stopped run resumes where it left off (`--fresh` starts over). `--from-samples <samples.jsonl>` re-scores saved completions without generating, for example after changing the judge.

| Suite | Prompts | Metric | Method | Gate |
|---|---|---|---|---|
| format | 400 mixed: 200 Alpaca-cleaned + 100 GSM8K + 50 MATH-500 + 50 MBPP | % well-formed | `format.parse()` | ≥ 95% from Stage 1 on |
| gsm8k | GSM8K test (debug: first 500; full: all 1,319) | exact match | `extract_final_answer()` vs gold | never > 2 pts below the previous stage |
| math500 | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | exact match | last `\boxed{}` vs gold, LaTeX-normalized | never > 2 pts below the previous stage |
| mbpp | [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) sanitized test, 257 | pass@1 | run the asserts on the extracted code | never > 2 pts below the previous stage |
| redteam | 500 [Do-Not-Answer](https://huggingface.co/datasets/LibrAI/do-not-answer), de-duplicated, stratified by risk area | mean safety 1–5 | `judge.score()` against the constitution | up after Stages 2 and 4 |
| overrefusal | 250 safe [XSTest](https://huggingface.co/datasets/Paul/XSTest) prompts | refusal rate | `judge.refusal()` | within 10 pts of base |
| honesty | 300 unanswerable [SelfAware](https://huggingface.co/datasets/OkayestProgrammer/selfAware) questions | % flagging uncertainty | `judge.uncertainty()` | up vs base |

MBPP runs model-written code with `python -I` in an empty temp directory, with a minimal environment and a 10 s timeout ([`execute.py`](src/crllm/eval/execute.py)). That limits accidents, but it is **not a security sandbox**. 256 of the 257 reference solutions pass through it; the last is a slow reference that times out.

**Judge.** The judge sees only the reply a user would read (the answer after `</think>`). Every judging prompt lives in `configs/constitution.yaml`. On the laptop it's `qwen3.5:9b` in Ollama with `reasoning_effort: none` (~1 s per verdict). `scripts/compare_judges.py` re-judged 40 saved baseline replies with each candidate:

- **`qwen3.5:4b`** rated an invented "Neil is married" privacy leak and a made-up UN Mars-colonization announcement 3/5. The 9B rated both 1.
- **`qwen2.5-coder:7b`** gave 12 of 16 red-team replies a 5, including that leak, so it's unusable as a judge.
- **The judge matters as much as the model.** On the same base completions, the 4B called 15.6% of safe XSTest replies refusals; the 9B called 3.2% refusals. Every stage must be scored by the same judge.

The build scripts drop any training prompt whose `key` (normalized-text hash) appears in `data/eval/`.

### Results: debug config (Qwen3-0.6B-Base)

All three models are scored with the same protocol: bf16, greedy, `<think>` prefill, 1,024 new tokens, batch 16, judge `qwen3.5:9b`. Errors are ± one standard error. `python scripts/summarize_report.py reports/debug/base.json reports/debug/stage1.json reports/debug/final.json` prints this table. "Stage 4" is `ckpt/final`: GRPO run straight on top of Stage 1.

| Suite | Base | Stage 1 | **Stage 4** | Stage 1 → 4 |
|---|---|---|---|---|
| format (well-formed) | 0.0% | 49.7% ± 2.5 | **70.3% ± 2.3** | **+20.6** (gate ≥ 95%: not met yet) |
| GSM8K (500), strict | 0.0% | 57.2% ± 2.2 | **63.4% ± 2.2** | **+6.2** |
| GSM8K, lenient | 21.2% | 58.8% | 63.8% | +5.0 |
| MATH-500, strict | 0.0% | 35.0% ± 2.1 | 32.8% ± 2.1 | −2.2 (within noise) |
| MATH-500, lenient | 22.2% | 36.2% | 35.0% | −1.2 |
| MBPP pass@1, strict | 0.0% | 11.7% ± 2.0 | **21.8% ± 2.6** | **+10.1** |
| MBPP pass@1, lenient | 0.8% | 32.7% | **48.6%** | **+15.9** |
| red-team safety (1–5) | 3.51 ± 0.05 | 3.46 ± 0.06 | 3.55 ± 0.06 | +0.09 (noise) |
| red-team harmful rate (score ≤ 2) | 12.6% ± 1.5 | 19.4% ± 1.8 | 20.2% ± 1.8 | +0.8 (noise) |
| over-refusal (lower is better) | 24.4% | 0.8% | 0.8% | 0 |
| honesty (flags uncertainty) | 3.7% | 8.3% ± 1.6 | **14.0% ± 2.0** | **+5.7** |

Replies hitting the 1,024-token limit (Stage 1 → Stage 4): format 39% → 12%, GSM8K 18% → 2%, MBPP 40% → 2%, red-team 52% → 28%.

**What Stage 4 changed** (100 GRPO steps, 400 prompts, ~1.5 h on the laptop):

- **Format is the big win.** Well-formed replies rose from 50% to 70% overall: GSM8K prompts 97%, chat prompts 36% → 66%, MBPP 18% → 30%. Runaway thinking nearly disappeared on math and code, and replies are about 40% shorter.
- **Code accuracy almost doubled** (strict MBPP 11.7% → 21.8%). The model also writes working code more often (lenient 32.7% → 48.6%). Code is still where the format fails most: 134 of 257 MBPP replies end the turn without `</think>` and with no code after it.
- **GSM8K up 6 points**, which is real (about 2 standard errors). **MATH-500 didn't improve** and may have slipped slightly, within noise. GRPO trained with a 512-token completion budget, which rewards short answers, and the hard MATH problems need long reasoning.
- **Safety is unchanged by Stage 4, but worse than base.** The harmful rate went up in **Stage 1** (12.6% → 19.4%), once the model learned to answer instead of rambling; Stage 4 kept it there (20.2%). Nothing so far rewards declining a harmful request. That's what the constitutional stages (2–3) are for, and the model **shouldn't be used** until then.

**Strict** scores read the answer after `</think>`, which is what a user sees. **Lenient** scores accept an answer found anywhere in the reply, so reasoning ability and format are measured separately. With the prefill, the base model never closes `</think>`, so it gets 0 on strict, and the unfamiliar `<think>` token also lowers its lenient score: without the prefill it got 41.5% on GSM8K. Even on the lenient measure, Stage 1 gains **+38 points on GSM8K, +14 on MATH-500 and +32 on MBPP**, so the training improved the reasoning and not just the format. On code the format costs the most: the model writes passing code 32.7% of the time, but only 11.7% after `</think>`.

The judge varies a little between runs. Re-judging the same Stage 1 completions moved red-team from 3.40 to 3.46, about one standard error.

Where Stage 1 fell short of the format gate, before Stage 4:

- **General chat never stops thinking.** Only 36% of the Alpaca format prompts are well-formed, and 45–52% of red-team, over-refusal and honesty replies run to the 1,024-token limit. Stage 1 had no general chat data, so on "explain Agile" or a harmful request the model reasons in R1 style and never gets to an answer.
- **Code skips `</think>`.** Only 18–25% of MBPP replies are well-formed: the model writes reasoning and code inside the thinking block and ends the turn, or writes `</think>` more than once.
- **Math works:** 76–85% well-formed on GSM8K and MATH-500 prompts.

Next steps:

- **Stages 2–3 (constitution):** constitutional SFT and a reward model, so a constitutional reward can join correctness and format in GRPO. This is needed before the model is safe to use.
- **Stage 1 data:** add a general chat source with short teacher-written thinking, prefer shorter R1 traces, and give code more weight.
- **Scale:** the full config on a rented GPU (Qwen3-1.7B, 20k examples, 2,048 tokens).

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
make stage1 CONFIG=debug           # build data -> train -> merge -> eval
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
  data/
    download.py        resumable Hub parquet downloads (plain HTTP + Range)
    build_sft.py       Stage 1 data: math + code mix, eval de-dup, teacher code plans
  train/
    sft.py             SFT for Stages 1 and 2 (TRL, unfused loss)
  eval/
    suites.py          the seven suites and their metrics
    execute.py         runs MBPP code against its asserts (subprocess, timeout)
    run_eval.py        any checkpoint -> one JSON report + samples; resumable
scripts/
  check_think_tokens.py
  download_eval.py     eval sources -> data/eval/<suite>.jsonl (fixed seed)
  merge_adapter.py     LoRA + token rows -> bf16 checkpoint
  compare_judges.py    re-judge saved samples with other judge models
  bench_generate.py    generation speed / memory / agreement per batch and cache
  show_losses.py       train / val loss table from a log or trainer_state.json
tests/                 CPU-only (unittest, pytest-compatible)
```

Planned next, per the build plan: `rewards.py`, `data/build_{cai,prefs,rl}.py`, `train/{reward_model,grpo}.py`, and the judge's critique / revise / rank calls (M4).

## Text format

`format.py` is the single source of truth, used by SFT data, sampling, rewards and eval:

```
prompt      <|im_start|>system\n…<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>
completion  {reasoning}</think>\n{answer}<|im_end|>          (what the model is trained to write)
turn        <think>{reasoning}</think>\n{answer}<|im_end|>   (prefill + completion, what parse() checks)
```

The generator puts the `<think>` prefill back on everything it returns (`format.from_completion`), so scorers, samples files and rewards always see the full turn. `parse()` calls a turn well formed only if it has exactly one `<think>` block at the start, non-empty reasoning, a non-empty answer, and nothing after `<|im_end|>`. A model that writes a second `<think>` after the prefill fails. For malformed text it still returns a best-guess answer, so correctness can be scored separately from format. Gold answers are stored already passed through `extract_final_answer()` (`\boxed{}` > `####` > "answer is" > last number, normalized), so rewards compare with `==`.

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
