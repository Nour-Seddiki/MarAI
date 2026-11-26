# Usage: make test | make check | make debug | make stage1 CONFIG=full ...
# CONFIG picks the overlay in configs/ (debug or full); base.yaml is always loaded first.
CONFIG ?= debug
PY ?= python
CFG = configs/$(CONFIG).yaml

.PHONY: test check think-tokens eval-data eval-base stage1 stage2 stage3 stage4 debug full

test:
	$(PY) -m unittest discover -s tests -v

# M7 adds a 5-step smoke run here.
check: test

# M1: are <think>/</think> embeddings trained in the base model? (downloads the base model)
think-tokens:
	$(PY) scripts/check_think_tokens.py --config $(CFG)

# M2: one-off download of the eval suites into data/eval/
eval-data:
	$(PY) scripts/download_eval.py --config $(CFG)

eval-base:
	$(PY) -m crllm.eval.run_eval --config $(CFG) --ckpt base

stage1:
	$(PY) -m crllm.data.build_sft --config $(CFG)
	$(PY) -m crllm.train.sft --config $(CFG) --stage 1
	$(PY) scripts/merge_adapter.py --config $(CFG) --stage 1
	$(PY) -m crllm.eval.run_eval --config $(CFG) --ckpt stage1

stage2:
	$(PY) -m crllm.data.build_cai --config $(CFG)
	$(PY) -m crllm.train.sft --config $(CFG) --stage 2
	$(PY) scripts/merge_adapter.py --config $(CFG) --stage 2
	$(PY) -m crllm.eval.run_eval --config $(CFG) --ckpt stage2

stage3:
	$(PY) -m crllm.data.build_prefs --config $(CFG)
	$(PY) -m crllm.train.reward_model --config $(CFG)

stage4:
	$(PY) -m crllm.data.build_rl --config $(CFG)
	$(PY) -m crllm.train.grpo --config $(CFG)
	$(PY) scripts/merge_adapter.py --config $(CFG) --stage 4
	$(PY) -m crllm.eval.run_eval --config $(CFG) --ckpt final

debug:
	$(MAKE) test eval-base stage1 stage2 stage3 stage4 CONFIG=debug

full:
	$(MAKE) check eval-base stage1 stage2 stage3 stage4 CONFIG=full
