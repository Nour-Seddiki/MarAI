"""Stage 4 (GRPO) reward functions, in TRL's signature: f(prompts, completions, **dataset_columns).

Completions are what the model wrote after the <think> prefill; every function first puts
the prefill back (format.from_completion) and scores the full turn. Correctness is STRICT:
it reads only the answer after </think>, i.e. what a user would see, so the two rewards
pull together - an unclosed <think> earns neither.

  format_reward       1.0 if the turn is well formed (exactly one closed <think> block,
                      non-empty reasoning and answer, nothing after <|im_end|>), else 0.0
  correctness_reward  math: 1.0 if the final answer matches the gold (LaTeX-normalized)
                      code: share of the unit-test asserts that pass (partial credit)
                      chat: None (no ground truth; TRL's nansum skips it)

Weights come from cfg.grpo.reward_weights, in REWARD_FUNCS order.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from crllm.eval.execute import fraction_passed
from crllm.format import extract_code, extract_final_answer, from_completion, last_boxed, math_answers_equal, parse


STOP_IDS: set[int] = set()  # set by train/grpo.py: <|im_end|> and <|endoftext|> ids


def _turns(prompts, completions) -> list[str]:
    return [from_completion(p, c) for p, c in zip(prompts, completions)]


def format_reward(prompts, completions, completion_ids=None, **kw) -> list[float]:
    """1.0 for a well-formed turn that actually ENDED. TRL decodes completions with special
    tokens skipped, so a reply cut off at max_completion_length ("...</think>\\nThe answer is")
    would parse as well formed; the token ids show whether it stopped on <|im_end|>."""
    out = []
    for i, t in enumerate(_turns(prompts, completions)):
        ok = parse(t).well_formed
        if ok and STOP_IDS and completion_ids is not None:
            ids = list(completion_ids[i])
            ok = bool(ids) and ids[-1] in STOP_IDS
        out.append(1.0 if ok else 0.0)
    return out


def math_correct(answer_text: str, gold: str) -> float:
    pred = last_boxed(answer_text) or extract_final_answer(answer_text)
    return 1.0 if math_answers_equal(pred, gold) else 0.0


def correctness_reward(prompts, completions, task, answer=None, tests=None, test_imports=None,
                       code_workers: int = 4, **kw) -> list[float | None]:
    n = len(completions)
    answer = answer or [None] * n
    tests = tests or [[]] * n
    test_imports = test_imports or [[]] * n
    answers = [parse(t).answer for t in _turns(prompts, completions)]
    out: list[float | None] = [None] * n
    code_jobs = []
    for i in range(n):
        if task[i] == "math":
            out[i] = math_correct(answers[i], answer[i])
        elif task[i] == "code":
            code_jobs.append(i)
    if code_jobs:
        with ThreadPoolExecutor(max_workers=code_workers) as ex:
            scores = ex.map(lambda i: fraction_passed(extract_code(answers[i]), list(tests[i] or []),
                                                      list(test_imports[i] or [])), code_jobs)
            for i, s in zip(code_jobs, scores):
                out[i] = s
    return out


REWARD_FUNCS = [correctness_reward, format_reward]  # order matches cfg.grpo.reward_weights
