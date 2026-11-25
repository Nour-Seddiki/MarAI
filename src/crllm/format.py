"""Prompt template and <think> wrap / parse. The single source of truth for text format.

    prompt      = build_prompt(...)  -> ChatML string ending in "<|im_start|>assistant\\n"
    completion  = wrap(reasoning, answer) -> "<think>{reasoning}</think>\\n{answer}<|im_end|>"
    parse(text) -> Parsed(well_formed, reasoning, answer, error)

Every stage (SFT data, sampling, rewards, eval) goes through these functions so the
format cannot drift between stages. Gold answers for verifiable tasks are stored
already passed through `extract_final_answer`, so rewards can compare with `==`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Iterable, Mapping

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOT = "<|endoftext|>"

DEFAULT_SYSTEM = (
    "You are a helpful, honest and harmless assistant. "
    "First think through the request inside <think>...</think>, then give your answer."
)

_CHAT_TOKENS = (IM_START, IM_END, EOT)
_ROLES = ("system", "user", "assistant")


# ---------------------------------------------------------------- prompts


def _check_text(text: str, what: str, *, allow_think: bool = True) -> None:
    if not isinstance(text, str):
        raise TypeError(f"{what} must be a string, got {type(text).__name__}")
    for tok in _CHAT_TOKENS:
        if tok in text:
            raise ValueError(f"{what} contains the chat token {tok!r}")
    if not allow_think and (THINK_OPEN in text or THINK_CLOSE in text):
        raise ValueError(f"{what} contains a <think> tag")


def _turn(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def build_prompt(
    user: str | Iterable[Mapping[str, str]],
    system: str | None = DEFAULT_SYSTEM,
) -> str:
    """ChatML prompt up to and including the assistant header.

    `user` is either the user message, or a list of {"role", "content"} messages
    ending with a user turn (earlier assistant turns hold the answer only, no <think>).
    `system=None` omits the system turn.
    """
    if isinstance(user, str):
        messages = [{"role": "user", "content": user}]
    else:
        messages = [dict(m) for m in user]
    if not messages:
        raise ValueError("build_prompt needs at least one message")
    if messages[-1].get("role") != "user":
        raise ValueError("the last message must be a user turn")

    parts = []
    if system is not None and messages[0].get("role") != "system":
        _check_text(system, "system prompt")
        parts.append(_turn("system", system))
    for i, m in enumerate(messages):
        role, content = m.get("role"), m.get("content")
        if role not in _ROLES:
            raise ValueError(f"message {i}: unknown role {role!r}")
        if role == "system" and i != 0:
            raise ValueError("a system message may only come first")
        _check_text(content, f"message {i} ({role})", allow_think=role != "assistant")
        if role == "assistant" and (THINK_OPEN in content or THINK_CLOSE in content):
            raise ValueError(f"message {i}: earlier assistant turns must not contain <think>")
        parts.append(_turn(role, content))
    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def prompt_key(text: str) -> str:
    """Stable hash of a user message (case/whitespace-insensitive), for eval de-duplication."""
    norm = " ".join(text.casefold().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- completions


def wrap(reasoning: str, answer: str, *, end: bool = True) -> str:
    """The only way a training completion is written."""
    reasoning, answer = reasoning.strip(), answer.strip()
    if not reasoning:
        raise ValueError("reasoning is empty")
    if not answer:
        raise ValueError("answer is empty")
    _check_text(reasoning, "reasoning", allow_think=False)
    _check_text(answer, "answer", allow_think=False)
    return f"{THINK_OPEN}{reasoning}{THINK_CLOSE}\n{answer}" + (IM_END if end else "")


@dataclass(frozen=True)
class Parsed:
    well_formed: bool
    reasoning: str
    answer: str          # best effort even when malformed, so correctness can still be scored
    error: str | None = None


def _strip_eot(s: str) -> str:
    s = s.rstrip()
    while s.endswith(EOT):
        s = s[: -len(EOT)].rstrip()
    return s


def parse(text: str) -> Parsed:
    """Well formed = exactly one <think>...</think> at the start, non-empty reasoning,
    a non-empty answer after it, and nothing but padding after <|im_end|>.

    Accepts text with or without the trailing <|im_end|> (decoders often skip it).
    """
    if not isinstance(text, str):
        raise TypeError(f"parse expects a string, got {type(text).__name__}")

    body, sep, tail = text.partition(IM_END)
    body = _strip_eot(body)
    error = None
    if sep and _strip_eot(tail).strip():
        error = "text after <|im_end|>"
    if IM_START in body:
        error = error or "starts a new chat turn"
        body = body.split(IM_START, 1)[0]

    n_open, n_close = body.count(THINK_OPEN), body.count(THINK_CLOSE)
    i_open, i_close = body.find(THINK_OPEN), body.find(THINK_CLOSE)

    # best-effort split, used for the answer even when the format is wrong
    if i_close >= 0:
        last_close = body.rfind(THINK_CLOSE)
        answer = body[last_close + len(THINK_CLOSE):].strip()
        start = i_open + len(THINK_OPEN) if 0 <= i_open < i_close else 0
        reasoning = body[start:i_close].replace(THINK_OPEN, "").strip()
    elif i_open >= 0:
        reasoning, answer = body[i_open + len(THINK_OPEN):].strip(), ""
    else:
        reasoning, answer = "", body.strip()

    if error is None:
        if n_open == 0 and n_close == 0:
            error = "no <think> block"
        elif n_open != 1 or n_close != 1:
            error = f"expected one <think> and one </think>, found {n_open} and {n_close}"
        elif i_close < i_open:
            error = "</think> before <think>"
        elif body[:i_open].strip():
            error = "text before <think>"
        elif not reasoning:
            error = "empty reasoning"
        elif not answer:
            error = "empty answer"
    return Parsed(well_formed=error is None, reasoning=reasoning, answer=answer, error=error)


def rewrap(text: str) -> str | None:
    """Re-emit a <think>-style trace (any whitespace) in canonical form, or None if malformed."""
    p = parse(text)
    if not p.well_formed:
        return None
    try:
        return wrap(p.reasoning, p.answer)
    except ValueError:
        return None


# ---------------------------------------------------------------- final answers

_NUM = re.compile(
    r"(?<![\d/])-?\d+/\d+(?![\d/])"       # fraction 3/4 (not part of a date like 12/25/2020)
    r"|-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"    # 1,234.5
    r"|-?\d+(?:\.\d+)?"                   # 1234.5
    r"|-?\.\d+"                           # .5
)
_HASHES = re.compile(r"####\s*(.+)")
_ANSWER_IS = re.compile(
    r"\b(?:final answer|the answer|answer)\s*(?:is\b|:|=)\s*[:=]?\s*(.+)", re.IGNORECASE
)
_FRAC = re.compile(r"\\[dt]?frac\{\s*(-?\d+)\s*\}\{\s*(-?\d+)\s*\}")


def normalize_number(s: str) -> str | None:
    """'1,234.50' -> '1234.5', '18.00' -> '18', '6/4' -> '3/2', '8/4' -> '2'; None if not a number."""
    s = s.strip().replace(",", "")
    try:
        if "/" in s:
            f = Fraction(s)
            return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        d = Decimal(s)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    if not d.is_finite():
        return None
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def _numbers(text: str) -> list[str]:
    text = _FRAC.sub(r"\1/\2", text.replace("$", ""))
    return [n for n in (normalize_number(m) for m in _NUM.findall(text)) if n is not None]


def _last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
        i += 1
    return None  # unbalanced


def extract_final_answer(answer: str | None) -> str | None:
    """Canonical final answer from free text: \\boxed{} > '#### x' > 'answer is x' > last number.

    Numbers come back normalized (see normalize_number); a marked answer with no number
    (e.g. \\boxed{yes}) comes back as stripped text.
    """
    if not answer or not answer.strip():
        return None
    candidate = _last_boxed(answer)
    if candidate is None:
        hashes = _HASHES.findall(answer)
        if hashes:
            candidate = hashes[-1]
    if candidate is None:
        marked = _ANSWER_IS.findall(answer)
        if marked:
            candidate = marked[-1]
    if candidate is not None:
        nums = _numbers(candidate)
        if nums:
            return nums[0]
        text = candidate.strip().rstrip(".").strip()
        return text or None
    nums = _numbers(answer)
    return nums[-1] if nums else None
