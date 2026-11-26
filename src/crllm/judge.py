"""One judge client for any OpenAI-compatible chat endpoint (Ollama, a vLLM server, a hosted API).

Eval verdicts (this module, M2):  score() 1-5 safety, refusal() COMPLY/REFUSE, uncertainty() bool.
Data building (M4):               critique(), revise(), rank() reuse the same transport.

All prompts come from configs/constitution.yaml. Every verdict returns the parsed value
or None when the judge's reply can't be parsed, plus the raw text for the samples file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import httpx

from crllm.config import Constitution, JudgeCfg, load_constitution

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    value: Any          # parsed result, or None if the reply could not be parsed
    raw: str            # judge's reply text (or the error message)
    ok: bool


_JSON = re.compile(r"\{.*?\}", re.S)


def parse_json(text: str) -> dict | None:
    """First JSON object in `text` (judges wrap JSON in prose or ``` fences)."""
    if not text:
        return None
    for m in _JSON.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_score(text: str) -> int | None:
    obj = parse_json(text)
    if obj is not None and "score" in obj:
        try:
            v = int(round(float(obj["score"])))
        except (TypeError, ValueError):
            return None
        return v if 1 <= v <= 5 else None
    return None


def _parse_label(text: str) -> str | None:
    obj = parse_json(text)
    v = str(obj.get("label", "")).strip().upper() if obj else ""
    return v if v in ("COMPLY", "REFUSE") else None


def _parse_flag(text: str) -> bool | None:
    obj = parse_json(text)
    if not obj or "flags_uncertainty" not in obj:
        return None
    v = obj["flags_uncertainty"]
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


class Judge:
    def __init__(self, jcfg: JudgeCfg, constitution: Constitution | None = None,
                 transport: Callable[[list[dict], int], str] | None = None):
        self.cfg = jcfg
        self.constitution = constitution or load_constitution()
        self._transport = transport or self._http_chat
        self._client: httpx.Client | None = None

    # ------------------------------------------------------------ transport

    def _http_chat(self, messages: list[dict], max_tokens: int) -> str:
        if self._client is None:
            key = os.environ.get(self.cfg.api_key_env, "not-needed")
            self._client = httpx.Client(
                base_url=self.cfg.base_url.rstrip("/"), timeout=self.cfg.timeout_s,
                headers={"Authorization": f"Bearer {key}"},
            )
        body: dict[str, Any] = {
            "model": self.cfg.model, "messages": messages,
            "temperature": self.cfg.temperature, "max_tokens": max_tokens,
        }
        if self.cfg.reasoning_effort is not None:
            body["reasoning_effort"] = self.cfg.reasoning_effort
        r = self._client.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

    def chat(self, prompt: str, max_tokens: int | None = None) -> str:
        last: Exception | None = None
        for attempt in range(1, self.cfg.retries + 1):
            try:
                return self._transport([{"role": "user", "content": prompt}], max_tokens or self.cfg.max_tokens)
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last = e
                log.warning("judge call failed (attempt %d/%d): %s", attempt, self.cfg.retries, e)
                time.sleep(2 * attempt)
        raise RuntimeError(f"judge unreachable at {self.cfg.base_url}: {last}")

    def list_models_error(self) -> str | None:
        """Cheap availability check that does not load the model (GET /models). None = available."""
        if self._transport != self._http_chat:
            return None  # injected transport (tests)
        try:
            key = os.environ.get(self.cfg.api_key_env, "not-needed")
            r = httpx.get(self.cfg.base_url.rstrip("/") + "/models", timeout=10,
                          headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        if ids and self.cfg.model not in ids:
            return f"model {self.cfg.model!r} not served at {self.cfg.base_url} (has: {', '.join(ids[:8])})"
        return None

    def ping(self) -> str | None:
        """None if the judge answers, else the error text."""
        try:
            self._transport([{"role": "user", "content": 'Reply with JSON only: {"ok": true}'}], 16)
            return None
        except Exception as e:  # noqa: BLE001 - any failure means "unavailable"
            return f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------ verdicts

    def _clip(self, text: str) -> str:
        n = self.cfg.max_response_chars
        return text if len(text) <= n else text[:n] + "\n[... truncated]"

    def _verdict(self, template: str, parser: Callable[[str], Any], max_tokens: int, **fields) -> Verdict:
        prompt = self.constitution.render(template, **fields)
        try:
            raw = self.chat(prompt, max_tokens)
        except RuntimeError as e:
            return Verdict(None, str(e), False)
        value = parser(raw)
        return Verdict(value, raw, value is not None)

    def score(self, prompt: str, response: str) -> Verdict:
        return self._verdict("score", _parse_score, 32, constitution=self.constitution.as_text(),
                             prompt=prompt, response=self._clip(response))

    def refusal(self, prompt: str, response: str) -> Verdict:
        return self._verdict("refusal", _parse_label, 32, prompt=prompt, response=self._clip(response))

    def uncertainty(self, prompt: str, response: str) -> Verdict:
        return self._verdict("uncertainty", _parse_flag, 32, prompt=prompt, response=self._clip(response))

    def map(self, fn: Callable[..., Verdict], items: Sequence[tuple[str, str]], desc: str = "") -> list[Verdict]:
        """Run a verdict over (prompt, response) pairs with cfg.concurrency workers, in order."""
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as ex:
            out = list(ex.map(lambda pr: fn(*pr), items))
        bad = sum(not v.ok for v in out)
        log.info("judged %d %s in %.0fs (%d unparsed)", len(out), desc, time.time() - t0, bad)
        return out
