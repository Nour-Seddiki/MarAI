"""Run model-written Python against asserts, for MBPP pass@1.

Each program runs as `python -I <file>` in its own empty temp directory with a minimal
environment and a timeout. That limits accidents (hangs, stray files, imports from the
repo) but it is NOT a security sandbox: the code runs as your user with network access.
Only run it on outputs of models you are training yourself.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecResult:
    passed: bool
    status: str      # "passed" | "failed" | "timeout" | "no_code"
    detail: str      # last stderr line, for the samples file


def _minimal_env() -> dict[str, str]:
    keep = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "LANG", "LC_ALL")
    env = {k: v for k, v in os.environ.items() if k.upper() in keep}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_tests(code: str | None, tests: list[str], imports: list[str] | None = None,
              timeout_s: float = 10.0) -> ExecResult:
    if not code:
        return ExecResult(False, "no_code", "")
    program = "\n".join([*(imports or []), code, "", *tests, ""])
    with tempfile.TemporaryDirectory(prefix="crllm_exec_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, env=_minimal_env(), capture_output=True,
                               text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout", f"> {timeout_s:.0f}s")
    if p.returncode == 0:
        return ExecResult(True, "passed", "")
    last = (p.stderr.strip().splitlines() or [""])[-1]
    return ExecResult(False, "failed", last[:200])


def run_many(jobs: list[tuple[str | None, list[str], list[str]]], workers: int = 4,
             timeout_s: float = 10.0) -> list[ExecResult]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda j: run_tests(j[0], j[1], j[2], timeout_s), jobs))


_COUNTER = """
import json as _json
_tests = _json.loads({tests!r})
_passed = 0
for _src in _tests:
    try:
        exec(_src, globals())
        _passed += 1
    except BaseException:
        pass
print("__CRLLM_PASSED__", _passed, len(_tests))
"""


def fraction_passed(code: str | None, tests: list[str], imports: list[str] | None = None,
                    timeout_s: float = 5.0) -> float:
    """Share of asserts that pass (each runs independently), for partial-credit rewards. 0.0 on any failure."""
    if not code or not tests:
        return 0.0
    import json

    program = "\n".join([*(imports or []), code, _COUNTER.format(tests=json.dumps(tests))])
    with tempfile.TemporaryDirectory(prefix="crllm_exec_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, env=_minimal_env(), capture_output=True,
                               text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return 0.0
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("__CRLLM_PASSED__"):
            _, k, n = line.split()
            return int(k) / max(1, int(n))
    return 0.0  # the solution itself crashed before the tests ran
