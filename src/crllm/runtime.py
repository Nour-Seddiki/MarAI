"""Process-level guards for long runs on the Windows laptop. No-ops elsewhere.

Call `apply_runtime(cfg.runtime)` at the top of every training / sampling entry point,
before the first CUDA allocation:
  * keep_awake            - Modern Standby suspends runs ~30 min after the last real input.
  * pin_p_cores           - background-launched runs land on E-cores and go ~5x slower.
  * cuda_memory_fraction  - on WDDM the CUDA cache spills into shared RAM instead of OOMing.
"""
from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000

IS_WINDOWS = sys.platform == "win32"


def _kernel32():
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    k32.SetThreadExecutionState.restype = wintypes.DWORD
    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    k32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
    ]
    k32.GetProcessAffinityMask.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    return k32


def keep_awake() -> bool:
    """Hold the system and display awake until this process exits. Changes no power settings."""
    if not IS_WINDOWS:
        return False
    k32 = _kernel32()
    prev = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    if prev == 0:
        log.warning("SetThreadExecutionState failed; the laptop may still go to standby")
        return False
    log.info("keep-awake on (released when the process exits)")
    return True


def pin_to_cores(mask: int, above_normal: bool = False) -> int | None:
    """Restrict this process to the logical CPUs in `mask`. Returns the mask now in effect."""
    if not IS_WINDOWS:
        return None
    import ctypes

    k32 = _kernel32()
    proc = k32.GetCurrentProcess()
    if not k32.SetProcessAffinityMask(proc, mask):
        log.warning("SetProcessAffinityMask(0x%X) failed, error %d", mask, ctypes.get_last_error())
        return None
    if above_normal and not k32.SetPriorityClass(proc, ABOVE_NORMAL_PRIORITY_CLASS):
        log.warning("SetPriorityClass failed, error %d", ctypes.get_last_error())
    cur, sys_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(proc, ctypes.byref(cur), ctypes.byref(sys_mask)):
        return None
    if cur.value != mask:
        log.warning("affinity reads back as 0x%X, wanted 0x%X", cur.value, mask)
    else:
        log.info("pinned to CPUs mask 0x%X%s", cur.value, " at AboveNormal" if above_normal else "")
    return cur.value


def cap_cuda_memory(fraction: float) -> bool:
    """Cap the CUDA caching allocator so it releases cache instead of spilling into shared RAM."""
    import torch

    if not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(fraction)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    log.info("CUDA memory capped at %.0f%% (%.1f of %.1f GiB)", fraction * 100, fraction * total, total)
    return True


def apply_runtime(rt) -> dict[str, object]:
    """Apply a RuntimeCfg. Returns what was actually applied, for run.json / logs."""
    applied: dict[str, object] = {}
    if rt.keep_awake:
        applied["keep_awake"] = keep_awake()
    if rt.pin_p_cores:
        applied["affinity_mask"] = pin_to_cores(rt.p_core_mask, rt.above_normal_priority)
    if rt.cuda_memory_fraction is not None:
        applied["cuda_memory_fraction"] = rt.cuda_memory_fraction if cap_cuda_memory(rt.cuda_memory_fraction) else None
    return applied
