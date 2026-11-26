"""Lock files that clear themselves when the run that made them is gone.

A run that is closed with the window's X, killed, or crashes never reaches its
cleanup, so its lock stays behind and every later start refused to run. Each
lock records the pid that made it; a lock is stale when no process with that
pid is alive, the process is not Python, or it started after the lock was
written (Windows reused the pid). Only stale locks are removed.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys

_SLACK_S = 5.   # the lock is written a moment after its process starts


def _lock_pid(lock: Path) -> int | None:
    try:
        text = lock.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text.isdigit():
        return int(text)
    try:
        data = json.loads(text)
    except ValueError:
        return None
    pid = data.get("pid") if isinstance(data, dict) else None
    return pid if isinstance(pid, int) else None


def _python_started_before(pid: int, written_epoch: float) -> bool:
    """True when pid is a live Python process that started before the lock was written."""
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False   # no such process (access denied would mean another user's process, not ours)
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
            return False
        size = wintypes.DWORD(1024)
        name = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
            return True   # cannot tell: treat as alive
        if not Path(name.value).name.lower().startswith("python"):
            return False
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return True
        created = (times[0].dwHighDateTime << 32 | times[0].dwLowDateTime) / 1e7 - 11644473600
        return created <= written_epoch + _SLACK_S
    finally:
        kernel.CloseHandle(handle)


def holder_alive(lock: Path) -> bool:
    """True when the lock exists and the run that wrote it is still running."""
    try:
        written = lock.stat().st_mtime
    except FileNotFoundError:
        return False
    pid = _lock_pid(lock)
    if pid is None:
        return True   # unreadable lock: leave it for a person to inspect
    return _python_started_before(pid, written)


def clear_if_stale(lock: Path) -> bool:
    """Remove the lock when its run has ended; True when it was removed."""
    if lock.exists() and not holder_alive(lock):
        lock.unlink(missing_ok=True)
        print(f"Removed a leftover {lock.name} from a run that has already ended.", flush=True)
        return True
    return False


def claim(lock: Path, content: bytes, busy_message: str) -> None:
    """Create the lock exclusively, first clearing one left by an ended run."""
    clear_if_stale(lock)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(busy_message) from None
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
