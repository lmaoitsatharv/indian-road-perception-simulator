"""Resolve local runtime paths without embedding a developer's installation."""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess


def linux_path(path):
    """Use a native Linux path or the standard WSL mount for a drive path."""
    value = os.fspath(path)
    if re.match(r"^[A-Za-z]:[\\/]", value):
        windows = PureWindowsPath(value)
        return "/mnt/" + windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
    if value.startswith("\\\\") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Use an absolute local drive path or a native Linux path")
    return Path(path).expanduser().resolve().as_posix()


def wsl_distribution(explicit=None):
    return explicit or os.environ.get("PROJECT_WSL_DISTRIBUTION") or "Ubuntu-24.04"


def wsl_python(distribution=None, explicit=None):
    """Resolve the target WSL user's home, or use an explicit absolute interpreter."""
    configured = explicit or os.environ.get("PROJECT_WSL_PYTHON")
    if configured:
        if not configured.startswith("/") or "\n" in configured or "\r" in configured:
            raise ValueError("PROJECT_WSL_PYTHON must be an absolute Linux interpreter path")
        return configured
    selected = wsl_distribution(distribution)
    result = subprocess.run(
        ["wsl.exe", "-d", selected, "--exec", "/usr/bin/printenv", "HOME"],
        check=True, capture_output=True, text=True,
    )
    home = result.stdout.strip()
    if not home.startswith("/") or "\n" in home or "\r" in home:
        raise ValueError("Could not resolve the selected WSL user's HOME")
    return home.rstrip("/") + "/.venvs/idd-tf220/bin/python"


def matlab_executable(explicit=None):
    """Find MATLAB through a caller setting, environment variable or PATH."""
    configured = explicit or os.environ.get("MATLAB_EXECUTABLE")
    candidate = configured or shutil.which("matlab") or shutil.which("matlab.exe")
    if not candidate:
        raise FileNotFoundError("Set MATLAB_EXECUTABLE, pass --matlab, or add MATLAB to PATH")
    located = shutil.which(candidate) or candidate
    path = Path(located).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Configured MATLAB executable was not found")
    return path
