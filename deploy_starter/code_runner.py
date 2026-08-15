"""Restricted Python code execution sandbox for the code runner sub-agent.

Pure standard library — no AgentScope dependency — so it can be unit-tested
locally without the Bailian runtime. Runs user code in a fresh ``python -c``
subprocess with a whitelist of safe import roots and a hard timeout.

The course teaches NumPy/Pandas/Matplotlib, whose packages import ``os``/``sys``/
``platform``/``_io`` etc. internally — often from C-extension or lazy-loading
code whose caller frame is ``importlib``, not the library itself. A strict
whitelist would break them. The sandbox therefore:

  1. pre-imports the data libs the code actually uses (before locking imports),
  2. allows every import while a trusted lib is mid-import (the flag), and
  3. allows imports whose caller frame is inside a trusted lib (lazy loading).

The learner's own ``import os`` / ``import subprocess`` is still rejected.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap

DEFAULT_TIMEOUT = 5
MAX_OUTPUT_BYTES = 4096

# The course's data-analysis subject libraries. Importable by the learner, and
# trusted to import their own transitive dependencies (os/sys/...) internally.
TRUSTED_ROOTS = ("numpy", "pandas", "matplotlib")

# Modules a learner may legitimately import for the demo curriculum (Python
# data-analysis basics) plus the subject libraries. Everything else — os, sys,
# subprocess, socket, shutil, pathlib, importlib, ctypes, etc. — is blocked at
# the learner's level (see the import hook in SANDBOX_PREAMBLE).
SAFE_MODULES = {
    # Python standard library
    "math", "random", "collections", "itertools", "functools", "json", "re",
    "statistics", "heapq", "bisect", "string", "datetime", "typing", "copy",
    "decimal", "fractions", "numbers", "operator", "enum", "dataclasses",
    "uuid", "hashlib", "time",
    # Course's subject libraries
    "numpy", "pandas", "matplotlib",
}

# Prepended to the learner's code. {__data_libs__} is the subset of
# TRUSTED_ROOTS the code actually imports, pre-imported before the lock so
# their heavy/lazy internals resolve with the real import machinery.
SANDBOX_PREAMBLE = textwrap.dedent(
    """
    import builtins as _sandbox_builtins
    import sys as _sandbox_sys
    _sandbox_orig_import = _sandbox_builtins.__import__

    _sandbox_libs = {}
    for _lib_name in {__data_libs__}:
        try:
            if _lib_name == "matplotlib":
                _mpl = _sandbox_orig_import("matplotlib")
                _mpl.use("Agg")  # headless backend, no display needed
                _sandbox_orig_import("matplotlib.pyplot")
                _sandbox_orig_import("matplotlib.backends.backend_agg")
                _sandbox_libs["matplotlib"] = _mpl
            else:
                _sandbox_libs[_lib_name] = _sandbox_orig_import(_lib_name)
        except Exception:
            pass  # library not installed — learner gets a normal ImportError

    _sandbox_safe = {__safe__}
    _sandbox_trusted = {__trusted__}
    _sandbox_trusted_import = False

    def _sandbox_import(name, globals=None, locals=None, fromlist=(), level=0):
        global _sandbox_trusted_import
        root = name.split(".")[0]
        # While a trusted lib is mid-import, allow every transitive import
        # (C-extension deps like numpy._core need _io/os/... from importlib).
        if _sandbox_trusted_import:
            return _sandbox_orig_import(name, globals, locals, fromlist, level)
        if root in _sandbox_libs and not fromlist:
            return _sandbox_libs[root]
        if root in _sandbox_safe:
            if root in _sandbox_trusted:
                _sandbox_trusted_import = True
                try:
                    return _sandbox_orig_import(name, globals, locals, fromlist, level)
                finally:
                    _sandbox_trusted_import = False
            return _sandbox_orig_import(name, globals, locals, fromlist, level)
        # Lazy Python-level import from inside a trusted lib (e.g. pandas
        # "from pandas import ArrowStringArray" at DataFrame() time). The
        # learner's own code runs with __name__ == "__main__", never matching.
        try:
            caller = _sandbox_sys._getframe(1)
            caller_name = (
                caller.f_globals.get("__name__", "") if caller.f_globals else ""
            )
        except Exception:
            caller_name = ""
        if caller_name.split(".")[0] in _sandbox_trusted:
            return _sandbox_orig_import(name, globals, locals, fromlist, level)
        raise ImportError("module '%s' is not allowed in the sandbox" % name)

    _sandbox_builtins.__import__ = _sandbox_import
    """
)


def _read_flat_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    result: dict = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if value.lower() == "true":
                    result[key] = True
                elif value.lower() == "false":
                    result[key] = False
                elif value.isdigit():
                    result[key] = int(value)
                else:
                    result[key] = value
    except Exception:
        pass
    return result


_config = _read_flat_config()


def is_enabled() -> bool:
    return bool(_config.get("CODE_RUNNER_ENABLED", True))


def _default_timeout() -> int:
    return int(_config.get("CODE_RUNNER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT))


def _truncate(value: str) -> str:
    if value is None:
        return ""
    if len(value) > MAX_OUTPUT_BYTES:
        return value[:MAX_OUTPUT_BYTES] + "\n... [output truncated]"
    return value


def _detect_data_libs(code: str) -> list[str]:
    """Return the subset of TRUSTED_ROOTS referenced via ``import``/``from``."""
    libs: list[str] = []
    for lib in TRUSTED_ROOTS:
        if re.search(rf"\b(?:import|from)\s+{lib}\b", code):
            libs.append(lib)
    return libs


def _build_preamble(code: str) -> str:
    return (
        SANDBOX_PREAMBLE.replace("{__safe__}", repr(sorted(SAFE_MODULES)))
        .replace("{__trusted__}", repr(TRUSTED_ROOTS))
        .replace("{__data_libs__}", repr(_detect_data_libs(code)))
    )


def run_python(code: str, timeout: int | None = None) -> dict:
    """Execute learner code in the sandbox and return a structured result.

    Returns a dict with ``success``, ``stdout``, ``stderr`` and ``exit_code``.
    Never raises — all failures are captured into the returned dict.
    """
    if not is_enabled():
        return {
            "success": False,
            "stdout": "",
            "stderr": "代码执行已禁用（CODE_RUNNER_ENABLED=false）。",
            "exit_code": -1,
        }

    code = (code or "").strip()
    if not code:
        return {
            "success": False,
            "stdout": "",
            "stderr": "没有可执行的代码。",
            "exit_code": -1,
        }

    if timeout is None:
        timeout = _default_timeout()

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"  # force the sandboxed child to emit UTF-8
    env["MPLBACKEND"] = "Agg"  # headless matplotlib backend (no display needed)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _build_preamble(code) + "\n" + code],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"代码执行超时（>{timeout}s）。",
            "exit_code": -1,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "success": False,
            "stdout": "",
            "stderr": f"沙箱启动失败: {exc}",
            "exit_code": -1,
        }

    return {
        "success": proc.returncode == 0,
        "stdout": _truncate(proc.stdout),
        "stderr": _truncate(proc.stderr),
        "exit_code": proc.returncode,
    }
