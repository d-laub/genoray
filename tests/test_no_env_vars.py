"""The GENORAY_* configuration channel is gone; keep it gone.

Every knob is a Python or CLI argument. An environment read is invisible to the
`pipeline config` banner, cannot be validated, and silently wins over an
explicit argument -- the exact combination that cost a downstream user a day of
misdiagnosis on genoray 4.0.1 (PR #174).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# `svar1_reader.rs` uses `std::env::temp_dir()` for scratch paths, which is not
# configuration. Nothing else in src/ may read the environment.
_ALLOWED = {"src/svar1_reader.rs"}

_ENV_READ = re.compile(r"std::env::var|env::var_os")
_GENORAY_ENV_NAME = re.compile(r"GENORAY_[A-Z_]+")

# Directories that never carry hand-authored source and whose contents (build
# artifacts, caches, VCS metadata) can be arbitrarily large or non-UTF-8.
_SKIP_DIRS = {".git", "__pycache__", "target", ".pytest_cache", ".mypy_cache"}


def _iter_files(*relative_roots: str) -> list[Path]:
    files = []
    for root in relative_roots:
        for path in (REPO / root).rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.relative_to(REPO).parts):
                continue
            files.append(path)
    return sorted(files)


def _grep(pattern: re.Pattern[str], *relative_roots: str) -> list[str]:
    """Reimplementation of `rg -n <pattern> <relative_roots...>`.

    No subprocess, no `rg` dependency: this guard is the single mechanism
    stopping the environment-variable channel from growing back, so it must
    never error for an environmental reason (a runner image without `rg`
    installed would turn a real regression into an opaque `FileNotFoundError`
    instead of a red assertion naming the offending line).
    """
    offenders = []
    for path in _iter_files(*relative_roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not a source file `rg` would match either
        rel = path.relative_to(REPO).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{rel}:{lineno}:{line}")
    return offenders


def test_no_environment_reads_in_rust_sources():
    offenders = [
        line
        for line in _grep(_ENV_READ, "src")
        if not any(line.startswith(allowed) for allowed in _ALLOWED)
    ]
    assert offenders == [], (
        "environment reads found in src/; every knob must be an explicit "
        "argument:\n" + "\n".join(offenders)
    )


def test_no_genoray_env_var_names_anywhere_in_the_package():
    offenders = _grep(_GENORAY_ENV_NAME, "src", "python")
    assert offenders == [], "GENORAY_* names found in shipped code:\n" + "\n".join(
        offenders
    )
