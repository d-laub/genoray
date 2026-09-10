"""The GENORAY_* configuration channel is gone; keep it gone.

Every knob is a Python or CLI argument. An environment read is invisible to the
`pipeline config` banner, cannot be validated, and silently wins over an
explicit argument -- the exact combination that cost a downstream user a day of
misdiagnosis on genoray 4.0.1 (PR #174).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# `svar1_reader.rs` uses `std::env::temp_dir()` for scratch paths, which is not
# configuration. Nothing else in src/ may read the environment.
_ALLOWED = {"src/svar1_reader.rs"}

_ENV_READ = r"std::env::var|env::var_os"


def _rg(pattern: str, *paths: str) -> list[str]:
    proc = subprocess.run(
        ["rg", "--no-heading", "--line-number", pattern, *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):  # 1 == no matches
        pytest.fail(f"rg failed: {proc.stderr}")
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def test_no_environment_reads_in_rust_sources():
    offenders = [
        line
        for line in _rg(_ENV_READ, "src")
        if not any(line.startswith(allowed) for allowed in _ALLOWED)
    ]
    assert offenders == [], (
        "environment reads found in src/; every knob must be an explicit "
        "argument:\n" + "\n".join(offenders)
    )


def test_no_genoray_env_var_names_anywhere_in_the_package():
    offenders = _rg(r"GENORAY_[A-Z_]+", "src", "python")
    assert offenders == [], "GENORAY_* names found in shipped code:\n" + "\n".join(
        offenders
    )
