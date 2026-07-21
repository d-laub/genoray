"""Executable regression for the #135 `from_vcf` concurrent-chromosome livelock.

The two regimes need DIFFERENT-SCALE cohorts (see
`docs/superpowers/specs/2026-07-20-svar2-from-vcf-livelock-diagnosis.md`), so
each test is gated on its own env var:

- ``REPRO_CONTROL_BCF`` -> a cohort small enough to COMPLETE single-lane in
  600 s (e.g. the synthetic ``scripts/from_vcf_livelock/generate_repro.py``
  cohort). Used by ``test_single_concurrent_completes``.
- ``REPRO_LIVELOCK_BCF`` -> a REAL-SCALE cohort (thousands of samples, tens of
  GB) that actually trips the livelock at >=2 concurrent chromosomes. The
  synthetic cohort does NOT reproduce it (it completes in every regime), so
  pointing this at the synthetic cohort would make the strict-xfail XPASS and
  fail. Used by ``test_multi_concurrent_completes``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _cohort(env_var):
    # An unset/empty env var must SKIP, not run: Path("") == Path("."), whose
    # .exists() is True, so guarding on the raw Path would run against ".".
    val = os.environ.get(env_var, "")
    return Path(val) if val else None


CONTROL_BCF = _cohort("REPRO_CONTROL_BCF")
LIVELOCK_BCF = _cohort("REPRO_LIVELOCK_BCF")


def _repro(bcf, threads, out, timeout):
    return subprocess.run(
        [
            sys.executable,
            "scripts/from_vcf_livelock/repro.py",
            "--bcf",
            str(bcf),
            "--out",
            str(out),
            "--threads",
            str(threads),
            "--timeout",
            str(timeout),
        ]
    ).returncode


@pytest.mark.skipif(
    CONTROL_BCF is None or not CONTROL_BCF.exists(),
    reason="set REPRO_CONTROL_BCF to a small cohort that completes single-lane",
)
def test_single_concurrent_completes(tmp_path):
    # threads=6 -> 1 concurrent chromosome -> memory bounded -> completes.
    assert _repro(CONTROL_BCF, 6, tmp_path / "ctrl", 600) == 0


@pytest.mark.skipif(
    LIVELOCK_BCF is None or not LIVELOCK_BCF.exists(),
    reason="set REPRO_LIVELOCK_BCF to a real-scale cohort that trips the livelock",
)
@pytest.mark.xfail(
    reason=(
        "#135 livelock: >=2 concurrent chromosomes buffer chunks unbounded and "
        "OOM/hang (never advance past ordinal 0 at scale). Flips to pass once "
        "Phase 2 adds reader-side memory backpressure."
    ),
    strict=True,
)
def test_multi_concurrent_completes(tmp_path):
    # threads=32 -> 5 concurrent chromosomes -> unbounded buffering -> OOM/hang.
    assert _repro(LIVELOCK_BCF, 32, tmp_path / "multi", 300) == 0
