import os
import subprocess
import sys
import pytest
from pathlib import Path

BCF = Path(
    os.environ.get("REPRO_BCF", "")
)  # set to $CLAUDE_JOB_DIR/tmp/repro/cohort.bcf


def _repro(threads, out, timeout):
    return subprocess.run(
        [
            sys.executable,
            "scripts/from_vcf_livelock/repro.py",
            "--bcf",
            str(BCF),
            "--out",
            str(out),
            "--threads",
            str(threads),
            "--timeout",
            str(timeout),
        ]
    ).returncode


@pytest.mark.skipif(not BCF.exists(), reason="set REPRO_BCF to a generated cohort.bcf")
def test_single_concurrent_completes(tmp_path):
    assert _repro(6, tmp_path / "ctrl", 600) == 0


@pytest.mark.skipif(not BCF.exists(), reason="set REPRO_BCF to a generated cohort.bcf")
@pytest.mark.xfail(
    reason="#135 livelock: >=2 concurrent chromosomes never commit a chunk", strict=True
)
def test_multi_concurrent_completes(tmp_path):
    # Flips from xfail to pass once the livelock is fixed (Phase 2).
    assert _repro(32, tmp_path / "multi", 300) == 0
