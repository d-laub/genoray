from __future__ import annotations

import subprocess
import sys

from genoray import Tuning
from genoray._cli._tuning_flags import VcfTuningFlags


def test_flags_convert_to_a_tuning():
    flags = VcfTuningFlags(reader_workers=20, overshard=40, sample_interval=0)
    assert flags.to_tuning() == Tuning(
        reader_workers=20, overshard=40, sample_interval=0
    )


def test_unset_flags_stay_none():
    assert VcfTuningFlags().to_tuning() == Tuning()


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    # `-m genoray._cli`, not `-m genoray`: the package has no `__main__.py`, and
    # the console script points at `genoray._cli.__main__:app`. Getting this
    # wrong is silent -- the subprocess dies with an empty stdout, so a
    # `flag not in out` assertion passes for the wrong reason. Hence the
    # returncode check below: every flag assertion in this file reads stdout,
    # and an empty stdout must fail loudly rather than agree with whatever was
    # asked of it.
    proc = subprocess.run(
        [sys.executable, "-m", "genoray._cli", *args],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (args, proc.returncode, proc.stderr)
    return proc


def test_vcf_only_flags_are_absent_from_the_pgen_command():
    # Rejected by argument parsing, not at runtime: a flag the backend cannot
    # use should not be spellable.
    proc = _cli("write", "pgen", "--help")
    assert "--overshard" not in proc.stdout
    assert "--reader-workers" not in proc.stdout


def test_shared_flags_are_present_on_every_write_command():
    for sub in ("vcf", "pgen"):
        out = _cli("write", sub, "--help").stdout
        for flag in (
            "--dense-cap",
            "--merge-threads",
            "--sample-interval",
            "--log-filter",
        ):
            assert flag in out, (sub, flag)


def test_vcf_command_has_the_sharded_flags():
    out = _cli("write", "vcf", "--help").stdout
    assert "--reader-workers" in out
    assert "--overshard" in out
    assert "--concurrent-chroms" in out
