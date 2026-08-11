"""Shared access to the `vcfixture bulk` Rust CLI.

Extracted from `pgen_corpus.py` when `vcf_corpus.py` began needing the same
binary: the VCF corpus is the PGEN pipeline stopped one step early (bulk emits
the variant file; only the PGEN path then runs plink2 over it), so resolution,
profile choice, and argument construction belong in one place.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

PROFILE = Path(__file__).parent / "profiles" / "germline-1kgp-varskew.json"


def resolve_vcfixture() -> Path:
    """Locate the `vcfixture` bulk CLI.

    This is NOT the PyPI `vcfixture` package pinned in pixi.toml -- that ships
    no console script (no entry_points.txt, no bin/vcfixture). The bulk
    generator is a separate Rust binary on its own version line. Shelling out
    to a bare `vcfixture` therefore passes on a dev box that happens to have it
    built and fails in CI with FileNotFoundError, so resolution is explicit and
    the error says how to fix it.
    """
    env = os.environ.get("VCFIXTURE_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"VCFIXTURE_BIN={env} is not a file")
    found = shutil.which("vcfixture")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "vcfixture bulk CLI not found. It is a Rust binary, separate from the "
        "PyPI `vcfixture` package (which has no CLI). Install it with "
        "`cargo install vcfixture --features cli`, or point VCFIXTURE_BIN at "
        "an existing build."
    )


def cli_version() -> str:
    """The bulk CLI's own version, recorded in every manifest.

    v0.5.0 is an explicit BREAKING output change -- "generated output for a
    given seed differs from v0.4.0 ... existing corpora must be regenerated" --
    so a manifest that cannot distinguish the two invites silently pooling
    incompatible corpora into one fit. Cached corpora key on this string.
    """
    out = subprocess.run(
        [str(resolve_vcfixture()), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # `clap` prints "vcfixture 0.5.0"; keep only the version token.
    return out.split()[-1] if out else ""


def bulk(
    *,
    samples: int,
    variants: int,
    contigs: Sequence[str],
    seed: int,
    fmt: str,
    out: Path,
) -> None:
    """Run `vcfixture bulk` for one corpus. `fmt` is "bcf", "vcf-gz" or "vcf"."""
    subprocess.run(
        [
            str(resolve_vcfixture()),
            "bulk",
            "--profile",
            str(PROFILE),
            "--samples",
            str(samples),
            "--contigs",
            ",".join(contigs),
            "--records",
            str(variants),
            "--payload",
            "gt-only",
            "--format",
            fmt,
            "--seed",
            str(seed),
            "-o",
            str(out),
        ],
        check=True,
    )
