"""Build a PGEN matched to a BCF for apples-to-apples SVAR2 conversion benchmarks."""

from __future__ import annotations

import subprocess
from pathlib import Path


def plink2_cmd(bcf: Path, out_prefix: Path) -> list[str]:
    # multiallelics kept split as-is; SVAR2 atomizes downstream regardless.
    return [
        "plink2",
        "--bcf",
        str(bcf),
        "--make-pgen",
        "--out",
        str(out_prefix),
    ]


def make_pgen(bcf: Path, out_prefix: Path) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(plink2_cmd(bcf, out_prefix), check=True)
    return out_prefix.with_suffix(".pgen")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("bcf", type=Path)
    ap.add_argument("out_prefix", type=Path)
    args = ap.parse_args()
    print(make_pgen(args.bcf, args.out_prefix))
