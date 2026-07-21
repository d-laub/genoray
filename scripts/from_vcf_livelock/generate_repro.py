#!/usr/bin/env python
"""Generate a synthetic multi-contig BCF carrying a ``VAF Number=A Float``
FORMAT field, for reproducing the ``SparseVar2.from_vcf`` concurrent-chromosome
livelock (genoray #135).

Wraps the mechanism documented in ``scripts/from_vcf_livelock/README.md``:

1. ``vcfixture bulk --payload gt-vaf`` generates a BCF whose ``VAF`` FORMAT
   field is declared ``Number=1`` (native limitation of the ``bulk``
   generator — see the README for why no CLI flag can produce ``Number=A``
   directly).
2. Because the bundled ``vcfixture`` profiles are biallelic-only
   (``multiallelic_rate=0.0``), every record has exactly one ALT allele, so
   the on-disk ``VAF`` values (one float per sample) are already valid for a
   ``Number=A`` field. Only the header declaration needs to change — a plain
   ``bcftools reheader`` (no data rewrite) is exact for these profiles. See
   the README's "Caveat for Task 2 / future profiles" section: this reheader
   shortcut would NOT be valid if multiallelic records were ever introduced.

Usage:
    python scripts/from_vcf_livelock/generate_repro.py \\
        --out <dir> --samples 40 --contigs chr1,chr2,chr3,chr4,chr5,chr6 \\
        --target-size 8MB --seed 0

Writes ``<dir>/cohort.bcf`` and ``<dir>/cohort.bcf.csi`` and prints the final
BCF path.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

_VCFIXTURE_FALLBACK = Path("/tmp/vcfixture-cli/bin/vcfixture")

_VAF_NUMBER_1 = (
    '##FORMAT=<ID=VAF,Number=1,Type=Float,Description="Variant allele frequency">'
)
_VAF_NUMBER_A = (
    '##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele frequency">'
)


def resolve_vcfixture() -> Path:
    """Resolve the ``vcfixture`` binary from PATH, falling back to the
    Task-1 install location. Raises with an actionable message if neither
    exists.
    """
    on_path = shutil.which("vcfixture")
    if on_path is not None:
        return Path(on_path)
    if _VCFIXTURE_FALLBACK.exists():
        return _VCFIXTURE_FALLBACK
    raise FileNotFoundError(
        "vcfixture binary not found on PATH or at "
        f"{_VCFIXTURE_FALLBACK}. Install it with:\n"
        "  CARGO_TARGET_DIR=/tmp/genoray-target-svar2 cargo install vcfixture "
        "--version 0.3.0 --features cli --root /tmp/vcfixture-cli --locked"
    )


def generate(
    out_dir: Path,
    samples: int,
    contigs: str,
    target_size: str,
    seed: int,
) -> Path:
    """Generate the repro cohort BCF (+CSI) under ``out_dir`` and return the
    path to the final ``cohort.bcf``.
    """
    vcfixture = resolve_vcfixture()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_bcf = out_dir / "_raw.bcf"
    header_txt = out_dir / "_header.txt"
    final_bcf = out_dir / "cohort.bcf"

    # 1. Generate with the gt-vaf payload (VAF present, but Number=1).
    subprocess.run(
        [
            str(vcfixture),
            "bulk",
            "--samples",
            str(samples),
            "--contigs",
            contigs,
            "--target-size",
            target_size,
            "--seed",
            str(seed),
            "--payload",
            "gt-vaf",
            "-o",
            str(raw_bcf),
        ],
        check=True,
    )

    # 2. Dump the header and patch the VAF FORMAT line's Number from 1 to A.
    header = subprocess.run(
        ["bcftools", "view", "-h", str(raw_bcf)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if _VAF_NUMBER_1 in header:
        header = header.replace(_VAF_NUMBER_1, _VAF_NUMBER_A)
    elif _VAF_NUMBER_A not in header:
        raise RuntimeError(
            "vcfixture bulk output does not contain the expected VAF "
            "FORMAT header line (neither Number=1 nor Number=A found); "
            "the vcfixture bulk output format may have changed. See "
            "scripts/from_vcf_livelock/README.md for the reference mechanism."
        )
    header_txt.write_text(header)

    # 3. Swap the header in place (record bytes are untouched — bcftools
    #    reheader does not re-encode data, so this is cheap and exact for
    #    these biallelic-only profiles).
    subprocess.run(
        [
            "bcftools",
            "reheader",
            "-h",
            str(header_txt),
            "-o",
            str(final_bcf),
            str(raw_bcf),
        ],
        check=True,
    )

    # 4. Ensure a CSI index is written for the final BCF.
    subprocess.run(["bcftools", "index", "-f", str(final_bcf)], check=True)

    # Clean up intermediates.
    raw_bcf.unlink(missing_ok=True)
    raw_csi = raw_bcf.with_suffix(raw_bcf.suffix + ".csi")
    raw_csi.unlink(missing_ok=True)
    header_txt.unlink(missing_ok=True)

    return final_bcf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic multi-contig BCF with a VAF Number=A "
            "Float FORMAT field, for the genoray #135 from_vcf livelock "
            "repro."
        )
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output directory; writes <out>/cohort.bcf + .csi",
    )
    parser.add_argument(
        "--samples",
        required=True,
        type=int,
        help="number of samples",
    )
    parser.add_argument(
        "--contigs",
        required=True,
        help="comma-separated contig names, e.g. chr1,chr2,chr3",
    )
    parser.add_argument(
        "--target-size",
        required=True,
        help="approximate output size, e.g. 8MB, 500MB",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="RNG seed",
    )
    args = parser.parse_args()

    final_bcf = generate(
        out_dir=args.out,
        samples=args.samples,
        contigs=args.contigs,
        target_size=args.target_size,
        seed=args.seed,
    )
    print(str(final_bcf))


if __name__ == "__main__":
    main()
