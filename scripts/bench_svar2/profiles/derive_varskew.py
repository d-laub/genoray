"""Derive a variant-skewed vcfixture profile from a fitted one.

`vcfixture bulk`'s `Size::Records` splits records across contigs
proportional to each contig's `density_per_kb` (`distribute_by_density` in
vcfixture's src/bulk/mod.rs). For the human profiles those densities are
near-uniform -- ~21-26 variants/kb across the autosomes -- so a 22-contig
corpus comes out at max/min 1.38x where the real cohort is 6.07x. An even
corpus makes contig-level scheduling unmeasurable: every contig finishes at
once, so longest-first dispatch and makespan tails look like no-ops.

`density_per_kb` is read in EXACTLY ONE place in vcfixture -- that split --
and nothing else consumes it (positions, gap distribution, SFS, class mix
all ignore it). So overwriting it with `n_variants` changes the
apportionment and nothing else about the generated data.

This is a workaround for vcfixture-rs#15 (no way to express per-contig
counts through `Size`). If that lands, prefer it: this transform depends on
`density_per_kb` having a single consumer, and would break silently if that
ever stops holding.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

DERIVED_NAME = "germline-1kgp-varskew"


def derive_varskew(profile: dict) -> dict:
    """Return a copy of `profile` whose per-contig `density_per_kb` is the
    contig's fitted `n_variants`.

    Only the ratios between contigs matter to `distribute_by_density`, so
    the absolute units of the rewritten field are irrelevant -- it is a
    split weight after this transform, not a density.
    """
    out = copy.deepcopy(profile)
    for contig in out["fitted"]["contigs"]:
        contig["density_per_kb"] = float(contig["n_variants"])
    out["name"] = DERIVED_NAME
    out.setdefault("provenance", {})
    out["provenance"]["derived_from"] = profile.get("name", "unknown")
    out["provenance"]["derivation"] = (
        "density_per_kb overwritten with n_variants so vcfixture's "
        "distribute_by_density apportions records by true per-contig variant "
        "counts instead of near-uniform densities (vcfixture-rs#15)"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "source",
        type=Path,
        help="path to a fitted vcfixture profile JSON (e.g. "
        "vcfixture-rs/profiles/germline-1kgp.json)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).parent / f"{DERIVED_NAME}.json",
    )
    args = ap.parse_args()
    derived = derive_varskew(json.loads(args.source.read_text()))
    args.out.write_text(json.dumps(derived, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
