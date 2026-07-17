# scripts/bench_from_vcf_list/generate_cohort.py
"""Generate a synthetic single-sample somatic cohort for from_vcf_list benchmarks.

Each file gets `shared_frac` of its sites drawn from a shared pool (so the k-way
merge actually joins) and the rest private (fan-out). Reproducible by seed.
"""

from __future__ import annotations
import argparse
import random
import subprocess
import zlib
from collections.abc import Sequence
from pathlib import Path

_BASES = "ACGT"

# Private somatic variants are ~never shared, so the union must grow with the cohort.
# Sizing the position space from n_files keeps the collision rate constant as N grows;
# a fixed pool would saturate and hide every O(V x N) cost downstream (issue #120).
_POSITIONS_PER_PRIVATE_VARIANT = 20

_HTYPE = {"VAF": "Float", "DP": "Integer"}  # extend as fields are added


def _split_shared_private(n_variants: int, shared_frac: float) -> tuple[int, int]:
    """Split `n_variants` into (n_shared, n_priv). `n_shared + n_priv == n_variants`
    always holds — the single source of truth for `required_contig_len`,
    `union_positions`, and `generate_cohort` so they never drift apart."""
    n_shared = int(n_variants * shared_frac)
    n_priv = n_variants - n_shared
    return n_shared, n_priv


def required_contig_len(n_files: int, n_variants: int, shared_frac: float) -> int:
    _, n_priv = _split_shared_private(n_variants, shared_frac)
    return max(1_000_000, n_files * n_priv * _POSITIONS_PER_PRIVATE_VARIANT)


def union_positions(
    n_files: int, n_variants: int, shared_frac: float, contig_len: int
) -> float:
    """Expected distinct positions across the cohort — the birthday-problem estimate
    used to assert the cohort is realistic (not saturating)."""
    n_shared, n_priv = _split_shared_private(n_variants, shared_frac)
    span = max(contig_len, 1)
    p = n_priv / span
    return n_shared + span * (1.0 - (1.0 - p) ** n_files)


def _header(
    sample: str, contigs: list[str], contig_len: int, format_fields: list[str]
) -> str:
    lines = ["##fileformat=VCFv4.2"]
    lines += [f"##contig=<ID={c},length={contig_len}>" for c in contigs]
    lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    lines += [
        f'##FORMAT=<ID={f},Number=1,Type={_HTYPE.get(f, "Float")},Description="bench">'
        for f in format_fields
    ]
    lines.append(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}")
    return "\n".join(lines) + "\n"


def _sample_col(rng: random.Random, format_fields: list[str]) -> tuple[str, str]:
    """FORMAT key and value columns. Values are per-sample, which is what makes
    staging cost F x N per variant in the current pipeline."""
    keys = ["GT"] + list(format_fields)
    vals = ["1/1"]
    for f in format_fields:
        vals.append(
            str(rng.randint(1, 300))
            if _HTYPE.get(f) == "Integer"
            else f"{rng.random():.4f}"
        )
    return ":".join(keys), ":".join(vals)


def _alt(rng: random.Random, ref: str, indel: bool) -> tuple[str, str]:
    if indel:  # simple 1bp insertion anchored on ref
        return ref, ref + rng.choice(_BASES)
    alt = rng.choice([b for b in _BASES if b != ref])
    return ref, alt


def _contig_salt(contig: str) -> int:
    """Deterministic per-contig salt (str hashing is randomized per-process by
    default, so we can't rely on hash())."""
    return zlib.crc32(contig.encode())


def generate_cohort(
    out_dir: Path,
    n_files: int,
    n_variants: int,
    *,
    contigs: list[str] | None = None,
    contig_len: int | None = None,
    shared_frac: float = 0.1,
    indel_frac: float = 0.1,
    seed: int = 0,
    format_fields: Sequence[str] = (),
) -> Path:
    """Write `n_files` single-sample VCF/BCFs plus a manifest to `out_dir`.

    `n_variants` is **per contig, per file**: each contig in `contigs` gets its
    own `n_variants`-sized shared/private split, so the total variants per file
    is `n_variants * len(contigs)`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contigs = list(contigs) if contigs else ["1"]
    format_fields = list(format_fields)
    if contig_len is None:
        contig_len = required_contig_len(n_files, n_variants, shared_frac)

    # n_variants is per contig, per file: each contig is an independent coordinate
    # space (real cohorts have roughly as many private sites on chr2 as chr1), and
    # keeping the per-contig formula matches union_positions/required_contig_len,
    # which are single-contig quantities.
    n_shared, n_priv = _split_shared_private(n_variants, shared_frac)

    # Every position's REF/ALT is a deterministic function of (contig, position)
    # alone, so any two files that emit the same position ALWAYS agree -- whether
    # it's a shared-pool site or a coincidental private/private collision. This
    # matters at scale: with many files, private positions collide across files
    # (birthday paradox in a finite contig), and genoray's no_reference k-way merge
    # hard-errors on a cross-file REF disagreement. Deriving from the position (not
    # a per-file rng) mirrors a real cohort called against one reference.
    shared_by_contig: dict[str, set[int]] = {}
    for c in contigs:
        crng = random.Random(seed * 1_000_003 + _contig_salt(c))
        shared_by_contig[c] = (
            set(crng.sample(range(1, contig_len), n_shared)) if n_shared else set()
        )

    variant_cache: dict[tuple[str, int], tuple[str, str]] = {}

    def variant_at(contig: str, pos: int) -> tuple[str, str]:
        v = variant_cache.get((contig, pos))
        if v is None:
            prng = random.Random(
                (seed * 1_000_003) ^ (pos * 2_654_435_761) ^ _contig_salt(contig)
            )
            ref = prng.choice(_BASES)
            is_indel = prng.random() < indel_frac
            v = _alt(prng, ref, is_indel)
            variant_cache[(contig, pos)] = v
        return v

    manifest_paths: list[str] = []
    for i in range(n_files):
        lines: list[str] = []
        for c in contigs:
            shared_set = shared_by_contig[c]
            frng = random.Random(seed * 100003 + i + _contig_salt(c))
            # Draw private positions disjoint from the shared pool so per-file
            # counts stay exact. Oversample to absorb collisions, then take the
            # first n_priv.
            priv_pos: list[int] = []
            if n_priv:
                pool = frng.sample(
                    range(1, contig_len), min(n_priv * 2, contig_len - 1)
                )
                priv_pos = [p for p in pool if p not in shared_set][:n_priv]
            positions = sorted(shared_set | set(priv_pos))
            for pos in positions:
                r, a = variant_at(c, pos)
                keys, vals = _sample_col(frng, format_fields)
                lines.append(f"{c}\t{pos}\t.\t{r}\t{a}\t.\tPASS\t.\t{keys}\t{vals}\n")
        sample = f"S{i:05d}"
        plain = out_dir / f"sample_{i:05d}.vcf"
        plain.write_text(
            _header(sample, contigs, contig_len, format_fields) + "".join(lines)
        )
        gz = out_dir / f"sample_{i:05d}.vcf.gz"
        with open(gz, "wb") as fh:
            subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
        subprocess.run(["bcftools", "index", str(gz)], check=True)
        plain.unlink()
        manifest_paths.append(str(gz))
    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(manifest_paths) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-files", type=int, required=True)
    ap.add_argument(
        "--n-variants",
        type=int,
        required=True,
        help="variants per contig, per file (total per file = n_variants * n_contigs)",
    )
    ap.add_argument(
        "--contig",
        dest="contigs",
        action="append",
        default=None,
        help="repeatable; defaults to a single contig '1'",
    )
    ap.add_argument(
        "--contig-len",
        type=int,
        default=None,
        help="defaults to required_contig_len(n_files, n_variants, shared_frac)",
    )
    ap.add_argument("--shared-frac", type=float, default=0.1)
    ap.add_argument("--indel-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--format-field",
        dest="format_fields",
        action="append",
        default=[],
        help="repeatable FORMAT field to emit per-sample (e.g. VAF, DP)",
    )
    a = ap.parse_args()
    m = generate_cohort(
        a.out_dir,
        a.n_files,
        a.n_variants,
        contigs=a.contigs,
        contig_len=a.contig_len,
        shared_frac=a.shared_frac,
        indel_frac=a.indel_frac,
        seed=a.seed,
        format_fields=a.format_fields,
    )
    print(m)


if __name__ == "__main__":
    main()
