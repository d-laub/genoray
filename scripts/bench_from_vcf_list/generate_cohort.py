# scripts/bench_from_vcf_list/generate_cohort.py
"""Generate a synthetic single-sample somatic cohort for from_vcf_list benchmarks.

Each file gets `shared_frac` of its sites drawn from a shared pool (so the k-way
merge actually joins) and the rest private (fan-out). Reproducible by seed.
"""

from __future__ import annotations
import argparse
import random
import subprocess
from pathlib import Path

_BASES = "ACGT"
_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID={contig},length={length}>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
)


def _alt(rng: random.Random, ref: str, indel: bool) -> tuple[str, str]:
    if indel:  # simple 1bp insertion anchored on ref
        return ref, ref + rng.choice(_BASES)
    alt = rng.choice([b for b in _BASES if b != ref])
    return ref, alt


def generate_cohort(
    out_dir: Path,
    n_files: int,
    n_variants: int,
    *,
    contig: str = "chr1",
    contig_len: int = 1_000_000,
    shared_frac: float = 0.1,
    indel_frac: float = 0.1,
    seed: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n_shared = int(n_variants * shared_frac)
    shared_pos = sorted(rng.sample(range(1, contig_len), n_shared)) if n_shared else []
    shared_set = set(shared_pos)

    # Every position's REF/ALT is a deterministic function of the position alone,
    # so any two files that emit the same position ALWAYS agree -- whether it's a
    # shared-pool site or a coincidental private/private collision. This matters at
    # scale: with many files, private positions collide across files (birthday
    # paradox in a finite contig), and genoray's no_reference k-way merge
    # hard-errors on a cross-file REF disagreement. Deriving from the position (not
    # a per-file rng) mirrors a real cohort called against one reference.
    variant_cache: dict[int, tuple[str, str]] = {}

    def variant_at(pos: int) -> tuple[str, str]:
        v = variant_cache.get(pos)
        if v is None:
            prng = random.Random((seed * 1_000_003) ^ (pos * 2_654_435_761))
            ref = prng.choice(_BASES)
            is_indel = prng.random() < indel_frac
            v = _alt(prng, ref, is_indel)
            variant_cache[pos] = v
        return v

    manifest_paths: list[str] = []
    for i in range(n_files):
        frng = random.Random(seed * 100003 + i)
        n_priv = n_variants - n_shared
        # Draw private positions disjoint from the shared pool so per-file counts
        # stay exact. Oversample to absorb collisions, then take the first n_priv.
        priv_pos: list[int] = []
        if n_priv:
            pool = frng.sample(range(1, contig_len), min(n_priv * 2, contig_len - 1))
            priv_pos = [p for p in pool if p not in shared_set][:n_priv]
        positions = sorted(shared_set | set(priv_pos))
        lines = []
        for pos in positions:
            r, a = variant_at(pos)
            lines.append(f"{contig}\t{pos}\t.\t{r}\t{a}\t.\tPASS\t.\tGT\t1/1\n")
        sample = f"S{i:05d}"
        plain = out_dir / f"sample_{i:05d}.vcf"
        plain.write_text(
            _HEADER.format(contig=contig, length=contig_len, sample=sample)
            + "".join(lines)
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
    ap.add_argument("--n-variants", type=int, required=True)
    ap.add_argument("--contig", default="chr1")
    ap.add_argument("--contig-len", type=int, default=1_000_000)
    ap.add_argument("--shared-frac", type=float, default=0.1)
    ap.add_argument("--indel-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = generate_cohort(
        a.out_dir,
        a.n_files,
        a.n_variants,
        contig=a.contig,
        contig_len=a.contig_len,
        shared_frac=a.shared_frac,
        indel_frac=a.indel_frac,
        seed=a.seed,
    )
    print(m)


if __name__ == "__main__":
    main()
