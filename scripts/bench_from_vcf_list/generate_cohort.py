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
    # Pin each shared site's REF/ALT ONCE (from the top-level rng) so every file
    # that includes it emits an identical variant. genoray's k-way merge joins on
    # (pos, ilen, alt), so shared sites must match on REF/ALT to actually join --
    # and in no_reference mode a coincidental (ilen, alt) match with a differing
    # REF would hard-error the conversion. Randomizing per file would defeat both.
    shared_variants: dict[int, tuple[str, str]] = {}
    for pos in shared_pos:
        ref = rng.choice(_BASES)
        is_indel = rng.random() < indel_frac
        shared_variants[pos] = _alt(rng, ref, is_indel)
    shared_set = set(shared_pos)
    manifest_paths: list[str] = []
    for i in range(n_files):
        frng = random.Random(seed * 100003 + i)
        n_priv = n_variants - n_shared
        # Draw private positions disjoint from the shared pool so a private site
        # never clobbers a shared site's pinned REF/ALT and per-file counts stay
        # exact. Oversample to absorb collisions, then take the first n_priv.
        priv_pos: list[int] = []
        if n_priv:
            pool = frng.sample(range(1, contig_len), min(n_priv * 2, contig_len - 1))
            priv_pos = [p for p in pool if p not in shared_set][:n_priv]
        variants: dict[int, tuple[str, str]] = dict(shared_variants)
        for pos in priv_pos:
            ref = frng.choice(_BASES)
            is_indel = frng.random() < indel_frac
            variants[pos] = _alt(frng, ref, is_indel)
        lines = []
        for pos in sorted(variants):
            r, a = variants[pos]
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
