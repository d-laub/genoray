"""Generate a synthetic multi-sample, single/multi-contig bgzipped VCF for
SVAR2 sharded-reader benchmarks.

The sharded VCF path (orchestrator.rs) needs: an indexed .vcf.gz, OverlapMode::Pos
(the whole-contig default), and >1 planned shard. Records are biallelic SNPs with
a GT-only FORMAT, which is the AoU chr22 shape the PR#140 motivating run used.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

BASES = np.array(["A", "C", "G", "T"])
# Genotype token pool weighted toward hom-ref, like a real cohort at common sites.
GT_TOKENS = np.array(["0|0", "0|1", "1|0", "1|1"])
GT_WEIGHTS = np.array([0.72, 0.12, 0.12, 0.04])


def gen(
    out: Path,
    n_samples: int,
    n_variants: int,
    contigs: list[str],
    contig_len: int,
    seed: int,
    threads: int,
) -> None:
    rng = np.random.default_rng(seed)
    samples = [f"S{i:06d}" for i in range(n_samples)]

    header = ["##fileformat=VCFv4.2"]
    for c in contigs:
        header.append(f"##contig=<ID={c},length={contig_len}>")
    header.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    header.append(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)
    )

    proc = subprocess.Popen(
        ["bgzip", "-c", "-@", str(threads)],
        stdin=subprocess.PIPE,
        stdout=out.open("wb"),
    )
    assert proc.stdin is not None
    w = proc.stdin
    w.write(("\n".join(header) + "\n").encode())

    per_contig = n_variants // len(contigs)
    block = 2000
    for contig in contigs:
        # Sorted unique positions, 1-based, spread across the contig. Sample with
        # replacement and dedupe (collisions are rare at per_contig << contig_len);
        # `choice(..., replace=False)` would permute the whole contig-length space.
        pos = np.unique(rng.integers(1, contig_len, size=int(per_contig * 1.2)))
        while pos.size < per_contig:
            pos = np.unique(
                np.concatenate([pos, rng.integers(1, contig_len, size=per_contig)])
            )
        pos = pos[:per_contig]
        for start in range(0, per_contig, block):
            stop = min(start + block, per_contig)
            n = stop - start
            ref_i = rng.integers(0, 4, size=n)
            alt_off = rng.integers(1, 4, size=n)
            ref = BASES[ref_i]
            alt = BASES[(ref_i + alt_off) % 4]
            gt_idx = rng.choice(4, size=(n, n_samples), p=GT_WEIGHTS)
            gts = GT_TOKENS[gt_idx]
            lines = []
            for r in range(n):
                lines.append(
                    f"{contig}\t{pos[start + r]}\t.\t{ref[r]}\t{alt[r]}\t.\tPASS\t.\tGT\t"
                    + "\t".join(gts[r])
                )
            w.write(("\n".join(lines) + "\n").encode())

    w.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"bgzip failed: {rc}")
    subprocess.run(["tabix", "-f", "-p", "vcf", str(out)], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--variants", type=int, default=100_000)
    p.add_argument("--contigs", type=str, default="chr22")
    p.add_argument("--contig-len", type=int, default=50_818_468)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    gen(
        a.out,
        a.samples,
        a.variants,
        a.contigs.split(","),
        a.contig_len,
        a.seed,
        a.threads,
    )
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
