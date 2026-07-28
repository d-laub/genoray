"""Generate the sweep plans from the spec's scale points.

Generated rather than committed so a change to `size_corpus`'s rule cannot
silently disagree with a stale JSON file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from scripts.bench_svar2.records import SweepPoint
from scripts.bench_svar2.scale_corpus import size_corpus

CELLS_BUDGET = 1_400_000_000
SCALE_SAMPLES = (250, 1_000, 4_000, 16_000, 64_000, 250_000, 500_000)
# Validate the predicted knee against a real sweep at only three points.
KNEE_VALIDATION_SAMPLES = (250, 16_000, 500_000)
KNEE_WORKERS = (1, 2, 3, 5, 7, 11)
CONTIG_COUNTS = (1, 2, 8, 22)
HOLDOUT = {"samples": 100_000, "variants": 28_000, "format_fields": ("DP", "GQ", "AD")}


def _point(
    corpus: Path,
    workers: int,
    chunk_size: int,
    threads: int,
    concurrent: int | None = None,
) -> SweepPoint:
    return SweepPoint(
        corpus=str(corpus),
        reader_workers=workers,
        concurrent_chroms=concurrent,
        shard_htslib=0,
        overshard=4,
        chunk_size=chunk_size,
        threads=threads,
        reps=3,
        rss_ceiling_mb=60_000,
    )


def build(corpus_dir: Path, threads: int) -> dict[str, list[SweepPoint]]:
    scale, contig, holdout = [], [], []

    for s in SCALE_SAMPLES:
        _, cs = size_corpus(s, CELLS_BUDGET)
        corpus = corpus_dir / f"s{s}.manifest.json"
        # One w=1 run per point predicts the knee; only three points get swept.
        scale.append(_point(corpus, 1, cs, threads))
        if s in KNEE_VALIDATION_SAMPLES:
            for w in KNEE_WORKERS:
                if w != 1:
                    scale.append(_point(corpus, w, cs, threads))

    # Contig axis at fixed cohort: hold TOTAL readers constant (12) and vary the
    # split, which is what separates "too few readers" from "wrong contig".
    _, cs = size_corpus(4_000, CELLS_BUDGET)
    for c in CONTIG_COUNTS:
        corpus = corpus_dir / f"s4000_c{c}.manifest.json"
        for concurrent in (1, min(c, 4)):
            workers = max(1, 12 // concurrent)
            contig.append(_point(corpus, workers, cs, threads, concurrent=concurrent))

    holdout.append(_point(corpus_dir / "holdout.manifest.json", 1, 2_000, threads))

    return {"scale": scale, "contig": contig, "holdout": holdout}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=48)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, points in build(a.corpus_dir, a.threads).items():
        path = a.out_dir / f"{name}.json"
        path.write_text(json.dumps([dataclasses.asdict(pt) for pt in points], indent=2))
        print(f"{path}: {len(points)} points")


if __name__ == "__main__":
    main()
