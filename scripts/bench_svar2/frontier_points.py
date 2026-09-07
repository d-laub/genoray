"""Reader-frontier plan for issue #169.

Emits a `sweep.py --plan` JSON file. `sweep.py` owns corpus-manifest loading,
`code_id` stamping and resume; `probe.py` owns the instrumented child run and
already parses `pending_highwater` out of the `genoray::monitor` trace stream.
Neither is reimplemented here -- this module is only the point list.

Run as a module so a ProcessPool worker can re-import it by NAME. Python
3.14's Linux default start method is forkserver, and a worker whose function
lives in a `spec_from_file_location`-loaded module dies with
ModuleNotFoundError -> BrokenProcessPool.

    python -m scripts.bench_svar2.frontier_points \\
        --manifest wide.manifest.json --out plan.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from scripts.bench_svar2.records import SweepPoint

# One contig, so contig concurrency is pinned at 1 and `reader_workers` is the
# only thing moving. The three worker counts bracket the old default (3), the
# new depth target (W_TARGET=8), and the count the reporter actually used (20).
READER_WORKERS = (3, 8, 20)
CHUNK_SIZE = 5_000


def points(manifest: str) -> list[SweepPoint]:
    """`SweepPoint.corpus` is the path to a corpus MANIFEST json (written by
    `scale_corpus.py` next to the .vcf.gz), not the .vcf.gz itself --
    `sweep.py` reads shape from the manifest so a corpus can be relocated."""
    return [
        SweepPoint(
            corpus=manifest,
            reader_workers=w,
            concurrent_chroms=1,
            shard_htslib=0,
            overshard=4,
            chunk_size=CHUNK_SIZE,
            threads=32,
            reps=1,
        )
        for w in READER_WORKERS
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(
        json.dumps([dataclasses.asdict(pt) for pt in points(a.manifest)], indent=2)
    )


if __name__ == "__main__":
    main()
