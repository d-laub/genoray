"""Dense vs sparse chunk emission on a synthetic cohort store.

Not a CI gate -- a cohort-scale store is too expensive to build per run. Run it
by hand when changing either kernel:

    pixi run python scripts/bench_sparse_ranges.py --samples 20 --regions 400

Reports per-chunk wall time and the realized fill, which is the number that
decides whether the sparse path is worth anything: at the All of Us chr22 grid
it is 0.45%.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from conftest import build_svar2_singleton_store  # noqa: E402

from genoray import SparseVar2  # noqa: E402


def _time(fn, reps: int) -> float:
    """Minimum over `reps` runs: scheduling noise only ever adds time."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    # The fixture puts one singleton per sample inside a 40 bp contig, so 20 is
    # its ceiling. Raising it means a bigger reference, not a bigger flag.
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--regions", type=int, default=400)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    # A fresh directory per run: the conversion pipeline writes a new store and
    # will not overwrite one left behind by the last invocation.
    tmp = Path(tempfile.mkdtemp(prefix="genoray-bench-sparse-"))
    store = build_svar2_singleton_store(tmp, n_samples=args.samples)
    sv = SparseVar2(store)

    rng = np.random.default_rng(0)
    starts = rng.integers(0, 20, size=args.regions).astype(np.int64)
    ends = np.minimum(starts + 8, 40)

    dense = _time(
        lambda: [c for c in sv._find_ranges_chunked("chr1", starts, ends).chunks],
        args.reps,
    )
    sparse = _time(
        lambda: [
            c for c in sv._find_ranges_chunked_sparse("chr1", starts, ends).chunks
        ],
        args.reps,
    )

    chunks = list(sv._find_ranges_chunked_sparse("chr1", starts, ends).chunks)
    n = sum(len(c.cell_id) for c in chunks)
    cells = args.regions * sv.n_samples * sv.ploidy
    print(f"regions={args.regions} samples={sv.n_samples} ploidy={sv.ploidy}")
    print(f"fill    {n}/{cells} = {100 * n / cells:.2f}%")
    print(f"dense   {1000 * dense:8.2f} ms")
    print(f"sparse  {1000 * sparse:8.2f} ms  ({dense / sparse:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
