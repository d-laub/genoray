"""Per-sample mutation count matrices."""

from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np
import polars as pl
from numpy.typing import NDArray

from .codebook import Kind, N_CODES, code_ranges, labels


@nb.njit(parallel=True, nogil=True, cache=True)
def _count_kernel(
    data_codes: NDArray[np.int16],
    offsets: NDArray[np.int64],
    ploidy: np.int64,
    n_samples: np.int64,
    n_codes: np.int64,
    per_sample: np.bool_,
    out: NDArray[np.int64],
) -> None:
    """out[sample, code] accumulator over genotype entries.

    Parallelized over samples: each thread owns a disjoint row of ``out``.
    When ``per_sample`` is True, a code is counted at most once per sample.
    """
    for sample in nb.prange(n_samples):  # type: ignore[misc]
        for slot in range(sample * ploidy, (sample + 1) * ploidy):
            o_s, o_e = offsets[slot], offsets[slot + 1]
            for j in range(o_s, o_e):
                code = data_codes[j]
                if code < 0 or code >= n_codes:
                    continue
                if per_sample:
                    out[sample, code] = 1
                else:
                    out[sample, code] += 1


def count_matrix(
    entry_codes: np.ndarray,
    offsets: np.ndarray,
    ploidy: int,
    n_samples: int,
    sample_names: list[str],
    kind: Kind,
    per_sample: bool,
) -> "pl.DataFrame":
    counts = np.zeros((n_samples, N_CODES), dtype=np.int64)
    _count_kernel(
        entry_codes.astype(np.int16),
        offsets.astype(np.int64),
        np.int64(ploidy),
        np.int64(n_samples),
        np.int64(N_CODES),
        np.bool_(per_sample),
        counts,
    )
    lo, hi = code_ranges()[kind]
    block = counts[:, lo:hi]  # (n_samples, n_categories)
    out: dict[str, Any] = {"MutationType": labels(kind)}
    for s_i, name in enumerate(sample_names):
        out[name] = block[s_i]
    return pl.DataFrame(out)
