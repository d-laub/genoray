from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import polars as pl
from numpy.typing import NDArray
from seqpro.rag import Ragged

from .._types import DOSAGE_TYPE, V_IDX_TYPE
from ..exprs import ILEN

NUMERIC = TypeVar("NUMERIC", bound=np.number)


def _write_filtered_index(src: Path, dst: Path, pl_filter: pl.Expr | None) -> None:
    """Stream a (possibly filtered) genoray index from ``src`` to ``dst``.

    When ``pl_filter`` is None this is byte-equivalent to copying. Otherwise the
    filter is applied lazily; ALT is normalized to list[str] for the filter and
    re-joined to the on-disk comma-Utf8 form so the SVAR index format is
    unchanged. ILEN is computed on-the-fly if absent so ILEN-dependent
    expressions (e.g. ``is_snp``) work correctly, then dropped from the output
    to preserve the original on-disk schema. ILEN is always computed when absent
    from the on-disk schema — even if the filter doesn't reference it — to avoid
    introspecting the opaque Polars expression.
    """
    if pl_filter is None:
        shutil.copy(src, dst)
        return
    lf = pl.scan_ipc(src)
    schema = lf.collect_schema()
    alt_is_utf8 = schema["ALT"] == pl.Utf8
    ilen_added = "ILEN" not in schema
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").str.split(","))
    if ilen_added:
        lf = lf.with_columns(ILEN=ILEN)
    lf = lf.filter(pl_filter)
    if ilen_added:
        lf = lf.drop("ILEN")
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").list.join(","))
    lf.sink_ipc(dst, compression="zstd")


def _subset_var_idxs_and_recompute_af(
    out_path: Path,
    n_total: int,
    n_out: int,
    ploidy: int,
    with_dosages: bool,
) -> tuple[NDArray[V_IDX_TYPE], NDArray[np.float32]]:
    """After concat, drop variants whose MAC across the (already sample-subset)
    output is 0 and remap surviving variant ids to a compacted range.

    A MAC=0 variant contributes no entries to ``variant_idxs.npy`` (it is never
    non-ref in any kept sample/ploidy slot), so dropping it requires only a
    remap of the stored ids — no entries are removed and offsets are unchanged.

    ``with_dosages`` is accepted for call-site symmetry with ``from_vcf`` /
    ``from_pgen``; dosages are stored per-entry and need no remap since no
    entries are dropped.

    Returns ``(survivor_rows, af)`` where ``survivor_rows`` indexes into the
    ``n_total`` candidate rows (use it to subset the index frame) and ``af`` is
    the recomputed allele frequency for each survivor.
    """
    vi_path = out_path / "variant_idxs.npy"
    # If file is missing or empty, treat every candidate as MAC=0 (no entries
    # were ever written), so all variants will be dropped by the check below.
    if not vi_path.exists() or vi_path.stat().st_size == 0:
        mac: NDArray[np.int64] = np.zeros(n_total, dtype=np.int64)
        var_idxs = None
    else:
        var_idxs = np.memmap(vi_path, dtype=V_IDX_TYPE, mode="r+")
        mac = np.bincount(np.asarray(var_idxs, dtype=np.int64), minlength=n_total)
    survivor_mask = mac > 0
    n_surv = int(survivor_mask.sum())
    if n_surv == 0:
        raise ValueError(
            "all selected variants have MAC=0 in the chosen sample subset; "
            "nothing to write"
        )
    n_dropped = n_total - n_surv
    if n_dropped:
        warnings.warn(
            f"dropping {n_dropped} variant(s) with MAC=0 in the output sample set",
            stacklevel=2,
        )
    if var_idxs is not None:
        remap = np.empty(n_total, dtype=V_IDX_TYPE)
        remap[survivor_mask] = np.arange(n_surv, dtype=V_IDX_TYPE)
        # every referenced id survives by construction, so this never hits a gap
        var_idxs[:] = remap[np.asarray(var_idxs, dtype=np.int64)]
        var_idxs.flush()
        del var_idxs

    survivor_rows = np.flatnonzero(survivor_mask).astype(V_IDX_TYPE)
    af = (mac[survivor_mask] / (n_out * ploidy)).astype(np.float32)
    return survivor_rows, af


def _build_working_index(
    src_index_path: Path, pl_filter: pl.Expr | None
) -> tuple[pl.DataFrame, bool, bool]:
    """Load the source index, apply ``pl_filter`` (if any), and return a working
    frame with ALT as list[str], an ILEN list column, and an ``index`` column
    holding each row's position (the SVAR variant id). Also returns
    ``(alt_is_utf8, ilen_added)`` so the on-disk format can be reconstructed.
    """
    lf = pl.scan_ipc(src_index_path)
    schema = lf.collect_schema()
    alt_is_utf8 = schema["ALT"] == pl.Utf8
    ilen_added = "ILEN" not in schema
    if alt_is_utf8:
        lf = lf.with_columns(pl.col("ALT").str.split(","))
    if ilen_added:
        lf = lf.with_columns(ILEN=ILEN)
    if pl_filter is not None:
        lf = lf.filter(pl_filter)
    df = lf.collect().with_row_index("index")
    return df, alt_is_utf8, ilen_added


def _write_index_from_working(
    working_df: "pl.DataFrame",
    rows: "NDArray[V_IDX_TYPE]",
    dst: Path,
    alt_is_utf8: bool,
    ilen_added: bool,
    af: "NDArray[np.float32] | None" = None,
) -> None:
    """Write the rows of *working_df* selected by *rows* (in the given order) to
    *dst* in the canonical SVAR on-disk index format: ALT re-joined to comma-Utf8
    if it was originally Utf8, ILEN dropped if we added it, and the helper
    ``index`` column dropped. If *af* is given, (re)sets an ``AF`` column.

    *rows* must be a numpy array of positional row offsets into *working_df*
    (i.e. values from the ``index`` column, which equals row position since
    ``_build_working_index`` calls ``with_row_index`` after any filter).
    """
    frame = working_df[rows.tolist()]
    if af is not None:
        if "AF" in frame.columns:
            frame = frame.drop("AF")
        frame = frame.with_columns(AF=pl.Series(af))
    if ilen_added and "ILEN" in frame.columns:
        frame = frame.drop("ILEN")
    if alt_is_utf8:
        frame = frame.with_columns(pl.col("ALT").list.join(","))
    frame = frame.drop("index")
    frame.write_ipc(dst, compression="zstd")


def _open_genos(path: Path, shape: tuple[int | None, ...], mode: Literal["r", "r+"]):
    # Load the memory-mapped files
    var_idxs = np.memmap(path / "variant_idxs.npy", dtype=V_IDX_TYPE, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = Ragged[V_IDX_TYPE].from_offsets(var_idxs, shape, offsets)
    return sp_genos


def _open_fmt(
    name: str,
    type_: NUMERIC | np.dtype[NUMERIC] | type[NUMERIC],
    path: Path,
    shape: tuple[int | None, ...],
    mode: Literal["r", "r+"],
) -> Ragged[NUMERIC]:
    # Load the memory-mapped files
    data = np.memmap(path / f"{name}.npy", dtype=type_, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = Ragged.from_offsets(data, shape, offsets)
    return sp_genos  # type: ignore[bad-return]


def _write_genos(path: Path, sp_genos: Ragged[V_IDX_TYPE]):
    path.mkdir(parents=True, exist_ok=True)

    var_idxs = np.memmap(
        path / "variant_idxs.npy",
        shape=sp_genos.data.shape,
        dtype=sp_genos.data.dtype,
        mode="w+",
    )
    var_idxs[:] = sp_genos.data
    var_idxs.flush()

    offsets = np.memmap(
        path / "offsets.npy",
        shape=sp_genos.offsets.shape,
        dtype=sp_genos.offsets.dtype,
        mode="w+",
    )
    offsets[:] = sp_genos.offsets
    offsets.flush()


def _write_dosages(path: Path, dosages: NDArray[DOSAGE_TYPE]):
    path.mkdir(parents=True, exist_ok=True)

    dosages_memmap = np.memmap(
        path / "dosages.npy",
        shape=dosages.shape,
        dtype=dosages.dtype,
        mode="w+",
    )
    dosages_memmap[:] = dosages
    dosages_memmap.flush()
