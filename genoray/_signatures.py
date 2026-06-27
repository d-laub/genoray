"""COSMIC mutational-signature refitting.

Ports the core of SigProfilerAssignment: a sparse forward-selection refit that
decomposes a mutation catalogue into per-sample activities against a set of
reference signatures. Pure numpy/scipy/polars; no SigProfiler dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import pooch
from numpy.typing import NDArray
from joblib import Parallel, delayed
from scipy.optimize import nnls

from ._mutcat import labels

Kind = Literal["SBS96", "DBS78", "ID83"]


def _cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Cosine similarity of two vectors; 0.0 if either has zero norm."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _nnls(W: NDArray[np.floating], m: NDArray[np.floating]) -> NDArray[np.float64]:
    """Non-negative least squares: argmin_{h>=0} ||W h - m||."""
    h, _ = nnls(W.astype(np.float64), m.astype(np.float64))
    return h


def _fit_one(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    *,
    max_delta: float,
    min_activity: float,
) -> tuple[NDArray[np.float64], float]:
    """Refit one sample by sparse forward selection.

    Returns ``(activities, cosine)`` where ``activities`` has one entry per
    reference signature (column of ``W``), zero for unselected signatures, and
    ``cosine`` is the reconstruction cosine similarity of the final fit.
    """
    n_sigs = W.shape[1]
    full = np.zeros(n_sigs, dtype=np.float64)
    if float(np.sum(m)) == 0.0:
        return full, 0.0

    active: list[int] = []
    remaining = list(range(n_sigs))
    best_cos = 0.0

    # Forward selection: add the signature that most improves cosine, until the
    # improvement falls below max_delta.
    while remaining:
        best = None  # (cos, sig_index, h_subvector)
        for c in remaining:
            cand = active + [c]
            h_sub = _nnls(W[:, cand], m)
            recon = W[:, cand] @ h_sub
            cos = _cosine(m, recon)
            if best is None or cos > best[0]:
                best = (cos, c, h_sub)
        assert best is not None
        cos, c, _ = best
        if cos - best_cos < max_delta:
            break
        best_cos = cos
        active.append(c)
        remaining.remove(c)

    if not active:
        return full, 0.0

    # Prune signatures below min_activity (as a fraction of total), re-fitting
    # survivors until the active set is stable.
    while True:
        h_sub = _nnls(W[:, active], m)
        total = float(h_sub.sum())
        if total == 0.0:
            return full, 0.0
        keep = [
            active[i] for i in range(len(active)) if h_sub[i] / total >= min_activity
        ]
        if len(keep) == len(active):
            break
        if not keep:
            # everything pruned: keep the single largest contributor
            keep = [active[int(np.argmax(h_sub))]]
        active = keep

    h_sub = _nnls(W[:, active], m)
    recon = W[:, active] @ h_sub
    cos = _cosine(m, recon)
    for i, sig in enumerate(active):
        full[sig] = h_sub[i]
    return full, cos


def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    max_delta: float = 0.01,
    min_activity: float = 0.005,
    n_jobs: int = -1,
    backend: str = "loky",
) -> pl.DataFrame:
    """Refit a mutation catalogue against reference signatures.

    Parameters
    ----------
    catalogue
        A ``mutation_matrix``-shaped frame: a ``MutationType`` column followed by
        one numeric column per sample.
    reference
        A ``MutationType`` column followed by one column per reference signature.
        Columns need not be pre-normalized; each is scaled to sum 1 so reported
        activities are in mutation-count units.
    max_delta
        Minimum cosine-similarity improvement to keep adding a signature
        (forward-selection stop criterion).
    min_activity
        Minimum fractional contribution; signatures below this are pruned.
    n_jobs
        Number of parallel workers for the per-sample refit (passed to
        ``joblib.Parallel``). ``-1`` (default) uses all cores; ``1`` runs
        serially. Results are identical regardless of ``n_jobs``.
    backend
        ``joblib`` backend (default ``"loky"``, process-based). Samples are
        refit independently, so a process backend avoids GIL contention from
        the forward-selection orchestration.

    Returns
    -------
    pl.DataFrame
        One row per sample: a ``Sample`` column, one Float column per reference
        signature (activities, 0.0 if unselected), and a ``cosine_similarity``
        column for the final reconstruction.

    Raises
    ------
    ValueError
        If a ``MutationType`` present in the catalogue is missing from the
        reference (rows cannot be aligned).
    """
    if "MutationType" not in catalogue.columns:
        raise ValueError("catalogue must have a 'MutationType' column.")
    if "MutationType" not in reference.columns:
        raise ValueError("reference must have a 'MutationType' column.")

    sample_cols = [c for c in catalogue.columns if c != "MutationType"]
    sig_cols = [c for c in reference.columns if c != "MutationType"]

    # Align reference rows to the catalogue's row order by joining on MutationType.
    # maintain_order="left" is required: without it, Polars' hash-join is
    # non-deterministic under concurrent workloads and can reorder left-frame rows.
    aligned = catalogue.select("MutationType").join(
        reference, on="MutationType", how="left", maintain_order="left"
    )
    missing = aligned.filter(pl.col(sig_cols[0]).is_null())
    if missing.height > 0:
        bad = missing["MutationType"].to_list()
        raise ValueError(
            f"reference is missing MutationType rows present in the catalogue: {bad}"
        )

    W = aligned.select(sig_cols).to_numpy().astype(np.float64)  # (n_types, n_sigs)
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0.0] = 1.0  # avoid div-by-zero for empty signatures
    W = W / col_sums  # normalize each signature column to sum 1

    M = (
        catalogue.select(sample_cols).to_numpy().astype(np.float64)
    )  # (n_types, n_samples)

    activities = np.zeros((len(sample_cols), len(sig_cols)), dtype=np.float64)
    cosines = np.zeros(len(sample_cols), dtype=np.float64)
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_fit_one)(W, M[:, j], max_delta=max_delta, min_activity=min_activity)
        for j in range(len(sample_cols))
    )
    for j, (h, cos) in enumerate(results):
        activities[j] = h
        cosines[j] = cos

    out: dict[str, object] = {"Sample": sample_cols}
    for i, sig in enumerate(sig_cols):
        out[sig] = activities[:, i]
    out["cosine_similarity"] = cosines
    return pl.DataFrame(out)


# ---------------------------------------------------------------------------
# COSMIC reference signature loader (pooch-backed)
# ---------------------------------------------------------------------------

# COSMIC reference signatures (v3.4). The filename convention is
# COSMIC_v{ver}_{SBS,DBS,ID}_{genome}.txt with a `Type` header column.
# Map (kind, version, genome) -> (url, known_hash). known_hash is None until
# pinned; pooch will warn but still download when None.
_COSMIC_REGISTRY: dict[tuple[str, str, str], tuple[str, str | None]] = {
    # URLs point to the AlexandrovLab/SigProfilerAssignment GitHub mirror of the
    # official COSMIC v3.4 release files. Hashes verified by direct download on
    # 2026-06-11.
    ("SBS96", "3.4", "GRCh38"): (
        "https://raw.githubusercontent.com/AlexandrovLab/SigProfilerAssignment"
        "/main/SigProfilerAssignment/data/Reference_Signatures/GRCh38"
        "/COSMIC_v3.4_SBS_GRCh38.txt",
        "sha256:c93fa5b0f857ef56d81b753a9543e2fa250a6df04841b20e3f88f575a9dea9e2",
    ),
    ("DBS78", "3.4", "GRCh38"): (
        "https://raw.githubusercontent.com/AlexandrovLab/SigProfilerAssignment"
        "/main/SigProfilerAssignment/data/Reference_Signatures/GRCh38"
        "/COSMIC_v3.4_DBS_GRCh38.txt",
        "sha256:ef337893e86ffd534df4e0b507b7638f7414aa452fd708ea1572175620acc5e8",
    ),
    ("ID83", "3.4", "GRCh37"): (
        "https://raw.githubusercontent.com/AlexandrovLab/SigProfilerAssignment"
        "/main/SigProfilerAssignment/data/Reference_Signatures/GRCh37"
        "/COSMIC_v3.4_ID_GRCh37.txt",
        "sha256:aa53aafb8a629c8d8df5908df0a9e5d6bda09425a01b0bb37d838f2dc4a20224",
    ),
}

_KIND_TOKEN = {"SBS96": "SBS", "DBS78": "DBS", "ID83": "ID"}


def _load_signature_file(path: str | Path) -> pl.DataFrame:
    """Parse a COSMIC-style signature TSV into a ``MutationType``-first frame."""
    df = pl.read_csv(Path(path), separator="\t")
    first = df.columns[0]
    if first != "MutationType":
        df = df.rename({first: "MutationType"})
    return df


def cosmic_signatures(
    kind: Kind,
    *,
    version: str = "3.4",
    genome: str = "GRCh38",
) -> pl.DataFrame:
    """Fetch (and cache) the COSMIC reference signatures for ``kind``.

    Parameters
    ----------
    kind
        One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
    version
        COSMIC signature release (default ``"3.4"``).
    genome
        Reference build for SBS/DBS (``"GRCh37"`` or ``"GRCh38"``). Ignored for
        ID83 (indel signatures are build-independent in the COSMIC release).

    Returns
    -------
    pl.DataFrame
        A ``MutationType`` column (in genoray's canonical codebook order for
        ``kind``) followed by one column per COSMIC signature, ready to pass to
        :func:`fit_signatures`.
    """
    if kind not in _KIND_TOKEN:
        raise ValueError(f"Unknown kind {kind!r}; choose from {list(_KIND_TOKEN)}.")
    eff_genome = "GRCh37" if kind == "ID83" else genome
    key = (kind, version, eff_genome)
    if key not in _COSMIC_REGISTRY:
        raise ValueError(
            f"No COSMIC URL registered for {key}. Register it in "
            "genoray/_signatures.py:_COSMIC_REGISTRY."
        )
    url, known_hash = _COSMIC_REGISTRY[key]
    local = pooch.retrieve(url=url, known_hash=known_hash)
    df = _load_signature_file(local)

    # Reindex to genoray's canonical row order so it aligns with mutation_matrix.
    order = labels(kind)
    df = pl.DataFrame({"MutationType": order}).join(
        df, on="MutationType", how="left", maintain_order="left"
    )

    sig_cols = [c for c in df.columns if c != "MutationType"]
    null_types = df.filter(pl.col(sig_cols[0]).is_null())["MutationType"].to_list()
    if null_types:
        raise ValueError(
            f"COSMIC {kind} file is missing {len(null_types)} expected MutationType "
            f"rows (codebook/COSMIC mismatch): {null_types[:5]}"
        )

    return df
