"""COSMIC mutational-signature refitting.

A sparse forward-selection refit that decomposes a mutation catalogue into
per-sample activities against a set of reference signatures. Pure
numpy/scipy/polars; no SigProfiler dependency.

The *shape* follows SigProfilerAssignment's ``add_signatures`` -- greedily add
the signature that most improves the fit, stop once the improvement is too
small, then prune negligible activities -- but this is not a port, and the
difference is larger than the defaults. SigProfilerAssignment's ``cosmic_fit``
does not select forward at all: it runs one NNLS over the *entire* signature set
and then eliminates backward on relative L2 error. It also scores on relative L2
rather than cosine, force-includes SBS1/SBS5 as protected background signatures,
force-adds known co-occurring partners (``connected_sigs=True``), and rescales
and integer-rounds activities so they sum to the sample's total burden. None of
that is reproduced here. See the audit issue for the full comparison.

See ``fit_signatures``' ``criterion`` argument for the choice of stop rule, and
why the default is not the statistically consistent one.
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

from ._mutcat import Kind, labels


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


#: Forward-selection stop rules. See ``fit_signatures``.
Criterion = Literal["cosine", "bic"]


def _poisson_ll(m: NDArray[np.floating], e: NDArray[np.floating]) -> float:
    """Poisson log-likelihood of observed counts ``m`` under expected counts ``e``.

    Drops the ``-log(m!)`` term. It depends only on the data, so it cancels in
    every likelihood *difference* -- which is all the ``"bic"`` criterion uses.
    """
    e = np.clip(np.asarray(e, dtype=np.float64), 1e-12, None)
    return float(np.sum(np.asarray(m, dtype=np.float64) * np.log(e) - e))


def _fit_one(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    *,
    max_delta: float,
    min_activity: float,
    criterion: Criterion = "cosine",
) -> tuple[NDArray[np.float64], float]:
    """Refit one sample by sparse forward selection.

    Returns ``(activities, cosine)`` where ``activities`` has one entry per
    reference signature (column of ``W``), zero for unselected signatures, and
    ``cosine`` is the reconstruction cosine similarity of the final fit. The
    returned cosine is the fit's reconstruction quality under either criterion;
    only the *stop rule* changes.
    """
    n_sigs = W.shape[1]
    full = np.zeros(n_sigs, dtype=np.float64)
    if float(np.sum(m)) == 0.0:
        return full, 0.0

    active: list[int] = []
    remaining = list(range(n_sigs))
    best_cos = 0.0
    best_ll = -np.inf
    # BIC observation count: each mutation is one Poisson event, so the penalty
    # for one extra free parameter is log(total burden). Clamped at 2 so a
    # 1-mutation sample gets a non-zero penalty instead of log(1) == 0, which
    # would accept every candidate.
    log_n_obs = float(np.log(max(float(np.sum(m)), 2.0)))

    # Forward selection: add the signature that most improves the fit, until the
    # improvement no longer clears the stop criterion.
    while remaining:
        best = None  # (score, sig_index, cosine)
        for c in remaining:
            cand = active + [c]
            h_sub = _nnls(W[:, cand], m)
            recon = W[:, cand] @ h_sub
            cos = _cosine(m, recon)
            score = cos if criterion == "cosine" else _poisson_ll(m, recon)
            if best is None or score > best[0]:
                best = (score, c, cos)
        assert best is not None
        score, c, cos = best
        if criterion == "cosine":
            if score - best_cos < max_delta:
                break
            best_cos = score
        else:
            # One more signature is one more free parameter. Accept it only when
            # twice the log-likelihood gain clears the BIC penalty. Unlike the
            # cosine delta, this scales with burden: more mutations buy more
            # power to resolve a real but low-activity signature.
            if 2.0 * (score - best_ll) < log_n_obs:
                break
            best_ll = score
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
    criterion: Criterion = "cosine",
    n_jobs: int = 1,
    backend: str = "loky",
) -> pl.DataFrame:
    """Refit a mutation catalogue against reference signatures.

    Args:
        catalogue: A ``mutation_matrix``-shaped frame: a ``MutationType`` column followed by
            one numeric column per sample.
        reference: A ``MutationType`` column followed by one column per reference signature.
            Columns need not be pre-normalized; each is scaled to sum 1 so reported
            activities are in mutation-count units.
        max_delta: Minimum cosine-similarity improvement to keep adding a signature.
            Only used when ``criterion="cosine"``; ignored otherwise.
        min_activity: Minimum fractional contribution; signatures below this are pruned.
        criterion: Forward-selection stop rule.

            ``"cosine"`` (default) stops when the best candidate improves cosine
            similarity by less than ``max_delta``. Cosine is scale-invariant, so this
            threshold is blind to mutation burden: a sample with 100,000 mutations
            gets no more power to resolve a real signature than one with 100. On
            synthetic mixtures of four known COSMIC signatures it plateaus around 3.3
            of 4 recovered and does **not** improve with burden -- and around 2.6 of 4
            when the mixture includes the flat-spectrum family (SBS5, SBS40a), whose
            contribution is what a too-coarse threshold absorbs first.

            ``"bic"`` stops when twice the Poisson log-likelihood gain fails to clear
            the Bayesian information criterion penalty ``log(total burden)`` for the
            one added free parameter. The likelihood gain grows with burden while the
            penalty grows only logarithmically, so the rule is burden-aware and
            consistent: on the same synthetic mixtures it recovers 4.00 of 4 from
            ~1,000 mutations upward (~10,000 with flat signatures present) with
            0.00-0.03 false positives per sample.

            The trade-off runs the other way at very low burden, where ``"bic"``
            over-selects (~1.4 false positives per sample at 100 mutations). The
            default stays ``"cosine"`` for backward compatibility; prefer ``"bic"``
            for whole-genome catalogues.
        n_jobs: Number of parallel workers for the per-sample refit (passed to
            ``joblib.Parallel``). ``1`` (default) runs serially; ``-1`` uses all
            cores. Results are identical regardless of ``n_jobs``.
        backend: ``joblib`` backend (default ``"loky"``, process-based). Samples are
            refit independently, so a process backend avoids GIL contention from
            the forward-selection orchestration.

    Returns:
        pl.DataFrame: One row per sample: a ``Sample`` column, one Float column per reference
        signature (activities, 0.0 if unselected), and a ``cosine_similarity``
        column for the final reconstruction.

    Raises:
        ValueError: If ``criterion`` is not one of ``"cosine"`` / ``"bic"``, or if a
            ``MutationType`` present in the catalogue is missing from the reference
            (rows cannot be aligned).
    """
    if criterion not in ("cosine", "bic"):
        raise ValueError(f"criterion must be 'cosine' or 'bic', got {criterion!r}.")
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
        delayed(_fit_one)(
            W,
            M[:, j],
            max_delta=max_delta,
            min_activity=min_activity,
            criterion=criterion,
        )
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

    Args:
        kind: One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
        version: COSMIC signature release (default ``"3.4"``).
        genome: Reference build for SBS/DBS (``"GRCh37"`` or ``"GRCh38"``). Ignored for
            ID83 (indel signatures are build-independent in the COSMIC release).

    Returns:
        pl.DataFrame: A ``MutationType`` column (in genoray's canonical codebook order for
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
