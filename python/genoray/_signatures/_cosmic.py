"""COSMIC reference signature loader (pooch-backed)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pooch

from .._mutcat import Kind, labels

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
