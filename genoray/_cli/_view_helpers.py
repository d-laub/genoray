"""Helpers for the `genoray view` subcommand."""

from __future__ import annotations

import re

import polars as pl

_REGION_RE = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def parse_regions_arg(s: str) -> pl.DataFrame:
    """Parse a bcftools-style ``-r`` value into a 0-based half-open DataFrame.

    Accepts a single ``chrom:start-end`` (1-based inclusive) or a comma-
    separated list. Returns columns ``chrom`` (Utf8), ``start`` (Int32),
    ``end`` (Int32).
    """
    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for piece in (p.strip() for p in s.split(",") if p.strip()):
        m = _REGION_RE.match(piece)
        if m is None:
            raise ValueError(
                f"region {piece!r} does not match 'chrom:start-end' (1-based inclusive)"
            )
        chroms.append(m["chrom"])
        starts.append(int(m["start"]) - 1)  # 1-based -> 0-based
        ends.append(int(m["end"]))
    return pl.DataFrame(
        {"chrom": chroms, "start": starts, "end": ends},
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
    )
