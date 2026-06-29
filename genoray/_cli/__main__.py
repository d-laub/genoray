#! /usr/bin/env python

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import polars as pl
from cyclopts import App, Parameter, validators

app = App(
    help_on_error=True,
    version=f"[magenta]genoray[/magenta] {version('genoray')}",
    version_format="rich",
    help="Tools for genoray, including SVAR files.",
)


@app.command
def index(source: Path):
    """Create a genoray index for a VCF or PGEN file."""
    from genoray import VCF
    from genoray._pgen import _write_index
    from genoray._utils import variant_file_type

    file_type = variant_file_type(source)
    if file_type == "vcf":
        vcf = VCF(source)
        vcf._write_gvi_index()
    elif file_type == "pgen":
        index = source.with_suffix(".pvar")

        if not index.exists():
            index = source.with_suffix(".pvar.zst")

        if not index.exists():
            raise FileNotFoundError("No index file found.")

        index = index.with_suffix(f"{index.suffix}.gvi")
        _write_index(index)
    else:
        raise ValueError(f"Unsupported file type: {source}")


@app.command
def write(
    source: Path,
    out: Path,
    max_mem: str = "1g",
    overwrite: bool = False,
    dosages: str | None = None,
    threads: int | None = None,
    no_symbolic: Annotated[bool, Parameter(name="--no-symbolic", negative="")] = False,
    no_breakend: Annotated[bool, Parameter(name="--no-breakend", negative="")] = False,
    haploid: Annotated[bool, Parameter(name="--haploid", negative="")] = False,
) -> None:
    """
    Convert a VCF or PGEN file to a SVAR file.

    Parameters
    ----------
    source
        Path to the input VCF or PGEN file.
    out
        Path to the output SVAR file.
    max_mem
        Maximum memory to use for conversion e.g. 1g, 250 MB, etc.
    overwrite
        Whether to overwrite the output file if it exists.
    dosages
        Whether to write dosages.
        If `source` is a PGEN, this must be a path to a PGEN of dosages.
        If `source` is a VCF, this must be the name of the FORMAT field to use for dosages.
        If not provided, dosages will not be written.
    threads
        Number of threads to use for conversion. Defaults to the number of available CPU cores.
    no_symbolic
        If set, drop records whose ALT contains a symbolic allele
        (``<DEL>``, ``<INS>``, etc.) per VCF 4.x. Applies to both VCF and PGEN
        sources. Recommended for SV-bearing cohorts (e.g. 1kGP SNV_INDEL_SV
        panels) to avoid emitting literal ``<DEL>`` ASCII bytes into downstream
        haplotype buffers.
    no_breakend
        If set, drop records whose ALT contains a breakend (BND) in mate-pair or
        single-breakend notation (``G[chr2:321[``, ``]chr2:321]G``, ``.TGCA``,
        ``TGCA.``). A distinct ALT class from symbolic alleles; combine with
        ``--no-symbolic`` to drop everything haplotype consumers cannot expand.
    haploid
        If set, OR-collapse the ploidy axis into a single haploid call per
        sample (a variant present on any haplotype becomes one call) and record
        ``ploidy=1`` in the output metadata. Intended for unphased somatic data.
    """
    from genoray import PGEN, VCF, SparseVar, exprs
    from genoray._utils import variant_file_type

    file_type = variant_file_type(source)

    if dosages is None:
        with_dosages = False
    else:
        with_dosages = True

    if threads is None:
        threads = -1

    # Compose the requested ALT-class filters. Each flag contributes a polars
    # expr (used by both VCF index and PGEN) and a record-level predicate (used
    # by the VCF cyvcf2 genotype scan). The two representations of each flag are
    # kept in parity via genoray.exprs.
    pl_terms: list[pl.Expr] = []
    record_preds: list[Callable[[list[str]], bool]] = []
    if no_symbolic:
        pl_terms.append(~exprs.is_symbolic)
        record_preds.append(exprs._record_is_symbolic)
    if no_breakend:
        pl_terms.append(~exprs.is_breakend)
        record_preds.append(exprs._record_is_breakend)

    if pl_terms:
        from functools import reduce
        from operator import and_

        pl_filter = reduce(and_, pl_terms)

        def record_filter(
            rec: Any,
            _preds: tuple[Callable[[list[str]], bool], ...] = tuple(record_preds),
        ) -> bool:
            return not any(pred(rec.ALT) for pred in _preds)
    else:
        pl_filter = None
        record_filter = None

    if file_type == "vcf":
        if dosages is not None and Path(dosages).exists():
            raise ValueError(
                "The `dosages` argument appears to be a path to an existing file, but VCF requires a FORMAT field name."
            )

        # VCF requires both filters together (or neither): a cyvcf2 callable
        # applied during the genotype scan and a matching polars expr applied to
        # the index; both must express the same predicate.
        vcf = VCF(
            source,
            dosage_field=dosages,
            filter=record_filter,
            pl_filter=pl_filter,
        )
        SparseVar.from_vcf(
            out,
            vcf,
            max_mem,
            overwrite,
            with_dosages=with_dosages,
            n_jobs=threads,
            haploid=haploid,
        )
    elif file_type == "pgen":
        pgen = PGEN(
            source,
            dosage_path=dosages,
            filter=pl_filter,
        )
        SparseVar.from_pgen(
            out,
            pgen,
            max_mem,
            overwrite,
            with_dosages=with_dosages,
            n_jobs=threads,
            haploid=haploid,
        )
    else:
        raise ValueError(f"Unsupported file type: {source}")


@app.command
def view(
    source: Annotated[
        Path,
        Parameter(
            validator=validators.Path(exists=True, dir_okay=True, file_okay=False)
        ),
    ],
    out: Path,
    *,
    regions: Annotated[str | None, Parameter(name=["--regions", "-r"])] = None,
    regions_file: Annotated[
        Path | None,
        Parameter(
            name=["--regions-file", "-R"],
            validator=validators.Path(exists=True, dir_okay=False, file_okay=True),
        ),
    ] = None,
    samples: Annotated[str | None, Parameter(name=["--samples", "-s"])] = None,
    samples_file: Annotated[
        Path | None,
        Parameter(
            name=["--samples-file", "-S"],
            validator=validators.Path(exists=True, dir_okay=False, file_okay=True),
        ),
    ] = None,
    fields: Annotated[list[str] | None, Parameter(name=["--fields", "-f"])] = None,
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
    overwrite: bool = False,
    threads: Annotated[int | None, Parameter(name=["--threads", "-@"])] = None,
    progress: bool = False,
) -> None:
    """Write a subset of an SVAR to a new SVAR directory.

    At least one of --regions/--regions-file or --samples/--samples-file is
    required. The omitted side defaults to "all" (all samples or all variants).

    Parameters
    ----------
    source
        Path to the input SVAR directory.
    out
        Path to the output SVAR directory.
    regions
        Inline region(s): a single ``chrom:start-end`` (1-based inclusive, bcftools
        convention) or a comma-separated list, e.g. ``chr1:1-100,chr2:200-300``.
        Mutually exclusive with --regions-file.
    regions_file
        Path to a BED file (0-based half-open) of regions. Mutually exclusive
        with --regions.
    samples
        Comma-separated list of sample names to keep, e.g. ``A,B,C``. Mutually
        exclusive with --samples-file.
    samples_file
        Path to a file of sample names (one per line). Mutually exclusive with
        --samples.
    fields
        Optional FORMAT fields to carry over (e.g. ``-f GT -f GQ``). Defaults to
        all available fields. Use ``--empty-fields`` to carry over none.
    merge_overlapping
        If set, silently merge overlapping regions instead of raising.
    regions_overlap
        How variants are matched to regions: ``pos`` (default; match if the
        variant POS falls in the range), ``record`` (match by VCF record extent),
        or ``variant`` (match by full variant extent including ILEN).
    overwrite
        Overwrite the output directory if it already exists.
    threads
        Number of threads. Defaults to all available CPUs.
    progress
        If set, show a phase-level progress bar while writing the view.
    """
    import polars as pl

    from genoray import SparseVar

    from ._view_helpers import parse_regions_arg

    # No-op guard
    if (
        regions is None
        and regions_file is None
        and samples is None
        and samples_file is None
    ):
        raise ValueError(
            "at least one of --regions/--regions-file or --samples/--samples-file is required"
        )

    # Mutex within each pair
    if regions is not None and regions_file is not None:
        raise ValueError("--regions and --regions-file are mutually exclusive")
    if samples is not None and samples_file is not None:
        raise ValueError("--samples and --samples-file are mutually exclusive")

    sv = SparseVar(source)

    # Resolve regions arg
    if regions is not None:
        regions_arg: pl.DataFrame | Path = parse_regions_arg(regions)
    elif regions_file is not None:
        regions_arg = regions_file
    else:
        # "all variants" — one row per contig spanning [0, max_pos+1).
        # Use the lazy per-contig stats so we never materialize the full index.
        stats = sv._contig_stats  # columns: CHROM, n, pos_max
        regions_arg = stats.select(
            chrom=pl.col("CHROM"),
            start=pl.lit(0, dtype=pl.Int32),
            end=(pl.col("pos_max") + 1).cast(pl.Int32),
        )

    # Resolve samples arg
    if samples is not None:
        samples_arg: list[str] | Path = [s for s in samples.split(",") if s]
    elif samples_file is not None:
        samples_arg = samples_file
    else:
        samples_arg = list(sv.available_samples)

    sv.write_view(
        regions=regions_arg,
        samples=samples_arg,
        output=out,
        fields=fields,
        merge_overlapping=merge_overlapping,
        regions_overlap=regions_overlap,
        overwrite=overwrite,
        threads=threads,
        progress=progress,
    )


if __name__ == "__main__":
    app()
