#! /usr/bin/env python

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import polars as pl
from cyclopts import App, Parameter, validators

from genoray._svar2_ops import Mode

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


write = App(
    name="write", help="Convert a VCF/PGEN to a SparseVar file (SVAR2 by default)."
)
app.command(write)


@write.default
def write_svar2(
    source: Path,
    out: Path,
    *,
    reference: Annotated[Path | None, Parameter(name="--reference")] = None,
    no_reference: Annotated[
        bool, Parameter(name="--no-reference", negative="")
    ] = False,
    ploidy: int = 2,
    chunk_size: int | None = None,
    threads: Annotated[int | None, Parameter(name=["--threads", "-@"])] = None,
    long_allele_capacity: int = 8 * 1024 * 1024,
    overwrite: bool = False,
    skip_symbolics_and_breakends: Annotated[
        bool, Parameter(name="--skip-symbolics-and-breakends", negative="")
    ] = False,
    check_ref: Annotated[Literal["e", "x"], Parameter(name="--check-ref")] = "e",
) -> None:
    """
    Convert a bgzipped VCF, BCF, or PLINK2 PGEN to an SVAR2 store (the default, better-across-the-board format).

    Parameters
    ----------
    source
        Path to a bgzipped VCF (``.vcf.gz``), BCF (``.bcf``), or PLINK2 PGEN
        (``.pgen``, with its ``.pvar``/``.pvar.zst`` and ``.psam`` siblings).
        VCF/BCF inputs are auto-indexed (``.csi``) if no index is present.
    out
        Path to the output SVAR2 directory.
    reference
        Path to a reference FASTA (with ``.fai``). Used to validate REF and
        left-align indels. Exactly one of ``--reference`` or ``--no-reference``
        is required.
    no_reference
        Skip REF validation and indel left-alignment; the input is trusted to be
        already normalized. Use only for pre-normalized (e.g. ``bcftools norm``)
        inputs.
    ploidy
        Ploidy of the samples. Default 2. VCF/BCF only — PGEN is diploid.
    chunk_size
        Variants per conversion chunk. Defaults to 25000 for VCF/BCF, and to a
        memory-derived value for PGEN.
    threads
        Number of threads. Defaults to all available cores.
    long_allele_capacity
        Advanced: byte budget for the streaming long-allele buffer.
    overwrite
        Overwrite the output directory if it exists.
    skip_symbolics_and_breakends
        Drop records whose ALT is symbolic (``<DEL>``, ``<INS>``, …) or a
        breakend, instead of erroring. The SVAR2 core cannot expand either
        class into nucleotides, so they are dropped together. (On
        ``genoray write svar1`` the two classes are filtered independently
        via ``--no-symbolic`` / ``--no-breakend``.)
    check_ref
        REF-vs-reference policy (ignored with ``--no-reference``). ``e``
        (default) aborts on the first REF/FASTA disagreement; ``x`` drops the
        offending record and continues. Mirrors ``bcftools norm --check-ref``.
    """
    from genoray import SparseVar2

    skip_out_of_scope = skip_symbolics_and_breakends
    if source.suffix == ".pgen":
        if ploidy != 2:
            raise ValueError(
                "PGEN is diploid; --ploidy is only meaningful for VCF/BCF sources."
            )
        dropped = SparseVar2.from_pgen(
            out,
            source,
            reference,
            no_reference=no_reference,
            skip_out_of_scope=skip_out_of_scope,
            chunk_size=chunk_size,
            threads=threads,
            overwrite=overwrite,
            long_allele_capacity=long_allele_capacity,
            check_ref=check_ref,
        )
    else:
        dropped = SparseVar2.from_vcf(
            out,
            source,
            reference,
            no_reference=no_reference,
            skip_out_of_scope=skip_out_of_scope,
            ploidy=ploidy,
            chunk_size=chunk_size if chunk_size is not None else 25_000,
            threads=threads,
            overwrite=overwrite,
            long_allele_capacity=long_allele_capacity,
            check_ref=check_ref,
        )
    if skip_out_of_scope:
        print(f"Dropped {dropped} out-of-scope (symbolic/breakend) ALT alleles.")


@write.command(name="svar1")
def write_svar1(
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
    Convert a VCF or PGEN file to a SVAR 1.0 file.

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
        (``<DEL>``, ``<INS>``, etc.) per VCF 4.x.
    no_breakend
        If set, drop records whose ALT contains a breakend (BND).
    haploid
        If set, OR-collapse the ploidy axis into a single haploid call per
        sample and record ``ploidy=1`` in the output metadata.
    """
    from genoray import PGEN, VCF, Filter, SparseVar, exprs
    from genoray._utils import variant_file_type

    file_type = variant_file_type(source)

    if dosages is None:
        with_dosages = False
    else:
        with_dosages = True

    if threads is None:
        threads = -1

    pl_terms: list[pl.Expr] = []
    record_preds: list[Callable[[list[str]], bool]] = []
    if no_symbolic:
        pl_terms.append(~exprs.is_symbolic)
        record_preds.append(exprs._record_is_symbolic)
    if no_breakend:
        pl_terms.append(~exprs.is_breakend)
        record_preds.append(exprs._record_is_breakend)

    vcf_filter: Filter | None
    if pl_terms:
        from functools import reduce
        from operator import and_

        pl_filter = reduce(and_, pl_terms)

        def record_filter(
            rec: Any,
            _preds: tuple[Callable[[list[str]], bool], ...] = tuple(record_preds),
        ) -> bool:
            return not any(pred(rec.ALT) for pred in _preds)

        vcf_filter = Filter(record=record_filter, expr=pl_filter)
    else:
        pl_filter = None
        vcf_filter = None

    if file_type == "vcf":
        if dosages is not None and Path(dosages).exists():
            raise ValueError(
                "The `dosages` argument appears to be a path to an existing file, but VCF requires a FORMAT field name."
            )

        vcf = VCF(
            source,
            dosage_field=dosages,
            filter=vcf_filter,
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


view = App(
    name="view",
    help="Write a region/sample subset of an SVAR2 store (SVAR1 via `view svar1`).",
)
app.command(view)


@view.default
def view_svar2(
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
    reference: Annotated[Path | None, Parameter(name="--reference")] = None,
    merge_overlapping: bool = False,
    regions_overlap: Literal["pos", "record", "variant"] = "pos",
    reroute: bool | None = None,
    overwrite: bool = False,
    threads: Annotated[int | None, Parameter(name=["--threads", "-@"])] = None,
    progress: bool = False,
) -> None:
    """Write a region/sample subset of an SVAR2 store.

    At least one of --regions/--regions-file or --samples/--samples-file is
    required. The omitted side defaults to "all" (all samples or all variants,
    the latter meaning one region per contig spanning that contig).

    Parameters
    ----------
    source
        Path to the input SVAR2 directory.
    out
        Path to the output SVAR2 directory.
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
        Optional INFO/FORMAT fields to carry over. Defaults to unset, meaning
        no fields are carried through (genotypes only) — this always
        succeeds, even on a store that has INFO/FORMAT fields. Both
        ``reroute`` and ``--no-reroute`` carry fields through.
    reference
        Optional path to a reference FASTA. Recomputes the ``mutcat``
        mutational-signature sidecar for the output on both ``reroute`` and
        ``--no-reroute``.
    merge_overlapping
        If set, silently merge overlapping regions instead of raising.
    regions_overlap
        How variants are matched to regions: ``pos`` (default; match if the
        variant POS falls in the range), ``record`` (match by VCF record extent),
        or ``variant`` (match by full variant extent including ILEN).
    reroute
        Whether to rerun the var_key/dense routing cost model on the subset.
        ``--reroute`` forces it on (size-optimal). ``--no-reroute`` forces it
        off: directly slices each variant's existing on-disk representation
        -- representation-preserving regardless of the subset's
        sample/carrier counts -- recommended for somatic/rare-variant subsets
        or memory-constrained runs. Omitting both flags (the default) is
        ``"auto"``: resolves to ``--no-reroute``'s behavior when any FORMAT
        field is carried, ``--reroute``'s otherwise -- a dense variant
        re-routed to var_key has no slot for a non-carrier sample's FORMAT
        value, so ``"auto"`` prefers fidelity whenever FORMAT is in play and
        takes the size-optimal re-route otherwise.
    overwrite
        Overwrite the output directory if it already exists.
    threads
        Number of threads. Defaults to all available CPUs.
    progress
        If set, show a phase-level progress bar while writing the view.
    """
    import polars as pl

    from genoray import SparseVar2

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

    sv = SparseVar2(source)

    # Resolve regions arg
    if regions is not None:
        regions_arg: pl.DataFrame | Path = parse_regions_arg(regions)
    elif regions_file is not None:
        regions_arg = regions_file
    else:
        # "all variants" — SVAR2 has no contig-length metadata, so span each
        # contig with [0, i32::MAX) (POS is i32; regions_overlap="pos" keeps
        # every variant on that contig).
        regions_arg = pl.DataFrame(
            {"chrom": sv.contigs},
            schema={"chrom": pl.Utf8},
        ).select(
            chrom=pl.col("chrom"),
            start=pl.lit(0, dtype=pl.Int32),
            end=pl.lit(2**31 - 1, dtype=pl.Int32),
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
        reference=reference,
        merge_overlapping=merge_overlapping,
        regions_overlap=regions_overlap,
        reroute="auto" if reroute is None else reroute,
        overwrite=overwrite,
        threads=threads,
        progress=progress,
    )


@view.command(name="svar1")
def view_svar1(
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


@app.command
def concat(
    out: Path,
    sources: list[Path],
    *,
    mode: Mode = "copy",
    overwrite: bool = False,
) -> None:
    """Concatenate disjoint-contig SVAR2 stores into one."""
    from genoray import SparseVar2

    SparseVar2.concat(out, sources, mode=mode, overwrite=overwrite)


@app.command
def split(
    source: Path,
    out: Path,
    *,
    contigs: Annotated[str | None, Parameter(name=["--contigs", "-c"])] = None,
    mode: Mode = "copy",
    overwrite: bool = False,
) -> None:
    """Split an SVAR2 store by contig.

    With --contigs: subset into one store at OUT. Without: explode into
    OUT/{contig}.svar2.
    """
    from genoray import SparseVar2

    sv = SparseVar2(source)
    if contigs is not None:
        sv.subset_contigs(
            out,
            [c for c in contigs.split(",") if c],
            mode=mode,
            overwrite=overwrite,
        )
    else:
        sv.split_by_contig(out, mode=mode, overwrite=overwrite)


if __name__ == "__main__":
    app()
