from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from natsort import natsorted

import genoray._core as _core
from genoray._svar2_batch import _BatchQueryMixin
from genoray._svar2_decode import _DecodeMixin
from genoray._svar2_fields import (
    StoredField,
    _load_field_manifest,
    _resolve_fields,
    _resolve_read_fields,
)
from genoray._svar2_mutcat import _MutcatMixin
from genoray._svar2_ops import Mode, _assert_concat_compatible, _load_meta, _write_store

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from genoray._svar2_fields import FormatField, InfoField


def _ensure_bgzipped(source: Path) -> None:
    """Reject a plain (uncompressed) VCF — it can't be tabix/csi-indexed."""
    is_bcf = source.suffix == ".bcf"
    is_vcfgz = source.name.endswith(".vcf.gz")
    if not (is_bcf or is_vcfgz):
        raise ValueError(
            f"{source} must be a BCF (.bcf) or bgzipped VCF (.vcf.gz); bgzip it first."
        )


def _ensure_index(source: Path) -> None:
    """Build a .csi index next to `source` if it has no .csi/.tbi index."""
    csi = source.with_name(source.name + ".csi")
    tbi = source.with_name(source.name + ".tbi")
    if csi.exists() or tbi.exists():
        return
    _core.index_vcf(str(source))


class SparseVar2(_BatchQueryMixin, _DecodeMixin, _MutcatMixin):
    """Reader for a finished SVAR2 store (M6a skeleton).

    Loads the top-level ``meta.json`` and opens one native
    :class:`genoray._core.PyContigReader` per contig. Query methods land in M6b
    (raw two-channel result) and M6c (decoded ``seqpro.rag.Ragged``).
    """

    def __init__(
        self, path: str | Path, *, fields: "Sequence[str] | None" = None
    ) -> None:
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        self.format_version: int = meta["format_version"]
        self.available_samples: list[str] = list(meta["samples"])
        self.contigs: list[str] = list(meta["contigs"])
        self.ploidy: int = meta["ploidy"]
        self.available_fields: dict[str, StoredField] = _load_field_manifest(meta)
        #: The fields this reader decodes. Empty unless opted into via
        #: ``fields=`` / :meth:`with_fields` — decoding a field costs extra I/O.
        self._fields: list[StoredField] = _resolve_read_fields(
            fields, self.available_fields
        )
        self._readers: dict[str, _core.PyContigReader] = {
            contig: _core.PyContigReader(
                str(self.path), contig, len(self.available_samples), self.ploidy
            )
            for contig in self.contigs
        }

    @property
    def n_samples(self) -> int:
        return len(self.available_samples)

    def with_fields(self, fields: "Sequence[str]") -> "SparseVar2":
        """A new reader over the same store that also decodes ``fields``.

        Keys are those of :attr:`available_fields`: the bare field name when it
        is unique across INFO/FORMAT, else bcftools-style ``INFO/DP`` /
        ``FORMAT/DP``.
        """
        return SparseVar2(self.path, fields=fields)

    def subset_contigs(
        self,
        output: str | Path,
        contigs: str | Sequence[str],
        *,
        mode: Mode = "copy",
        overwrite: bool = False,
    ) -> None:
        """Write a new SVAR2 store containing only `contigs` (metadata + file copy)."""
        output = Path(output)
        wanted = [contigs] if isinstance(contigs, str) else list(contigs)
        missing = [c for c in wanted if c not in self.contigs]
        if missing:
            raise ValueError(f"contigs not in store: {missing}")
        if output.resolve() == self.path.resolve():
            raise ValueError("cannot write a subset in place (output == source)")
        kept = [c for c in self.contigs if c in set(wanted)]  # preserve source order
        meta = _load_meta(self.path)
        meta["contigs"] = kept
        _write_store(output, {c: self.path for c in kept}, meta, mode, overwrite)

    def split_by_contig(
        self, out_dir: str | Path, *, mode: Mode = "copy", overwrite: bool = False
    ) -> list[Path]:
        """Explode into one single-contig store per contig at out_dir/{contig}.svar2."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for c in self.contigs:
            p = out_dir / f"{c}.svar2"
            self.subset_contigs(p, [c], mode=mode, overwrite=overwrite)
            paths.append(p)
        return paths

    @classmethod
    def concat(
        cls,
        output: str | Path,
        sources: Sequence[str | Path | "SparseVar2"],
        *,
        mode: Mode = "copy",
        overwrite: bool = False,
    ) -> None:
        """Concatenate disjoint-contig SVAR2 stores (identical samples/ploidy/fields) into one."""
        paths = [Path(s.path if isinstance(s, SparseVar2) else s) for s in sources]
        if not paths:
            raise ValueError("concat requires at least one source")
        metas = [_load_meta(p) for p in paths]
        _assert_concat_compatible(metas)
        contig_sources: dict[str, Path] = {}
        for p, m in zip(paths, metas):
            for c in m["contigs"]:
                if c in contig_sources:
                    raise ValueError(
                        f"contig {c!r} appears in multiple sources; concat requires disjoint contigs"
                    )
                contig_sources[c] = p
        merged_contigs = natsorted(contig_sources)
        meta = dict(metas[0])
        meta["contigs"] = merged_contigs
        _write_store(
            Path(output),
            {c: contig_sources[c] for c in merged_contigs},
            meta,
            mode,
            overwrite,
        )

    def write_view(
        self,
        regions: "str | tuple[str, int, int] | Path | object",
        samples: "str | Sequence[str] | Path",
        output: str | Path,
        fields: "Sequence[str] | None" = None,
        reference: str | Path | None = None,
        *,
        merge_overlapping: bool = False,
        regions_overlap: "Literal['pos', 'record', 'variant']" = "pos",
        reroute: "Literal['auto', True]" = "auto",
        overwrite: bool = False,
        threads: int | None = None,
        progress: bool = False,
    ) -> None:
        """Write a region/sample subset of this store to `output` by re-running
        the ordinary conversion pipeline over the finished store's own records
        (rather than re-reading the original VCF/PGEN).

        `regions`/`samples` accept the same inputs as the query methods (region
        string, `(chrom, start, end)` tuple, BED path, or a samples sequence /
        path to a sample list). `regions_overlap` controls how a variant's span
        is matched against the requested regions (`"pos"`/`"record"`/`"variant"`
        — see `_normalize_regions`/`_resolve_kept_rows`); `merge_overlapping`
        silently merges overlapping input regions instead of raising.

        `fields` defaults to `None`, meaning *no* fields are carried through
        (genotypes only) — this always succeeds, even on a store that has
        INFO/FORMAT fields (`available_fields` non-empty). Field
        carry-through itself is not yet implemented on this path: passing
        any non-empty `fields` (any field other than `"mutcat"`) makes the
        backend raise `ValueError`. `"mutcat"` is always excluded from
        `fields` — pass `reference=` to recompute it instead of copying
        (unimplemented; currently only affects `mutcat` filtering, since the
        backend does not yet recompute anything from `reference`).

        `reroute` selects the output representation: `"auto"` (equivalent to
        `True`) reruns the full var_key/dense routing cost model on the
        subset, which is the only mode currently implemented.
        `reroute=False` (byte-preserving passthrough of each variant's
        original representation) is NOT implemented and raises
        `NotImplementedError`.

        `progress` is accepted for interface parity with other long-running
        entry points but is currently a no-op (no progress bar is shown).

        Raises `FileExistsError` if `output` exists and `overwrite=False`, and
        `ValueError` if `output` resolves to this store's own path (writing a
        view in place is not supported).
        """
        from genoray._contigs import ContigNormalizer
        from genoray._svar._regions import (
            _normalize_regions,
            _normalize_samples,
            _validate_fields,
        )

        output = Path(output)
        if reroute == "auto":
            reroute = True
        if reroute is not True:
            raise NotImplementedError(
                "reroute=False (preserve source representation) is not "
                "implemented; only reroute=True is supported (see the "
                "concat/split/write-view design doc)."
            )
        if fields is not None and "mutcat" in fields and reference is None:
            raise ValueError(
                "'mutcat' cannot be copied through write_view; pass "
                "reference= to recompute it."
            )
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} exists; pass overwrite=True")
        if output.resolve() == self.path.resolve():
            raise ValueError(
                "output resolves to the same path as the source; cannot "
                "write a view in place"
            )
        cnorm = ContigNormalizer(self.contigs)
        regions_df = _normalize_regions(regions, cnorm)
        caller_samples = _normalize_samples(samples, self.available_samples)
        if not caller_samples:
            raise ValueError("write_view requires at least one sample")
        if fields is None:
            # `_validate_fields(None, available)` returns *all* available
            # fields (its semantics for the read path), not "none" -- the
            # write_view default must mean "genotypes only" so a plain
            # `write_view(...)` call succeeds on any store, including one
            # with INFO/FORMAT fields.
            fields_to_write: list[str] = []
        else:
            fields_to_write = [
                f
                for f in _validate_fields(
                    fields, cast("dict[str, Any]", self.available_fields)
                )
                if f != "mutcat"
            ]
        region_tuples = [
            (row["chrom"], int(row["start"]), int(row["end"]))
            for row in regions_df.iter_rows(named=True)
        ]
        _core.run_view_pipeline(
            str(self.path),
            str(output),
            self.contigs,
            caller_samples,
            region_tuples,
            regions_overlap,
            merge_overlapping,
            fields_to_write,
            str(reference) if reference is not None else None,
            threads,
            overwrite,
        )

    @classmethod
    def from_vcf(
        cls,
        out: str | Path,
        source: str | Path,
        reference: str | Path | None = None,
        *,
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        ploidy: int = 2,
        chunk_size: int = 25_000,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
        signatures: bool = False,
        info_fields: Sequence[str | InfoField] | None = None,
        format_fields: Sequence[str | FormatField] | None = None,
    ) -> int:
        """Convert a bgzipped VCF or BCF to an SVAR2 store.

        Exactly one of `reference` or `no_reference=True` is required. With a
        reference, indels are validated against and left-aligned to the FASTA;
        with `no_reference`, validation and left-alignment are skipped and the
        input is trusted to be already normalized. Returns the number of
        out-of-scope (symbolic/breakend) ALTs dropped (0 unless
        `skip_out_of_scope`).

        signatures: if True, classify SBS96/ID83 codes during the write and
        store the mutcat sidecar (factored into the dense/var_key cost model).
        Requires a reference; raises if `no_reference=True`.

        info_fields, format_fields: scalar-numeric (Integer/Float, and Flag for
        INFO) header fields to carry through to the SVAR2 store. Each entry is
        either a bare field name (dtype auto-narrowed from the header, no
        default fill) or an :class:`InfoField`/:class:`FormatField` spec
        (explicit `dtype`/`default`). `default` fills VCF-missing entries;
        otherwise a reserved sentinel/NaN is written. FORMAT fields are
        genotype-aligned: non-carrier values are dropped for var_key-routed
        variants.
        """
        from cyvcf2 import VCF as _CyVCF

        out = Path(out)
        source = Path(source)

        if (reference is None) == (not no_reference):
            raise ValueError(
                "provide exactly one of `reference` (a FASTA path) or "
                "`no_reference=True`"
            )
        if signatures and no_reference:
            raise ValueError("signatures=True requires a reference (not no_reference).")
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.parent.mkdir(parents=True, exist_ok=True)

        _ensure_bgzipped(source)
        _ensure_index(source)

        v = _CyVCF(str(source))
        samples = list(v.samples)
        contigs = [c for c in natsorted(v.seqnames) if next(v(c), None) is not None]
        if not contigs:
            raise ValueError(f"No variants found in {source}.")

        reference_path = None if no_reference else str(reference)
        flds = _resolve_fields(str(source), info_fields, format_fields)
        info = [t for t in flds if t[1] == "info"]
        format_ = [t for t in flds if t[1] == "format"]
        return _core.run_conversion_pipeline(
            str(source),
            reference_path,
            contigs,
            str(out),
            samples,
            chunk_size,
            ploidy,
            threads,  # max_threads; None => auto
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            info,
            format_,
        )

    @classmethod
    def from_pgen(
        cls,
        out: str | Path,
        source: str | Path,
        reference: str | Path | None = None,
        *,
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        chunk_size: int | None = None,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
        signatures: bool = False,
    ) -> int:
        """Convert a PLINK2 PGEN to an SVAR2 store.

        Genotypes are read through the ``pgenlib`` package; variant metadata comes
        from the sibling ``.pvar``/``.pvar.zst`` and sample names from the ``.psam``.

        Exactly one of `reference` or `no_reference=True` is required, with the same
        meaning as :meth:`from_vcf`: with a reference, indels are validated against
        and left-aligned to the FASTA; with `no_reference`, both are skipped and the
        input is trusted to be already normalized. Returns the number of
        out-of-scope (symbolic/breakend) ALTs dropped (0 unless `skip_out_of_scope`).

        PGEN is diploid, so there is no `ploidy` parameter.

        chunk_size: variants per conversion chunk. Defaults to a value derived from
        a memory budget, since a packed dense chunk costs
        ``chunk_size * n_samples * 2 / 8`` bytes.

        Not supported (and silently ignored rather than errored, where noted):

        - **Dosages.** SVAR2 stores no dosages; a ``.pgen`` dosage track is ignored
          and hardcalls are read as usual.
        - **INFO/FORMAT fields.** PGEN has no FORMAT; ``.pvar`` INFO extraction is
          not implemented.
        - **Sample subsetting.** All samples in the ``.psam`` are converted, matching
          :meth:`from_vcf`.

        Haplotype resolution for *unphased* heterozygotes follows the allele-code
        order ``pgenlib`` returns — the same caveat :meth:`from_vcf` carries for
        unphased ``GT``.
        """
        from genoray._pgen import _read_psam

        out = Path(out)
        source = Path(source)

        if (reference is None) == (not no_reference):
            raise ValueError(
                "provide exactly one of `reference` (a FASTA path) or "
                "`no_reference=True`"
            )
        if signatures and no_reference:
            raise ValueError("signatures=True requires a reference (not no_reference).")
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        if source.suffix != ".pgen":
            raise ValueError(f"Expected a .pgen file, got {source}")
        if not source.exists():
            raise FileNotFoundError(source)

        pvar = _find_pvar(source)
        psam = source.with_suffix(".psam")
        if not psam.exists():
            raise FileNotFoundError(psam)
        out.parent.mkdir(parents=True, exist_ok=True)

        samples = cast("list[str]", _read_psam(psam).tolist())
        n_samples = len(samples)
        if n_samples == 0:
            raise ValueError(f"No samples found in {psam}.")

        contigs, ranges, allele_idx_offsets = _pvar_contig_ranges(pvar)
        if not contigs:
            raise ValueError(f"No variants found in {pvar}.")

        if chunk_size is None:
            chunk_size = _auto_chunk_size(n_samples)

        import pgenlib

        # One reader per contig: readers seek independently, so concurrent contigs
        # must not share one. `allele_idx_offsets` is required (not just used) once
        # any variant in the file is multiallelic -- it is a file-wide array, so
        # every contig's reader is constructed with the same one.
        readers = [
            pgenlib.PgenReader(
                bytes(source), n_samples, allele_idx_offsets=allele_idx_offsets
            )
            for _ in contigs
        ]

        return _core.run_pgen_conversion_pipeline(
            str(source),
            str(pvar),
            None if no_reference else str(reference),
            contigs,
            ranges,
            str(out),
            samples,
            chunk_size,
            threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            readers,
        )


def _find_pvar(pgen: Path) -> Path:
    """Locate the `.pvar` / `.pvar.zst` sibling of `pgen`."""
    for suffix in (".pvar", ".pvar.zst"):
        cand = pgen.with_suffix(suffix)
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"No .pvar or .pvar.zst found next to {pgen}. "
        f"Looked for {pgen.with_suffix('.pvar')} and {pgen.with_suffix('.pvar.zst')}."
    )


def _pvar_contig_ranges(
    pvar: Path,
) -> tuple[list[str], list[tuple[int, int]], NDArray[np.uintp]]:
    """Contigs in `.pvar` file order, each one's half-open `[lo, hi)` variant
    index range, and the file-wide `allele_idx_offsets` array `pgenlib.PgenReader`
    requires once any variant in the file is multiallelic.

    `allele_idx_offsets` has length `n_variants + 1`: `offsets[0] = 0` and
    `offsets[i+1] = offsets[i] + 1 + n_alts(i)`, where `n_alts(i)` is the number of
    comma-separated ALT tokens of variant `i` -- including a variant whose ALT is
    the bare `.` sentinel (no ALT observed), which still counts as 1 token/2 total
    alleles, matching `pgenlib`'s on-disk model (see the comment at its
    computation below). It is a single, file-wide array -- every per-contig reader
    is constructed with the same one, not a per-contig slice.

    Raises if a contig's variants are not contiguous -- SVAR2 converts one contig at
    a time from a variant index range, which requires the `.pvar` to be grouped by
    contig (as plink2 always writes it).
    """
    import polars as pl

    from genoray._pgen import _scan_pvar

    df = _scan_pvar(pvar).select("#CHROM", "ALT").with_row_index("vidx").collect()

    # `_scan_pvar` opens the .pvar with `null_values="."`, so a monomorphic
    # variant's ALT ('.' -- no alternate allele observed) reads as a polars
    # null, and `.list.len()` on a null is null. Left un-guarded, `.to_numpy()`
    # would upcast to float64 (NaN for that slot) and
    # `np.cumsum(..., out=<uintp>)` would silently reinterpret that NaN as a
    # huge garbage integer -- for that variant *and every one after it*, since
    # cumsum is prefix-summed.
    #
    # The count to fill in is 1, not 0: `pgenlib`'s on-disk `allele_idx_offsets`
    # model reserves a minimum of 2 allele slots (REF + one ALT slot) per
    # variant, even when plink2 has no observed ALT to report -- the ALT
    # column's bare '.' is a *display* convention for "no ALT was observed",
    # not "no ALT slot exists". Verified directly against `pgenlib`: building
    # `allele_idx_offsets` with a 0-count (1 total allele) for a '.'-ALT
    # variant makes `PgenReader.read_alleles_range` segfault on every
    # multiallelic variant after it (offsets one short of what the file
    # actually stores); a 1-count (2 total alleles, matching every other
    # biallelic row) reads correctly and matches the VCF-derived genotypes.
    # (Rust's `PvarReader` separately empties `alts` for a '.' ALT -- that's
    # about which alleles the *conversion spine* atomizes, an independent
    # question from what `pgenlib` needs to step through the file.)
    n_alts_col = df["ALT"].str.split(",").list.len().fill_null(1)
    if n_alts_col.null_count():  # pragma: no cover - defensive, should be unreachable
        raise ValueError(
            f"Could not determine the ALT-allele count for every variant in {pvar}; "
            "expected only null ALTs (from the '.' monomorphic sentinel) to be null "
            "here, but some remained null after filling."
        )
    n_alts = n_alts_col.to_numpy()
    if n_alts.dtype.kind not in "iu" or (n_alts < 0).any():
        raise ValueError(
            f"Computed a negative or non-integer ALT-allele count while parsing {pvar} "
            f"(dtype={n_alts.dtype}); this indicates a malformed ALT column."
        )
    allele_idx_offsets = np.empty(len(n_alts) + 1, dtype=np.uintp)
    allele_idx_offsets[0] = 0
    np.cumsum(n_alts + 1, out=allele_idx_offsets[1:])

    grouped = (
        df.lazy()
        .group_by("#CHROM", maintain_order=True)
        .agg(
            pl.col("vidx").min().alias("lo"),
            pl.col("vidx").max().alias("hi"),
            pl.len().alias("n"),
        )
        .collect()
    )
    contigs: list[str] = []
    ranges: list[tuple[int, int]] = []
    for chrom, lo, hi, n in grouped.iter_rows():
        if hi - lo + 1 != n:
            raise ValueError(
                f"Contig {chrom!r} is not contiguous in {pvar} "
                f"(spans indices {lo}..{hi} but has {n} variants). "
                "SVAR2 requires a .pvar grouped by contig."
            )
        contigs.append(str(chrom))
        ranges.append((int(lo), int(hi) + 1))
    return contigs, ranges, allele_idx_offsets


# Target byte size of one packed dense chunk (chunk_size * n_samples * ploidy / 8).
_DENSE_CHUNK_TARGET_BYTES = 256 * 1024 * 1024


def _auto_chunk_size(n_samples: int, ploidy: int = 2) -> int:
    """Variants per chunk, derived from a memory budget rather than a fixed count.

    A packed dense chunk costs `chunk_size * n_samples * ploidy / 8` bytes, so a
    fixed 25k chunk that is fine at 200 samples is not at 500k.
    """
    bits_per_variant = n_samples * ploidy
    by_budget = (_DENSE_CHUNK_TARGET_BYTES * 8) // max(bits_per_variant, 1)
    return max(1024, min(25_000, int(by_budget)))
