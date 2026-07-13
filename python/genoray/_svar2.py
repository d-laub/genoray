from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from genoray._svar2_fields import FormatField, InfoField


def _resolve_vcf_sources(sources: "str | Path | Sequence[str | Path]") -> list[Path]:
    """Resolve `sources` (as accepted by :meth:`SparseVar2.from_vcf_list`) to a
    concrete, ordered list of VCF/BCF paths.

    `sources` may be:

    - a `Sequence` (list/tuple, not `str`/`Path`) of file paths, taken as-is
      and in the given order.
    - a single directory `Path`: every `*.vcf.gz` then every `*.bcf` directly
      inside it (non-recursive), each group name-sorted (`natsort`).
    - a single file `Path` ending in `.vcf.gz` or `.bcf`: that one file.
    - any other single file `Path`: treated as a manifest -- one path per
      line, blank lines and `#`-prefixed comment lines skipped, relative
      entries resolved against the manifest's parent directory.
    """
    if isinstance(sources, (str, Path)):
        path = Path(sources)
        if path.is_dir():
            paths = natsorted(path.glob("*.vcf.gz")) + natsorted(path.glob("*.bcf"))
        elif path.name.endswith(".vcf.gz") or path.suffix == ".bcf":
            paths = [path]
        else:
            paths = []
            for line in path.read_text().splitlines():
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                entry_path = Path(entry)
                if not entry_path.is_absolute():
                    entry_path = path.parent / entry_path
                paths.append(entry_path)
    else:
        paths = [Path(s) for s in sources]

    if not paths:
        raise ValueError(f"no VCF/BCF files found in {sources}")
    return paths


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

    @classmethod
    def from_vcf_list(
        cls,
        out: str | Path,
        sources: "str | Path | Sequence[str | Path]",
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
        info_fields: "Sequence[str | InfoField] | None" = None,
        format_fields: "Sequence[str | FormatField] | None" = None,
    ) -> int:
        """Build one SVAR2 store from many **single-sample** VCFs/BCFs via a
        native k-way merge (no `bcftools merge`, no intermediate multi-sample
        VCF).

        Each file in `sources` must have exactly one sample column; that
        sample becomes one sample in the resulting store, named after its
        VCF header sample name (duplicates across files are rejected). A
        site present in some input files but absent from another is filled
        **hom-ref (`0`)** for the samples that lack it. An in-file `./.`
        (missing) call is *not* separately preserved once merged: SVAR2's
        sparse layout stores only ALT-carrying entries, so a missing hap and
        a hom-ref hap both produce zero entries and are indistinguishable
        through `decode` or `region_counts`. (The `-1` missing sentinel is a
        dense `genoray.VCF`/`genoray.PGEN` convention; it is not part of
        SVAR2's decode.)
        The merge is join-on-atom: files are merged one contig at a time by
        walking each file's already-sorted record stream in lockstep, so a
        variant is one shared row in the output store iff its normalized
        (pos, ref, alt) atom matches exactly across files, not merely its
        position.

        `sources` accepts three forms (resolved by module-level
        `_resolve_vcf_sources`):

        - a `Sequence` of paths -- explicit, in the given order.
        - a single directory `Path` -- every `*.vcf.gz` then every `*.bcf`
          directly inside it (non-recursive), each group name-sorted.
        - a single file `Path` -- if it ends in `.vcf.gz`/`.bcf`, that one
          file; otherwise treated as a manifest (one path per line, blank
          and `#`-comment lines skipped, relative entries resolved against
          the manifest's directory).

        As with :meth:`from_vcf`, each input VCF's records must already be
        position-sorted per contig -- an unsorted file silently corrupts the
        k-way merge.

        `reference` is **required** in the current implementation
        (`no_reference=True` is not yet supported here, unlike `from_vcf`).

        `info_fields`/`format_fields`: same declaration API as :meth:`from_vcf`
        (resolved against the FIRST file in `sources`' header). INFO fields
        merge **first-carrier-wins**: when a site is shared across files, the
        value comes from the lowest-numbered (earliest in `sources` order)
        file that carries the atom, not the last or the max. FORMAT fields
        remain per-sample, exactly as in `from_vcf`: each sample gets its own
        file's value, and a sample that doesn't carry the atom gets the
        field's default.

        Returns the number of out-of-scope (symbolic/breakend) ALTs dropped
        (0 unless `skip_out_of_scope`).
        """
        from cyvcf2 import VCF as _CyVCF

        out = Path(out)

        if (reference is None) == (not no_reference):
            raise ValueError(
                "provide exactly one of `reference` (a FASTA path) or "
                "`no_reference=True`"
            )
        if signatures and no_reference:
            raise ValueError("signatures=True requires a reference (not no_reference).")
        if no_reference:
            raise NotImplementedError(
                "no_reference is not yet supported by from_vcf_list; pass a reference"
            )
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )

        paths = _resolve_vcf_sources(sources)
        out.parent.mkdir(parents=True, exist_ok=True)

        samples: list[str] = []
        contig_set: set[str] = set()
        for path in paths:
            _ensure_bgzipped(path)
            _ensure_index(path)
            v = _CyVCF(str(path))
            if len(v.samples) != 1:
                raise ValueError(
                    f"{path} is not single-sample (has {len(v.samples)} samples)"
                )
            samples.append(v.samples[0])
            contig_set.update(c for c in v.seqnames if next(v(c), None) is not None)

        dupes = sorted({s for s in samples if samples.count(s) > 1})
        if dupes:
            raise ValueError(f"duplicate sample names across inputs: {dupes}")

        contigs = natsorted(contig_set)
        if not contigs:
            raise ValueError("No variants found in any input.")

        # Field specs are resolved against the FIRST file's header -- every
        # input is single-sample and expected to share a header schema (same
        # assumption the reference/samples handling already makes).
        flds = _resolve_fields(str(paths[0]), info_fields, format_fields)
        info = [t for t in flds if t[1] == "info"]
        format_ = [t for t in flds if t[1] == "format"]

        return _core.run_vcf_list_conversion_pipeline(
            [str(p) for p in paths],
            None if no_reference else str(reference),
            contigs,
            str(out),
            samples,
            chunk_size,
            ploidy,
            threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            info,
            format_,
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
