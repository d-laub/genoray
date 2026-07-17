from __future__ import annotations

import json
from collections.abc import Sequence as AbcSequence
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import polars as pl
from natsort import natsorted

import genoray._core as _core
from genoray._contigs import _MITO_ALIASES
from genoray._svar2_batch import _BatchQueryMixin
from genoray._svar2_decode import _DecodeMixin
from genoray._svar2_fields import (
    _META_DTYPE,
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


def _resolve_vcf_sources(sources: "str | Path | Sequence[str | Path]") -> list[Path]:
    """Resolve `sources` (as accepted by :meth:`SparseVar2.from_vcf_list`) to a
    concrete, ordered list of VCF/BCF paths.

    `sources` may be:

    - a `Sequence` (list/tuple, not `str`/`Path`) of file paths, taken as-is
      and in the given order.
    - a single directory `Path`: every `*.vcf.gz`/`*.vcf.bgz` then every
      `*.bcf` directly inside it (non-recursive), each group name-sorted
      (`natsort`).
    - a single file `Path` ending in `.vcf.gz`, `.vcf.bgz`, or `.bcf`: that
      one file.
    - any other single file `Path`: treated as a manifest -- one path per
      line, blank lines and `#`-prefixed comment lines skipped, relative
      entries resolved against the manifest's parent directory.
    """
    if isinstance(sources, (str, Path)):
        path = Path(sources)
        if path.is_dir():
            paths = natsorted(
                [*path.glob("*.vcf.gz"), *path.glob("*.vcf.bgz")]
            ) + natsorted(path.glob("*.bcf"))
        elif _is_bgzipped_vcf(path) or path.suffix == ".bcf":
            paths = [path]
        elif path.suffix == ".vcf":
            # Without this, a bare `.vcf` single-path `sources` falls into the
            # manifest branch below: every `##`/`#CHROM` header line reads as
            # a `#`-comment (skipped), then every data line is treated as a
            # *path* -- producing a bewildering downstream error far from the
            # real problem. `_ensure_bgzipped` always raises here (this
            # branch is reached only when the suffix is exactly `.vcf`, which
            # is neither `.bcf` nor a recognized bgzipped VCF suffix); the
            # `paths = []` afterward is unreachable but keeps this branch's
            # static type honest.
            _ensure_bgzipped(path)
            paths = cast("list[Path]", [])
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


def _is_bgzipped_vcf(source: Path) -> bool:
    return source.name.endswith((".vcf.gz", ".vcf.bgz"))


def _ensure_bgzipped(source: Path) -> None:
    """Reject a plain (uncompressed) VCF — it can't be tabix/csi-indexed."""
    is_bcf = source.suffix == ".bcf"
    if not (is_bcf or _is_bgzipped_vcf(source)):
        raise ValueError(
            f"{source} must be a BCF (.bcf) or bgzipped VCF "
            "(.vcf.gz/.vcf.bgz); bgzip it first."
        )


def _ensure_index(source: Path) -> None:
    """Build a .csi index next to `source` if it has no .csi/.tbi index."""
    csi = source.with_name(source.name + ".csi")
    tbi = source.with_name(source.name + ".tbi")
    if csi.exists() or tbi.exists():
        return
    _core.index_vcf(str(source))


def _is_region_triplet(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and isinstance(value[2], int)
    )


def _normalize_svar2_regions(
    regions: "str | tuple[str, int, int] | PathLike | object | None",
    available_contigs: "Sequence[str]",
    *,
    merge_overlapping: bool,
) -> list[tuple[str, int, int]]:
    """Normalize caller regions to coalesced, backend-agnostic genomic intervals.

    This is intentionally a pre-conversion/pre-slice helper: it reuses the v1
    coordinate parser/sample conventions, then returns per-contig 0-based
    half-open intervals. It is mode-independent -- `regions_overlap` (which
    variants a region's POS/record/anchor-trimmed-extent must overlap) is
    applied downstream (Rust `query_window`/`parse_overlap_mode`), not here.
    """
    from genoray._contigs import ContigNormalizer
    from genoray._svar._regions import _normalize_regions

    if regions is None:
        return []

    cnorm = ContigNormalizer(available_contigs)

    if isinstance(regions, AbcSequence) and not isinstance(
        regions, (str, bytes, PathLike)
    ):
        if _is_region_triplet(regions):
            frames = [_normalize_regions(regions, cnorm)]
        else:
            frames = [_normalize_regions(r, cnorm) for r in regions]
        regions_df = pl.concat(frames) if frames else pl.DataFrame()
    else:
        regions_df = _normalize_regions(regions, cnorm)

    if regions_df.height == 0:
        raise ValueError("No requested regions match VCF contigs.")

    rows = regions_df.sort(["chrom", "start", "end"]).iter_rows(named=True)

    merged: list[tuple[str, int, int]] = []
    for row in rows:
        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])
        if start < 0:
            raise ValueError(f"region start must be >= 0; got {start} for {chrom}")
        if end <= start:
            raise ValueError(
                f"region end must be greater than start; got {chrom}:{start}-{end}"
            )

        if merged and merged[-1][0] == chrom and start <= merged[-1][2]:
            prev_chrom, prev_start, prev_end = merged[-1]
            if start < prev_end and not merge_overlapping:
                raise ValueError(
                    "regions overlap; pass merge_overlapping=True to dedupe"
                )
            merged[-1] = (prev_chrom, prev_start, max(prev_end, end))
        else:
            merged.append((chrom, start, end))

    return merged


def _reject_multiregion_variant(
    region_ranges: "list[tuple[str, int, int]]", regions_overlap: str
) -> None:
    """`regions_overlap="variant"` is only sound with at most ONE region per
    contig: a variant whose anchor-trimmed extent spans the gap between two
    disjoint regions on the same contig is handled inconsistently across the
    conversion readers (double-counted in the VCF path, dropped in the
    PGEN/SVAR1 paths). Until per-region ownership is unified, reject it up
    front. `pos`/`record` modes are unaffected (POS belongs to exactly one
    coalesced region)."""
    if regions_overlap != "variant":
        return
    from collections import Counter

    counts = Counter(chrom for chrom, _s, _e in region_ranges)
    multi = sorted(c for c, n in counts.items() if n > 1)
    if multi:
        raise ValueError(
            "regions_overlap='variant' currently supports at most one region "
            f"per contig, but these contigs have multiple: {multi}. Convert "
            "them in separate calls, coalesce with merge_overlapping=True if "
            "they are adjacent/overlapping, or use regions_overlap='pos'/'record'."
        )


def _canonical_contig_id(name: str) -> str:
    """The `chr`-prefix-insensitive, mito-alias-aware identity that
    :class:`genoray._contigs.ContigNormalizer` treats as "the same contig" --
    used here only to *detect* (not resolve) a cohort mixing `chr1`/`1`-style
    naming across input files, mirroring `ContigNormalizer`'s own rule
    (`contig_map`'s `chr`-stripping and `_MITO_ALIASES` grouping).
    """
    if name in _MITO_ALIASES:
        return "MT"
    return name[3:] if name.startswith("chr") else name


def _check_consistent_contig_naming(
    per_file_contigs: "list[tuple[Path, set[str]]]",
) -> None:
    """Raise if the cohort mixes naming schemes for the same logical contig
    across input files (e.g. file A calls it ``chr1``, file B calls it ``1``).

    `from_vcf_list`'s native k-way merge (`VcfListRecordSource`) matches
    contigs by exact per-file string, not through `ContigNormalizer` -- each
    contig name is opened literally against every file's own header. A cohort
    that mixes naming schemes would otherwise silently produce two separate
    entries in the union contig list (e.g. ``["1", "chr1", ...]``); each
    "contig" then converts using only the files whose spelling matches, with
    every other file's column filled hom-ref via the existing
    contig-missing-from-header skip -- with no error and no warning.
    `from_vcf` cannot have this bug structurally (one file, one scheme); this
    entry point can, because it merges N independently-produced files.
    """
    spellings: dict[str, dict[str, list[Path]]] = {}
    for path, contigs in per_file_contigs:
        for c in contigs:
            spellings.setdefault(_canonical_contig_id(c), {}).setdefault(c, []).append(
                path
            )

    conflicts = {k: v for k, v in spellings.items() if len(v) > 1}
    if not conflicts:
        return

    lines: list[str] = []
    for canonical, by_spelling in sorted(conflicts.items()):
        for spelling, paths in sorted(by_spelling.items()):
            names = ", ".join(str(p) for p in paths)
            lines.append(f"  {spelling!r} (contig {canonical!r}): {names}")
    raise ValueError(
        "from_vcf_list: inconsistent contig naming across input files -- the "
        "same contig is spelled differently in different files (e.g. 'chr1' "
        "vs '1'). The native k-way merge matches contigs by an exact "
        "per-file string, so a mixed cohort would silently be treated as if "
        "these were different contigs, filling every file that uses the "
        "OTHER spelling hom-ref with no error. Conflicting spellings:\n"
        + "\n".join(lines)
        + "\nNormalize contig names across all inputs first (e.g. `bcftools "
        "annotate --rename-chrs`) before calling from_vcf_list."
    )


# Rough per-file FD cost of `from_vcf_list` opening N single-sample VCFs
# concurrently: one for the data file, one for its .tbi/.csi index (htslib's
# `IndexedReader` holds both). `_FD_SAFETY_MARGIN` covers stdio, the output
# writer/monitor threads' files, and other process-wide overhead.
_FD_PER_INPUT_FILE = 2
_FD_SAFETY_MARGIN = 64


def _check_fd_budget(n_files: int) -> None:
    """Guard against FD exhaustion before opening `n_files` inputs concurrently
    (`from_vcf_list` holds one `IndexedReader` per file per contig -- see
    `VcfListRecordSource`), and raise an error that actually names the real
    problem.

    Without this, hitting a common soft `RLIMIT_NOFILE` (e.g. 1024) at large
    N surfaces as htslib's *"Failed to open VCF/BCF index ... (is there a
    .tbi or .csi file?)"* for some arbitrary file near the ceiling -- sending
    users to debug a nonexistent indexing problem instead of the real
    open-file limit. There is no batched/hierarchical merge to fall back on
    (explicit future work); the only fix at this entry point is raising the
    ulimit.
    """
    import resource

    needed = n_files * _FD_PER_INPUT_FILE + _FD_SAFETY_MARGIN
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= needed:
        return

    # Try to raise the soft limit toward the hard ceiling ourselves first --
    # cheap, and transparent whenever the hard limit already allows it.
    if hard == resource.RLIM_INFINITY or hard >= needed:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
            return
        except (ValueError, OSError):
            pass

    hard_str = "unlimited" if hard == resource.RLIM_INFINITY else str(hard)
    raise ValueError(
        f"from_vcf_list needs to open {n_files} input files concurrently "
        f"(~{needed} file descriptors, including index files and process "
        f"overhead), but the current open-file limit is {soft} (hard limit "
        f"{hard_str}). Raise it before retrying, e.g. `ulimit -n {needed}` "
        "(or higher, up to the hard limit) in the shell that launches this "
        "process. from_vcf_list does not batch the merge hierarchically to "
        "work around this ceiling -- see its docstring."
    )


def _validate_check_ref(check_ref: str) -> str:
    """Validate a `check_ref` mode string. Returns it unchanged on success."""
    if check_ref not in ("e", "x"):
        raise ValueError(
            f'check_ref must be "e" (error) or "x" (exclude), got {check_ref!r}'
        )
    return check_ref


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
        output = Path(output)
        if any(output.resolve() == p.resolve() for p in paths):
            raise ValueError(
                "cannot write concat output in place (output == one of the sources)"
            )
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
        reroute: "bool | Literal['auto']" = "auto",
        overwrite: bool = False,
        threads: int | None = None,
        progress: bool = False,
    ) -> None:
        """Write a region/sample subset of this store to `output`.

        `regions`/`samples` accept the same inputs as the query methods (region
        string, `(chrom, start, end)` tuple, BED path, or a samples sequence /
        path to a sample list). `regions_overlap` controls how a variant's span
        is matched against the requested regions (`"pos"`/`"record"`/`"variant"`
        — see `_normalize_regions`/`_resolve_kept_rows`); `merge_overlapping`
        silently merges overlapping input regions instead of raising.

        `fields` defaults to `None`, meaning *no* fields are carried through
        (genotypes only) — this always succeeds, even on a store that has
        INFO/FORMAT fields (`available_fields` non-empty). `"mutcat"` is
        always excluded from `fields` — pass `reference=` to recompute it
        instead of copying.

        Both `reroute=True` and `reroute=False` go through the same slicer
        backend, which carries `fields` and recomputes `mutcat` from
        `reference` (when given) on either path:

        - `reroute=True` reruns the var_key/dense routing cost model over the
          subset. This is *size-optimal* (each variant is re-routed to
          whichever representation is smaller for the subset's sample/carrier
          counts).
        - `reroute=False` directly slices each variant's *existing* on-disk
          representation (byte-copy, no cost model) — representation-
          preserving regardless of the subset's sample/carrier counts.
          Recommended when the subset is expected to route the same way as
          the source anyway (e.g. slicing somatic/rare-variant cohorts, where
          nearly every variant is already var_key-routed) or when the view
          must be produced under tight memory constraints.
        - `"auto"` (default) resolves to `False` when any FORMAT field is
          carried (any entry of `fields` other than `"mutcat"` whose
          `available_fields[...].category == "format"`), `True` otherwise. A
          dense->var_key flip stores one value per *carrier call* and has no
          slot for a non-carrier sample's FORMAT value, so re-routing a
          source-dense variant under a FORMAT-carrying view would silently
          drop that value; `"auto"` prefers fidelity in that case and takes
          the size-optimal re-route otherwise (genotype-only / INFO-only
          views, which have no per-sample slot to lose).

        `progress` is accepted for interface parity with other long-running
        entry points but is currently a no-op (no progress bar is shown).
        `threads` caps the number of contigs sliced concurrently (autodetected
        from available CPUs when `None`), same convention as `from_vcf`.
        Peak memory is O(output size) **per in-flight contig** times
        `threads`; with `reference=` given, each in-flight contig additionally
        holds that contig's reference sequence in memory.

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
        if reroute != "auto" and not isinstance(reroute, bool):
            raise ValueError(f"reroute must be 'auto', True, or False; got {reroute!r}")
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
        reference_str = str(reference) if reference is not None else None
        if reroute == "auto":
            # A dense->var_key flip stores one value per CARRIER CALL and has
            # no slot for a non-carrier sample's FORMAT value, so re-routing a
            # source-dense variant would silently drop it. Prefer fidelity
            # (reroute=False, preserve each variant's source representation)
            # when FORMAT is in play; take the size-optimal re-route otherwise
            # -- that is also where the win is: genotype-only / INFO-only
            # views have no per-sample slot to lose.
            carries_format = any(
                self.available_fields[key].category == "format"
                for key in fields_to_write
            )
            reroute = not carries_format
        field_tuples = [
            (sf.name, sf.category, _META_DTYPE[sf.dtype], sf.default)
            for sf in (self.available_fields[key] for key in fields_to_write)
        ]
        _core.run_slice_view(
            str(self.path),
            str(output),
            self.contigs,
            caller_samples,
            region_tuples,
            regions_overlap,
            merge_overlapping,
            field_tuples,
            reference_str,
            reroute,
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
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        samples: "str | Sequence[str] | PathLike | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: "Literal['pos', 'record', 'variant']" = "pos",
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
        check_ref: Literal["e", "x"] = "e",
    ) -> int:
        """Convert a bgzipped VCF or BCF to an SVAR2 store.

        Exactly one of `reference` or `no_reference=True` is required. With a
        reference, indels are validated against and left-aligned to the FASTA;
        with `no_reference`, validation and left-alignment are skipped and the
        input is trusted to be already normalized. Returns the number of
        out-of-scope (symbolic/breakend) ALTs dropped (0 unless
        `skip_out_of_scope`).

        `regions` restricts conversion to one or more indexed VCF fetch
        intervals. Region strings use Genoray's existing convention:
        ``"chrom:start-end"`` is 1-based inclusive and is converted to a
        0-based half-open interval; tuple/BED/frame inputs are already 0-based
        half-open. Overlapping regions raise unless `merge_overlapping=True`.

        `regions_overlap` controls which variants a region keeps, matching
        bcftools --regions-overlap: "pos" (POS inside [start,end)), "record"
        (POS in [start,end+1), so an indel at the region's last base is
        kept), or "variant" (the anchor-trimmed variant extent overlaps the
        region). In "variant" mode a multiallelic record is kept whole if ANY
        of its alleles truly overlaps the region; individual non-overlapping
        alleles are not dropped.

        `samples` selects and reorders VCF samples by name, preserving caller
        order and de-duplicating first occurrences.

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

        check_ref: policy for a record whose REF disagrees with the reference
        FASTA (ignored when `no_reference=True`). `"e"` (default) raises and
        aborts the build — matching `bcftools norm --check-ref e`. `"x"` drops
        the offending record (including a REF that runs past the contig end)
        and continues, logging a per-contig count. Comparison is
        case-insensitive, so soft-masked (lowercase) reference bases match.
        """
        from cyvcf2 import VCF as _CyVCF
        from genoray._svar._regions import _normalize_samples

        if regions_overlap not in {"pos", "record", "variant"}:
            raise ValueError(
                "regions_overlap must be one of 'pos', 'record', or 'variant'; "
                f"got {regions_overlap!r}"
            )

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
        available_samples = list(v.samples)
        selected_samples = (
            available_samples
            if samples is None
            else _normalize_samples(samples, available_samples)
        )
        if not selected_samples:
            raise ValueError("from_vcf requires at least one sample")

        all_contigs = list(v.seqnames)
        region_ranges = _normalize_svar2_regions(
            regions,
            all_contigs,
            merge_overlapping=merge_overlapping,
        )
        _reject_multiregion_variant(region_ranges, regions_overlap)

        if region_ranges:
            ranges_by_contig: dict[str, list[tuple[int, int]]] = {}
            for chrom, start, end in region_ranges:
                ranges_by_contig.setdefault(chrom, []).append((start, end))

            contigs = []
            for chrom in natsorted(ranges_by_contig):
                has_variant = any(
                    next(v(f"{chrom}:{start + 1}-{end}"), None) is not None
                    for start, end in ranges_by_contig[chrom]
                )
                if has_variant:
                    contigs.append(chrom)
            if not contigs:
                raise ValueError(
                    f"No variants found in requested regions for {source}."
                )

            region_ranges = [
                (chrom, start, end)
                for chrom in contigs
                for start, end in ranges_by_contig[chrom]
            ]
        else:
            contigs = [c for c in natsorted(v.seqnames) if next(v(c), None) is not None]
            # Whole-contig conversion: fill explicit [0, len) ranges so the Rust
            # VCF path can sub-contig shard -- its shard planner needs concrete
            # intervals, and an empty region list disables sharding. `regions`
            # was None here, so `regions_overlap` stays the default "pos", which
            # is exactly what the sharded path requires for byte-identity. A
            # contig with variants but no reported length simply gets no range
            # (Rust then reads it whole via the single reader).
            contig_lengths = {
                chrom: int(length)
                for chrom, length in zip(v.seqnames, getattr(v, "seqlens", []) or [])
                if length is not None and int(length) > 0
            }
            region_ranges = [
                (chrom, 0, contig_lengths[chrom])
                for chrom in contigs
                if chrom in contig_lengths
            ]
        if not contigs:
            raise ValueError(f"No variants found in {source}.")

        reference_path = None if no_reference else str(reference)
        flds = _resolve_fields(str(source), info_fields, format_fields)
        info = [t for t in flds if t[1] == "info"]
        format_ = [t for t in flds if t[1] == "format"]
        _validate_check_ref(check_ref)
        return _core.run_conversion_pipeline(
            str(source),
            reference_path,
            contigs,
            str(out),
            selected_samples,
            chunk_size,
            ploidy,
            threads,  # max_threads; None => auto
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            info,
            format_,
            check_ref,
            region_ranges,
            regions_overlap,
        )

    @classmethod
    def from_vcf_shards(
        cls,
        out: str | Path,
        sources: "Sequence[tuple[str | Path, object]]",
        reference: str | Path | None = None,
        *,
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        samples: "str | Sequence[str] | PathLike | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: "Literal['pos', 'record', 'variant']" = "pos",
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        ploidy: int = 2,
        chunk_size: int | None = None,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
        signatures: bool = False,
        info_fields: Sequence[str | InfoField] | None = None,
        format_fields: Sequence[str | FormatField] | None = None,
        check_ref: Literal["e", "x"] = "e",
    ) -> int:
        """Convert position-sharded multi-sample VCFs into one SVAR2 store.

        ``sources`` is an ordered sequence of ``(path, ownership_regions)``
        pairs. Every path must be an indexed VCF/BCF for the same cohort: the
        sample names and their order must be identical. Ownership regions use
        the normal Genoray region inputs and are 0-based half-open for tuple,
        BED, or frame inputs. Across all sources they must be disjoint; gaps
        and empty owned intervals are allowed.

        Indexed readers consume and normalize padded native source intervals
        concurrently under ``threads=``. Normalized-POS ownership keeps each
        atom exactly once when it crosses a physical source-file boundary. No
        concatenated VCF is materialized.

        ``chunk_size=None`` chooses a cohort-size-aware value targeting a
        256 MiB packed dense chunk, avoiding multi-gigabyte chunks for very
        large cohorts.

        The other conversion options match :meth:`from_vcf`. This first public
        version supports ``regions_overlap='pos'``; record/variant-extent
        selection requires a second predicate in addition to source ownership
        and is rejected rather than approximated.
        """
        from cyvcf2 import VCF as _CyVCF
        from genoray._svar._regions import _normalize_samples

        if regions_overlap != "pos":
            raise ValueError(
                "from_vcf_shards currently supports only regions_overlap='pos'; "
                "source ownership is POS-based"
            )
        if not isinstance(sources, AbcSequence) or isinstance(
            sources, (str, bytes, PathLike)
        ):
            raise TypeError(
                "sources must be a sequence of (path, ownership_regions) pairs"
            )
        if not sources:
            raise ValueError("from_vcf_shards requires at least one source")

        out = Path(out)
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

        paths: list[Path] = []
        vcfs: list[Any] = []
        ownership: list[tuple[int, str, int, int]] = []
        per_file_contigs: list[tuple[Path, set[str]]] = []
        cohort_samples: list[str] | None = None
        try:
            for path_index, entry in enumerate(sources):
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise TypeError(
                        "each source must be a (path, ownership_regions) pair"
                    )
                path = Path(entry[0])
                owned_input = entry[1]
                _ensure_bgzipped(path)
                _ensure_index(path)
                vcf = _CyVCF(str(path))
                paths.append(path)
                vcfs.append(vcf)
                try:
                    file_samples = list(vcf.samples)
                    file_seqnames = list(vcf.seqnames)
                finally:
                    # Cohort headers can contain hundreds of thousands of sample
                    # names. Keep at most one cyvcf2 header resident during
                    # preflight instead of retaining one per physical source.
                    vcf.close()
                    vcfs.pop()
                if cohort_samples is None:
                    cohort_samples = file_samples
                elif file_samples != cohort_samples:
                    raise ValueError(
                        f"VCF shard {path} does not have the identical sample order "
                        "required by the first source"
                    )

                file_contigs = set(file_seqnames)
                per_file_contigs.append((path, file_contigs))
                owned = _normalize_svar2_regions(
                    owned_input,
                    file_seqnames,
                    merge_overlapping=True,
                )
                for chrom, start, end in owned:
                    ownership.append((path_index, chrom, start, end))

            assert cohort_samples is not None
            if not cohort_samples:
                raise ValueError("from_vcf_shards requires at least one sample")
            _check_consistent_contig_naming(per_file_contigs)

            # Validate physical ownership BEFORE applying the caller's optional
            # query subset. An overlap is a malformed shard map even if the
            # requested region happens not to touch it.
            ownership.sort(key=lambda x: (x[1], x[2], x[3], x[0]))
            for left, right in zip(ownership, ownership[1:]):
                if left[1] == right[1] and right[2] < left[3]:
                    raise ValueError(
                        "VCF shard ownership intervals overlap: "
                        f"{left[1]}:{left[2]}-{left[3]} and "
                        f"{right[1]}:{right[2]}-{right[3]}"
                    )

            all_contigs = natsorted({chrom for _i, chrom, _s, _e in ownership})
            if not all_contigs:
                raise ValueError("from_vcf_shards resolved no ownership intervals")
            requested = _normalize_svar2_regions(
                regions,
                all_contigs,
                merge_overlapping=merge_overlapping,
            )

            if requested:
                requested_by_contig: dict[str, list[tuple[int, int]]] = {}
                for chrom, start, end in requested:
                    requested_by_contig.setdefault(chrom, []).append((start, end))
                selected_ownership: list[tuple[int, str, int, int]] = []
                for path_index, chrom, own_start, own_end in ownership:
                    for query_start, query_end in requested_by_contig.get(chrom, []):
                        start = max(own_start, query_start)
                        end = min(own_end, query_end)
                        if start < end:
                            selected_ownership.append((path_index, chrom, start, end))
            else:
                selected_ownership = ownership

            if not selected_ownership:
                raise ValueError("No requested regions match VCF shard ownership.")
            contigs = natsorted({chrom for _i, chrom, _s, _e in selected_ownership})
            selected_samples = (
                cohort_samples
                if samples is None
                else _normalize_samples(samples, cohort_samples)
            )
            if not selected_samples:
                raise ValueError("from_vcf_shards requires at least one sample")

            if chunk_size is None:
                chunk_size = _auto_chunk_size(len(selected_samples), ploidy)

            flds = _resolve_fields(str(paths[0]), info_fields, format_fields)
            for path in paths[1:]:
                other = _resolve_fields(str(path), info_fields, format_fields)
                if other != flds:
                    raise ValueError(
                        f"VCF shard {path} has incompatible requested INFO/FORMAT headers"
                    )
        finally:
            for vcf in vcfs:
                vcf.close()

        info = [field for field in flds if field[1] == "info"]
        format_ = [field for field in flds if field[1] == "format"]
        assert chunk_size is not None
        _validate_check_ref(check_ref)
        return _core.run_vcf_shards_conversion_pipeline(
            [str(path) for path in paths],
            selected_ownership,
            None if no_reference else str(reference),
            contigs,
            str(out),
            selected_samples,
            chunk_size,
            ploidy,
            threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            info,
            format_,
            check_ref,
        )

    @classmethod
    def from_pgen(
        cls,
        out: str | Path,
        source: str | Path,
        reference: str | Path | None = None,
        *,
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        samples: "str | Sequence[str] | PathLike | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: "Literal['pos', 'record', 'variant']" = "pos",
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        chunk_size: int | None = None,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
        signatures: bool = False,
        check_ref: Literal["e", "x"] = "e",
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

        `regions` restricts conversion to one or more `.pvar` variant-index
        ranges. Region strings use Genoray's existing convention:
        ``"chrom:start-end"`` is 1-based inclusive and is converted to a
        0-based half-open interval; tuple/BED/frame inputs are already 0-based
        half-open. Overlapping regions raise unless `merge_overlapping=True`.

        `regions_overlap` controls which variants a region keeps, matching
        bcftools --regions-overlap: "pos" (POS inside [start,end)), "record"
        (POS in [start,end+1), so an indel at the region's last base is
        kept), or "variant" (the anchor-trimmed variant extent overlaps the
        region). In "variant" mode a multiallelic record is kept whole if ANY
        of its alleles truly overlaps the region; individual non-overlapping
        alleles are not dropped.

        `samples` selects and reorders `.psam` samples by name, preserving
        caller order and de-duplicating first occurrences -- the store's
        `available_samples` (and every decoded column) matches that order
        exactly, regardless of each sample's original `.psam` position.

        Not supported (and silently ignored rather than errored, where noted):

        - **Dosages.** SVAR2 stores no dosages; a ``.pgen`` dosage track is ignored
          and hardcalls are read as usual.
        - **INFO/FORMAT fields.** PGEN has no FORMAT; ``.pvar`` INFO extraction is
          not implemented.

        Haplotype resolution for *unphased* heterozygotes follows the allele-code
        order ``pgenlib`` returns — the same caveat :meth:`from_vcf` carries for
        unphased ``GT``.

        check_ref: policy for a record whose REF disagrees with the reference
        FASTA (ignored when `no_reference=True`). `"e"` (default) raises and
        aborts the build — matching `bcftools norm --check-ref e`. `"x"` drops
        the offending record (including a REF that runs past the contig end)
        and continues, logging a per-contig count. Comparison is
        case-insensitive, so soft-masked (lowercase) reference bases match.
        """
        from genoray._pgen import _read_psam
        from genoray._svar._regions import _normalize_samples

        if regions_overlap not in {"pos", "record", "variant"}:
            raise ValueError(
                "regions_overlap must be one of 'pos', 'record', or 'variant'; "
                f"got {regions_overlap!r}"
            )

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

        all_psam_samples = cast("list[str]", _read_psam(psam).tolist())
        n_samples = len(all_psam_samples)
        if n_samples == 0:
            raise ValueError(f"No samples found in {psam}.")

        # `sample_perm[out]` = the position, within the sorted subset pgenlib
        # returns after `change_sample_subset`, that output column `out`
        # (caller order) should read from. Identity when no subsetting is
        # requested, so the Rust gather is byte-identical to a plain copy.
        subset_idx: "NDArray[np.uint32] | None" = None
        if samples is None:
            selected_samples = all_psam_samples
            sample_perm = list(range(n_samples))
        else:
            selected_samples = _normalize_samples(samples, all_psam_samples)
            if not selected_samples:
                raise ValueError("from_pgen requires at least one sample")
            psam_index = {s: i for i, s in enumerate(all_psam_samples)}
            sel_idx = [psam_index[s] for s in selected_samples]
            sorted_idx = sorted(set(sel_idx))
            pos_in_sorted = {orig: k for k, orig in enumerate(sorted_idx)}
            sample_perm = [pos_in_sorted[i] for i in sel_idx]
            subset_idx = np.asarray(sorted_idx, dtype=np.uint32)

        contigs, ranges, allele_idx_offsets = _pvar_contig_ranges(pvar)
        if not contigs:
            raise ValueError(f"No variants found in {pvar}.")

        region_ranges = _normalize_svar2_regions(
            regions,
            contigs,
            merge_overlapping=merge_overlapping,
        )
        _reject_multiregion_variant(region_ranges, regions_overlap)
        if region_ranges:
            covering = _pvar_covering_ranges(
                pvar, contigs, ranges, region_ranges, regions_overlap
            )
            contigs = [c for c in contigs if c in covering]
            ranges = [covering[c] for c in contigs]
            if not contigs:
                raise ValueError(
                    f"No variants found in requested regions for {source}."
                )

        if chunk_size is None:
            chunk_size = _auto_chunk_size(len(selected_samples))

        import pgenlib

        # A pool of P readers per contig: readers seek independently, so
        # concurrent shards (and concurrent contigs) must not share one. The
        # Rust side caps the actual shard count at
        # `min(processing_threads, len(pool))`, so `len(pool)` also bounds
        # intra-contig sharding. `allele_idx_offsets` is required (not just
        # used) once any variant in the file is multiallelic -- it is a
        # file-wide array, so every reader is constructed with the same one.
        # Readers are ALWAYS constructed with the full `.psam` sample count
        # (`n_samples`) -- the file's raw on-disk cohort size, independent of any
        # `samples=` subset, which is applied afterwards via change_sample_subset.
        #
        # P == 1 => PGEN sub-contig sharding is DISABLED (single reader per
        # contig, byte-identical to the serial path). Rationale, from a
        # reproducible sweep on carter-cn-02 over chr21c (~1M variants x 3202
        # samples): single-reader conversion is already fast (~33s) and is
        # bound by the shared executor/writer + reference I/O, NOT by pgenlib
        # decode. So sub-contig sharding cannot beat that floor -- measured
        # threads=24 sharding is NET SLOWER (44.9s vs 32.6s, 0.73x) because
        # concurrent readers add coordination overhead and, on `pgenlib`<0.92,
        # serialize on the CPython GIL (verified: 0 `nogil`/`prange` in
        # 0.91.0's .pyx). Nor does bumping to a GIL-releasing pgenlib help:
        # 0.94.1 parallelizes `read_alleles_range` decode via `prange`
        # (nogil=True), yet the same conversion is unchanged at ~33s across
        # OMP_NUM_THREADS 1..32 -- decode isn't the bottleneck, so the internal
        # parallelism buys nothing here. (An earlier 273s/340s serial/sharded
        # pair was a cluster-contention artifact and does not reproduce; the
        # slower-when-sharded conclusion holds at both scales.) The intra-contig
        # sharding machinery (plan_pgen_units, per-ordinal readers) is retained
        # and validated byte-identical at 1M-variant scale for re-enablement
        # only if a future reader/executor change shifts the bottleneck onto
        # decode. See docs/roadmap/svar2-conversion-decision-2026-07-15.md and
        # memory `pgenlib-holds-gil-sharded-reads`.
        P = 1
        readers = [
            [
                pgenlib.PgenReader(
                    bytes(source), n_samples, allele_idx_offsets=allele_idx_offsets
                )
                for _ in range(P)
            ]
            for _ in contigs
        ]
        if subset_idx is not None:
            # `readers` is nested (a per-shard pool per contig, P readers each),
            # so subset every reader in every pool.
            for pool in readers:
                for r in pool:
                    r.change_sample_subset(subset_idx)

        _validate_check_ref(check_ref)
        return _core.run_pgen_conversion_pipeline(
            str(source),
            str(pvar),
            None if no_reference else str(reference),
            contigs,
            ranges,
            str(out),
            selected_samples,
            chunk_size,
            threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            readers,
            check_ref,
            region_ranges,
            regions_overlap,
            sample_perm,
        )

    @classmethod
    def from_vcf_list(
        cls,
        out: str | Path,
        sources: "str | Path | Sequence[str | Path]",
        reference: str | Path | None = None,
        *,
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: "Literal['pos', 'record', 'variant']" = "pos",
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
        check_ref: Literal["e", "x"] = "e",
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
        - a single directory `Path` -- every `*.vcf.gz`/`*.vcf.bgz` then every
          `*.bcf` directly inside it (non-recursive), each group name-sorted.
        - a single file `Path` -- if it ends in
          `.vcf.gz`/`.vcf.bgz`/`.bcf`, that one file; otherwise treated as a
          manifest (one path per line, blank and `#`-comment lines skipped,
          relative entries resolved against the manifest's directory).

        As with :meth:`from_vcf`, each input VCF's records must already be
        position-sorted per contig; an unsorted file raises `ValueError`
        naming the offending file and positions rather than silently
        corrupting the k-way merge.

        Every input file must also use the **same contig naming scheme**
        (e.g. all `chr1`-style or all `1`-style) -- the merge matches contigs
        by an exact per-file string, so a cohort mixing schemes raises
        `ValueError` up front (naming the conflicting files/spellings)
        instead of silently producing a half-hom-ref-filled store.

        Opens all `N` input files concurrently (one file descriptor per file,
        per contig); at large `N` (roughly `N > (RLIMIT_NOFILE - 64) / 2`)
        this raises `ValueError` with the `ulimit -n` remedy rather than
        htslib's more confusing "no index?" error. There is no batched/
        hierarchical merge to fall back on for very large cohorts (future
        work) -- raise the open-file limit instead.

        Exactly one of `reference` or `no_reference=True` is required, with the
        same semantics as :meth:`from_vcf`: with a reference, atoms are
        validated against it and left-aligned before merging; with
        `no_reference`, both are skipped and each atom's REF is reconstructed
        from its own record's REF bytes. **Caveat specific to this method:**
        because merging is a per-contig k-way join on normalized (pos, ref,
        alt) atoms across *independently produced* files, skipping
        left-alignment under `no_reference` means a shared site only joins
        into one output row if every input file already represents it
        identically (e.g. all inputs came from the same caller, or were all
        already run through `bcftools norm` against the same reference). Two
        files encoding the same indel differently (different anchor base,
        different padding) will NOT join under `no_reference` -- they surface
        as two separate variants in the output store instead of one shared
        row, silently. `signatures=True` requires a reference (not
        `no_reference`).

        `info_fields`/`format_fields`: same declaration API as :meth:`from_vcf`
        (resolved against the FIRST file in `sources`' header). INFO fields
        merge **first-carrier-wins**: when a site is shared across files, the
        value comes from the lowest-numbered (earliest in `sources` order)
        file that carries the atom, not the last or the max. FORMAT fields
        remain per-sample, exactly as in `from_vcf`: each sample gets its own
        file's value, and a sample that doesn't carry the atom gets the
        field's default.

        `regions` restricts the merge to one or more indexed VCF fetch
        intervals, with the same convention as :meth:`from_vcf`:
        ``"chrom:start-end"`` is 1-based inclusive (converted to 0-based
        half-open); tuple/BED/frame inputs are already 0-based half-open.
        Overlapping regions raise unless `merge_overlapping=True`.
        `regions_overlap` controls which variants a region keeps, matching
        bcftools --regions-overlap: "pos" (POS inside [start,end)), "record"
        (POS in [start,end+1), so an indel at the region's last base is
        kept), or "variant" (the anchor-trimmed variant extent overlaps the
        region). In "variant" mode a multiallelic record is kept whole if ANY
        of its alleles truly overlaps the region; individual non-overlapping
        alleles are not dropped. The mode applies identically to every input
        file in the merge.

        `from_vcf_list` has no `samples` parameter -- each input is
        single-sample and the cohort is defined by `sources`.

        Returns the number of out-of-scope (symbolic/breakend) ALTs dropped
        (0 unless `skip_out_of_scope`).

        check_ref: policy for a record whose REF disagrees with the reference
        FASTA (ignored when `no_reference=True`). `"e"` (default) raises and
        aborts the build — matching `bcftools norm --check-ref e`. `"x"` drops
        the offending record (including a REF that runs past the contig end)
        and continues, logging a per-contig count. Comparison is
        case-insensitive, so soft-masked (lowercase) reference bases match.
        """
        from cyvcf2 import VCF as _CyVCF

        if regions_overlap not in {"pos", "record", "variant"}:
            raise ValueError(
                "regions_overlap must be one of 'pos', 'record', or 'variant'; "
                f"got {regions_overlap!r}"
            )

        out = Path(out)

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

        paths = _resolve_vcf_sources(sources)
        _check_fd_budget(len(paths))
        out.parent.mkdir(parents=True, exist_ok=True)

        samples: list[str] = []
        per_file_contigs: list[tuple[Path, set[str]]] = []
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
            file_contigs = {c for c in v.seqnames if next(v(c), None) is not None}
            per_file_contigs.append((path, file_contigs))
            contig_set.update(file_contigs)

        _check_consistent_contig_naming(per_file_contigs)

        sample_counts = Counter(samples)
        dupes = sorted(s for s, n in sample_counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate sample names across inputs: {dupes}")

        contigs = natsorted(contig_set)
        if not contigs:
            raise ValueError("No variants found in any input.")

        region_ranges = _normalize_svar2_regions(
            regions,
            sorted(contig_set),
            merge_overlapping=merge_overlapping,
        )
        _reject_multiregion_variant(region_ranges, regions_overlap)
        if regions is not None:
            ranges_by_contig: dict[str, list[tuple[int, int]]] = {}
            for chrom, start, end in region_ranges:
                ranges_by_contig.setdefault(chrom, []).append((start, end))
            contigs = [c for c in contigs if c in ranges_by_contig]
            if not contigs:
                raise ValueError("No requested regions match any input contig.")
            region_ranges = [
                (chrom, start, end)
                for chrom in contigs
                for start, end in ranges_by_contig[chrom]
            ]

        # Field specs are resolved against the FIRST file's header -- every
        # input is single-sample and expected to share a header schema (same
        # assumption the reference/samples handling already makes).
        flds = _resolve_fields(str(paths[0]), info_fields, format_fields)
        info = [t for t in flds if t[1] == "info"]
        format_ = [t for t in flds if t[1] == "format"]

        _validate_check_ref(check_ref)
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
            check_ref,
            region_ranges,
            regions_overlap,
        )

    @classmethod
    def from_svar1(
        cls,
        out: str | Path,
        source: str | Path,
        reference: str | Path | None = None,
        *,
        regions: "str | tuple[str, int, int] | PathLike | object | None" = None,
        samples: "str | Sequence[str] | PathLike | None" = None,
        merge_overlapping: bool = False,
        regions_overlap: "Literal['pos', 'record', 'variant']" = "pos",
        no_reference: bool = False,
        skip_out_of_scope: bool = False,
        chunk_size: int | None = None,
        threads: int | None = None,
        overwrite: bool = False,
        long_allele_capacity: int = 8 * 1024 * 1024,
        signatures: bool = False,
        check_ref: Literal["e", "x"] = "e",
    ) -> int:
        """Convert a SVAR1 (``SparseVar``) store to an SVAR2 store natively.

        Reads no VCF and no htslib: SVAR1 is already sparse, so this reconstructs
        variant records from SVAR1's arrays and reuses the same conversion spine
        as :meth:`from_vcf`.

        Exactly one of `reference` or `no_reference=True` is required, same meaning
        as :meth:`from_vcf`. `ploidy` is read from SVAR1's metadata. Returns the
        number of out-of-scope (symbolic/breakend) ALTs dropped.

        Only **biallelic** SVAR1 stores are supported (SVAR1's ``geno==1`` model);
        multiallelic input raises. All SVAR1 FORMAT fields (e.g. ``dosages``) are
        carried through; ``mutcat`` is dropped (pass `signatures=True` to recompute
        signatures from the reference). Because SVAR1 discarded non-carrier FORMAT
        values, a dense-routed variant's non-carrier cells are filled with the
        field's default/missing sentinel — field output is byte-identical to
        :meth:`from_vcf` only for var_key (carrier-only) routing.

        check_ref: policy for a record whose REF disagrees with the reference
        FASTA (ignored when `no_reference=True`). `"e"` (default) raises and
        aborts the build — matching `bcftools norm --check-ref e`. `"x"` drops
        the offending record (including a REF that runs past the contig end)
        and continues, logging a per-contig count. Comparison is
        case-insensitive, so soft-masked (lowercase) reference bases match.

        `regions` restricts conversion to one or more genomic ranges. Region
        strings use Genoray's existing convention: ``"chrom:start-end"`` is
        1-based inclusive and is converted to a 0-based half-open interval;
        tuple/BED/frame inputs are already 0-based half-open. Overlapping
        regions raise unless `merge_overlapping=True`. Unlike :meth:`from_pgen`,
        SVAR1 has no on-disk covering-range index to narrow against, so a
        selected contig's local variants are still scanned in full -- the
        per-record filter (in the Rust `Svar1RecordSource`) is what actually
        restricts the output.

        `regions_overlap` controls which variants a region keeps, matching
        bcftools --regions-overlap: "pos" (POS inside [start,end)), "record"
        (POS in [start,end+1), so an indel at the region's last base is
        kept), or "variant" (the anchor-trimmed variant extent overlaps the
        region). In "variant" mode a multiallelic record is kept whole if ANY
        of its alleles truly overlaps the region; individual non-overlapping
        alleles are not dropped. (SVAR1 is itself biallelic-only, so this only
        ever judges a single ALT.)

        `samples` selects and reorders SVAR1 samples by name, preserving
        caller order and de-duplicating first occurrences -- the store's
        `available_samples` (and every decoded column) matches that order
        exactly, regardless of each sample's original SVAR1 position.
        """
        from genoray._svar import SparseVar
        from genoray._svar._regions import _normalize_samples

        if regions_overlap not in {"pos", "record", "variant"}:
            raise ValueError(
                "regions_overlap must be one of 'pos', 'record', or 'variant'; "
                f"got {regions_overlap!r}"
            )

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
        if not source.exists():
            raise FileNotFoundError(source)

        sv1 = SparseVar(source)
        if not sv1._is_biallelic:
            raise ValueError(
                "from_svar1 supports only biallelic SVAR1 stores; this store has "
                "multiallelic variants. Re-create it biallelically first."
            )
        meta_samples, ploidy, contigs, fields = _read_svar1_metadata(source)
        if len(meta_samples) == 0:
            raise ValueError(f"No samples found in {source}.")

        # `sample_idx[out_s]` = the ORIGINAL SVAR1 sample index that output
        # column `out_s` (`selected_samples[out_s]`) reads from. Identity when
        # no subsetting is requested, so the Rust bucket remap is a no-op.
        if samples is None:
            selected_samples = meta_samples
            sample_idx = list(range(len(meta_samples)))
        else:
            selected_samples = _normalize_samples(samples, meta_samples)
            if not selected_samples:
                raise ValueError("from_svar1 requires at least one sample")
            sample_pos = {s: i for i, s in enumerate(meta_samples)}
            sample_idx = [sample_pos[s] for s in selected_samples]

        # `_svar1_index_arrays` must see the FULL contig list to compute
        # correct GLOBAL variant-id starts (the CSR spans the whole store) --
        # any region-based contig filtering happens AFTER, on its parallel
        # per-contig outputs, never by re-deriving `starts` from a narrowed
        # contig list.
        (
            starts,
            lens,
            pos_pc,
            ref_bytes_pc,
            ref_off_pc,
            alt_bytes_pc,
            alt_off_pc,
        ) = _svar1_index_arrays(source, contigs)

        region_ranges = _normalize_svar2_regions(
            regions,
            contigs,
            merge_overlapping=merge_overlapping,
        )
        _reject_multiregion_variant(region_ranges, regions_overlap)
        if region_ranges:
            region_contigs = {c for c, _, _ in region_ranges}
            keep = [i for i, c in enumerate(contigs) if c in region_contigs]
            if not keep:
                raise ValueError(
                    f"No variants found in requested regions for {source}."
                )
            contigs = [contigs[i] for i in keep]
            starts = [starts[i] for i in keep]
            lens = [lens[i] for i in keep]
            pos_pc = [pos_pc[i] for i in keep]
            ref_bytes_pc = [ref_bytes_pc[i] for i in keep]
            ref_off_pc = [ref_off_pc[i] for i in keep]
            alt_bytes_pc = [alt_bytes_pc[i] for i in keep]
            alt_off_pc = [alt_off_pc[i] for i in keep]

        format_tuples, src_dtypes = _svar1_fields_manifest(fields)

        if chunk_size is None:
            chunk_size = _auto_chunk_size(len(selected_samples), ploidy)

        out.parent.mkdir(parents=True, exist_ok=True)
        _validate_check_ref(check_ref)
        return _core.run_svar1_conversion_pipeline(
            str(source),
            None if no_reference else str(reference),
            contigs,
            starts,
            lens,
            str(out),
            selected_samples,
            ploidy,
            chunk_size,
            threads,
            long_allele_capacity,
            skip_out_of_scope,
            signatures,
            pos_pc,
            ref_bytes_pc,
            ref_off_pc,
            alt_bytes_pc,
            alt_off_pc,
            format_tuples,
            src_dtypes,
            check_ref,
            region_ranges,
            regions_overlap,
            sample_idx,
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


def _pvar_covering_ranges(
    pvar: Path,
    contigs: list[str],
    ranges: list[tuple[int, int]],
    region_ranges: list[tuple[str, int, int]],
    regions_overlap: str,
) -> dict[str, tuple[int, int]]:
    """Narrow each contig's `[lo, hi)` variant-index range (from
    `_pvar_contig_ranges`) to the covering range of its `region_ranges`, via a
    searchsorted over that contig's `.pvar` POS column.

    This is ONLY an optimization for how much of the `.pgen`/`.pvar` gets
    scanned -- the per-record Rust filter (`PgenRecordSource`, via
    `svar2_view::keeps`/`extent_overlaps`) is the source of truth for which
    variants are actually kept. The covering range returned here MUST include
    every variant the per-record filter would keep, or a real match would
    silently never reach the reader:

    - **Upper bound** is safe to narrow for every mode: a variant whose
      0-based POS lands at or past a region's end can never overlap it --
      even a `"variant"`-mode extent can only start at or after POS, never
      before it. `"record"` mode's effective end is one base past the
      nominal end (mirrors `keeps`'s inclusive upper bound).
    - **Lower bound** is only safe to narrow for `"pos"`/`"record"` (POS-
      membership rules: a variant whose POS precedes every region's start can
      never be kept). `"variant"` mode can keep a call whose POS precedes the
      region but whose anchor-trimmed *extent* (e.g. a deletion) reaches into
      it, so the lower bound stays at the contig's original `lo`.

    Returns only the contigs with >= 1 region AND >= 1 covered variant.
    """
    from genoray._pgen import _scan_pvar

    by_contig: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end in region_ranges:
        by_contig.setdefault(chrom, []).append((start, end))

    contig_range = dict(zip(contigs, ranges))
    pos_df = _scan_pvar(pvar).select("#CHROM", "POS").with_row_index("vidx").collect()

    out: dict[str, tuple[int, int]] = {}
    for chrom, regs in by_contig.items():
        if chrom not in contig_range:
            continue
        lo, hi = contig_range[chrom]
        # `.pvar` POS is 1-based; local index `i` (0-based, ascending within
        # the contig) maps to global vidx `lo + i`.
        pos = pos_df.filter(pl.col("#CHROM") == chrom).sort("vidx")["POS"].to_numpy()
        min_start = min(s for s, _ in regs)
        max_end = max(e for _, e in regs)

        eff_end = max_end + 1 if regions_overlap == "record" else max_end
        hi_local = int(np.searchsorted(pos, eff_end + 1, side="left"))

        if regions_overlap == "variant":
            lo_local = 0
        else:
            lo_local = int(np.searchsorted(pos, min_start + 1, side="left"))

        new_lo, new_hi = lo + lo_local, lo + hi_local
        if new_hi > new_lo:
            out[chrom] = (new_lo, new_hi)
    return out


def _read_svar1_metadata(
    source: Path,
) -> tuple[list[str], int, list[str], dict[str, str]]:
    """(samples, ploidy, contigs, fields) from a SVAR1 metadata.json."""
    meta = json.loads((source / "metadata.json").read_text())
    return (
        list(meta["samples"]),
        int(meta["ploidy"]),
        list(meta["contigs"]),
        dict(meta.get("fields", {})),
    )


def _pack_strings(values: list[str]) -> tuple[bytes, "np.ndarray"]:
    """Pack a list of ASCII allele strings into (concatenated bytes, i64 offsets)
    with offsets length len(values)+1."""
    encoded = [v.encode("ascii") for v in values]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(b) for b in encoded], out=offsets[1:])
    return b"".join(encoded), offsets


def _svar1_index_arrays(
    source: Path, contigs: list[str]
) -> tuple[
    list[int],
    list[int],
    list["np.ndarray"],
    list[bytes],
    list["np.ndarray"],
    list[bytes],
    list["np.ndarray"],
]:
    """Per-contig POS(0-based)/REF/ALT arrays + global contig start/len ranges.

    SVAR1's index.arrow is variant-major and contig-contiguous; POS is 1-based.
    """
    df = pl.read_ipc(source / "index.arrow", columns=["CHROM", "POS", "REF", "ALT"])
    # Global contig_start/len offsets below are only correct if index.arrow's
    # physical row order is contig-contiguous AND those runs appear in the same
    # order as `contigs` (from metadata.json). Verify via run-length encoding of
    # CHROM: a well-formed store has exactly one run per contig that has any rows,
    # in `contigs` order. `contigs` is the source's full header contig dictionary,
    # though, so it routinely includes contigs with zero surviving variants (no
    # rows in index.arrow at all) -- those legitimately contribute no RLE run, so
    # they're dropped from the expected side rather than required to match.
    run_order = df["CHROM"].rle().struct.field("value").to_list()
    present = set(df["CHROM"].unique().to_list())
    expected_order = [c for c in contigs if c in present]
    if run_order != expected_order:
        raise ValueError(
            f"{source / 'index.arrow'} is not contig-contiguous in the order given "
            f"by metadata.json's contigs list, so SVAR1->SVAR2 conversion cannot "
            f"safely assign global variant-id ranges per contig.\n"
            f"Expected contig run order (metadata.contigs, dropping contigs with no "
            f"rows in index.arrow): {expected_order}\n"
            f"Actual CHROM run order (from index.arrow): {run_order}"
        )
    # ALT is comma-Utf8 on disk; biallelic => a single token per row.
    starts: list[int] = []
    lens: list[int] = []
    pos_pc: list[np.ndarray] = []
    ref_b: list[bytes] = []
    ref_o: list[np.ndarray] = []
    alt_b: list[bytes] = []
    alt_o: list[np.ndarray] = []
    cursor = 0
    for c in contigs:
        sub = df.filter(pl.col("CHROM") == c)
        n = sub.height
        starts.append(cursor)
        lens.append(n)
        cursor += n
        pos_pc.append((sub["POS"].to_numpy().astype(np.int64) - 1).astype(np.uint32))
        rb, ro = _pack_strings(sub["REF"].to_list())
        ab, ao = _pack_strings(sub["ALT"].to_list())
        ref_b.append(rb)
        ref_o.append(ro)
        alt_b.append(ab)
        alt_o.append(ao)
    return starts, lens, pos_pc, ref_b, ref_o, alt_b, alt_o


def _svar1_fields_manifest(
    fields: dict[str, str],
) -> tuple[list[tuple[str, str, str, None, None]], list[str]]:
    """Map SVAR1 metadata.fields -> (FORMAT FieldSpec tuples, source numpy dtypes).

    Every SVAR1 custom field is FORMAT. `mutcat` is dropped (signature machinery).
    htype is inferred from the numpy dtype; storage dtype is left None (Auto).
    """
    tuples: list[tuple[str, str, str, None, None]] = []
    src_dtypes: list[str] = []
    for name, np_dtype in fields.items():
        if name == "mutcat":
            continue
        htype = "float" if np.dtype(np_dtype).kind == "f" else "int"
        tuples.append((name, "format", htype, None, None))
        src_dtypes.append(np_dtype)
    return tuples, src_dtypes


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
