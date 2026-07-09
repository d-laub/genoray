from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast, overload

import cyvcf2
from hirola import HashTable
import numpy as np
import oxbow
import polars as pl
import pyranges as pr
from loguru import logger
from more_itertools import mark_ends
from natsort import natsorted
from numpy.typing import ArrayLike, NDArray
from seqpro.rag import OFFSET_TYPE
from tqdm.auto import tqdm
from typing_extensions import Self, assert_never

from ._modes import make_array_mode, make_tuple_mode
from ._types import POS_MAX, POS_TYPE
from ._utils import (
    ContigNormalizer,
    atomic_write_path,
    format_memory,
    hap_ilens,
    parse_memory,
)
from ._var_ranges import var_counts, var_indices
from .exprs import ILEN, symbolic_ilen

V_IDX_TYPE = np.uint32
"""Dtype for VCF variant indices (uint32). This determines the maximum number of unique variants in a file."""

_CHECK_LEN_EVERY_N = 20


class DosageFieldError(RuntimeError): ...


class MultiallelicDosageError(RuntimeError): ...


GDTYPE = TypeVar("GDTYPE", np.int8, np.int16)

_Genos8Base = make_array_mode("Genos8", np.int8, 3, genos=True)
_Genos16Base = make_array_mode("Genos16", np.int16, 3, genos=True)
_DosagesBase = make_array_mode("Dosages", np.float32, 2)


class Genos8(_Genos8Base):
    pass


class Genos16(_Genos16Base):
    pass


class Dosages(_DosagesBase):
    pass


_Genos8DosagesBase = make_tuple_mode(
    "Genos8Dosages", (Genos8, Dosages), genos_dtype=np.int8
)
_Genos16DosagesBase = make_tuple_mode(
    "Genos16Dosages", (Genos16, Dosages), genos_dtype=np.int16
)


class Genos8Dosages(_Genos8DosagesBase):
    pass


class Genos16Dosages(_Genos16DosagesBase):
    pass


T = TypeVar("T", Genos8, Genos16, Dosages, Genos8Dosages, Genos16Dosages)
L = TypeVar("L", Genos8, Genos16, Genos8Dosages, Genos16Dosages)
G = TypeVar("G", Genos8, Genos16)
GD = TypeVar("GD", Genos8Dosages, Genos16Dosages)


class _Index:
    gr: pr.PyRanges
    """PyRanges for range queries, just has Chromosome, Start, End, and index columns."""
    df: pl.DataFrame
    """All the other columns in the index that aren't #CHROM, start, end, or index. Facilitates
    index -> attribute lookups."""

    def __init__(self, gr: pr.PyRanges, df: pl.DataFrame):
        self.gr = gr
        self.df = df


@dataclass(frozen=True)
class Filter:
    """A cyvcf2 record predicate paired with its matching ``.gvi`` polars expression.

    Both travel together so the record scan (``record``) and the index-level
    filter (``expr``) can never diverge. ``record`` should return True for
    variants to keep; ``expr`` must be an equivalent predicate over the index
    columns (``CHROM``, ``POS``, ``REF``, ``ALT``, ``ILEN``).
    """

    record: Callable[[cyvcf2.Variant], bool]
    expr: pl.Expr


class VCF:
    """Create a VCF reader.

    Parameters
    ----------
    path
        Path to the VCF file.
    filter
        A :class:`Filter` bundling a cyvcf2 record predicate with its matching
        ``.gvi`` polars expression, or ``None`` to disable filtering.

        .. note::
            The ``record`` predicate needs to be tolerant to missing fields. For example, if you
            access an INFO or FORMAT field, not all variants are guaranteed to have the same fields.
            The `cyvcf2.Variant <https://brentp.github.io/cyvcf2/docstrings.html#cyvcf2.cyvcf2.Variant>`_
            API provides the :meth:`.get <dict.get>` method on the INFO and FORMAT attributes. For example,
            :code:`lambda v: v.INFO.get("AF", 0) > 0.05` will skip any variants with an AF <= 0.05 or a
            missing AF by treating missing AFs as 0.

        .. note::
            The ``expr`` polars expression will be applied to the polars DataFrame returned by
            :meth:`get_record_info`. It is not applied to the VCF file itself, so it will not be
            able to use the cyvcf2.Variant API. For example, if you want to filter variants by INFO
            field, you can use:
            :code:`pl.col("AF") > 0.05`
            but you can not use:
            :code:`lambda v: v.INFO.get("AF", 0) > 0.05`
            because the expression will be applied to the polars DataFrame, not the VCF file.
    read_as
        Type of data to read from the VCF file. Can be VCF.Genos, VCF.Dosages, or VCF.GenosDosages.
    phasing
        Whether to include phasing information on genotypes. If True, the ploidy axis will be length 3 such that
        phasing is indicated by the 3rd value: 0 = unphased, 1 = phased. If False, the ploidy axis will be length 2.
    dosage_field
        Name of the dosage field to read from the VCF file. Required if read_as is VCF.Dosages, VCF.Genos8Dosages,
        or VCF.Genos16Dosages.
    progress
        Whether to show a progress bar while reading the VCF file.
    """

    path: Path
    """Path to the VCF file."""
    available_samples: list[str]
    """List of available samples in the VCF file."""
    contigs: list[str]
    """Naturally sorted list of available contigs in the VCF file."""
    ploidy: int = 2
    """Ploidy of the VCF file. This is currently always 2 since we use cyvcf2."""
    _filter: Filter | None
    """The record predicate + matching polars expression currently in effect."""
    phasing: bool
    """Whether to include phasing information on genotypes. If True, the ploidy axis will be length 3 such that
    phasing is indicated by the 3rd value: 0 = unphased, 1 = phased. If False, the ploidy axis will be length 2."""
    dosage_field: str | None
    """Name of the dosage field to read from the VCF file. Required if you want to use modes that include dosages."""
    _pbar: tqdm | None
    """A progress bar to use while reading variants. This will be incremented per variant
    during any calls to a read function."""
    _s_sorter: NDArray[np.intp] | slice
    _samples: list[str]
    _c_norm: ContigNormalizer
    _index: pl.DataFrame | None
    _vcf: cyvcf2.VCF

    Genos8 = Genos8
    """Mode for :code:`int8` genotypes :code:`(samples ploidy variants)`"""
    Genos16 = Genos16
    """Mode for :code:`int16` genotypes :code:`(samples ploidy variants)`"""
    Dosages = Dosages
    """Mode for dosages :code:`(samples variants) float32`"""
    Genos8Dosages = Genos8Dosages
    """Mode for :code:`int8` genotypes :code:`(samples ploidy variants) int8` and dosages :code:`(samples variants) float32`"""
    Genos16Dosages = Genos16Dosages
    """Mode for :code:`int16` genotypes :code:`(samples ploidy variants) int16` and dosages :code:`(samples variants) float32`"""

    def __init__(
        self,
        path: str | Path,
        filter: Filter | None = None,
        phasing: bool = False,
        dosage_field: str | None = None,
        progress: bool = False,
        with_gvi_index: bool = True,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"VCF file {self.path} does not exist.")

        self._filter = filter
        self.phasing = phasing
        self.dosage_field = dosage_field
        self.progress = progress
        self._pbar = None
        self._index = None

        vcf = cyvcf2.VCF(path)
        self.available_samples = vcf.samples
        self.contigs = natsorted(vcf.seqnames)
        self._c_norm = ContigNormalizer(vcf.seqnames)
        avail = np.asarray(self.available_samples)
        self._s2i = HashTable(max=len(avail) * 2, dtype=avail.dtype)  # type: ignore[bad-argument-type]
        self._s2i.add(avail)

        self.set_samples(None)

        if with_gvi_index and self._valid_index() and self._filter is None:
            self._load_index()

    def _open(self) -> cyvcf2.VCF:
        return cyvcf2.VCF(self.path, samples=self._samples, lazy=True)

    @property
    def filter(self) -> Filter | None:
        """The :class:`Filter` currently in effect, or ``None`` if no filter is set.

        Assigning ``vcf.filter = vcf.filter`` round-trips.
        """
        return self._filter

    def _index_path(self) -> Path:
        """Path to the index file."""
        base = Path(f"{self.path}.gvi")
        if base.exists():
            return base
        else:
            return base.with_suffix(".gvi.zst")

    @filter.setter
    def filter(self, value: Filter | None):
        """Set the record + index filter together, or clear it with ``None``.

        Assign a :class:`Filter` bundling the cyvcf2 record predicate (for the
        genotype scan) and the matching polars expression (for the ``.gvi``
        index), or ``None`` to disable filtering. Changing the filter
        invalidates the in-memory index.
        """
        if value is not None and not isinstance(value, Filter):
            raise TypeError(
                f"VCF.filter must be a genoray.Filter or None; got {type(value).__name__}."
            )
        self._index = None
        self._filter = value

    @property
    def nbytes(self) -> int:
        """Total in-memory footprint, in bytes, of resident (non-mmap'd) data
        structures held by this reader. Currently this is the gvi variant
        index (CHROM/POS/REF/ALT/ILEN). Returns 0 before the index is loaded.
        """
        if self._index is None:
            return 0
        return int(self._index.estimated_size())

    @property
    def current_samples(self) -> list[str]:
        """List of samples currently being read from the VCF file."""
        return self._samples

    @property
    def n_samples(self) -> int:
        """Number of samples currently selected."""
        return len(self._samples)

    def set_samples(self, samples: ArrayLike | None) -> Self:
        """Set the samples to read from the VCF file. Modifies the VCF reader in place and returns it.

        Parameters
        ----------
        samples
            List of sample names to read from the VCF file.

        Returns
        -------
            The VCF reader with the specified samples.
        """
        if samples is not None:
            samples = cast(list[str], np.atleast_1d(samples).tolist())

        if samples is None or samples == self.available_samples:
            self._samples = self.available_samples
            self._s_sorter = slice(None)
            self._vcf = self._open()
            return self

        if missing := set(samples).difference(self.available_samples):
            raise ValueError(
                f"Samples {missing} not found in the VCF file. "
                f"Available samples: {self.available_samples}"
            )

        self._samples = samples
        avail_indices = self._s2i.get(np.asarray(samples))
        vcf_order = np.argsort(avail_indices, kind="stable")
        if np.all(vcf_order == np.arange(len(samples))):
            self._s_sorter = slice(None)
        else:
            self._s_sorter = np.argsort(vcf_order, kind="stable")
        self._vcf = self._open()
        return self

    @contextmanager
    def using_pbar(self, pbar: tqdm):
        """Create a context where the given progress bar will be incremented by any calls to a read method.

        Parameters
        ----------
        pbar
            Progress bar to use while reading variants. This will be incremented per variant
            during any calls to a read function.
        """
        self._pbar = pbar
        try:
            yield self
        finally:
            self._pbar = None

    def n_vars_in_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
    ) -> NDArray[np.uint32]:
        """Return the start and end indices of the variants in the given ranges.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.

        Returns
        -------
            Shape: :code:`(ranges)`. Number of variants in the given ranges.
        """
        if self._index is None:
            return self._n_vars_no_index(contig, starts, ends)
        else:
            return self._n_vars_with_index(contig, starts, ends)

    def _n_vars_no_index(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
    ) -> NDArray[np.uint32]:
        """Return the start and end indices of the variants in the given ranges.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.

        Returns
        -------
            Shape: :code:`(ranges)`. Number of variants in the given ranges.
        """
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE)).clip(min=0)
        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros_like(starts, np.uint32)

        out = np.empty_like(starts, np.uint32)
        starts = starts + 1  # 1-based
        for i, (s, e) in enumerate(zip(starts, ends)):
            coord = f"{c}:{s}-{e}"
            if self._filter is None:
                out[i] = sum(1 for _ in self._vcf(coord))
            else:
                out[i] = sum(self._filter.record(v) for v in self._vcf(coord))

        return out

    def _n_vars_with_index(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
    ) -> NDArray[np.uint32]:
        """Return the start and end indices of the variants in the given ranges.

        Parameters
        ----------
        index
            Index to use for counting variants.
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.

        Returns
        -------
        n_variants
            Shape: :code:`(ranges)`. Number of variants in the given ranges.
        """
        if self._index is None:
            raise RuntimeError(
                "Index not loaded. Call `_load_index()` before using this method."
            )

        return var_counts(self._c_norm, self._index, contig, starts, ends)

    def _var_idxs(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
    ) -> tuple[NDArray[np.integer], NDArray[OFFSET_TYPE]]:
        """Get variant indices and the number of indices per range.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.

        Returns
        -------
            Shape: (tot_variants). Variant indices for the given ranges.

            Shape: (ranges+1). Offsets to get variant indices for each range.
        """
        if self._index is None:
            raise RuntimeError(
                "Index not loaded. Call `_load_index()` before using this method."
            )

        return var_indices(V_IDX_TYPE, self._c_norm, self._index, contig, starts, ends)

    def _norm_or_warn(self, contig: str) -> str | None:
        c = self._c_norm.norm(contig)
        if c is None:
            logger.warning(
                f"Query contig {contig} not found in VCF file, even after "
                "normalizing for UCSC/Ensembl nomenclature."
            )
        return c

    def _empty(self, mode: type[T], n_variants: int = 0) -> T:
        return mode.empty(self.n_samples, self.ploidy + self.phasing, n_variants)

    def _empty_gen(
        self, mode: type[L], end: POS_TYPE
    ) -> Generator[tuple[L, POS_TYPE, NDArray[V_IDX_TYPE]]]:
        return (
            (self._empty(mode), end, np.empty(0, dtype=V_IDX_TYPE)) for _ in range(1)
        )

    def read(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = POS_MAX,
        mode: type[T] = Genos16,
        out: T | None = None,
    ) -> T:
        """Read genotypes and/or dosages for a range.

        Parameters
        ----------
        contig
            Contig name.
        start
            0-based start position.
        end
            0-based, exclusive end position.
        mode
            Type of data to read.
        out
            Output array to fill with genotypes and/or dosages. If None, a new array will be created.

        Returns
        -------
            Genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        """
        if (
            issubclass(mode, (Dosages, Genos8Dosages, Genos16Dosages))
            and self.dosage_field is None
        ):
            raise ValueError(
                "Dosage field not specified. Set the VCF reader's `dosage_field` parameter."
            )

        c = self._norm_or_warn(contig)
        if c is None:
            return self._empty(mode)  # type: ignore[bad-return]

        start = max(0, start)  # type: ignore

        vcf = self._vcf(f"{c}:{int(start + 1)}-{end}")  # range string is 1-based
        if out is None:
            if self._index is not None:
                n_variants = self.n_vars_in_ranges(c, start, end)[0]  # type: ignore[bad-argument-type]
                if n_variants == 0:
                    return self._empty(mode)  # type: ignore[bad-return]
                data = mode.empty(
                    self.n_samples, self.ploidy + self.phasing, n_variants
                )
            else:
                data = None
        else:
            data = out

        if issubclass(mode, (Genos8, Genos16)):
            data, _ = self._fill_genos(vcf, data, mode=mode)
        elif issubclass(mode, Dosages):
            assert self.dosage_field is not None
            data, _ = self._fill_dosages(vcf, data, self.dosage_field)
        elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
            assert self.dosage_field is not None
            data, _ = self._fill_genos_and_dosages(
                vcf, data, self.dosage_field, mode=mode
            )
        else:
            assert_never(mode)  # type: ignore[bad-argument-type]

        return cast(T, data)

    def chunk(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = POS_MAX,
        max_mem: int | str = "4g",
        mode: type[T] = Genos16,
    ) -> Generator[T]:
        """Iterate over genotypes and/or dosages for a range in chunks limited by :code:`max_mem`.

        Parameters
        ----------
        contig
            Contig name.
        start
            0-based start position.
        end
            0-based, exclusive end position.
        max_mem
            Maximum memory to use for each chunk. Can be an integer or a string with a suffix
            (e.g. "4g", "2 MB").
        mode
            Type of data to read.

        Returns
        -------
            Generator of genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        """
        if (
            issubclass(mode, (Dosages, Genos8Dosages, Genos16Dosages))
            and self.dosage_field is None
        ):
            raise ValueError(
                "Dosage field not specified. Set the VCF reader's `dosage_field` parameter."
            )

        max_mem = parse_memory(max_mem)

        c = self._norm_or_warn(contig)
        if c is None:
            yield self._empty(mode)  # type: ignore[invalid-yield]
            return

        start = max(0, start)  # type: ignore

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = max_mem // mem_per_v
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                + f" Memory per variant: {format_memory(mem_per_v)}."
            )

        buffer = mode.empty(self.n_samples, self.ploidy + self.phasing, vars_per_chunk)
        if isinstance(buffer, (Genos8, Genos16)):
            gt_buffer = buffer
            ds_buffer = None
        elif isinstance(buffer, Dosages):
            gt_buffer = None
            ds_buffer = buffer
        else:
            gt_buffer, ds_buffer = buffer

        vcf = self._vcf(f"{c}:{int(start + 1)}-{end}")  # range string is 1-based
        if self._filter is not None:
            vcf = filter(self._filter.record, vcf)
        if self.progress and self._pbar is None:
            vcf = tqdm(vcf, desc="Reading VCF", unit=" variant")
        i = 0
        for v in vcf:
            if gt_buffer is not None:
                if self.phasing:
                    # (s p+1) np.int16
                    gt_buffer[..., i] = v.genotype.array()[self._s_sorter]
                else:
                    gt_buffer[..., i] = v.genotype.array()[
                        self._s_sorter, : self.ploidy
                    ]

            if ds_buffer is not None:
                assert self.dosage_field is not None
                ds_buffer[..., i] = self._extract_dosage(v, self.dosage_field)[
                    self._s_sorter
                ]

            i += 1

            if self._pbar is not None:
                self._pbar.update()

            if i == vars_per_chunk:
                yield buffer  # type: ignore[invalid-yield]
                i = 0

        if i != 0:
            buffer = []
            if gt_buffer is not None:
                gt_buffer = gt_buffer[..., :i]
                buffer.append(gt_buffer)
            if ds_buffer is not None:
                ds_buffer = ds_buffer[..., :i]
                buffer.append(ds_buffer)  # type: ignore[bad-argument-type]
            buffer = tuple(buffer)

            if len(buffer) == 1:
                yield buffer[0]  # type: ignore[invalid-yield]
            else:
                yield buffer  # type: ignore

    def _chunk_ranges_with_length(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
        max_mem: int | str = "4g",
        mode: type[L] = Genos16,
    ) -> Generator[
        Generator[
            tuple[L, POS_TYPE, NDArray[V_IDX_TYPE]]  # data, end, chunk_idxs
        ]
    ]:
        """Read genotypes and/or dosages in chunks approximately limited by :code:`max_mem`.
        Will extend the range so that the returned data corresponds to haplotypes that have at least as much
        length as the original range.

        Parameters
        ----------
        contig
            Contig name.
        start
            0-based start positions.
        end
            0-based, exclusive end positions.
        max_mem
            Maximum memory to use for each chunk. Can be an integer or a string with a suffix
            (e.g. "4g", "2 MB").
        mode
            Type of data to read.

        Returns
        -------
            Generator of chunks of genotypes and/or dosages and the 0-based end position of the final variant
            in the chunk. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise will be a
            tuple of arrays.
        """
        if (
            issubclass(mode, (Genos8Dosages, Genos16Dosages))
            and self.dosage_field is None
        ):
            raise ValueError(
                "Dosage field not specified. Set the VCF reader's `dosage_field` parameter."
            )
        mode = cast(type[L], mode)

        max_mem = parse_memory(max_mem)
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE)).clip(min=0)
        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

        c = self._norm_or_warn(contig)
        if c is None:
            for e in ends:
                yield self._empty_gen(mode, e)  # type: ignore[invalid-yield]
            return

        n_variants = self.n_vars_in_ranges(c, starts, ends)
        tot_variants = n_variants.sum()
        if tot_variants == 0:
            for e in ends:
                yield self._empty_gen(mode, e)  # type: ignore[invalid-yield]
            return

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = min(max_mem // mem_per_v, tot_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        v_idx, v_offsets = self._var_idxs(c, starts, ends)  # starts still 0-based here

        starts = starts + 1  # cyvcf2 queries are 1-based
        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

        for ri, (s, e, n) in enumerate(zip(starts, ends, n_variants)):
            if n == 0:
                yield self._empty_gen(mode, e)  # type: ignore[invalid-yield]
                continue

            range_idxs = v_idx[v_offsets[ri] : v_offsets[ri + 1]].astype(V_IDX_TYPE)
            yield self._chunk_with_length_helper(
                n, vars_per_chunk, c, s, e, mode, range_idxs
            )

    def _chunk_with_length_helper(
        self,
        n: int,
        vars_per_chunk: int,
        contig: str,
        start: POS_TYPE,
        end: POS_TYPE,
        mode: type[L],
        range_idxs: NDArray[V_IDX_TYPE],
    ) -> Generator[tuple[L, POS_TYPE, NDArray[V_IDX_TYPE]]]:
        if (
            issubclass(mode, (Genos8Dosages, Genos16Dosages))
            and self.dosage_field is None
        ):
            raise ValueError(
                "Dosage field not specified. Set the VCF reader's `dosage_field` parameter."
            )

        n_chunks, final_chunk = divmod(n, vars_per_chunk)
        if final_chunk == 0:
            # perfectly divisible so there is no final chunk
            chunk_sizes = np.full(n_chunks, vars_per_chunk)
        elif n_chunks == 0:
            # n_vars < vars_per_chunk, so we just use the remainder
            chunk_sizes = np.array([final_chunk])
        else:
            # have a final chunk that is smaller than the rest
            chunk_sizes = np.full(n_chunks + 1, vars_per_chunk)
            chunk_sizes[-1] = final_chunk

        vcf = self._vcf(f"{contig}:{start}-{end}")
        hap_lens = np.full((self.n_samples, self.ploidy), end - start, dtype=np.int32)
        consumed = 0
        for _, is_last, chunk_size in mark_ends(chunk_sizes):
            ilens = np.empty(chunk_size, dtype=np.int32)
            if issubclass(mode, (Genos8, Genos16)):
                out = cast(
                    Genos8 | Genos16,
                    mode.empty(self.n_samples, self.ploidy + self.phasing, chunk_size),
                )
                out, last_end = self._fill_genos(vcf, out, ilens)
                hap_lens += hap_ilens(out[:, : self.ploidy], ilens)
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                self.dosage_field = cast(str, self.dosage_field)
                out = mode.empty(self.n_samples, self.ploidy + self.phasing, chunk_size)
                out, last_end = self._fill_genos_and_dosages(
                    vcf, out, self.dosage_field, ilens
                )
                hap_lens += hap_ilens(out[0][:, : self.ploidy], ilens)
            else:
                assert_never(mode)  # type: ignore[bad-argument-type]

            # chunk_size may carry a numpy uint dtype (from vars_per_chunk); adding a
            # python int to a numpy uint64 upcasts to float64, which is invalid as a
            # slice bound, so normalize to a plain int first.
            chunk_size = int(chunk_size)
            base_idxs = range_idxs[consumed : consumed + chunk_size]
            consumed += chunk_size

            if not is_last:
                yield cast(L, out), cast(POS_TYPE, last_end), base_idxs
                continue

            if issubclass(mode, (Genos8, Genos16)):
                ls_ext, last_end = self._ext_with_length(  # type: ignore[bad-specialization]
                    contig, start, end, hap_lens, mode, last_end
                )
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                self.dosage_field = cast(str, self.dosage_field)
                ls_ext, last_end = self._ext_with_length(  # type: ignore[bad-specialization]
                    contig,
                    start,
                    end,
                    hap_lens,
                    mode,
                    last_end,
                    dosage_field=self.dosage_field,
                )
            else:
                assert_never(mode)  # type: ignore[bad-argument-type]

            if len(ls_ext) > 0:
                if issubclass(mode, (Genos8, Genos16)):
                    out = np.concatenate([out, *ls_ext], axis=-1)
                else:
                    out = tuple(
                        np.concatenate([o, *ls], axis=-1)
                        for o, ls in zip(out, zip(*ls_ext))
                    )

                last_in_range = int(range_idxs[-1])
                ext_idxs = np.arange(
                    last_in_range + 1,
                    last_in_range + 1 + len(ls_ext),
                    dtype=V_IDX_TYPE,
                )
                chunk_idxs = np.concatenate([base_idxs, ext_idxs])
            else:
                chunk_idxs = base_idxs

            yield cast(L, out), cast(POS_TYPE, last_end), chunk_idxs

    @overload
    def get_record_info(
        self,
        contig: str | None = None,
        start: int | np.integer | None = None,
        end: int | np.integer | None = None,
        fields: list[str] | None = None,
        info: list[str] | None = None,
        lazy: Literal[False] = ...,
    ) -> pl.DataFrame: ...
    @overload
    def get_record_info(
        self,
        contig: str | None = None,
        start: int | np.integer | None = None,
        end: int | np.integer | None = None,
        fields: list[str] | None = None,
        info: list[str] | None = None,
        *,
        lazy: Literal[True],
    ) -> pl.LazyFrame: ...
    @overload
    def get_record_info(
        self,
        contig: str | None = None,
        start: int | np.integer | None = None,
        end: int | np.integer | None = None,
        fields: list[str] | None = None,
        info: list[str] | None = None,
        lazy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame: ...
    def get_record_info(
        self,
        contig: str | None = None,
        start: int | np.integer | None = None,
        end: int | np.integer | None = None,
        fields: list[str] | None = None,
        info: list[str] | None = None,
        lazy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Get a DataFrame of any non-FORMAT fields in the VCF for a given range or the entire VCF.
        Will filter variants if the VCF instance has a filter function.

        Parameters
        ----------
        contig
            Contig name. If None, will read the entire VCF.
        start
            0-based start position.
        end
            0-based, exclusive end position.
        fields
            List of non-FORMAT, non-INFO fields to include. Returns all by default.
        info
            List of INFO fields to include. Returns all by default.
        """
        if (start is not None or end is not None) and contig is None:
            raise ValueError("start and end must be None if no contig is specified.")

        if start is None:
            start = 0
        if end is None:
            end = POS_MAX

        if contig is not None:
            region = f"{contig}:{start + 1}-{end}"
        else:
            region = None

        if fields is not None:
            fields = [f.lower() for f in fields]

        if info is not None:
            info = [f.lower() for f in info]

        reader = self._oxbow_reader()

        df = (
            cast(
                pl.LazyFrame,
                reader(
                    self.path,
                    samples=[],
                    fields=fields,
                    info_fields=info,
                    regions=region,
                ).pl(lazy=True),
            )
            .rename(lambda c: c.upper())
            .with_columns(pl.col("CHROM").cast(pl.Enum(self.contigs)))
        )

        if self._filter is not None:
            df = df.filter(self._filter.expr)

        if not lazy:
            df = df.collect()

        return df

    def _oxbow_reader(self) -> Callable:
        """Return the oxbow reader callable appropriate for this file's extension."""
        if self.path.suffix == ".bcf":
            return oxbow.from_bcf
        elif re.search(r"\.vcf(\.gz)?$", self.path.name) is not None:
            return oxbow.from_vcf
        else:
            raise ValueError(f"Unsupported file extension: {self.path.suffix}")

    def _declared_info_fields(self, candidates: tuple[str, ...]) -> list[str]:
        """Return which of ``candidates`` are declared as INFO fields in the VCF header.

        Uses ``header_iter()`` rather than ``get_header_type()`` because the latter
        matches both INFO and FORMAT declarations; a FORMAT-only field must NOT be
        treated as an INFO field (it would error when passed to oxbow's info_fields=).
        """
        info_ids: set[str] = {
            h.info()["ID"]
            for h in self._vcf.header_iter()
            if h.info().get("HeaderType") == "INFO"
        }
        return [c for c in candidates if c in info_ids]

    def _fetch_info_cols(self, info_names: list[str]) -> pl.LazyFrame:
        """Fetch a set of uppercase INFO field names directly from oxbow and unnest the
        returned struct, returning a LazyFrame with POS and those INFO columns as
        top-level columns. POS is retained so the caller can cross-check alignment
        against the base frame before positional concat.

        Returns a LazyFrame with columns: POS, <info_names...>
        """
        reader = self._oxbow_reader()

        # oxbow requires uppercase INFO field names; returns them nested in an 'info' struct
        raw = (
            cast(
                pl.LazyFrame,
                reader(
                    self.path,
                    samples=[],
                    fields=["pos"],
                    info_fields=info_names,
                ).pl(lazy=True),
            )
            .with_columns(pl.col("pos").alias("POS"))
            .drop("pos")
            .unnest("info")
        )
        return raw

    def _write_gvi_index(
        self,
        fields: list[str] | None = None,
        info: list[str] | None = None,
        overwrite: bool = True,
        only_biallelic: bool = False,
    ) -> None:
        """Writes record information to disk, ignoring any filtering. At a minimum this index will
        include columns `CHROM`, `POS` (1-based), `REF`, `ALT`, and `ILEN`.

        Parameters
        ----------
        fields
            List of non-FORMAT, non-INFO fields to include. At a minimum this index will include
            columns `CHROM`, `POS` (1-based), `REF`, `ALT`, and `ILEN`.
        info
            List of INFO fields to include.
        overwrite
            Whether to overwrite the index file if it exists.
        only_biallelic
            Whether to only use the first ALT alleles for each variant (i.e. assume all variants are biallelic). Better compression if True.
        """
        if self._valid_index() and not overwrite:
            raise FileExistsError(
                f"A valid index file {self._index_path()} already exists. Use overwrite=True to overwrite."
            )

        _fields: set[str] = {"CHROM", "POS", "REF", "ALT"}
        if fields is not None:
            _fields.update(fields)

        # Pull SVLEN/END/IMPRECISE when the header declares them so symbolic SVs
        # can be sized. Requesting an undeclared INFO field can error in oxbow.
        sv_info = self._declared_info_fields(("SVLEN", "END", "IMPRECISE"))
        user_info_upper = {i.upper() for i in info} if info else set()
        extra_sv = [f for f in sv_info if f.upper() not in user_info_upper]

        filt = self._filter
        self._filter = None
        try:
            index = self.get_record_info(fields=list(_fields), info=info, lazy=True)
        finally:
            self._filter = filt

        # Fetch SV helper columns directly (oxbow requires uppercase and returns a struct).
        # Both oxbow reads cover the identical full record set (no region, no filter, same
        # file order), which is what makes the positional horizontal concat correct.
        # WARNING: region-scoping or pre-concat filtering either call would silently
        # misalign SVLEN/END/IMPRECISE to wrong variants, corrupting ILEN.
        # POS is cross-checked element-wise to confirm the two reads are in identical order.
        if extra_sv:
            index_df = index.collect()
            sv_cols_df = self._fetch_info_cols(extra_sv).collect()
            if index_df.height != sv_cols_df.height:
                raise ValueError(
                    f"Row count mismatch between base index ({index_df.height}) and SV INFO "
                    f"columns ({sv_cols_df.height}); positional concat would misalign ILEN."
                )
            base_pos = index_df.get_column("POS")
            sv_pos = sv_cols_df.get_column("POS")
            if not base_pos.equals(sv_pos):
                raise ValueError(
                    "POS mismatch between base index and SV INFO columns; "
                    "positional concat would misalign ILEN. This is a bug — please report it."
                )
            # Drop POS from sv_cols_df to avoid duplicate column before horizontal concat
            sv_cols_df = sv_cols_df.drop("POS")
            index = pl.concat([index_df, sv_cols_df], how="horizontal").lazy()

        # Ensure the columns symbolic_ilen references exist (nulls when absent).
        schema = index.collect_schema()
        for col in ("SVLEN", "END", "IMPRECISE"):
            if col not in schema.names():
                dtype = pl.Boolean if col == "IMPRECISE" else pl.Int64
                index = index.with_columns(pl.lit(None, dtype=dtype).alias(col))

        # SVLEN from oxbow is List(Int32) (Number=A); coerce to scalar via list.first()
        schema = index.collect_schema()
        coerce: list[pl.Expr] = []
        for col in ("SVLEN", "END"):
            if col in schema.names() and isinstance(schema[col], pl.List):
                coerce.append(pl.col(col).list.first().alias(col))
        if coerce:
            index = index.with_columns(coerce)

        index = index.with_columns(ILEN=symbolic_ilen())

        # Drop ALL of {SVLEN, END, IMPRECISE} that the user did not explicitly request
        # via info=. This covers both fetched helper cols AND null-placeholder cols added
        # above, so non-SV indexes don't gain stray all-null columns.
        sv_cols_in_frame = {"SVLEN", "END", "IMPRECISE"} & set(
            index.collect_schema().names()
        )
        drop_cols = [c for c in sv_cols_in_frame if c not in user_info_upper]
        if drop_cols:
            index = index.drop(drop_cols)

        with atomic_write_path(self._index_path()) as _tmp:
            index.with_columns(pl.col("ALT").list.join(",")).collect().write_ipc(
                _tmp, compression="zstd"
            )

    def _load_index(self) -> Self:
        """Load the index from disk, applying the filter expression if provided. You must
        ensure that the filter expression is exactly equivalent to the vcf.filter function.
        If a filter expression is not given and the VCF has a filter function, then one pass
        over the VCF will be made to infer what records should be filtered.

        Parameters
        ----------
        filter
            Filter expression to apply to the index. This should be a pl.Expr object that
            is equivalent to the VCF filter function. If None, the filter function will be
            used to filter the index.
        """
        if not self._valid_index():
            raise FileNotFoundError(
                f"Index file {self._index_path()} does not exist or is out-of-date. "
                "Please (re)create the index using `_write_gvi_index()`."
            )

        logger.info("Loading genoray index.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index = pl.scan_ipc(
                self._index_path(), row_index_name="index"
            ).with_columns(pl.col("CHROM").cast(pl.Enum(self.contigs)))

        # Normalize ALT (on-disk comma-Utf8) to list[str] BEFORE applying the
        # filter so the in-memory schema documented in genoray.exprs holds and
        # list-typed expressions (is_symbolic, is_biallelic) work on this path.
        schema = index.collect_schema()
        if schema["ALT"] == pl.Utf8:
            index = index.with_columns(pl.col("ALT").str.split(","))

        if self._filter is not None:
            index = index.filter(self._filter.expr)

        if "ILEN" not in schema:
            index = index.with_columns(ILEN=ILEN)

        self._index = index.collect()

        return self

    def _valid_index(self) -> bool:
        """Check if the index is valid. Needs to exist and have a modified time greater than
        the VCF file."""
        if not self._index_path().exists():
            return False

        vcf_mtime = self.path.stat().st_mtime_ns
        index_mtime = self._index_path().stat().st_mtime_ns
        return index_mtime > vcf_mtime

    def _fill_genos(
        self,
        vcf: cyvcf2.VCF,
        out: Genos8 | Genos16 | None,
        ilens: NDArray[np.int32] | None = None,
        mode: type[Genos8 | Genos16] | None = None,
    ) -> tuple[Genos8 | Genos16, int]:
        if self._filter is not None:
            vcf = filter(self._filter.record, vcf)

        if out is None:
            assert mode is not None
            assert ilens is None, "caller should not provide ilens if out is None"

            out_ls: list[NDArray[np.int16]] = []

            for i, v in enumerate(vcf):
                if self.phasing:
                    # (s p+1) np.int16
                    out_ls.append(v.genotype.array())
                else:
                    # (s p) np.int16
                    out_ls.append(v.genotype.array()[:, : self.ploidy])

                if self._pbar is not None:
                    self._pbar.update()

            if len(out_ls) == 0:
                return self._empty(mode), 0

            # (s p v)
            out = cast(
                Genos8 | Genos16,
                np.stack(out_ls, axis=-1, dtype=mode._gdtype)[self._s_sorter],
            )

            return out, v.end  # type: ignore

        #! assumes n_variants > 0
        n_variants = out.shape[-1]

        if self.progress and self._pbar is None:
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit=" variant")
        elif self._pbar is not None and self._pbar.total is None:
            self._pbar.total = n_variants
            self._pbar.refresh()

        i = 0
        for i, v in enumerate(vcf):
            if self.phasing:
                # (s p+1) np.int16
                out[..., i] = v.genotype.array()[self._s_sorter]
            else:
                # (s p) np.int16
                out[..., i] = v.genotype.array()[self._s_sorter, : self.ploidy]

            if ilens is not None:
                ilens[i] = len(v.ALT[0]) - len(v.REF)

            if self._pbar is not None:
                self._pbar.update()

            if i == n_variants - 1:
                break

        if i != n_variants - 1:
            raise ValueError("Not enough variants found in the given range.")

        return out, v.end  # type: ignore

    def _extract_dosage(
        self, v: cyvcf2.Variant, dosage_field: str
    ) -> NDArray[np.float32]:
        """Fetch, validate, and squeeze the per-sample dosage for one record."""
        d = v.format(dosage_field)
        if d is None:
            raise DosageFieldError(
                f"Dosage field '{dosage_field}' not found for record {v!r}"
            )
        if d.shape[1] > 1:
            raise MultiallelicDosageError(
                f"Multiallelic dosages are not supported, encountered in VCF record {v!r}"
            )
        return d.squeeze(1)

    def _fill_dosages(
        self, vcf: cyvcf2.VCF, out: Dosages | None, dosage_field: str
    ) -> tuple[Dosages, int]:
        if self._filter is not None:
            vcf = filter(self._filter.record, vcf)

        if out is None:
            out_ls: list[NDArray[np.float32]] = []
            for v in vcf:
                out_ls.append(self._extract_dosage(v, dosage_field))

                if self._pbar is not None:
                    self._pbar.update()

            if len(out_ls) == 0:
                return Dosages.empty(self.n_samples, self.ploidy + self.phasing, 0), 0

            _out = cast(
                Dosages, np.stack(out_ls, axis=-1, dtype=np.float32)[self._s_sorter]
            )

            return _out, v.end  # type: ignore

        #! assumes n_variants > 0
        n_variants = out.shape[-1]

        if self.progress and self._pbar is None:
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit=" variant")
        elif self._pbar is not None and self._pbar.total is None:
            self._pbar.total = n_variants
            self._pbar.refresh()

        i = 0
        for i, v in enumerate(vcf):
            # (samples alts)
            out[..., i] = self._extract_dosage(v, dosage_field)[self._s_sorter]

            if self._pbar is not None:
                self._pbar.update()

            if i == n_variants - 1:
                break

        if i != n_variants - 1:
            raise ValueError("Not enough variants found in the given range.")

        return out, v.end  # type: ignore

    def _fill_genos_and_dosages(
        self,
        vcf: cyvcf2.VCF,
        out: Genos8Dosages | Genos16Dosages | None,
        dosage_field: str,
        ilens: NDArray[np.int32] | None = None,
        mode: type[Genos8Dosages | Genos16Dosages] | None = None,
    ) -> tuple[Genos8Dosages | Genos16Dosages, int]:
        if self._filter is not None:
            vcf = filter(self._filter.record, vcf)

        if out is None:
            assert mode is not None
            assert ilens is None, "caller should not provide ilens if out is None"

            geno_ls: list[NDArray[np.int16]] = []
            dosage_ls: list[NDArray[np.float32]] = []
            for i, v in enumerate(vcf):
                if self.phasing:
                    # (s p+1) np.int16
                    geno_ls.append(v.genotype.array())
                else:
                    # (s p) np.int16
                    geno_ls.append(v.genotype.array()[:, : self.ploidy])

                dosage_ls.append(self._extract_dosage(v, dosage_field))

                if self._pbar is not None:
                    self._pbar.update()

            if len(geno_ls) == 0:
                out = self._empty(mode)
                return out, 0

            genos = cast(
                Genos8 | Genos16,
                np.stack(geno_ls, axis=-1, dtype=mode._gdtype)[self._s_sorter],
            )
            dosages = cast(
                Dosages, np.stack(dosage_ls, axis=-1, dtype=np.float32)[self._s_sorter]
            )

            out = cast(Genos8Dosages | Genos16Dosages, (genos, dosages))
            return out, v.end  # type: ignore

        #! assumes n_variants > 0
        n_variants = out[0].shape[-1]

        if self.progress and self._pbar is None:
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit=" variant")
        elif self._pbar is not None and self._pbar.total is None:
            self._pbar.total = n_variants
            self._pbar.refresh()

        i = 0
        for i, v in enumerate(vcf):
            if self.phasing:
                # (s p+1) np.int16
                out[0][..., i] = v.genotype.array()[self._s_sorter]
            else:
                out[0][..., i] = v.genotype.array()[self._s_sorter, : self.ploidy]

            out[1][..., i] = self._extract_dosage(v, dosage_field)[self._s_sorter]

            if ilens is not None:
                ilens[i] = len(v.ALT[0]) - len(v.REF)

            if self._pbar is not None:
                self._pbar.update()

            if i == n_variants - 1:
                break

        if i != n_variants - 1:
            raise ValueError("Not enough variants found in the given range.")

        return out, v.end  # type: ignore

    def _mem_per_variant(self, mode: type[T]) -> int:
        """Calculate the memory required per variant for the given mode.

        Returns
        -------
        int
            Memory required per variant in bytes.
        """
        mem = mode.nbytes_per_variant(self.n_samples, self.ploidy + self.phasing)
        if isinstance(self._s_sorter, np.ndarray):
            mem *= 2  # a copy is made to reorder by samples
        return mem

    def _ext_with_length(
        self,
        contig: str,
        start: int | np.integer,
        end: int | np.integer,
        hap_lens: NDArray[np.int32],
        mode: type,
        last_end: int,
        *,
        dosage_field: str | None = None,
    ) -> tuple[list, int]:
        ploidy = self.ploidy + self.phasing
        length = end - start
        ext_start = end
        coord = f"{contig}:{ext_start + 1}"

        out_ls: list = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="no intervals found for", category=UserWarning
            )
            for i, v in enumerate(self._vcf(coord)):
                if v.start < ext_start or (
                    self._filter is not None and not self._filter.record(v)
                ):
                    continue

                genos = v.genotype.array()[:, :ploidy, None].astype(mode._gdtype)
                if dosage_field is None:
                    out_ls.append(genos)
                else:
                    dosages = self._extract_dosage(v, dosage_field)[
                        self._s_sorter, None
                    ]
                    out_ls.append((genos, dosages))

                if v.is_indel:
                    ilen = len(v.ALT[0]) - len(v.REF)
                    dist = v.start - last_end
                    hap_lens += dist + np.where(
                        genos[:, : self.ploidy] == 1, ilen, 0
                    ).squeeze(-1)
                    last_end = cast(int, v.end)

                if i % _CHECK_LEN_EVERY_N == 0 and (hap_lens >= length).all():
                    break

        if len(out_ls) > 0:
            last_end = cast(int, v.end)  # type: ignore | bound by len(out_ls) > 0

        return out_ls, last_end
