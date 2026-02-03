from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Literal, TypeVar, cast, overload

import cyvcf2
import numpy as np
import oxbow
import polars as pl
import pyranges as pr
from loguru import logger
from more_itertools import mark_ends
from natsort import natsorted
from numpy.typing import ArrayLike, NDArray
from phantom import Phantom
from seqpro.rag import OFFSET_TYPE
from tqdm.auto import tqdm
from typing_extensions import Self, TypeGuard, assert_never

from ._types import POS_MAX, POS_TYPE
from ._utils import ContigNormalizer, format_memory, hap_ilens, parse_memory
from ._var_ranges import var_counts, var_indices
from .exprs import ILEN

"""Dtype for VCF range indices. This determines the maximum size of a contig in genoray.
We have to use int64 because this is what htslib uses for CSI indexes."""

V_IDX_TYPE = np.uint32
"""Dtype for VCF variant indices (uint32). This determines the maximum number of unique variants in a file."""


class DosageFieldError(RuntimeError): ...


class MultiallelicDosageError(RuntimeError): ...


GDTYPE = TypeVar("GDTYPE", np.int8, np.int16)


def _is_genos8(obj: Any) -> TypeGuard[NDArray[np.int8]]:
    return (
        isinstance(obj, np.ndarray)
        and obj.dtype.type == np.int8
        and obj.ndim == 3
        and obj.shape[1] in (2, 3)
    )


class Genos8(NDArray[np.int8], Phantom, predicate=_is_genos8):
    _gdtype = np.int8

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Genos8:
        return cls.parse(np.empty((n_samples, ploidy, n_variants), np.int8))


def _is_genos16(obj: Any) -> TypeGuard[NDArray[np.int16]]:
    return (
        isinstance(obj, np.ndarray)
        and obj.dtype.type == np.int16
        and obj.ndim == 3
        and obj.shape[1] in (2, 3)
    )


class Genos16(NDArray[np.int16], Phantom, predicate=_is_genos16):
    _gdtype = np.int16

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Genos16:
        return cls.parse(np.empty((n_samples, ploidy, n_variants), np.int16))


def _is_dosages(obj: Any) -> TypeGuard[NDArray[np.float32]]:
    return (
        isinstance(obj, np.ndarray) and obj.dtype.type == np.float32 and obj.ndim == 2
    )


class Dosages(NDArray[np.float32], Phantom, predicate=_is_dosages):
    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Dosages:
        return cls.parse(np.empty((n_samples, n_variants), np.float32))


def _is_genos8_dosages(obj) -> TypeGuard[tuple[Genos8, Dosages]]:
    """Check if the object is a tuple of genotypes and dosages.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    -------
    bool
        True if the object is a tuple of genotypes and dosages, False otherwise.
    """
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], Genos8)
        and isinstance(obj[1], Dosages)
    )


class Genos8Dosages(tuple[Genos8, Dosages], Phantom, predicate=_is_genos8_dosages):
    _gdtype = np.int8

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Genos8Dosages:
        return cls.parse(
            (
                Genos8.empty(n_samples, ploidy, n_variants),
                Dosages.empty(n_samples, ploidy, n_variants),
            )
        )


def _is_genos16_dosages(obj) -> TypeGuard[tuple[Genos8, Dosages]]:
    """Check if the object is a tuple of genotypes and dosages.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    -------
    bool
        True if the object is a tuple of genotypes and dosages, False otherwise.
    """
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], Genos16)
        and isinstance(obj[1], Dosages)
    )


class Genos16Dosages(tuple[Genos16, Dosages], Phantom, predicate=_is_genos16_dosages):
    _gdtype = np.int16

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Genos16Dosages:
        return cls.parse(
            (
                Genos16.empty(n_samples, ploidy, n_variants),
                Dosages.empty(n_samples, ploidy, n_variants),
            )
        )


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


class VCF:
    """Create a VCF reader.

    Parameters
    ----------
    path
        Path to the VCF file.
    filter
        Function to filter variants. Should return True for variants to keep.

        .. note::
            To avoid KeyErrors, this function needs to be tolerant to missing fields. For example, if you
            access an INFO or FORMAT field, not all variants are guaranteed to have the same fields.
            The `cyvcf2.Variant <https://brentp.github.io/cyvcf2/docstrings.html#cyvcf2.cyvcf2.Variant>`_
            API provides the :meth:`.get <dict.get>` method on the INFO and FORMAT attributes. For example,
            :code:`lambda v: v.INFO.get("AF", 0) > 0.05` will skip any variants with an AF <= 0.05 or a
            missing AF by treating missing AFs as 0.
    pl_filter
        Polars expression to filter variants. Should return True for variants to keep. Must match the filter function.

        .. note::
            This expression will be applied to the polars DataFrame returned by :meth:`get_record_info`.
            It is not applied to the VCF file itself, so it will not be able to use the cyvcf2.Variant API.
            For example, if you want to filter variants by INFO field, you can use:
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
    ploidy = 2
    """Ploidy of the VCF file. This is currently always 2 since we use cyvcf2."""
    _filter: Callable[[cyvcf2.Variant], bool] | None
    """Function to filter variants. Should return True for variants to keep."""
    _pl_filter: pl.Expr | None
    """Polars expression to filter variants. Should return True for variants to keep. Must match the filter function."""
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
        filter: Callable[[cyvcf2.Variant], bool] | None = None,
        pl_filter: pl.Expr | None = None,
        phasing: bool = False,
        dosage_field: str | None = None,
        progress: bool = False,
        with_gvi_index: bool = True,
    ):
        if (filter is not None and pl_filter is None) or (
            filter is None and pl_filter is not None
        ):
            raise ValueError(
                "If a filter function is provided, a polars expression must also be provided, and vice versa."
            )

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"VCF file {self.path} does not exist.")

        self._filter = filter
        self._pl_filter = pl_filter
        self.phasing = phasing
        self.dosage_field = dosage_field
        self.progress = progress
        self._pbar = None
        self._index = None

        vcf = cyvcf2.VCF(path)
        self.available_samples = vcf.samples
        self.contigs = natsorted(vcf.seqnames)
        self._c_norm = ContigNormalizer(vcf.seqnames)

        self.set_samples(None)

        if with_gvi_index and self._valid_index() and self._filter is None:
            self._load_index()

    def _open(self) -> cyvcf2.VCF:
        return cyvcf2.VCF(self.path, samples=self._samples, lazy=True)

    @property
    def filter(self) -> Callable[[cyvcf2.Variant], bool] | None:
        """Function to filter variants. Should return True for variants to keep."""
        return self._filter

    def _index_path(self) -> Path:
        """Path to the index file."""
        base = Path(f"{self.path}.gvi")
        if base.exists():
            return base
        else:
            return base.with_suffix(".gvi.zst")

    @filter.setter
    def filter(self, filter: Callable[[cyvcf2.Variant], bool] | None):
        """Changing the filter invalidates the in-memory index."""
        self._index = None
        self._filter = filter

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

        vcf = self._open()
        _, s_idx, _ = np.intersect1d(vcf.samples, samples, return_indices=True)
        self._samples = samples
        self._s_sorter = np.argsort(s_idx)
        self._vcf = vcf
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
                out[i] = sum(self._filter(v) for v in self._vcf(coord))

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
        ploidy = self.ploidy + self.phasing

        c = self._c_norm.norm(contig)
        if c is None:
            logger.warning(
                f"Query contig {contig} not found in VCF file, even after normalizing for UCSC/Ensembl nomenclature."
            )
            return mode.empty(self.n_samples, ploidy, 0)

        start = max(0, start)  # type: ignore

        vcf = self._vcf(f"{c}:{int(start + 1)}-{end}")  # range string is 1-based
        if out is None:
            if self._index is not None:
                n_variants = self.n_vars_in_ranges(c, start, end)[0]
                if n_variants == 0:
                    return mode.empty(self.n_samples, ploidy, 0)
            else:
                n_variants = None

            if issubclass(mode, (Genos8, Genos16)):
                if n_variants is None:
                    data = None
                else:
                    data = mode.empty(self.n_samples, ploidy, n_variants)
                data, _ = self._fill_genos(vcf, data, mode=mode)
            elif issubclass(mode, Dosages):
                assert self.dosage_field is not None
                if n_variants is None:
                    data = None
                else:
                    data = mode.empty(self.n_samples, ploidy, n_variants)
                data, _ = self._fill_dosages(vcf, data, self.dosage_field)
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                assert self.dosage_field is not None
                if n_variants is None:
                    data = None
                else:
                    data = mode.empty(self.n_samples, ploidy, n_variants)
                data, _ = self._fill_genos_and_dosages(
                    vcf, data, self.dosage_field, mode=mode
                )
            else:
                assert_never(mode)

            out = cast(T, data)
        else:
            if isinstance(out, (Genos8, Genos16)):
                self._fill_genos(vcf, out)
            elif isinstance(out, Dosages):
                assert self.dosage_field is not None
                self._fill_dosages(vcf, out, self.dosage_field)
            elif isinstance(out, (Genos8Dosages, Genos16Dosages)):
                assert self.dosage_field is not None
                self._fill_genos_and_dosages(vcf, out, self.dosage_field)
            else:
                assert_never(mode)

        return out

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

        ploidy = self.ploidy + self.phasing

        max_mem = parse_memory(max_mem)

        c = self._c_norm.norm(contig)
        if c is None:
            logger.warning(
                f"Query contig {contig} not found in VCF file, even after normalizing for UCSC/Ensembl nomenclature."
            )
            yield mode.empty(self.n_samples, ploidy, 0)
            return

        start = max(0, start)  # type: ignore

        n_variants: int = self.n_vars_in_ranges(c, start, end)[0]
        if n_variants == 0:
            yield mode.empty(self.n_samples, ploidy, 0)
            return

        if self._pbar is not None and self._pbar.total is None:
            self._pbar.total = n_variants
            self._pbar.refresh()

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        n_chunks, final_chunk = divmod(n_variants, vars_per_chunk)
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

        vcf = self._vcf(f"{c}:{int(start + 1)}-{end}")  # range string is 1-based
        ploidy = self.ploidy + self.phasing
        for chunk_size in chunk_sizes:
            if issubclass(mode, (Genos8, Genos16)):
                out = mode.empty(self.n_samples, ploidy, chunk_size)
                self._fill_genos(vcf, out)
            elif issubclass(mode, Dosages):
                out = mode.empty(self.n_samples, ploidy, chunk_size)
                self._fill_dosages(
                    vcf,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                out = mode.empty(self.n_samples, ploidy, chunk_size)
                self._fill_genos_and_dosages(
                    vcf,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                assert_never(mode)

            yield mode.parse(out)

    def _chunk_ranges_with_length(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = POS_MAX,
        max_mem: int | str = "4g",
        mode: type[L] = Genos16,
    ) -> Generator[
        Generator[
            tuple[L, int, int]  # data, end, n_extension_vars
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
        ploidy = self.ploidy + self.phasing
        mode = cast(type[L], mode)

        max_mem = parse_memory(max_mem)
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE)).clip(min=0)
        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

        c = self._c_norm.norm(contig)
        if c is None:
            logger.warning(
                f"Query contig {contig} not found in VCF file, even after normalizing for UCSC/Ensembl nomenclature."
            )
            for e in ends:
                yield ((mode.empty(self.n_samples, ploidy, 0), e, 0) for _ in range(1))
            return

        n_variants = self.n_vars_in_ranges(c, starts, ends)
        tot_variants = n_variants.sum()
        if tot_variants == 0:
            for e in ends:
                yield ((mode.empty(self.n_samples, ploidy, 0), e, 0) for _ in range(1))
            return

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = min(max_mem // mem_per_v, tot_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        starts = starts + 1  # cyvcf2 queries are 1-based
        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

        for s, e, n in zip(starts, ends, n_variants):
            if n == 0:
                yield ((mode.empty(self.n_samples, ploidy, 0), e, 0) for _ in range(1))
                continue

            yield self._chunk_with_length_helper(n, vars_per_chunk, c, s, e, mode)

    def _chunk_with_length_helper(
        self,
        n: int,
        vars_per_chunk: int,
        contig: str,
        start: POS_TYPE,
        end: POS_TYPE,
        mode: type[L],
    ) -> Generator[tuple[L, int, int]]:
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
        ploidy = self.ploidy + self.phasing
        hap_lens = np.full((self.n_samples, self.ploidy), end - start, dtype=np.int32)
        for _, is_last, chunk_size in mark_ends(chunk_sizes):
            ilens = np.empty(chunk_size, dtype=np.int32)
            if issubclass(mode, (Genos8, Genos16)):
                out = cast(
                    Genos8 | Genos16, mode.empty(self.n_samples, ploidy, chunk_size)
                )
                out, last_end = self._fill_genos(vcf, out, ilens)
                hap_lens += hap_ilens(out[:, : self.ploidy], ilens)
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                self.dosage_field = cast(str, self.dosage_field)
                out = mode.empty(self.n_samples, ploidy, chunk_size)
                out, last_end = self._fill_genos_and_dosages(
                    vcf, out, self.dosage_field, ilens
                )
                hap_lens += hap_ilens(out[0][:, : self.ploidy], ilens)
            else:
                assert_never(mode)

            if not is_last:
                yield cast(L, out), last_end, 0
                continue

            if issubclass(mode, (Genos8, Genos16)):
                ls_ext, last_end = self._ext_genos_with_length(
                    contig, start, end, hap_lens, mode, last_end
                )
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                self.dosage_field = cast(str, self.dosage_field)
                ls_ext, last_end = self._ext_genos_dosages_with_length(
                    contig,
                    start,
                    end,
                    hap_lens,
                    mode,
                    self.dosage_field,
                    last_end,
                )
            else:
                assert_never(mode)

            if len(ls_ext) > 0:
                if issubclass(mode, (Genos8, Genos16)):
                    out = np.concatenate([out, *ls_ext], axis=-1)
                else:
                    out = tuple(
                        np.concatenate([o, *ls], axis=-1)
                        for o, ls in zip(out, zip(*ls_ext))
                    )

            yield (
                cast(L, out),
                last_end,
                len(ls_ext),
            )

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

        if self.path.suffix == ".bcf":
            reader = oxbow.from_bcf
        elif re.search(r"\.vcf(\.gz)?$", self.path.name) is not None:
            reader = oxbow.from_vcf
        else:
            raise ValueError(f"Unsupported file extension: {self.path.suffix}")

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

        if self._pl_filter is not None:
            df = df.filter(self._pl_filter)

        if not lazy:
            df = df.collect()

        return df

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

        filt = self._pl_filter
        self._pl_filter = None
        try:
            index = self.get_record_info(fields=list(_fields), info=info, lazy=True)
        finally:
            self._pl_filter = filt

        index.with_columns(pl.col("ALT").list.join(",")).collect().write_ipc(
            self._index_path(), compression="zstd"
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

        if self._pl_filter is not None:
            index = index.filter(self._pl_filter)

        schema = index.collect_schema()
        if schema["ALT"] == pl.Utf8:
            index = index.with_columns(pl.col("ALT").str.split(","))

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
            vcf = filter(self._filter, vcf)

        if out is None:
            assert mode is not None
            assert ilens is None, "caller should not provide ilens if out is None"

            out_ls = []

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
                return mode.empty(self.n_samples, self.ploidy, 0), v.end

            # (s p v)
            out = cast(
                Genos8 | Genos16,
                np.stack(out_ls, axis=-1, dtype=mode._gdtype)[self._s_sorter],
            )

            return out, v.end

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

    def _fill_dosages(
        self, vcf: cyvcf2.VCF, out: Dosages | None, dosage_field: str
    ) -> tuple[Dosages, int]:
        if self._filter is not None:
            vcf = filter(self._filter, vcf)

        if out is None:
            out_ls = []
            for v in vcf:
                out_ls.append(v.format(dosage_field))
                if self._pbar is not None:
                    self._pbar.update()

            _out = np.stack(out_ls, axis=-1, dtype=Dosages._gdtype)[self._s_sorter]
            return _out, v.end

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
            d = v.format(dosage_field)
            if d is None:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                )
            if d.shape[1] > 1:
                raise MultiallelicDosageError(
                    f"Multiallelic dosages are not supported, encountered in VCF record {repr(v)}"
                )
            out[..., i] = d.squeeze(1)[self._s_sorter]

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
        if out is None:
            assert mode is not None
            assert ilens is None, "caller should not provide ilens if out is None"

            geno_ls = []
            dosage_ls = []
            for i, v in enumerate(vcf):
                if self.phasing:
                    # (s p+1) np.int16
                    geno_ls.append(v.genotype.array()[self._s_sorter])
                else:
                    # (s p) np.int16
                    geno_ls.append(v.genotype.array()[self._s_sorter, : self.ploidy])
                dosage_ls.append(v.format(dosage_field)[self._s_sorter])
                if self._pbar is not None:
                    self._pbar.update()

            genos = cast(
                Genos8 | Genos16,
                np.stack(geno_ls, axis=-1, dtype=mode._gdtype)[self._s_sorter],
            )
            dosages = cast(
                Dosages, np.stack(dosage_ls, axis=-1, dtype=np.float32)[self._s_sorter]
            )

            out = cast(Genos8Dosages | Genos16Dosages, (genos, dosages))
            return out, v.end

        #! assumes n_variants > 0
        n_variants = out[0].shape[-1]

        if self._filter is not None:
            vcf = filter(self._filter, vcf)

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

            d = v.format(dosage_field)
            if d is None:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                )
            if d.shape[1] > 1:
                raise MultiallelicDosageError(
                    f"Multiallelic dosages are not supported, encountered in VCF record {repr(v)}"
                )
            out[1][..., i] = d.squeeze(1)[self._s_sorter]

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
        """Calculate the memory required per variant for the given genotypes and dosages.

        Parameters
        ----------
        genotypes
            Whether to include genotypes.
        dosages
            Whether to include dosages.

        Returns
        -------
        int
            Memory required per variant in bytes.
        """
        mem = 0

        ploidy = self.ploidy + self.phasing

        if issubclass(mode, (Genos8, Genos16)):
            mem += self.n_samples * ploidy * mode._gdtype().itemsize
        elif issubclass(mode, Dosages):
            mem += self.n_samples * np.float32().itemsize
        elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
            mem += self.n_samples * ploidy * mode._gdtype().itemsize
            mem += self.n_samples * np.float32().itemsize
        else:
            assert_never(mode)

        return mem

    def _ext_genos_with_length(
        self,
        contig: str,
        start: int | np.integer,
        end: int | np.integer,
        hap_lens: NDArray[np.int32],
        mode: type[G],
        last_end: int,
    ) -> tuple[list[G], int]:
        ploidy = self.ploidy + self.phasing
        length = end - start
        ext_start = end
        coord = f"{contig}:{ext_start + 1}"

        _CHECK_LEN_EVERY_N = 20
        ls_genos: list[G] = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="no intervals found for", category=UserWarning
            )
            for i, v in enumerate(self._vcf(coord)):
                if v.start < ext_start or (
                    self._filter is not None and not self._filter(v)
                ):
                    continue

                # (s p, 1)
                genos = v.genotype.array()[:, :ploidy, None]
                genos = genos.astype(mode._gdtype)
                ls_genos.append(genos)

                if v.is_indel:
                    ilen = len(v.ALT[0]) - len(v.REF)
                    dist = v.start - last_end
                    hap_lens += dist + np.where(genos == 1, ilen, 0).squeeze(-1)
                    last_end = cast(int, v.end)

                if i % _CHECK_LEN_EVERY_N == 0 and (hap_lens >= length).all():
                    break

        if len(ls_genos) > 0:
            last_end = cast(int, v.end)  # type: ignore | guaranteed bound by len(ls) > 0

        return ls_genos, last_end

    def _ext_genos_dosages_with_length(
        self,
        contig: str,
        start: int | np.integer,
        end: int | np.integer,
        hap_lens: NDArray[np.int32],
        mode: type[GD],
        dosage_field: str,
        last_end: int,
    ) -> tuple[list[GD], int]:
        ploidy = self.ploidy + self.phasing
        length = end - start
        ext_start = end
        coord = f"{contig}:{ext_start + 1}"

        _CHECK_LEN_EVERY_N = 20
        ls_geno_dosages: list[GD] = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="no intervals found for", category=UserWarning
            )
            for i, v in enumerate(self._vcf(coord)):
                if v.start < ext_start or (
                    self._filter is not None and not self._filter(v)
                ):
                    continue
                # (s p 1)
                genos = v.genotype.array()[:, :ploidy, None]
                genos = genos.astype(mode._gdtype)

                dosages = v.format(dosage_field)
                if dosages is None:
                    raise DosageFieldError(
                        f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                    )
                # (s, 1, 1) or (s, 1)? -> (s)
                dosages = dosages.squeeze(1)[self._s_sorter, None]

                ls_geno_dosages.append((genos, dosages))  # type: ignore

                if v.is_indel:
                    ilen = len(v.ALT[0]) - len(v.REF)
                    dist = v.start - last_end
                    # (s p 1)
                    hap_lens += dist + np.where(genos == 1, ilen, 0).squeeze(-1)
                    last_end = cast(int, v.end)

                if i % _CHECK_LEN_EVERY_N == 0 and (hap_lens >= length).all():
                    break

        if len(ls_geno_dosages) > 0:
            last_end = cast(int, v.end)  # type: ignore | guaranteed bound by len(ls) > 0

        return ls_geno_dosages, last_end
