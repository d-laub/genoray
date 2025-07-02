from __future__ import annotations

from functools import partial
from io import TextIOWrapper
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Generator, TypeVar, cast

import numpy as np
import pgenlib
import polars as pl
import pyranges as pr
from hirola import HashTable
from loguru import logger
from more_itertools import mark_ends, windowed
from numpy.typing import ArrayLike, NDArray
from phantom import Phantom
from seqpro._ragged import OFFSET_TYPE, lengths_to_offsets
from typing_extensions import Self, TypeGuard, assert_never
from zstandard import ZstdDecompressor

from ._utils import ContigNormalizer, format_memory, hap_ilens, parse_memory
from .exprs import ILEN, is_biallelic

POS_TYPE = np.int64
"""Dtype for PGEN range indices. This determines the maximum size of a contig in genoray.
We have to use int64 because this is what PyRanges uses."""

V_IDX_TYPE = np.uint32
"""Dtype for PGEN variant indices (uint32). This determines the maximum number of unique variants in a file."""

INT64_MAX = np.iinfo(POS_TYPE).max


def _is_genos(obj: Any) -> TypeGuard[Genos]:
    return (
        isinstance(obj, np.ndarray)
        and obj.dtype.type == np.int32
        and obj.ndim == 3
        and obj.shape[1] == 2
    )


class Genos(NDArray[np.int32], Phantom, predicate=_is_genos):
    _dtype = np.int32

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Self:
        return cls.parse(np.empty((n_samples, ploidy, n_variants), dtype=cls._dtype))


def _is_dosages(obj: Any) -> TypeGuard[Dosages]:
    return (
        isinstance(obj, np.ndarray) and obj.dtype.type == np.float32 and obj.ndim == 2
    )


class Dosages(NDArray[np.float32], Phantom, predicate=_is_dosages):
    _dtype = np.float32

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Self:
        return cls.parse(np.empty((n_samples, n_variants), dtype=cls._dtype))


def _is_phasing(obj: Any) -> TypeGuard[Phasing]:
    return isinstance(obj, np.ndarray) and obj.dtype.type == np.bool_ and obj.ndim == 2


class Phasing(NDArray[np.bool_], Phantom, predicate=_is_phasing):
    _dtype = np.bool_

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Self:
        return cls.parse(np.empty((n_samples, n_variants), dtype=cls._dtype))


def _is_genos_phasing(obj) -> TypeGuard[GenosPhasing]:
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], Genos)
        and isinstance(obj[1], Phasing)
    )


class GenosPhasing(tuple[Genos, Phasing], Phantom, predicate=_is_genos_phasing):
    _dtypes = (np.int32, np.bool_)

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Self:
        return cls.parse(
            (
                Genos.empty(n_samples, ploidy, n_variants),
                Phasing.empty(n_samples, ploidy, n_variants),
            )
        )


def _is_genos_dosages(obj) -> TypeGuard[GenosDosages]:
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], Genos)
        and isinstance(obj[1], Dosages)
    )


class GenosDosages(tuple[Genos, Dosages], Phantom, predicate=_is_genos_dosages):
    _dtypes = (np.int32, np.float32)

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Self:
        return cls.parse(
            (
                Genos.empty(n_samples, ploidy, n_variants),
                Dosages.empty(n_samples, ploidy, n_variants),
            )
        )


def _is_genos_phasing_dosages(obj) -> TypeGuard[GenosPhasingDosages]:
    return (
        isinstance(obj, tuple)
        and len(obj) == 3
        and isinstance(obj[0], Genos)
        and isinstance(obj[1], Phasing)
        and isinstance(obj[2], Dosages)
    )


class GenosPhasingDosages(
    tuple[Genos, Phasing, Dosages], Phantom, predicate=_is_genos_phasing_dosages
):
    _dtypes = (np.int32, np.bool_, np.float32)

    @classmethod
    def empty(cls, n_samples: int, ploidy: int, n_variants: int) -> Self:
        return cls.parse(
            (
                Genos.empty(n_samples, ploidy, n_variants),
                Phasing.empty(n_samples, ploidy, n_variants),
                Dosages.empty(n_samples, ploidy, n_variants),
            )
        )


T = TypeVar("T", Genos, Dosages, GenosPhasing, GenosDosages, GenosPhasingDosages)
L = TypeVar("L", Genos, GenosPhasing, GenosDosages, GenosPhasingDosages)


class PGEN:
    """Create a PGEN reader.

    Parameters
    ----------
    path
        Path to the PGEN file. Only used for genotypes if a dosage path is provided as well.
    filter
        Polars expression to filter variants. Should return True for variants to keep. Will have at least the columns
        `CHROM`, `POS` (1-based), `REF`, `ALT`, and `ILEN` available to use.
    dosage_path
        Path to a dosage PGEN file. If None, the genotype PGEN file will be used for both genotypes and dosages.
    """

    available_samples: list[str]
    """List of available samples in the PGEN file."""
    _filter: pl.Expr | None
    """Polars expression to filter variants. Should return True for variants to keep."""
    ploidy = 2
    """Ploidy of the samples. The PGEN format currently only supports diploid (2)."""
    contigs: list[str]
    """Naturally sorted list of contig names in the PGEN file."""
    _index: pr.PyRanges
    _geno_pgen: pgenlib.PgenReader
    _dose_pgen: pgenlib.PgenReader
    _s_idx: NDArray[np.uint32] | slice
    _s_sorter: NDArray[np.intp] | slice
    _geno_path: Path
    _dose_path: Path | None
    _sei: StartsEndsIlens | None  # unfiltered so that var_idxs map correctly
    """Variant 0-based starts, ends, ILEN, and ALT alleles if the PGEN with filters is bi-allelic."""
    _s2i: HashTable
    _c_max_idxs: dict[str, int]

    Genos = Genos
    """:code:`(samples ploidy variants) int32`"""
    Dosages = Dosages
    """:code:`(samples variants) float32`
    
    .. note::
        PGEN does not support multi-allelic dosages. If you attempt to write one, you will get an
        error from PLINK 2.0.
    """
    GenosPhasing = GenosPhasing
    """:code:`(samples ploidy variants) int32` and :code:`(samples variants) bool`"""
    GenosDosages = GenosDosages
    """:code:`(samples ploidy variants) int32` and :code:`(samples variants) float32`"""
    GenosPhasingDosages = GenosPhasingDosages
    """:code:`(samples ploidy variants) int32`, :code:`(samples variants) bool`, and :code:`(samples variants) float32`"""

    def __init__(
        self,
        geno_path: str | Path,
        filter: pl.Expr | None = None,
        dosage_path: str | Path | None = None,
    ):
        geno_path = Path(geno_path)
        if geno_path.suffix != ".pgen":
            geno_path = geno_path.with_suffix(".pgen")
        self._geno_path = geno_path
        self._filter = filter

        samples = _read_psam(geno_path.with_suffix(".psam"))
        self.available_samples = cast(list[str], samples.tolist())
        self._s2i = HashTable(
            max=len(samples) * 2,  # type: ignore
            dtype=samples.dtype,
        )
        self._s2i.add(samples)
        self._s_idx = slice(None)
        self._s_sorter = slice(None)
        self._geno_pgen = pgenlib.PgenReader(bytes(geno_path), len(samples))

        if dosage_path is not None:
            dosage_path = Path(dosage_path)
            dose_samples = _read_psam(dosage_path.with_suffix(".psam"))
            if (samples != dose_samples).any():
                raise ValueError(
                    "Samples in dosage file do not match those in genotype file."
                )
            self._dose_pgen = pgenlib.PgenReader(bytes(Path(dosage_path)))
        else:
            self._dose_pgen = self._geno_pgen
        self._dose_path = dosage_path

        self._index, self._sei, self.contigs = _read_index(self._index_path(), filter)
        self._c_norm = ContigNormalizer(self.contigs)
        vars_per_contig = np.array([len(self._index[c]) for c in self.contigs]).cumsum()
        self._c_max_idxs = {c: v - 1 for c, v in zip(self.contigs, vars_per_contig)}

    @property
    def current_samples(self) -> list[str]:
        """List of samples that are currently being used, in order."""
        if isinstance(self._s_sorter, slice):
            return self.available_samples
        return cast(list[str], self._s2i.keys[self._s_idx].tolist())

    @property
    def n_samples(self) -> int:
        """Number of samples in the file."""
        if isinstance(self._s_sorter, slice):
            return len(self.available_samples)
        return len(self._s_sorter)

    @property
    def filter(self) -> pl.Expr | None:
        """Polars expression to filter variants. Should return True for variants to keep."""
        return self._filter

    @filter.setter
    def filter(self, filter: pl.Expr | None):
        """Set the Polars expression to filter variants. Should return True for variants to keep."""
        self._index, self._sei, _ = _read_index(self._index_path(), filter)
        self._filter = filter

    def _index_path(self) -> Path:
        """Path to the index file."""
        # check whether pvar or pvar.zst
        index = self._geno_path.with_suffix(".pvar")
        if not index.exists():
            index = self._geno_path.with_suffix(".pvar.zst")
        if not index.exists():
            raise FileNotFoundError("No index file found.")
        return index.with_suffix(f"{index.suffix}.gvi")

    def set_samples(self, samples: ArrayLike | None) -> Self:
        """Set the samples to use.

        Parameters
        ----------
        samples
            List of sample names to use. If None, all samples will be used.
        """
        if samples is not None:
            samples = np.atleast_1d(samples)

        if (
            samples is None
            or len(samples) == len(self.available_samples)
            and (samples == np.asarray(self.available_samples)).all()
        ):
            self._s_idx = slice(None)
            self._s_sorter = slice(None)
            return self

        s_idx = self._s2i.get(samples).astype(np.uint32)
        if len(missing := samples[s_idx == -1]) > 0:
            raise ValueError(f"Samples {missing} not found in the file.")

        self._s_idx = s_idx
        self._s_sorter = np.argsort(s_idx)
        # if dose path is None, then dose pgen is just a reference to geno pgen so
        # we're also (somewhat unsafely) mutating the dose pgen here
        self._geno_pgen.change_sample_subset(np.sort(s_idx))
        if self._dose_path is not None:
            self._dose_pgen.change_sample_subset(np.sort(s_idx))
        return self

    @property
    def dosage_path(self) -> Path | None:
        """Path to the dosage file."""
        return self._dose_path

    @dosage_path.setter
    def dosage_path(self, dosage_path: str | Path | None):
        """Set the path to the dosage file."""
        if dosage_path is not None:
            dosage_path = Path(dosage_path)
            dose_samples = _read_psam(dosage_path.with_suffix(".psam"))
            if (np.asarray(self.available_samples) != dose_samples).any():
                raise ValueError(
                    "Samples in dosage file do not match those in genotype file."
                )
            self._dose_pgen = pgenlib.PgenReader(bytes(Path(dosage_path)))
        else:
            self._dose_pgen = self._geno_pgen
        self._dose_path = dosage_path

    def __del__(self):
        self._geno_pgen.close()
        if self._dose_pgen is not None:
            self._dose_pgen.close()

    def n_vars_in_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
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
        n_variants
            Shape: :code:`(ranges)`. Number of variants in the given ranges.
        """
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros_like(starts, dtype=np.uint32)

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full(n_ranges, c),
                    "Start": starts,
                    "End": ends,
                }
            ).to_pandas(use_pyarrow_extension_array=True)
        )
        return (
            queries.count_overlaps(self._index)
            .df["NumberOverlaps"]
            .to_numpy()
            .astype(np.uint32)
        )

    def var_idxs(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
    ) -> tuple[NDArray[V_IDX_TYPE], NDArray[OFFSET_TYPE]]:
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
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.empty(0, V_IDX_TYPE), np.zeros(n_ranges + 1, OFFSET_TYPE)

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full(n_ranges, c),
                    "Start": starts,
                    "End": ends,
                }
            )
            .with_row_index("query")
            .to_pandas(use_pyarrow_extension_array=True)
        )
        join = pl.from_pandas(queries.join(self._index).df)

        if join.height == 0:
            return np.empty(0, V_IDX_TYPE), np.zeros(n_ranges + 1, OFFSET_TYPE)

        join = join.sort("query", "index")
        idxs = join["index"].to_numpy()
        lens = self.n_vars_in_ranges(c, starts, ends)
        offsets = lengths_to_offsets(lens)
        return idxs, offsets

    def read(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = INT64_MAX,
        mode: type[T] = Genos,
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
            Type of data to read. Can be :code:`Genos`, :code:`Dosages`, :code:`GenosPhasing`,
            :code:`GenosDosages`, or :code:`GenosPhasingDosages`.
        out
            Array to write the data to. If None, a new array will be created. The shape and dtype of the array
            should match the expected output shape for the given mode. For example, if mode is :code:`Genos`,
            the shape should be :code:`(samples ploidy variants)`. If mode is :code:`Dosages`, the shape should
            be :code:`(samples variants)`.

        Returns
        -------
            Genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        """
        c = self._c_norm.norm(contig)
        if c is None:
            return mode.empty(self.n_samples, self.ploidy, 0)

        var_idxs, _ = self.var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return mode.empty(self.n_samples, self.ploidy, 0)

        if issubclass(mode, Genos):
            if out is not None:
                out = mode.parse(out)
            _out = self._read_genos(var_idxs, out)
        elif issubclass(mode, Dosages):
            if out is not None:
                out = mode.parse(out)
            _out = self._read_dosages(var_idxs, out)
        elif issubclass(mode, GenosPhasing):
            if out is not None:
                out = mode.parse(out)
            _out = self._read_genos_phasing(var_idxs, out)
        elif issubclass(mode, GenosDosages):
            if out is not None:
                out = mode.parse(out)
            _out = self._read_genos_dosages(var_idxs, out)
        elif issubclass(mode, GenosPhasingDosages):
            if out is not None:
                out = mode.parse(out)
            _out = self._read_genos_phasing_dosages(var_idxs, out)
        else:
            assert_never(mode)

        return _out  # type: ignore

    def chunk(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = INT64_MAX,
        max_mem: int | str = "4g",
        mode: type[T] = Genos,
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
            Type of data to read. Can be :code:`Genos`, :code:`Dosages`, :code:`GenosPhasing`,
            :code:`GenosDosages`, or :code:`GenosPhasingDosages`.

        Returns
        -------
            Generator of genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        """
        max_mem = parse_memory(max_mem)

        c = self._c_norm.norm(contig)
        if c is None:
            logger.warning(
                f"Query contig {contig} not found in VCF file, even after normalizing for UCSC/Ensembl nomenclature."
            )
            yield mode.empty(self.n_samples, self.ploidy, 0)
            return

        var_idxs, _ = self.var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
            yield mode.empty(self.n_samples, self.ploidy, 0)
            return

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        n_chunks = -(-n_variants // vars_per_chunk)
        v_chunks = np.array_split(var_idxs, n_chunks)
        for var_idx in v_chunks:
            if issubclass(mode, Genos):
                _out = self._read_genos(var_idx)
            elif issubclass(mode, Dosages):
                _out = self._read_dosages(var_idx)
            elif issubclass(mode, GenosPhasing):
                _out = self._read_genos_phasing(var_idx)
            elif issubclass(mode, GenosDosages):
                _out = self._read_genos_dosages(var_idx)
            elif issubclass(mode, GenosPhasingDosages):
                _out = self._read_genos_phasing_dosages(var_idx)
            else:
                assert_never(mode)

            yield mode.parse(_out)

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        mode: type[T] = Genos,
    ) -> tuple[T, NDArray[OFFSET_TYPE]]:
        """Read genotypes and/or dosages for multiple ranges.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions.
        ends
            0-based, exclusive end positions.
        mode
            Type of data to read. Can be :code:`Genos`, :code:`Dosages`, :code:`GenosPhasing`,
            :code:`GenosDosages`, or :code:`GenosPhasingDosages`.

        Returns
        -------
            Genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.

            Shape: (ranges+1). Offsets to slice out data for each range from the variants axis like so:

        Examples
        --------
        .. code-block:: python

            data, offsets = reader.read_ranges(...)
            data[..., offsets[i] : offsets[i + 1]]  # data for range i

        Note that the number of variants for range :code:`i` is :code:`np.diff(offsets)[i]`.
        """
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            logger.warning(
                f"Query contig {contig} not found in VCF file, even after normalizing for UCSC/Ensembl nomenclature."
            )
            return mode.empty(self.n_samples, self.ploidy, 0), np.zeros(
                n_ranges + 1, OFFSET_TYPE
            )

        var_idxs, offsets = self.var_idxs(c, starts, ends)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return mode.empty(self.n_samples, self.ploidy, 0), np.zeros(
                n_ranges + 1, OFFSET_TYPE
            )

        if issubclass(mode, Genos):
            out = self._read_genos(var_idxs)
        elif issubclass(mode, Dosages):
            out = self._read_dosages(var_idxs)
        elif issubclass(mode, GenosPhasing):
            out = self._read_genos_phasing(var_idxs)
        elif issubclass(mode, GenosDosages):
            out = self._read_genos_dosages(var_idxs)
        elif issubclass(mode, GenosPhasingDosages):
            out = self._read_genos_phasing_dosages(var_idxs)
        else:
            assert_never(mode)

        return cast(T, out), offsets

    def chunk_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        max_mem: int | str = "4g",
        mode: type[T] = Genos,
    ) -> Generator[Generator[T]]:
        """Read genotypes and/or dosages for multiple ranges in chunks limited by :code:`max_mem`.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions.
        ends
            0-based, exclusive end positions.
        max_mem
            Maximum memory to use for each chunk. Can be an integer or a string with a suffix
            (e.g. "4g", "2 MB").
        mode
            Type of data to read. Can be :code:`Genos`, :code:`Dosages`, :code:`GenosPhasing`,
            :code:`GenosDosages`, or :code:`GenosPhasingDosages`.

        Returns
        -------
            Generator of generators of genotypes and/or dosages of each ranges' data. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.

        Examples
        --------
        .. code-block:: python

            gen = reader.read_ranges_chunks(...)
            for range_ in gen:
                if range_ is None:
                    continue
                for chunk in range_:
                    # do something with chunk
                    pass
        """
        max_mem = parse_memory(max_mem)

        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        c = self._c_norm.norm(contig)
        if c is None:
            logger.warning(
                f"Query contig {contig} not found in VCF file, even after normalizing for UCSC/Ensembl nomenclature."
            )
            for _ in range(len(starts)):
                yield (mode.empty(self.n_samples, self.ploidy, 0) for _ in range(1))
            return

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

        var_idxs, offsets = self.var_idxs(c, starts, ends)
        vars_per_range = np.diff(offsets)
        tot_variants = len(var_idxs)
        if tot_variants == 0:
            for _ in range(len(starts)):
                yield (mode.empty(self.n_samples, self.ploidy, 0) for _ in range(1))
            return

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = np.minimum(max_mem // mem_per_v, vars_per_range)
        if vars_per_chunk.min() == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        chunks_per_range = -(-vars_per_range // vars_per_chunk)

        for (o_s, o_e), n_chunks in zip(windowed(offsets, 2), chunks_per_range):
            if o_s == o_e:
                yield (mode.empty(self.n_samples, self.ploidy, 0) for _ in range(1))
                continue

            range_idxs = var_idxs[o_s:o_e]
            v_chunks = np.array_split(range_idxs, n_chunks)

            if issubclass(mode, Genos):
                read = self._read_genos
            elif issubclass(mode, Dosages):
                read = self._read_dosages
            elif issubclass(mode, GenosPhasing):
                read = self._read_genos_phasing
            elif issubclass(mode, GenosDosages):
                read = self._read_genos_dosages
            elif issubclass(mode, GenosPhasingDosages):
                read = self._read_genos_phasing_dosages
            else:
                assert_never(mode)

            yield (cast(T, read(var_idx)) for var_idx in v_chunks)

    def _chunk_ranges_with_length(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        max_mem: int | str = "4g",
        mode: type[L] = Genos,
    ) -> Generator[
        Generator[
            tuple[L, POS_TYPE, NDArray[V_IDX_TYPE]]  # data, end, chunk_idxs
        ]
    ]:
        """Read genotypes and/or dosages for multiple ranges in chunks approximately limited by :code:`max_mem`.
        Will extend the ranges so that the returned data corresponds to haplotypes that have at least as much
        length as the original ranges.

        .. note::

            Even if the reader is set to only return dosages, this method must read in genotypes to compute
            haplotype lengths so there is no performance difference between reading with/without genotypes.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions.
        ends
            0-based, exclusive end positions.
        max_mem
            Maximum memory to use for each chunk. Can be an integer or a string with a suffix
            (e.g. "4g", "2 MB").
        mode
            Type of data to read. Can be :code:`Genos`, :code:`Dosages`, :code:`GenosPhasing`,
            :code:`GenosDosages`, or :code:`GenosPhasingDosages`.

        Returns
        -------
            Generator of generators of genotypes and/or dosages of each ranges' data, plus an integer indicating
            the 0-based end position of the final variant in the chunk. Genotypes have shape
            :code:`(samples ploidy variants)` and dosages have shape :code:`(samples variants)`. Missing genotypes
            have value -1 and missing dosages have value np.nan. If just using genotypes or dosages, will be a
            single array, otherwise will be a tuple of arrays.

        Examples
        --------
        .. code-block:: python

            gen = reader.read_ranges_chunks(...)
            for range_ in gen:
                if range_ is None:
                    continue
                for chunk in range_:
                    # do something with chunk
                    pass
        """
        if self._sei is None:
            raise ValueError(
                "Cannot use chunk_ranges_with_length without variant start, end, and ilen info, which usually happens when multi-allelic"
                " variants are present."
            )

        max_mem = parse_memory(max_mem)

        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

        c = self._c_norm.norm(contig)
        if c is None:
            logger.warning(
                f"Query contig {contig} not found in VCF file, even after normalizing for UCSC/Ensembl nomenclature."
            )
            for e in ends:
                yield (
                    (
                        mode.empty(self.n_samples, self.ploidy, 0),
                        e,
                        np.empty(0, dtype=V_IDX_TYPE),
                    )
                    for _ in range(1)
                )
            # we have full length, no deletions in any of the ranges
            return

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))

        var_idxs, offsets = self.var_idxs(c, starts, ends)
        tot_variants = len(var_idxs)
        if tot_variants == 0:
            for e in ends:
                yield (
                    (
                        mode.empty(self.n_samples, self.ploidy, 0),
                        e,
                        np.empty(0, dtype=V_IDX_TYPE),
                    )
                    for _ in range(1)
                )
            # we have full length, no deletions in any of the ranges
            return

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = min(max_mem // mem_per_v, tot_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        if issubclass(mode, Genos):
            read = self._read_genos
        elif issubclass(mode, GenosPhasing):
            read = self._read_genos_phasing
        elif issubclass(mode, GenosDosages):
            read = self._read_genos_dosages
        elif issubclass(mode, GenosPhasingDosages):
            read = self._read_genos_phasing_dosages
        else:
            assert_never(mode)

        read = cast(Callable[[NDArray[np.uint32]], L], read)

        for i, (s, e) in enumerate(zip(starts, ends)):
            o_s, o_e = offsets[i], offsets[i + 1]
            range_idxs = var_idxs[o_s:o_e]
            n_variants = len(range_idxs)
            if n_variants == 0:
                # we have full length, no deletions in any of the ranges
                yield (
                    (
                        mode.empty(self.n_samples, self.ploidy, 0),
                        e,
                        np.empty(0, dtype=V_IDX_TYPE),
                    )
                    for _ in range(1)
                )
                continue

            n_chunks = -(-n_variants // vars_per_chunk)
            v_chunks = np.array_split(range_idxs, n_chunks)

            yield _gen_with_length(
                v_chunks=v_chunks,
                q_start=s,
                q_end=e,
                read=read,
                v_starts=self._sei.v_starts,
                v_ends=self._sei.v_ends,
                ilens=self._sei.ilens,
                contig_max_idx=self._c_max_idxs[c],
            )

    def _mem_per_variant(self, mode: type[T]) -> int:
        mem = 0

        if issubclass(mode, Genos):
            mem += self.n_samples * self.ploidy * mode._dtype().itemsize
        elif issubclass(mode, Dosages):
            mem += self.n_samples * mode._dtype().itemsize
        elif issubclass(mode, GenosPhasing):
            mem += self.n_samples * self.ploidy * mode._dtypes[0]().itemsize
            mem += self.n_samples * mode._dtypes[1]().itemsize
        elif issubclass(mode, GenosDosages):
            mem += self.n_samples * self.ploidy * mode._dtypes[0]().itemsize
            mem += self.n_samples * mode._dtypes[1]().itemsize
        elif issubclass(mode, GenosPhasingDosages):
            mem += self.n_samples * self.ploidy * mode._dtypes[0]().itemsize
            mem += self.n_samples * mode._dtypes[1]().itemsize
            mem += self.n_samples * mode._dtypes[2]().itemsize
        else:
            assert_never(mode)

        if isinstance(self._s_sorter, np.ndarray):
            mem *= 2  # have to make a copy to sort by samples

        return mem

    def _read_genos(
        self, var_idxs: NDArray[V_IDX_TYPE], out: Genos | None = None
    ) -> Genos:
        if out is None:
            _out = np.empty(
                (len(var_idxs), self.n_samples * self.ploidy), dtype=np.int32
            )
        else:
            _out = out
        self._geno_pgen.read_alleles_list(var_idxs, _out)
        _out = _out.reshape(len(var_idxs), self.n_samples, self.ploidy).transpose(
            1, 2, 0
        )[self._s_sorter]
        _out[_out == -9] = -1
        return Genos(_out)

    def _read_dosages(
        self, var_idxs: NDArray[V_IDX_TYPE], out: Dosages | None = None
    ) -> Dosages:
        if out is None:
            _out = np.empty((len(var_idxs), self.n_samples), dtype=np.float32)
        else:
            _out = out

        self._dose_pgen.read_dosages_list(var_idxs, _out)
        _out = _out.transpose(1, 0)[self._s_sorter]
        _out[_out == -9] = np.nan

        return Dosages.parse(_out)

    def _read_genos_dosages(
        self, var_idxs: NDArray[V_IDX_TYPE], out: GenosDosages | None = None
    ) -> GenosDosages:
        if out is None:
            _out = (None, None)
        else:
            _out = out

        genos = self._read_genos(var_idxs, _out[0])
        dosages = self._read_dosages(var_idxs, _out[1])

        return GenosDosages((genos, dosages))

    def _read_genos_phasing(
        self, var_idxs: NDArray[V_IDX_TYPE], out: GenosPhasing | None = None
    ) -> GenosPhasing:
        if out is None:
            genos = np.empty(
                (len(var_idxs), self.n_samples * self.ploidy), dtype=np.int32
            )
            phasing = np.empty((len(var_idxs), self.n_samples), dtype=np.bool_)
        else:
            genos = out[0]
            phasing = out[1]

        self._dose_pgen.read_alleles_and_phasepresent_list(var_idxs, genos, phasing)
        genos = genos.reshape(len(var_idxs), self.n_samples, self.ploidy).transpose(
            1, 2, 0
        )[self._s_sorter]
        genos[genos == -9] = -1
        phasing = phasing.transpose(1, 0)[self._s_sorter]

        return GenosPhasing.parse((genos, phasing))

    def _read_genos_phasing_dosages(
        self, var_idxs: NDArray[V_IDX_TYPE], out: GenosPhasingDosages | None = None
    ) -> GenosPhasingDosages:
        if out is None:
            _out = (None, None)
        else:
            _out = (GenosPhasing(out[:2]), out[2])

        genos_phasing = self._read_genos_phasing(var_idxs, _out[0])
        dosages = self._read_dosages(var_idxs, _out[1])

        return GenosPhasingDosages((*genos_phasing, dosages))


def _gen_with_length(
    v_chunks: list[NDArray[V_IDX_TYPE]],
    q_start: int,
    q_end: int,
    read: Callable[[NDArray[V_IDX_TYPE]], L],
    v_starts: NDArray[POS_TYPE],  # full dataset v_starts
    v_ends: NDArray[POS_TYPE],  # full dataset v_ends
    ilens: NDArray[np.int32],  # full dataset ilens
    contig_max_idx: int,
) -> Generator[tuple[L, POS_TYPE, NDArray[V_IDX_TYPE]]]:
    # * This implementation computes haplotype lengths as shorter than they actually are if a spanning deletion is present
    # * This will result in including more variants than needed, which is fine since we're extending var_idx by more than we
    # * need to anyway.
    #! Assume len(v_chunks) > 0 and all len(var_idx) > 0 is guaranteed by caller
    length = q_end - q_start

    _idx_extension = 20
    for _, is_last, var_idx in mark_ends(v_chunks):
        last_end = cast(POS_TYPE, v_ends[var_idx[-1]])
        if not is_last:
            yield read(var_idx), last_end, var_idx
            continue

        ext_s_idx: int = min(var_idx[-1] + 1, contig_max_idx)
        # end idx is 0-based inclusive
        ext_e_idx = min(ext_s_idx + _idx_extension - 1, contig_max_idx)
        _idx_extension *= 2
        if ext_s_idx == ext_e_idx:
            # no extension needed
            yield read(var_idx), last_end, var_idx
            return

        var_idx = np.concatenate(
            [var_idx, np.arange(ext_s_idx, ext_e_idx + 1, dtype=V_IDX_TYPE)]
        )
        last_idx: V_IDX_TYPE = var_idx[-1]
        last_end = cast(POS_TYPE, v_ends[var_idx[-1]])

        # (s p v)
        out = read(var_idx)

        if ext_s_idx == ext_e_idx:
            yield out, last_end, var_idx
            return

        initial_len = max(length, last_end - q_start)

        if isinstance(out, Genos):
            # (s p)
            hap_lens = np.full(out.shape[:-1], initial_len, dtype=np.int32)
            hap_lens += hap_ilens(out, ilens[var_idx])
        else:
            # (s p)
            hap_lens = np.full(out[0].shape[:-1], initial_len, dtype=np.int32)
            hap_lens += hap_ilens(out[0], ilens[var_idx])

        ls_ext: list[L] = []
        while (hap_lens < length).any():
            ext_s_idx = min(last_idx + 1, contig_max_idx)
            # end idx is 0-based inclusive
            ext_e_idx = min(ext_s_idx + _idx_extension - 1, contig_max_idx)
            _idx_extension *= 2
            if ext_s_idx == ext_e_idx:
                break

            ext_idx = np.arange(ext_s_idx, ext_e_idx + 1, dtype=V_IDX_TYPE)
            last_idx = ext_idx[-1]
            ext_out = read(ext_idx)
            ls_ext.append(ext_out)

            if isinstance(ext_out, Genos):
                ext_genos = ext_out
            else:
                ext_genos = ext_out[0]

            dist = v_starts[ext_idx[-1]] - last_end
            hap_lens += dist + hap_ilens(ext_genos, ilens[ext_idx])
            last_end = cast(POS_TYPE, v_ends[ext_idx[-1]])

        if len(ls_ext) == 0:
            yield out, last_end, var_idx
            return

        if isinstance(out, Genos):
            out = np.concatenate([out, *ls_ext], axis=-1)
        else:
            out = tuple(
                np.concatenate([o, *ls], axis=-1) for o, ls in zip(out, zip(*ls_ext))
            )

        var_idx = np.arange(var_idx[0], last_idx + 1, dtype=V_IDX_TYPE)
        yield (
            out,  # type: ignore
            last_end,
            var_idx,
        )


def _read_psam(path: Path) -> NDArray[np.str_]:
    with open(path.with_suffix(".psam")) as f:
        cols = [c.strip("#") for c in f.readline().strip().split()]

    psam = pl.read_csv(
        path.with_suffix(".psam"),
        separator="\t",
        has_header=False,
        skip_rows=1,
        new_columns=cols,
        schema_overrides={
            "FID": pl.Utf8,
            "IID": pl.Utf8,
            "SID": pl.Utf8,
            "PAT": pl.Utf8,
            "MAT": pl.Utf8,
            "SEX": pl.Utf8,
        },
    )
    samples = psam["IID"].to_numpy().astype(str)
    return samples


class StartsEndsIlens:
    v_starts: NDArray[POS_TYPE]
    """0-based starts, sorted."""
    v_ends: NDArray[POS_TYPE]
    """0-based exclusive ends, sorted by start."""
    ilens: NDArray[np.int32]
    """Indel lengths, sorted by start."""
    alt: pl.Series
    """Alternate alleles, sorted by start."""

    def __init__(
        self,
        v_starts: NDArray[POS_TYPE],
        v_ends: NDArray[POS_TYPE],
        ilens: NDArray[np.int32],
        alt: pl.Series,
    ):
        self.v_starts = v_starts
        self.v_ends = v_ends
        self.ilens = ilens
        self.alt = alt


def _valid_index(index_path: Path) -> bool:
    """Check if the index is valid. Needs to exist and have a modified time greater than
    the PVAR file."""
    if not index_path.exists():
        return False

    pvar_mtime = index_path.with_suffix("").stat().st_mtime_ns
    index_mtime = index_path.stat().st_mtime_ns
    return index_mtime > pvar_mtime


def _read_index(
    index_path: Path, filter: pl.Expr | None
) -> tuple[pr.PyRanges, StartsEndsIlens | None, list[str]]:
    if not _valid_index(index_path):
        logger.info("Genoray PVAR index not found or out-of-date, creating index.")
        _write_index(index_path)

    logger.info("Loading genoray index.")
    index = pl.scan_ipc(
        index_path, row_index_name="index", memory_map=False
    ).with_columns(
        Start=pl.col("POS") - 1,
        End=pl.col("POS") + pl.col("REF").str.len_bytes() - 1,
    )

    if filter is None:
        has_multiallelics = index.select((~is_biallelic).any()).collect().item()
    else:
        has_multiallelics = (
            index.filter(filter).select((~is_biallelic).any()).collect().item()
        )

    if has_multiallelics:
        sei = None
    else:
        # can keep the first alt for multiallelic sites since they're getting filtered out
        # anyway, so they won't be accessed
        # if the filter is changed, the index is invalidated and re-read (see filter setter)
        if filter is None:
            data = index.select("Start", "End", pl.col("ILEN", "ALT").list.first())
        else:
            data = index.with_columns(
                ILEN=pl.when(filter)
                .then(pl.col("ILEN").list.first())
                .otherwise(pl.lit(0))
            )
            data = data.select("Start", "End", "ILEN", pl.col("ALT").list.first())
        data = data.collect()
        v_starts = data["Start"].to_numpy()
        v_ends = data["End"].to_numpy()
        ilens = data["ILEN"].to_numpy()
        alt = data["ALT"]
        sei = StartsEndsIlens(v_starts, v_ends, ilens, alt)

    if filter is not None:
        index = index.filter(filter)

    index = index.select("index", "Start", "End", Chromosome="CHROM").collect()
    # PVAR contigs are not necessarily sorted, only guaranteed to be sorted within a contig
    contigs = index["Chromosome"].unique(maintain_order=True).to_list()
    pyr = pr.PyRanges(index.to_pandas(use_pyarrow_extension_array=True))
    return pyr, sei, contigs


# TODO: can this be implemented using the NCLS lib underlying PyRanges? Then we can
# pass np.memmap arrays directly instead of having to futz with DataFrames? This will likely make
# filtering less ergonomic/harder to make ergonomic though, but a memmap approach should be scalable
# to datasets with billions+ unique variants (reduce memory), reduce instantion time, but ~increase query time.
# Unless, NCLS creates a bunch of data structures in memory anyway.
def _write_index(index_path: Path):
    """Write PVAR index."""

    (
        _scan_pvar(index_path.with_suffix(""))
        .rename({"#CHROM": "CHROM"})
        .with_columns(ALT=pl.col("ALT").str.split(","))
        .with_columns(ILEN=ILEN)
        .sink_ipc(index_path)
    )


def _scan_pvar(pvar: Path):
    pvar_schema = {
        "#CHROM": pl.Utf8,
        "POS": pl.Int64,
        "ID": pl.Utf8,
        "REF": pl.Utf8,
        "ALT": pl.Utf8,
        "QUAL": pl.Float64,
        "FILTER": pl.Utf8,
        "INFO": pl.Utf8,
        "CM": pl.Float64,
    }

    cols = None
    is_pvar = False
    if pvar.suffix == ".zst":
        opener = ZstdFile
    else:
        opener = partial(open, mode="r")
    with opener(pvar) as f:
        for line in f:
            if line.startswith("##"):
                is_pvar = True
                continue
            if line.startswith("#"):
                is_pvar = True
            cols = [c for c in line.strip().split("\t")]
            break

    if not is_pvar:
        return _scan_bim(pvar)

    if cols is None:
        raise ValueError(f"No non-comment lines in PVAR file: {pvar}")

    if "FORMAT" in cols:
        raise RuntimeError("PVAR does not support the FORMAT column.")

    return pl.scan_csv(
        pvar,
        separator="\t",
        comment_prefix="##",
        schema={c: pvar_schema[c] for c in cols},
        null_values=".",
    )


class ZstdFile(TextIOWrapper):
    def __init__(self, path: Path):
        self.path = path
        self.reader = ZstdDecompressor().stream_reader(open(path, "rb"))
        super().__init__(self.reader, newline="\n", encoding="utf-8")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.reader.close()
        return super().__exit__(exc_type, exc_val, exc_tb)


def _scan_bim(bim: Path):
    with open(bim, "r") as f:
        n_cols = len(f.readline().strip().split("\t"))

    schema = {
        "#CHROM": pl.Categorical,
        "ID": pl.Utf8,
        "CM": pl.Float64,
        "POS": pl.Int32,
        "ALT": pl.Utf8,
        "REF": pl.Utf8,
    }

    if n_cols == 5:
        del schema["CM"]

    return pl.scan_csv(
        bim,
        separator="\t",
        has_header=False,
        schema=schema,
        null_values=".",
    ).filter(pl.col("POS") > 0)
