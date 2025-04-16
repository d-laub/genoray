from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator, TypeVar, cast

import numpy as np
import pgenlib
import polars as pl
import pyranges as pr
from hirola import HashTable
from more_itertools import mark_ends
from numpy.typing import ArrayLike, NDArray
from phantom import Phantom
from typing_extensions import Self, TypeGuard, assert_never

from ._types import (  #! we have to use INT64_MAX because this is what PyRanges uses!
    INT64_MAX,
    PGEN_R_DTYPE,
)
from ._utils import (
    ContigNormalizer,
    format_memory,
    hap_ilens,
    lengths_to_offsets,
    parse_memory,
)


def _is_genos(obj: Any) -> TypeGuard[Genos]:
    return (
        isinstance(obj, np.ndarray)
        and obj.dtype.type == np.int32
        and obj.ndim == 3
        and obj.shape[1] == 2
    )


class Genos(NDArray[np.int32], Phantom, predicate=_is_genos):
    _dtype = np.int32


def _is_dosages(obj: Any) -> TypeGuard[Dosages]:
    return (
        isinstance(obj, np.ndarray) and obj.dtype.type == np.float32 and obj.ndim == 2
    )


class Dosages(NDArray[np.float32], Phantom, predicate=_is_dosages):
    _dtype = np.float32


def _is_phasing(obj: Any) -> TypeGuard[Phasing]:
    return isinstance(obj, np.ndarray) and obj.dtype.type == np.bool_ and obj.ndim == 2


class Phasing(NDArray[np.bool_], Phantom, predicate=_is_phasing):
    _dtype = np.bool_


def _is_genos_phasing(obj) -> TypeGuard[GenosPhasing]:
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], Genos)
        and isinstance(obj[1], Phasing)
    )


class GenosPhasing(tuple[Genos, Phasing], Phantom, predicate=_is_genos_phasing):
    _dtypes = (np.int32, np.bool_)


def _is_genos_dosages(obj) -> TypeGuard[GenosDosages]:
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], Genos)
        and isinstance(obj[1], Dosages)
    )


class GenosDosages(tuple[Genos, Dosages], Phantom, predicate=_is_genos_dosages):
    _dtypes = (np.int32, np.float32)


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


T = TypeVar("T", Genos, Dosages, GenosPhasing, GenosDosages, GenosPhasingDosages)
L = TypeVar("L", Genos, GenosPhasing, GenosDosages, GenosPhasingDosages)


class PGEN:
    available_samples: list[str]
    """List of available samples in the PGEN file."""
    filter: pl.Expr | None
    """Polars expression to filter variants. Should return True for variants to keep."""
    ploidy = 2
    """Ploidy of the samples. The PGEN format currently only supports diploid (2)."""
    contigs: list[str]
    """List of contig names in the PGEN file."""
    _index: pr.PyRanges
    _geno_pgen: pgenlib.PgenReader
    _dose_pgen: pgenlib.PgenReader
    _s_idx: NDArray[np.uint32] | slice
    _s_sorter: NDArray[np.intp] | slice
    _dose_path: Path | None
    _sei: StartsEndsIlens | None  # unfiltered so that var_idxs map correctly

    Genos = Genos
    """:code:`(samples ploidy variants) int32`"""
    Dosages = Dosages
    """:code:`(samples variants) float32`"""
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
        """Create a PGEN reader.

        Parameters
        ----------
        path
            Path to the PGEN file. Only used for genotypes if a dosage path is provided as well.
        filter
            Polars expression to filter variants. Should return True for variants to keep.
        dosage_path
            Path to a dosage PGEN file. If None, the genotype PGEN file will be used for both genotypes and dosages.
        """
        geno_path = Path(geno_path)
        samples = _read_psam(geno_path.with_suffix(".psam"))

        self.filter = filter
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

        if not geno_path.with_suffix(".pvar.gvi").exists():
            _write_index(geno_path.with_suffix(".pvar"))
        self._index, self._sei = _read_index(
            geno_path.with_suffix(".pvar.gvi"), self.filter
        )
        self.contigs = self._index.chromosomes
        self._c_norm = ContigNormalizer(self._index.chromosomes)

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

    def set_samples(self, samples: list[str] | None) -> Self:
        """Set the samples to use.

        Parameters
        ----------
        samples
            List of sample names to use. If None, all samples will be used.
        """
        if samples is None or samples == self.available_samples:
            self._s_idx = slice(None)
            self._s_sorter = slice(None)
            return self

        _samples = np.atleast_1d(samples)
        s_idx = self._s2i.get(_samples).astype(np.uint32)
        if (missing := _samples[s_idx == -1]).any():
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
        starts = np.atleast_1d(np.asarray(starts, PGEN_R_DTYPE))

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros_like(starts, dtype=np.uint32)

        ends = np.atleast_1d(np.asarray(ends, PGEN_R_DTYPE))
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full_like(starts, contig),
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
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint64]]:
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
        starts = np.atleast_1d(np.asarray(starts, PGEN_R_DTYPE))

        c = self._c_norm.norm(contig)
        if c is None:
            return np.empty(0, np.uint32), np.zeros_like(starts, np.uint64)

        ends = np.atleast_1d(np.asarray(ends, PGEN_R_DTYPE))
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full(len(starts), c),
                    "Start": starts,
                    "End": ends,
                }
            )
            .with_row_index("query")
            .to_pandas(use_pyarrow_extension_array=True)
        )
        join = pl.from_pandas(queries.join(self._index).df)
        if join.height == 0:
            return np.empty(0, np.uint32), np.zeros_like(
                np.atleast_1d(starts), np.uint64
            )
        join = join.sort("query", "index")
        idxs = join["index"].to_numpy()
        lens = (
            join.group_by("query", maintain_order=True).agg(pl.len())["len"].to_numpy()
        )
        offsets = lengths_to_offsets(lens)
        return idxs, offsets

    def read(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = INT64_MAX,
        mode: type[T] = Genos,
        out: T | None = None,
    ) -> T | None:
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
            return

        var_idxs, _ = self.var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

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
            return

        var_idxs, _ = self.var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
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
    ) -> tuple[T, NDArray[np.uint64]] | None:
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
        c = self._c_norm.norm(contig)
        if c is None:
            return

        var_idxs, offsets = self.var_idxs(c, starts, ends)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

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

        return mode.parse(out), offsets

    def chunk_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        max_mem: int | str = "4g",
        mode: type[T] = Genos,
    ) -> Generator[Generator[T] | None]:
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

        c = self._c_norm.norm(contig)
        if c is None:
            return

        starts = np.atleast_1d(np.asarray(starts, PGEN_R_DTYPE))
        ends = np.atleast_1d(np.asarray(ends, PGEN_R_DTYPE))

        var_idxs, offsets = self.var_idxs(c, starts, ends)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        for i in range(len(offsets) - 1):
            o_s, o_e = offsets[i], offsets[i + 1]
            range_idxs = var_idxs[o_s:o_e]
            n_variants = len(range_idxs)

            if n_variants == 0:
                yield None
                continue

            n_chunks = -(-n_variants // vars_per_chunk)
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

            yield (mode.parse(read(var_idx)) for var_idx in v_chunks)

    def chunk_ranges_with_length(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        max_mem: int | str = "4g",
        mode: type[L] = Genos,
    ) -> Generator[Generator[L] | None]:
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
            Generator of generators of genotypes and/or dosages of each ranges' data. Genotypes have shape
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

        c = self._c_norm.norm(contig)
        if c is None:
            # we have full length, no deletions in any of the ranges
            return

        starts = np.atleast_1d(np.asarray(starts, PGEN_R_DTYPE))
        ends = np.atleast_1d(np.asarray(ends, PGEN_R_DTYPE))

        var_idxs, offsets = self.var_idxs(c, starts, ends)
        n_variants = len(var_idxs)
        if n_variants == 0:
            # we have full length, no deletions in any of the ranges
            return

        mem_per_v = self._mem_per_variant(mode)
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
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

        for i, s, e in zip(range(len(offsets) - 1), starts, ends):
            o_s, o_e = offsets[i], offsets[i + 1]
            range_idxs = var_idxs[o_s:o_e]
            n_variants = len(range_idxs)
            if n_variants == 0:
                # we have full length, no deletions in any of the ranges
                yield None
                return
            n_chunks = -(-n_variants // vars_per_chunk)
            v_chunks = np.array_split(range_idxs, n_chunks)

            yield _gen_with_length(
                v_chunks=v_chunks,
                length=e - s,
                read=read,
                v_starts=self._sei.v_starts,
                v_ends=self._sei.v_ends,
                ilens=self._sei.ilens,
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

        return mem

    def _read_genos(
        self, var_idxs: NDArray[np.uint32], out: Genos | None = None
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
        self, var_idxs: NDArray[np.uint32], out: Dosages | None = None
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
        self, var_idxs: NDArray[np.uint32], out: GenosDosages | None = None
    ) -> GenosDosages:
        if out is None:
            _out = (None, None)
        else:
            _out = out

        genos = self._read_genos(var_idxs, _out[0])
        dosages = self._read_dosages(var_idxs, _out[1])

        return GenosDosages((genos, dosages))

    def _read_genos_phasing(
        self, var_idxs: NDArray[np.uint32], out: GenosPhasing | None = None
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
        self, var_idxs: NDArray[np.uint32], out: GenosPhasingDosages | None = None
    ) -> GenosPhasingDosages:
        if out is None:
            _out = (None, None)
        else:
            _out = (GenosPhasing(out[:2]), out[2])

        genos_phasing = self._read_genos_phasing(var_idxs, _out[0])
        dosages = self._read_dosages(var_idxs, _out[1])

        return GenosPhasingDosages((*genos_phasing, dosages))


_IDX_EXTENSION = 20


def _gen_with_length(
    v_chunks: list[NDArray[np.uint32]],
    length: int,
    read: Callable[[NDArray[np.uint32]], L],
    v_starts: NDArray[PGEN_R_DTYPE],  # full dataset v_starts
    v_ends: NDArray[PGEN_R_DTYPE],  # full dataset v_ends
    ilens: NDArray[np.int32],  # full dataset ilens
) -> Generator[L]:
    # * This implementation computes haplotype lengths as shorter than they actually are if a spanning deletion is present
    # * This this will result in including more variants than needed, which is fine since we're extending var_idx by more than we
    # * need to anyway.
    #! Assume len(v_chunks) > 0 and all len(var_idx) > 0 is guaranteed by caller

    max_idx = len(ilens) - 1
    for _, is_last, var_idx in mark_ends(v_chunks):
        if not is_last:
            yield read(var_idx)
            continue

        ext_s_idx = min(var_idx[-1] + 1, max_idx)
        ext_e_idx = min(ext_s_idx + _IDX_EXTENSION - 1, max_idx)
        if ext_s_idx == ext_e_idx:
            yield read(var_idx)
            return

        var_idx = np.concat(
            [var_idx, np.arange(ext_s_idx, ext_e_idx + 1, dtype=np.uint32)]
        )
        out = read(var_idx)

        if isinstance(out, Genos):
            hap_lens = np.full(out.shape[:-1], length, dtype=np.int32)
            hap_lens += hap_ilens(out, ilens[var_idx])
        else:
            hap_lens = np.full(out[0].shape[:-1], length, dtype=np.int32)
            hap_lens += hap_ilens(out[0], ilens[var_idx])

        ls_ext: list[L] = []
        last_end = v_ends[var_idx[-1]]
        while (hap_lens < length).any() and ext_s_idx < max_idx:
            ext_s_idx = min(var_idx[-1] + 1, max_idx)
            ext_e_idx = min(ext_s_idx + _IDX_EXTENSION - 1, max_idx)
            if ext_s_idx == ext_e_idx:
                break

            var_idx = np.arange(ext_s_idx, ext_e_idx + 1, dtype=np.uint32)
            ext_out = read(var_idx)
            ls_ext.append(ext_out)

            if isinstance(ext_out, Genos):
                ext_genos = ext_out
            else:
                ext_genos = ext_out[0]

            dist = v_starts[var_idx[-1]] - last_end
            hap_lens += dist + hap_ilens(ext_genos, ilens[var_idx])
            last_end = v_ends[var_idx[-1]]

        if len(ls_ext) == 0:
            yield out
            return

        if isinstance(out, Genos):
            out = np.concat([out, *ls_ext], axis=-1)
        else:
            out = tuple(
                np.concat([o, *ls], axis=-1) for o, ls in zip(out, zip(*ls_ext))
            )
        yield out  # type: ignore


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


# TODO: can index be implemented using the NCLS lib underlying PyRanges? Then we can
# pass np.memmap arrays directly instead of having to futz with DataFrames. This will likely make
# filtering less ergonomic/harder to make ergonomic though, but a memmap approach should be scalable
# to datasets with billions+ unique variants (reduce memory), reduce instantion time, but increase query time.
# Unless, NCLS creates a bunch of data structures in memory anyway.
# TODO: support full .pvar and .bim spec see https://www.cog-genomics.org/plink/2.0/formats#pvar
def _write_index(pvar: Path):
    ILEN = pl.col("ALT").str.len_bytes().cast(pl.Int32) - pl.col("rlen").cast(pl.Int32)
    KIND = (
        pl.when(ILEN != 0)
        .then(pl.lit("INDEL"))
        .when(pl.col("rlen") == 1)  # ILEN == 0 and RLEN == 1
        .then(pl.lit("SNP"))
        .when(pl.col("rlen") > 1)  # ILEN == 0 and RLEN > 1
        .then(pl.lit("MNP"))  # ILEN == 0 and RLEN > 1
        .otherwise(pl.lit("OTHER"))
        .cast(pl.Categorical)
    )

    (
        pl.scan_csv(
            pvar,
            separator="\t",
            comment_prefix="##",
            schema_overrides={
                "#CHROM": pl.Categorical,
                "POS": pl.Int64,  #! MAKE SURE THIS MATCHES PGEN_R_DTYPE
            },
        )
        .with_columns(
            Chromosome="#CHROM",
            Start=pl.col("POS") - 1,
            End=pl.col("POS") + pl.col("REF").str.len_bytes() - 1,
            ALT=pl.col("ALT").str.split(","),
            rlen=pl.col("REF").str.len_bytes(),
        )
        .drop("#CHROM", "POS")
        .with_row_index("index")
        .explode("ALT")
        .with_columns(ilen=ILEN, kind=KIND)
        .drop("rlen")
        .group_by("index")
        .agg(pl.exclude("ALT", "ilen", "kind").first(), "ALT", "ilen", "kind")
        .drop("index")
        .sink_ipc(pvar.with_suffix(".pvar.gvi"))
    )


class StartsEndsIlens:
    v_starts: NDArray[PGEN_R_DTYPE]
    v_ends: NDArray[PGEN_R_DTYPE]
    ilens: NDArray[np.int32]

    def __init__(
        self,
        v_starts: NDArray[PGEN_R_DTYPE],
        v_ends: NDArray[PGEN_R_DTYPE],
        ilens: NDArray[np.int32],
    ):
        self.v_starts = v_starts
        self.v_ends = v_ends
        self.ilens = ilens


def _read_index(
    path: Path, filter: pl.Expr | None
) -> tuple[pr.PyRanges, StartsEndsIlens | None]:
    index = pl.scan_ipc(path, row_index_name="index", memory_map=False)

    if filter is None:
        has_multiallelics = (
            index.select((pl.col("ALT").list.len() != 1).any()).collect().item()
        )
    else:
        has_multiallelics = (
            index.filter(filter)
            .select((pl.col("ALT").list.len() != 1).any())
            .collect()
            .item()
        )

    if has_multiallelics:
        sei = None
    else:
        # can just leave the first alt for multiallelic sites since they're getting filtered out
        # anyway, so they won't be accessed
        data = index.select("Start", "End", pl.col("ilen").list.first()).collect()
        v_starts = data["Start"].to_numpy()
        v_ends = data["End"].to_numpy()
        ilens = data["ilen"].to_numpy()
        sei = StartsEndsIlens(v_starts, v_ends, ilens)

    if filter is not None:
        index = index.filter(filter)

    pyr = pr.PyRanges(
        index.select("Chromosome", "Start", "End", "index")
        .collect()
        .to_pandas(use_pyarrow_extension_array=True)
    )
    return pyr, sei
