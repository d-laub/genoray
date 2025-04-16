from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar, cast

import cyvcf2
import numpy as np
from more_itertools import mark_ends
from numpy.typing import ArrayLike, NDArray
from phantom import Phantom
from tqdm.auto import tqdm
from typing_extensions import Self, TypeGuard, assert_never

from ._types import UINT64_MAX, VCF_R_DTYPE
from ._utils import (
    ContigNormalizer,
    format_memory,
    hap_ilens,
    parse_memory,
)


class DosageFieldError(RuntimeError): ...


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


def _is_genos16(obj: Any) -> TypeGuard[NDArray[np.int16]]:
    return (
        isinstance(obj, np.ndarray)
        and obj.dtype.type == np.int16
        and obj.ndim == 3
        and obj.shape[1] in (2, 3)
    )


class Genos16(NDArray[np.int16], Phantom, predicate=_is_genos16):
    _gdtype = np.int16


def _is_dosages(obj: Any) -> TypeGuard[NDArray[np.float32]]:
    return (
        isinstance(obj, np.ndarray) and obj.dtype.type == np.float32 and obj.ndim == 2
    )


class Dosages(NDArray[np.float32], Phantom, predicate=_is_dosages): ...


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


T = TypeVar("T", Genos8, Genos16, Dosages, Genos8Dosages, Genos16Dosages)
L = TypeVar("L", Genos8, Genos16, Genos8Dosages, Genos16Dosages)
G = TypeVar("G", Genos8, Genos16)
GD = TypeVar("GD", Genos8Dosages, Genos16Dosages)


class VCF:
    path: Path
    """Path to the VCF file."""
    available_samples: list[str]
    """List of available samples in the VCF file."""
    contigs: list[str]
    """List of available contigs in the VCF file."""
    ploidy = 2
    """Ploidy of the VCF file. This is currently always 2 since we use cyvcf2."""
    filter: Callable[[cyvcf2.Variant], bool] | None
    """Function to filter variants. Should return True for variants to keep."""
    phasing: bool
    """Whether to include phasing information on genotypes. If True, the ploidy axis will be length 3 such that
    phasing is indicated by the 3rd value: 0 = unphased, 1 = phased. If False, the ploidy axis will be length 2."""
    dosage_field: str | None
    """Name of the dosage field to read from the VCF file. Required if you want to use modes that include dosages."""
    _pbar: tqdm | None
    """A progress bar to use while reading variants. This will be incremented per variant
    during any calls to a read function."""
    _vcf: cyvcf2.VCF
    _s_sorter: NDArray[np.intp] | slice
    _samples: list[str]
    _c_norm: ContigNormalizer

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
        phasing: bool = False,
        dosage_field: str | None = None,
        progress: bool = False,
    ):
        """Create a VCF reader.

        Parameters
        ----------
        path
            Path to the VCF file.
        filter
            Function to filter variants. Should return True for variants to keep.
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
        self.path = Path(path)
        self.filter = filter
        self.phasing = phasing
        self.dosage_field = dosage_field

        self._vcf = self._open()
        self.available_samples = self._vcf.samples
        self.contigs = self._vcf.seqnames
        self._c_norm = ContigNormalizer(self.contigs)
        self.progress = progress
        self._pbar = None
        self.set_samples(self._vcf.samples)

    def _open(self, samples: list[str] | None = None) -> cyvcf2.VCF:
        return cyvcf2.VCF(self.path, samples=samples, lazy=True)

    @property
    def current_samples(self) -> list[str]:
        """List of samples currently being read from the VCF file."""
        return self._samples

    @property
    def n_samples(self) -> int:
        """Number of samples in the VCF file."""
        return len(self._samples)

    def set_samples(self, samples: list[str] | None) -> Self:
        """Set the samples to read from the VCF file. Modifies the VCF reader in place and returns it.

        Parameters
        ----------
        samples
            List of sample names to read from the VCF file.

        Returns
        -------
            The VCF reader with the specified samples.
        """
        if samples is None or samples == self.available_samples:
            self._s_sorter = slice(None)
            self._samples = self.available_samples
            self._vcf = self._open(samples)
            return self

        if missing := set(samples).difference(self.available_samples):
            raise ValueError(
                f"Samples {missing} not found in the VCF file. "
                f"Available samples: {self.available_samples}"
            )
        self._vcf = self._open(samples)
        _, s_idx, _ = np.intersect1d(self._vcf.samples, samples, return_indices=True)
        self._samples = samples
        self._s_sorter = np.argsort(s_idx)
        return self

    def __del__(self):
        self._vcf.close()

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
        ends: ArrayLike = UINT64_MAX,
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
        starts = np.atleast_1d(np.asarray(starts, VCF_R_DTYPE))
        ends = np.atleast_1d(np.asarray(ends, VCF_R_DTYPE))

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros_like(starts, np.uint32)

        out = np.empty_like(starts, np.uint32)
        for i, (s, e) in enumerate(zip(starts, ends)):
            coord = f"{c}:{s + 1}-{e}"
            if self.filter is None:
                out[i] = sum(1 for _ in self._vcf(coord))
            else:
                out[i] = sum(self.filter(v) for v in self._vcf(coord))

        return out

    def read(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = UINT64_MAX,
        mode: type[T] = Genos16,
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

        c = self._c_norm.norm(contig)
        if c is None:
            return

        itr = self._vcf(f"{c}:{start + 1}-{end}")  # range string is 1-based
        ploidy = self.ploidy + self.phasing
        if out is None:
            n_variants: np.uint32 = self.n_vars_in_ranges(c, start, end)[0]
            if n_variants == 0:
                return

            if issubclass(mode, (Genos8, Genos16)):
                data = np.empty(
                    (self.n_samples, ploidy, n_variants), dtype=mode._gdtype
                )
                self._fill_genos(itr, data)
            elif issubclass(mode, Dosages):
                data = np.empty((self.n_samples, n_variants), dtype=np.float32)
                self._fill_dosages(
                    itr,
                    data,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                data = (
                    np.empty(
                        (self.n_samples, ploidy, n_variants),
                        dtype=mode._gdtype,
                    ),
                    np.empty((self.n_samples, n_variants), dtype=np.float32),
                )
                self._fill_genos_and_dosages(
                    itr,
                    data,
                    self.dosage_field,  # type: ignore | guaranteed to be str by init guard clause
                )
            else:
                assert_never(mode)

            out = mode.parse(data)
        else:
            if issubclass(mode, (Genos8, Genos16)):
                if not isinstance(out, (Genos8, Genos16)):
                    raise ValueError("Expected an int8 output array.")
                self._fill_genos(itr, out)
            elif issubclass(mode, Dosages):
                if not isinstance(out, Dosages):
                    raise ValueError("Expected a float32 output array.")
                self._fill_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                if not isinstance(out, (Genos8Dosages, Genos16Dosages)):
                    raise ValueError(
                        "Expected a 2-tuple of int8 and np.float32 arrays."
                    )
                self._fill_genos_and_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                assert_never(mode)

        return out

    def chunk(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = UINT64_MAX,
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

        c = self._c_norm.norm(contig)
        if c is None:
            return

        n_variants: int = self.n_vars_in_ranges(c, start, end)[0]
        if n_variants == 0:
            return

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

        itr = self._vcf(f"{c}:{start + 1}-{end}")  # range string is 1-based
        ploidy = self.ploidy + self.phasing
        for chunk_size in chunk_sizes:
            if issubclass(mode, (Genos8, Genos16)):
                out = np.empty((self.n_samples, ploidy, chunk_size), dtype=mode._gdtype)
                self._fill_genos(itr, out)
            elif issubclass(mode, Dosages):
                out = np.empty((self.n_samples, chunk_size), dtype=np.float32)
                self._fill_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                out = (
                    np.empty(
                        (self.n_samples, ploidy, chunk_size),
                        dtype=mode._gdtype,
                    ),
                    np.empty((self.n_samples, chunk_size), dtype=np.float32),
                )
                self._fill_genos_and_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                assert_never(mode)

            yield mode.parse(out)

    def chunk_with_length(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = UINT64_MAX,
        max_mem: int | str = "4g",
        mode: type[L] = Genos16,
    ) -> Generator[L]:
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
            Generator of genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
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

        c = self._c_norm.norm(contig)
        if c is None:
            return

        n_variants: int = self.n_vars_in_ranges(c, start, end)[0]
        if n_variants == 0:
            return

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

        itr = self._vcf(f"{c}:{start + 1}-{end}")  # range string is 1-based
        ploidy = self.ploidy + self.phasing
        hap_lens = np.full((self.n_samples, self.ploidy), end - start, dtype=np.int32)
        for _, is_last, chunk_size in mark_ends(chunk_sizes):
            ilens = np.empty(chunk_size, dtype=np.int32)
            if issubclass(mode, (Genos8, Genos16)):
                out = np.empty((self.n_samples, ploidy, chunk_size), dtype=mode._gdtype)
                last_end = self._fill_genos(itr, out, ilens)
                hap_lens += hap_ilens(out[:, : self.ploidy], ilens)
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                out = (
                    np.empty(
                        (self.n_samples, ploidy, chunk_size),
                        dtype=mode._gdtype,
                    ),
                    np.empty((self.n_samples, chunk_size), dtype=np.float32),
                )
                last_end = self._fill_genos_and_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                    ilens,
                )
                hap_lens += hap_ilens(out[0][:, : self.ploidy], ilens)
            else:
                assert_never(mode)

            mode = cast(type[L], mode)

            if not is_last:
                yield mode.parse(out)
                continue

            if issubclass(mode, (Genos8, Genos16)):
                ls_ext = self._ext_genos_with_length(
                    c, start, end, hap_lens, mode, last_end
                )
            elif issubclass(mode, (Genos8Dosages, Genos16Dosages)):
                ls_ext = self._ext_genos_dosages_with_length(
                    c,
                    start,
                    end,
                    hap_lens,
                    mode,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                    last_end,
                )
            else:
                assert_never(mode)

            if len(ls_ext) > 0:
                if issubclass(mode, (Genos8, Genos16)):
                    out = np.concat([out, *ls_ext], axis=-1)
                else:
                    out = tuple(
                        np.concat([o, *ls], axis=-1) for o, ls in zip(out, zip(*ls_ext))
                    )

            yield out  # type: ignore

    def _fill_genos(
        self,
        vcf: cyvcf2.VCF,
        out: NDArray[np.int8 | np.int16],
        ilens: NDArray[np.int32] | None = None,
    ) -> int:
        #! assumes n_variants > 0
        n_variants = out.shape[-1]

        if self.progress and self._pbar is None:
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit=" variant")

        i = 0
        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue

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

        return v.end  # type: ignore

    def _fill_dosages(
        self, vcf: cyvcf2.VCF, out: NDArray[np.float32], dosage_field: str
    ) -> int:
        #! assumes n_variants > 0
        n_variants = out.shape[-1]
        if self.progress and self._pbar is None:
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit=" variant")

        i = 0
        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue
            d = v.format(dosage_field)
            if d is None:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                )
            # (s, 1, 1) or (s, 1)? -> (s)
            out[..., i] = d.squeeze(1)[self._s_sorter]

            if self._pbar is not None:
                self._pbar.update()

            if i == n_variants - 1:
                break

        if i != n_variants - 1:
            raise ValueError("Not enough variants found in the given range.")

        return v.end  # type: ignore

    def _fill_genos_and_dosages(
        self,
        vcf: cyvcf2.VCF,
        out: tuple[NDArray[np.int8 | np.int16], NDArray[np.float32]],
        dosage_field: str,
        ilens: NDArray[np.int32] | None = None,
    ) -> int:
        #! assumes n_variants > 0
        n_variants = out[0].shape[-1]

        if self.progress and self._pbar is None:
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit=" variant")

        i = 0
        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue

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
            # (s, 1, 1) or (s, 1)? -> (s)
            out[1][..., i] = d.squeeze(1)[self._s_sorter]

            if ilens is not None:
                ilens[i] = len(v.ALT[0]) - len(v.REF)

            if self._pbar is not None:
                self._pbar.update()

            if i == n_variants - 1:
                break

        if i != n_variants - 1:
            raise ValueError("Not enough variants found in the given range.")

        return v.end  # type: ignore

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
    ) -> list[G]:
        ploidy = self.ploidy + self.phasing
        length = end - start
        ext_start = end
        coord = f"{contig}:{ext_start + 1}"

        _CHECK_LEN_EVERY_N = 20
        ls_genos: list[G] = []
        for i, v in enumerate(self._vcf(coord)):
            if v.start < ext_start or self.filter is not None and not self.filter(v):
                continue

            # (s p)
            genos = v.genotype.array()[:, :ploidy, None]
            genos = genos.astype(mode._gdtype)
            ls_genos.append(genos)

            if v.is_indel:
                ilen = len(v.ALT[0]) - len(v.REF)
                dist = v.start - last_end
                hap_lens += dist + np.where(genos == 1, ilen, 0)
                last_end = v.end

            if i % _CHECK_LEN_EVERY_N == 0 and (hap_lens >= length).all():
                break

        return ls_genos  # type: ignore

    def _ext_genos_dosages_with_length(
        self,
        contig: str,
        start: int | np.integer,
        end: int | np.integer,
        hap_lens: NDArray[np.int32],
        mode: type[GD],
        dosage_field: str,
        last_end: int,
    ) -> list[GD]:
        ploidy = self.ploidy + self.phasing
        length = end - start
        coord = f"{contig}:{end + 1}"

        _CHECK_LEN_EVERY_N = 20
        ls_geno_dosages: list[GD] = []
        for i, v in enumerate(self._vcf(coord)):
            if v.start < end or self.filter is not None and not self.filter(v):
                continue
            # (s p)
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
                hap_lens += dist + np.where(genos == 1, ilen, 0)
                last_end = v.end

            if i % _CHECK_LEN_EVERY_N == 0 and (hap_lens >= length).all():
                break

        return ls_geno_dosages
