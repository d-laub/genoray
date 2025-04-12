from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Generator, TypeVar, cast

import cyvcf2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from phantom import Phantom
from tqdm.auto import tqdm
from typing_extensions import Self, TypeGuard, assert_never

from ._types import Reader
from ._utils import (
    ContigNormalizer,
    format_memory,
    is_dtype,
    lengths_to_offsets,
    parse_memory,
)


class DosageFieldError(RuntimeError): ...


def _is_genos_dosages(obj) -> TypeGuard[tuple[Genos, Dosages]]:
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
        and isinstance(obj[0], Genos)
        and isinstance(obj[1], Dosages)
    )


class Genos(NDArray[np.int8], Phantom, predicate=partial(is_dtype, dtype=np.int32)): ...


class Dosages(
    NDArray[np.float32], Phantom, predicate=partial(is_dtype, dtype=np.float32)
): ...


class GenosDosages(tuple[Genos, Dosages], Phantom, predicate=_is_genos_dosages): ...


T = TypeVar("T", Genos, Dosages, GenosDosages)


class VCF(Reader[T]):
    path: Path
    available_samples: list[str]
    contigs: list[str]
    ploidy = 2
    filter: Callable[[cyvcf2.Variant], bool] | None
    dosage_field: str | None = None
    _vcf: cyvcf2.VCF
    _s_idx: NDArray[np.intp]
    _samples: list[str]
    _c_norm: ContigNormalizer
    _read_as: type[T]

    Genos = Genos
    Dosages = Dosages
    GenosDosages = GenosDosages

    def __init__(
        self,
        path: str | Path,
        filter: Callable[[cyvcf2.Variant], bool] | None = None,
        read_as: type[T] = Genos,
        dosage_field: str | None = None,
        progress: bool = False,
    ):
        if (read_as is Dosages or read_as is GenosDosages) and dosage_field is None:
            raise ValueError(
                "Dosage field not specified. Set the VCF reader's `dosage_field` parameter."
            )

        self.path = Path(path)
        self.filter = filter
        self.dosage_field = dosage_field

        self._vcf = self._open()
        self.available_samples = self._vcf.samples
        self.contigs = self._vcf.seqnames
        self._c_norm = ContigNormalizer(self.contigs)
        self.set_samples(self._vcf.samples)
        self._read_as = read_as
        self.progress = progress

    def _open(self, samples: list[str] | None = None) -> cyvcf2.VCF:
        return cyvcf2.VCF(self.path, samples=samples, lazy=True)

    @property
    def current_samples(self) -> list[str]:
        return self._samples

    def set_samples(self, samples: list[str]) -> Self:
        if missing := set(samples).difference(self.available_samples):
            raise ValueError(
                f"Samples {missing} not found in the VCF file. "
                f"Available samples: {self.available_samples}"
            )
        self._vcf = self._open(samples)
        _, s_idx, _ = np.intersect1d(self._vcf.samples, samples, return_indices=True)
        self._samples = samples
        self._s_idx = s_idx
        return self

    def __del__(self):
        self._vcf.close()

    def n_vars_in_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
    ) -> NDArray[np.uint32]:
        starts = np.atleast_1d(starts)
        ends = (
            np.full(len(starts), np.iinfo(np.int64).max)
            if ends is None
            else np.atleast_1d(ends)
        )

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros(len(starts), dtype=np.uint32)

        out = np.empty(len(starts), dtype=np.uint32)
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
        start: int = 0,
        end: int | None = None,
        out: T | None = None,
    ) -> T | None:
        c = self._c_norm.norm(contig)
        if c is None:
            return

        if end is None:
            end = np.iinfo(np.int64).max

        itr = self._vcf(f"{c}:{start + 1}-{end}")  # region string is 1-based
        if out is None:
            n_variants: np.uint32 = self.n_vars_in_ranges(c, start, end)[0]
            if n_variants == 0:
                return

            if self._read_as is Genos:
                data = np.empty(
                    (self.n_samples, self.ploidy, n_variants), dtype=np.int8
                )
                self._fill_genos(itr, data)
            elif self._read_as is Dosages:
                data = np.empty((self.n_samples, n_variants), dtype=np.float32)
                self._fill_dosages(
                    itr,
                    data,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                data = (
                    np.empty((self.n_samples, self.ploidy, n_variants), dtype=np.int8),
                    np.empty((self.n_samples, n_variants), dtype=np.float32),
                )
                self._fill_genos_and_dosages(
                    itr,
                    data,
                    self.dosage_field,  # type: ignore | guaranteed to be str by init guard clause
                )
            out = cast(T, data)
        else:
            if self._read_as is Genos:
                if not isinstance(out, Genos):
                    raise ValueError(
                        f"Expected output array of type {Genos.dtype}, got {type(out)}"
                    )
                self._fill_genos(itr, out)
            elif self._read_as is Dosages:
                if not isinstance(out, Dosages):
                    raise ValueError(
                        f"Expected output array of type {Dosages.dtype}, got {type(out)}"
                    )
                self._fill_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                if not isinstance(out, GenosDosages):
                    raise ValueError(
                        f"Expected output to be 2-tuple of arrays np.int8 and np.float32, but got {type(out)}"
                    )
                self._fill_genos_and_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )

        return out

    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
    ) -> Generator[T]:
        max_mem = parse_memory(max_mem)

        c = self._c_norm.norm(contig)
        if c is None:
            return

        if end is None:
            end = np.iinfo(np.int64).max

        n_variants: int = self.n_vars_in_ranges(c, start, end)[0]
        if n_variants == 0:
            return

        mem_per_v = self._mem_per_variant()
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

        itr = self._vcf(f"{c}:{start + 1}-{end}")  # region string is 1-based
        for chunk_size in chunk_sizes:
            if self._read_as is Genos:
                out = np.empty((self.n_samples, self.ploidy, chunk_size), dtype=np.int8)
                self._fill_genos(itr, out)
            elif self._read_as is Dosages:
                out = np.empty((self.n_samples, chunk_size), dtype=np.float32)
                self._fill_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                out = (
                    np.empty((self.n_samples, self.ploidy, chunk_size), dtype=np.int8),
                    np.empty((self.n_samples, chunk_size), dtype=np.float32),
                )
                self._fill_genos_and_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )

            yield cast(T, out)

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
    ) -> tuple[T, NDArray[np.uint32]] | None:
        c = self._c_norm.norm(contig)
        if c is None:
            return

        n_vars = self.n_vars_in_ranges(contig, starts, ends)

        tot_vars = n_vars.sum()
        if tot_vars == 0:
            return

        if self._read_as is GenosDosages:
            _out = (
                np.empty((self.n_samples, self.ploidy, tot_vars), dtype=np.int8),
                np.empty((self.n_samples, tot_vars), dtype=np.float32),
            )
        elif self._read_as is Dosages:
            _out = (np.empty((self.n_samples, tot_vars), dtype=np.float32),)
        else:
            _out = (np.empty((self.n_samples, self.ploidy, tot_vars), dtype=np.int8),)

        starts = np.atleast_1d(starts)
        if ends is None:
            ends = np.full(len(starts), np.iinfo(np.int64).max)
        else:
            ends = np.atleast_1d(ends)

        offsets = lengths_to_offsets(n_vars)
        for i, (s, e) in enumerate(zip(starts, ends)):
            o_s, o_e = offsets[i], offsets[i + 1]
            if o_s == o_e:
                continue

            if self._read_as is not GenosDosages:
                sub_out = _out[0][..., o_s:o_e]
            else:
                _out = cast(tuple[NDArray[np.int8], NDArray[np.float32]], _out)
                sub_out = (_out[0][..., o_s:o_e], _out[1][..., o_s:o_e])
            sub_out = cast(T, sub_out)
            self.read(contig, s, e, out=sub_out)

        if self._read_as is not GenosDosages:
            _out = cast(T, _out[0])
            return _out, n_vars

        _out = cast(T, _out)
        return _out, n_vars

    def _fill_genos(self, vcf: cyvcf2.VCF, out: NDArray[np.int8]):
        if self.progress:
            n_variants = out.shape[-1]
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit="variant")

        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue

            out[..., i] = v.genotype.array()[:, : self.ploidy]

            if i == out.shape[-1] - 1:
                break
        else:
            raise ValueError("Not enough variants found in the given range.")

    def _fill_dosages(
        self, vcf: cyvcf2.VCF, out: NDArray[np.float32], dosage_field: str
    ):
        if self.progress:
            n_variants = out.shape[-1]
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit="variant")

        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue
            d = v.format(dosage_field)
            if d is None:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                )
            # (s, 1, 1) or (s, 1)? -> (s)
            out[..., i] = d.squeeze()

            if i == out.shape[-1] - 1:
                break
        else:
            raise ValueError("Not enough variants found in the given range.")

    def _fill_genos_and_dosages(
        self,
        vcf: cyvcf2.VCF,
        out: tuple[NDArray[np.int8], NDArray[np.float32]],
        dosage_field: str,
    ):
        if self.progress:
            n_variants = out[0].shape[-1]
            vcf = tqdm(vcf, total=n_variants, desc="Reading VCF", unit="variant")

        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue

            out[0][..., i] = v.genotype.array()[:, : self.ploidy]

            d = v.format(dosage_field)
            if d is None:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                )
            # (s, 1, 1) or (s, 1)? -> (s)
            out[1][..., i] = d.squeeze()

            if i == out[0].shape[-1] - 1:
                break
        else:
            raise ValueError("Not enough variants found in the given range.")

    def _mem_per_variant(self) -> int:
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
        if issubclass(self._read_as, Genos):
            return self.n_samples * self.ploidy
        elif issubclass(self._read_as, Dosages):
            return self.n_samples * np.float32().itemsize
        elif issubclass(self._read_as, GenosDosages):
            return self.n_samples * self.ploidy + self.n_samples * np.float32().itemsize
        else:
            assert_never(self._read_as)
