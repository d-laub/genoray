from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Generator, TypeVar, cast

import numpy as np
import pgenlib
import polars as pl
import pyranges as pr
from hirola import HashTable
from numpy.typing import ArrayLike, NDArray
from phantom import Phantom
from typing_extensions import Self, TypeGuard, assert_never

from ._types import Reader
from ._utils import (
    ContigNormalizer,
    format_memory,
    is_dtype,
    lengths_to_offsets,
    parse_memory,
)


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


class Genos(
    NDArray[np.int32], Phantom, predicate=partial(is_dtype, dtype=np.int32)
): ...


class Dosages(
    NDArray[np.float32], Phantom, predicate=partial(is_dtype, dtype=np.float32)
): ...


class GenosDosages(tuple[Genos, Dosages], Phantom, predicate=_is_genos_dosages): ...


T = TypeVar(
    "T",
    Genos,
    Dosages,
    GenosDosages,
)


class PGEN(Reader[T]):
    available_samples: list[str]
    filter: pl.Expr | None
    ploidy = 2
    contigs: list[str]
    _index: pr.PyRanges
    _geno_pgen: pgenlib.PgenReader
    _dose_pgen: pgenlib.PgenReader
    _s_idx: NDArray[np.uint32]
    _read_as: type[T]

    Genos = Genos
    Dosages = Dosages
    GenosDosages = GenosDosages

    def __init__(
        self,
        geno_path: str | Path,
        filter: pl.Expr | None = None,
        read_as: type[T] = Genos,
        dosage_path: str | Path | None = None,
    ):
        # TODO: support dosages and allow user to either provide a second PGEN file for dosages
        # or else use the same PGEN file for both genotypes and dosages.
        # That being said, there's probably not much point for a user to use the same PGEN file
        # for genos and dosages since PLINK2 defines hardcalls as a simple threshold on the dosages
        # when dosages are available.
        if read_as is Dosages or read_as is GenosDosages:
            raise NotImplementedError("PGEN dosages are not yet supported.")

        geno_path = Path(geno_path)
        samples = _read_psam(geno_path.with_suffix(".psam"))

        self.filter = filter
        self.available_samples = cast(list[str], samples.tolist())
        self._s2i = HashTable(
            max=len(samples) * 2,  # type: ignore
            dtype=samples.dtype,
        )
        self._s2i.add(samples)
        self._s_idx = np.arange(len(samples), dtype=np.uint32)
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

        if not geno_path.with_suffix(".gvi").exists():
            _write_index(geno_path.with_suffix(".pvar"))
        self._index = _read_index(geno_path.with_suffix(".gvi"), self.filter)
        self.contigs = self._index.chromosomes
        self._c_norm = ContigNormalizer(self._index.chromosomes)
        self._read_as = read_as

    @property
    def current_samples(self) -> list[str]:
        return cast(list[str], self._s2i.keys[self._s_idx].tolist())

    def set_samples(self, samples: list[str]) -> Self:
        _samples = np.atleast_1d(samples)
        s_idx = self._s2i.get(_samples).astype(np.uint32)
        if (missing := _samples[s_idx == -1]).any():
            raise ValueError(f"Samples {missing} not found in the file.")
        self._s_idx = s_idx
        self._geno_pgen.change_sample_subset(np.sort(s_idx))
        return self

    def __del__(self):
        self._geno_pgen.close()
        if self._dose_pgen is not None:
            self._dose_pgen.close()

    def n_vars_in_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
    ) -> NDArray[np.uint32]:
        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros_like(np.atleast_1d(starts), dtype=np.uint32)

        starts = np.atleast_1d(starts)
        if ends is None:
            ends = np.full_like(starts, np.iinfo(np.int32).max)
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

    def _var_idxs(
        self, contig: str, starts: ArrayLike = 0, ends: ArrayLike | None = None
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
        idxs
            Shape: (tot_variants). Variant indices for the given ranges.
        offsets
            Shape: (ranges+1). Offsets to get variant indices for each range.
        """
        starts = np.atleast_1d(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.empty(0, np.uint32), np.zeros_like(
                np.atleast_1d(starts), np.uint64
            )

        starts = np.atleast_1d(starts)
        if ends is None:
            ends = np.full_like(starts, np.iinfo(np.int32).max)
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full_like(starts, contig),
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
        start: int = 0,
        end: int | None = None,
        out: T | None = None,
    ) -> T | None:
        c = self._c_norm.norm(contig)
        if c is None:
            return

        if end is None:
            end = np.iinfo(np.int64).max

        var_idxs, _ = self._var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

        # TODO: support dosages

        if out is None:
            data = np.empty((n_variants, self.n_samples * self.ploidy), dtype=np.int32)
        else:
            if not isinstance(out, Genos):
                raise ValueError(f"Expected a np.int32 array, got {type(out)}.")
            data = out

        self._geno_pgen.read_alleles_list(var_idxs, data)
        data = data.reshape(n_variants, self.n_samples, self.ploidy).transpose(1, 2, 0)[
            self._s_idx
        ]
        data[data == -9] = -1

        data = cast(T, data)

        return data

    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
    ) -> Generator[T]:
        # TODO: support dosages

        max_mem = parse_memory(max_mem)

        c = self._c_norm.norm(contig)
        if c is None:
            return

        if end is None:
            end = np.iinfo(np.int64).max

        var_idxs, _ = self._var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

        mem_per_v = self._mem_per_variant()
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        n_chunks = -(-n_variants // vars_per_chunk)
        v_chunks = np.array_split(var_idxs, n_chunks)
        for var_idx in v_chunks:
            chunk_size = len(var_idx)
            out = np.empty((chunk_size, self.n_samples * self.ploidy), dtype=np.int32)
            self._geno_pgen.read_alleles_list(var_idx, out)
            out = out.reshape(chunk_size, self.n_samples, self.ploidy).transpose(
                1, 2, 0
            )[self._s_idx]
            out[out == -9] = -1
            yield cast(T, out)

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
    ) -> tuple[T, NDArray[np.uint64]] | None:
        # TODO: support dosages

        starts = np.atleast_1d(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return

        var_idxs, offsets = self._var_idxs(c, starts, ends)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

        out = np.empty((n_variants, self.n_samples * self.ploidy), dtype=np.int32)

        self._geno_pgen.read_alleles_list(var_idxs, out)
        out = out.reshape(n_variants, self.n_samples, self.ploidy).transpose(1, 2, 0)[
            self._s_idx
        ]
        out[out == -9] = -1

        return cast(T, out), offsets

    def _mem_per_variant(self) -> int:
        if issubclass(self._read_as, Genos):
            return self.n_samples * self.ploidy * np.int32().itemsize
        elif issubclass(self._read_as, (Dosages, GenosDosages)):
            raise NotImplementedError("Dosages are not yet supported.")
        else:
            assert_never(self._read_as)


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


RLEN = pl.col("REF").str.len_bytes()
ALEN = pl.col("ALT").str.len_bytes()
ILEN = ALEN - RLEN
KIND = (
    pl.when(ILEN != 0)
    .then(pl.lit("INDEL"))
    .when(RLEN == 1)  # ILEN == 0 and RLEN == 1
    .then(pl.lit("SNP"))
    .otherwise(pl.lit("MNP"))  # ILEN == 0 and RLEN > 1
    .cast(pl.Categorical)
)


# TODO: index can likely be implemented using the NCLS lib underlying PyRanges and then we can
# pass np.memmap arrays directly instead of having to futz with DataFrames. This will likely make
# filtering less ergonomic/harder to make ergonomic though, but a memmap approach will be scalable
# to ultra-large datasets (100k+ individuals).
def _write_index(path: Path):
    (
        pl.scan_csv(
            path.with_suffix(".pvar"),
            separator="\t",
            comment_prefix="##",
            schema_overrides={"#CHROM": pl.Utf8, "POS": pl.Int32},
        )
        .select(
            Chromosome="#CHROM",
            Start=pl.col("POS") - 1,
            End=pl.col("POS") + RLEN - 1,
            kind=KIND,
        )
        .sink_ipc(path.with_suffix(".gvi"))
    )


def _read_index(path: Path, filter: pl.Expr | None) -> pr.PyRanges:
    index = pl.read_ipc(path, row_index_name="index", memory_map=False)
    if filter is not None:
        index = index.filter(filter)
    pyr = pr.PyRanges(index.drop("kind").to_pandas(use_pyarrow_extension_array=True))
    return pyr
