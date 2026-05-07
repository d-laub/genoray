# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses [Pixi](https://pixi.sh) for environment and task management. All development commands run inside the Pixi environment.

```bash
pixi s                  # activate the dev environment (installs all deps)
pixi run test           # run all tests (also regenerates test data via gen_from_vcf.sh)
pixi run prek-install   # install pre-commit and pre-push hooks (one-time setup)
pixi run bump-dry       # preview a version bump without committing
```

Run a single test file:
```bash
pixi run pytest tests/test_vcf.py
```

Skip network-dependent tests:
```bash
pixi run pytest -m "not network"
```

Linting/formatting is handled by `ruff` (via pre-commit hooks). To run manually:
```bash
ruff check genoray tests
ruff format genoray tests
```

## Architecture

`genoray` is a library for range-querying genetic variant data (genotypes and dosages) from VCF and PGEN files as NumPy arrays.

### Core modules

- **`genoray/_vcf.py`** — `VCF` class. Wraps `cyvcf2` and `oxbow` for VCF/BCF access. Defines Phantom-typed NumPy array wrappers (`Genos8`, `Genos16`) with shape `(samples, ploidy[+phasing], variants)`. Dosages have shape `(samples, variants)`.
- **`genoray/_pgen.py`** — `PGEN` class. Wraps `pgenlib` for PLINK2 PGEN access. Auto-creates a `.gvi` index on construction. Phantom types: `Genos` (int32), `Dosages` (float32), `Phasing` (bool). Both hardcall and dosage PGEN paths can be passed separately.
- **`genoray/_svar.py`** — `SparseVar` class. Converts dense genotype data to sparse representation using `seqpro.rag.Ragged`. Wraps both VCF and PGEN as backends.
- **`genoray/_var_ranges.py`** — Low-level utilities (`var_ranges`, `var_counts`, `var_indices`) that map genomic coordinate ranges to variant index slices, using `polars-bio` for interval overlap queries.
- **`genoray/_utils.py`** — Shared utilities: `ContigNormalizer` (handles `chr`-prefixed vs. unprefixed contig names), memory parsing, haplotype indel length calculations.
- **`genoray/exprs.py`** — Polars expressions for filtering the `.gvi` index (e.g., `is_snp`, `is_indel`, `is_biallelic`, `ILEN`). These are the filter expressions users pass to constructors.
- **`genoray/_types.py`** — Shared NumPy dtype constants (`POS_TYPE=np.int32`, `DOSAGE_TYPE=np.float32`, etc.).

### Key design patterns

- **Phantom types**: Array return types use `phantom-types` to encode shape/dtype constraints. All `Phantom` subclasses have `empty(n_samples, ploidy, n_variants)` classmethods.
- **`.gvi` index**: PGEN files get a zstd-compressed Polars DataFrame index written to `<prefix>.gvi` on first construction. This powers fast range queries.
- **Coordinate convention**: All ranges are 0-based, half-open `[start, end)`. Missing values are `-1` (genotypes) or `np.nan` (dosages).
- **`ContigNormalizer`**: Transparently maps `chr1 ↔ 1` style contig names so user queries work regardless of file naming scheme.
- **Chunking**: `chunk` / `chunk_ranges` yield NumPy array generators sized to `max_mem`. Memory estimates are based on per-variant byte costs.

### Commit convention

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, etc.). Version bumps are managed by `commitizen` (`cz bump`).
