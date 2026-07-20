# SVAR2 conversion: arbitrary fields (PGEN dosages, SVAR1 selection, CLI split)

**Date:** 2026-07-19
**Status:** Approved (design)
**Scope:** genoray SVAR2 conversion path — Python API, Rust pipeline, CLI, skill docs.

## Motivation

X→SVAR2 conversion should carry through arbitrary auxiliary fields, defaulting to
none. Most of this already exists:

- `from_vcf` / `from_vcf_list` accept `info_fields: Sequence[str | InfoField]` and
  `format_fields: Sequence[str | FormatField]` (default `None` = no fields).
- The reader uses a unified bcftools-style key space (`"INFO/x"` / `"FORMAT/x"`) via
  `SparseVar2(fields=...)` / `with_fields(...)`.
- `from_svar1` carries all SVAR1 FORMAT fields through automatically.
- FORMAT var_key/dense routing (cost model), packing, and finalize already exist and
  are reused throughout.

Three gaps remain, addressed here:

1. **PGEN dosages** — `from_pgen` has no field support: dosages are ignored, and the
   Rust PGEN pipeline hardcodes `fields = Vec::new()` with no dosage read path. Users
   store hard calls in one `.pgen` and VAFs/CCFs as dosage in *separate* `.pgen`
   files (pgenlib always derives hard calls from dosage when present, which breaks
   PLINK2's contract for VAF/CCF — hence the files must be separate).
2. **SVAR1 field selection** — `from_svar1` is all-or-nothing; add a selector.
3. **CLI** — the monolithic auto-detecting `write` exposes no field/dosage flags;
   split into per-source subcommands.

## Non-goals

- No change to the existing VCF `info_fields` / `format_fields` Python API (kept as
  two params; the unified bcftools-style string spec lives on the CLI only).
- No INFO/FORMAT extraction from PGEN `.pvar` (only dosages).
- No dense-per-sample dosage storage mode (dosages reuse the existing FORMAT
  cost-model routing).

## Design

### 1. PGEN dosages

**New public spec object** in `python/genoray/_svar2_fields.py`, exported from
`genoray` (`__init__.py` `_LAZY` + `__all__` + `TYPE_CHECKING`), mirroring
`InfoField` / `FormatField`:

```python
@dataclass(frozen=True)
class DosageField:
    name: str = "dosage"           # stored FORMAT-field key in the SVAR2 store
    source: str | Path = "self"    # "self" = the hardcall .pgen; else a separate dosage .pgen
    dtype: Literal["f16", "f32"] = "f32"
    default: float | None = None   # fill for missing dosage (else NaN sentinel)
```

**API:** `from_pgen(..., dosages: Sequence[DosageField] | None = None)`, default
`None` = no dosages (current behavior preserved). A sequence supports the
multi-file use case directly:

```python
SparseVar2.from_pgen(out, "calls.pgen", ref, dosages=[
    DosageField(name="VAF", source="vaf.pgen"),
    DosageField(name="CCF", source="ccf.pgen"),
])
```

**Semantics:**

- Hard calls always read from `source`. Each `DosageField` reads only its dosage
  track (`PgenReader.read_dosages_list` / `read_dosages_range`) from its own
  `source` — never that file's hard calls. `source="self"` reads dosages from the
  hardcall `.pgen`.
- Validation for each separate dosage `.pgen`:
  - `.psam` samples must match `source`'s (reuse the existing sample-match check in
    `_pgen.py`; raise on mismatch).
  - `.pvar` must align 1:1 with `source` (same variant count per contig range).
    Minimum: variant-count check per selected range; positions/alleles alignment is
    assumed (documented as a caller precondition).
- The same `samples=` subset (`sample_perm` / `change_sample_subset`) and `regions`
  narrowing applied to hardcall readers are applied identically to the dosage
  readers.
- Stored as a FORMAT field, auto-routed by the existing cost model (var_key
  carrier-only likely). Missing dosage → `NaN` (or `default`). **Caveat:** under
  var_key routing a non-carrier's dosage is dropped — acceptable for VAF/CCF where
  non-carriers ≈ 0; documented in the docstring (consistent with
  `svar2-sparse-hides-missingness`).
- Duplicate `name`s across the `dosages` sequence, or a `name` colliding with a
  reserved key (`mutcat`), raise.

**Rust** (`src/lib.rs` `run_pgen_conversion_pipeline`, `src/orchestrator.rs`
`process_chromosome`):

- `run_pgen_conversion_pipeline` grows two params: parallel dosage-reader pools
  (one `Vec<Vec<Py<PyAny>>>` per `DosageField`, same per-contig pool shape as the
  hardcall `readers`) and a `Vec<FieldSpec>` for the dosage fields (replacing the
  hardcoded empty `fields`).
- `process_chromosome` reads dosages per chunk via the dosage readers and feeds each
  per-sample `f32` array into the **existing** FORMAT finalize/routing path
  (`field_finalize`, var_key/dense packing). No new packing/routing code.
- `FieldSpec` for a dosage field is `category = format`, float dtype (`f16`/`f32`),
  with the `default`/NaN sentinel semantics already implemented for VCF FORMAT.

**Python plumbing** (`from_pgen`): build a dosage-reader pool per `DosageField`
exactly as the hardcall `readers` pool is built (same `n_samples`,
`allele_idx_offsets`, `change_sample_subset(subset_idx)`), resolve each into a
`FieldSpec` tuple, and pass both to the Rust call.

### 2. SVAR1 field selection

`from_svar1(..., fields: Sequence[str] | None = None)`.

- **Default `None` = carry all** SVAR1 fields (lossless format migration — the
  current behavior; silently dropping `dosages` on a migration would be surprising).
  This is the one place the default diverges from "no fields," by design.
- `fields=[...]` selects a subset by SVAR1 field name (e.g. `["dosages"]`);
  `fields=[]` carries none.
- Implementation: filter the `_svar1_fields_manifest(fields)` output
  (`format_tuples`, `src_dtypes`) by the selection before it reaches Rust; raise on
  an unknown field name (message lists available SVAR1 field names).

### 3. CLI restructure

Break the monolithic auto-detecting `write` (`@write.default`) into per-**source**
subcommands, target always SVAR2. **No bare-`write` default and no
backward-compatible auto-detect alias** — there are no production CLI consumers and
the SVAR2 write CLI shipped only days ago, so this is treated as a non-breaking
`feat:` (no 4.0 bump).

- `genoray write vcf SOURCE OUT [--fields "INFO/AF" --fields "FMT/AD" ...]`
  — single VCF/BCF and the vcf-list dir/manifest form (auto-detected within, as
  today, since they share htslib fields). `--fields` takes bcftools-style strings
  (`INFO/`, `FORMAT/`, `FMT/`), deduped (first occurrence wins), parsed into
  `info_fields` / `format_fields` for `from_vcf` / `from_vcf_list`.
- `genoray write pgen SOURCE OUT [--dosages NAME=PATH ...]`
  — each `--dosages` is `NAME=self` or `NAME=/path.pgen`, parsed into a
  `DosageField`. Repeatable.
- `genoray write svar1 SOURCE OUT [--fields NAME ...]`
  — SVAR1 → SVAR2 (**reassigned meaning**). `--fields` selects SVAR1 fields; omitted
  = carry all; `--empty-fields` carries none (matching the existing `view svar1`
  flag convention).

**Naming-collision resolution:** the current `write svar1` (convert *to* legacy
SVAR1 format) is renamed to top-level **`genoray write-svar1 SOURCE OUT`**, keeping
its existing `dosages` (FORMAT-name for VCF / dosage-`.pgen`-path for PGEN),
`--no-symbolic`, `--no-breakend`, `--haploid`, `max_mem` flags unchanged.

Shared flags (`--reference`/`--no-reference`, `--regions`/`-r`,
`--regions-file`/`-R`, `--samples`/`-s`, `--samples-file`/`-S`,
`--merge-overlapping`, `--regions-overlap`, `--ploidy` [VCF only],
`--chunk-size`, `--threads`/`-@`, `--long-allele-capacity`, `--overwrite`,
`--skip-symbolics-and-breakends`, `--check-ref`) are factored so each subcommand
carries the subset that applies to its source (e.g. no `--ploidy` on `pgen`).

### 4. Skill docs

`skills/genoray-api/SKILL.md` updated in the same PR (repo rule): new public
`DosageField`, `from_pgen(dosages=)`, `from_svar1(fields=)`, and the new CLI
surface (`write vcf` / `write pgen` / `write svar1`, `write-svar1`).

## Testing

- **PGEN dosages:** round-trip a PGEN with a same-file dosage track and with a
  separate dosage `.pgen`; assert decoded FORMAT field matches the source dosages
  for carriers; assert NaN/`default` for missing; assert multi-`DosageField`
  (VAF + CCF from two files). Sample-subset and region-subset parity. Mismatched
  `.psam` / variant-count raises. Use `vcfixture`/existing PGEN fixtures.
- **SVAR1 selection:** carry-all default byte-identical to current behavior;
  `fields=["dosages"]` subset; `fields=[]` none; unknown name raises.
- **CLI:** `write vcf --fields`, `write pgen --dosages NAME=PATH` (self and separate
  file), `write svar1 --fields`, `write-svar1` legacy parity. Bare `write` no longer
  resolves (asserts the removal).
- **Rust:** `cargo test --no-default-features` (per repo linking constraint);
  `maturin develop --release` before any Python-level dosage e2e verification (the
  `.so` is not rebuilt by `pixi run test`).

## Risks / notes

- PGEN dosage reads via pgenlib hold the GIL (`pgenlib-holds-gil-sharded-reads`);
  dosage reading is single-reader-per-contig like hardcalls (`P=1`), so no new
  contention.
- `.pvar` 1:1 alignment between hardcall and dosage files is a caller precondition
  beyond the variant-count check; document clearly.
- NFS `target/` linker issue: set `CARGO_TARGET_DIR=/tmp/...` for `cargo test` and
  commits (`genoray-nfs-linker-bus-error`).
