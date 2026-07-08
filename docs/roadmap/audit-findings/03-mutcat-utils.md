# Audit: _mutcat.py + small Python modules

## Summary
Overall these modules are functionally solid and the numerically hot paths are properly vectorized (numpy LUTs + numba kernels), so the "Python loop over arrays" smell is largely absent where it matters. The dominant problem is in `_mutcat.py`: an entire scalar classification path (`_classify_variants_scalar` plus `classify_sbs96`/`classify_id83`) is dead code that duplicates the shipped vectorized `classify_variants`, roughly 120 lines including a per-row Python loop and a byte-for-byte duplicated warning block. Secondary themes: stringly-typed dispatch (`SENTINELS` dict, `kind: str`, the `Kind` Literal defined in one module but not shared) where enums/newtypes would make invalid states unrepresentable; a duplicated 0-based-end Polars expression across all three `_var_ranges` functions; and `_utils.py` being a genuine junk-drawer with two confirmed-dead public helpers (`is_dtype`, and `POLARS_V_IDX_TYPE` in `_types.py`). `_reference.py` and `_types.py` are clean apart from a `DTYPE` TypeVar duplicated between `_types.py` and `_utils.py`.

## Findings

### [structure] Dead scalar classification path duplicates the vectorized one
- **Location:** _mutcat.py:683-739 (`_classify_variants_scalar`), 185-202 (`classify_sbs96`), 242-296 (`classify_id83`), 299-306 (`_microhomology_len`)
- **Severity:** high
- **Effort:** M
- **Risk:** none
- **Problem:** `_classify_variants_scalar` has zero callers (confirmed via grep); `classify_sbs96` and `classify_id83` are referenced only from inside it (lines 712, 721), so they are dead too, as is `_microhomology_len` (used only by `classify_id83`). This is a full second implementation of `classify_variants` (761-856): same docstring, same semantics, a per-row Python `for i in range(index.height)` loop (700), a per-iteration closure (`_fetch`, 718), and a byte-for-byte copy of the REF-mismatch warning (731-737 vs 848-854). It reads as a reference/prototype left in the shipped module. `classify_dbs78` (212) is the exception — still live via `_build_dbs_table` (528).
- **Recommendation:** Delete `_classify_variants_scalar`, `classify_sbs96`, `classify_id83`, and `_microhomology_len`. Keep `classify_dbs78` (or inline its logic into `_build_dbs_table`, its sole caller). Removes ~120 lines and the only per-row Python loop in the module.

### [structure] `_mutcat.py` mixes five responsibilities in 856 lines
- **Location:** _mutcat.py (whole file)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** The file bundles (1) codebook constants/label lists (18-173), (2) scalar single-variant classifiers (176-306, mostly dead per above), (3) numpy/numba LUT builders and vectorized code kernels (309-532), (4) the doublet-pairing entry-code kernel (536-625), and (5) count-matrix aggregation + DataFrame assembly (628-680). These are distinct concerns with distinct change cadences; the codebook tables and the count aggregation have nothing to do with each other.
- **Recommendation:** After deleting the dead scalar path, consider splitting into `_mutcat/codebook.py` (labels, offsets, `code_ranges`, `labels`, `SENTINELS`), `_mutcat/classify.py` (LUTs + kernels + `classify_variants` + `build_entry_codes`), and `_mutcat/count.py` (`count_matrix`). Lower priority than the dead-code removal; do it in the same pass while the seams are visible.

### [structure] `_utils.py` is a junk drawer spanning unrelated domains
- **Location:** _utils.py (whole file)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** One module holds contig-name normalization (`ContigNormalizer`), memory string parsing/formatting, genotype indel math (`hap_ilens`), file-type sniffing (`variant_file_type`), a numpy→polars dtype table, thread resolution, a numba-thread contextmanager, and three atomic-filesystem-write helpers (249-321). The atomic-write group is ~70 lines of I/O infrastructure unrelated to everything else; `hap_ilens` is genotype-domain logic; `ContigNormalizer` is a central public type large enough to stand alone.
- **Recommendation:** Split by domain: `_io.py` for `atomic_write_path`/`atomic_write_dir`/`_unique_sibling`, `_contigs.py` for `ContigNormalizer`, keep memory/dtype/thread helpers in `_utils.py`. Move `hap_ilens` next to the genotype code that consumes it.

### [consistency] `SENTINELS` is a stringly-typed dict instead of an IntEnum
- **Location:** _mutcat.py:136-141, and every `SENTINELS["..."]` lookup (e.g. 192, 219, 318, 368, 510, 695, 726, 781-787)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** Sentinel codes are keyed by bare strings, so a typo like `SENTINELS["UNCLASSFIED"]` is a runtime `KeyError` rather than a name error, and the values are re-wrapped as `np.int16(SENTINELS[...])` at many call sites. This is the "stringly-typed over enum" anti-pattern; the values are a small closed set of negative constants.
- **Recommendation:** Replace with an `IntEnum` (or module-level `Final[int]` constants: `UNCLASSIFIED = -2`, etc.). Enables `code == Sentinel.UNCLASSIFIED` comparisons and static checking of the names. Keep the `_REF_MISMATCH = -99` internal sentinel as a sibling constant.

### [consistency] `kind` dispatch is `str` in some places, `Literal` in others; `Kind` not shared
- **Location:** _mutcat.py:167 (`labels(kind: str)`), 158 (`code_ranges` str keys); _signatures.py:22 (`Kind = Literal["SBS96","DBS78","ID83"]`); _mutcat.py:662 (`count_matrix(..., kind: Literal[...])`)
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** The same three-way kind is expressed three ways: a `str` with a runtime `ValueError` (`labels`), an inline `Literal` (`count_matrix`), and a named `Kind` alias defined in `_signatures.py`. `_signatures.py` imports `labels` from `_mutcat` but redefines the type alias, so the canonical name lives downstream of the module that owns the codebook.
- **Recommendation:** Define `Kind = Literal["SBS96","DBS78","ID83"]` once in `_mutcat.py` (the codebook owner), import it into `_signatures.py`, and annotate `labels`, `code_ranges`, and `count_matrix` with it. The runtime check in `labels` (169-173) becomes a defense-in-depth guard rather than the primary contract.

### [consistency] 0-based exclusive-end expression duplicated across all three `_var_ranges` functions
- **Location:** _var_ranges.py:67-70, 131-132, 186-187
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** `pl.col("POS") - pl.col("ILEN").list.first().clip(upper_bound=0).fill_null(0)` (the null-ILEN-as-point-variant end computation, with its subtle rationale) is written three times across `var_ranges`, `var_indices`, and `var_counts`. Any change to the symbolic-SV handling must be made in three places consistently.
- **Recommendation:** Extract a module-level helper returning the `pl.Expr` (e.g. `_var_end_expr() -> pl.Expr`) and reuse it in all three. `var_ranges` uses a numpy variant (67-70) — express it via the shared Polars expr or a shared comment referencing the same rule.

### [consistency] `variant_file_type` has an implicit `None` return and no annotation
- **Location:** _utils.py:163-172
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** The function returns `"vcf"`, `"pgen"`, or falls off the end returning `None`, with no return-type annotation. Callers cannot see the closed return set, and the "not a recognized variant file" case is an implicit `None` rather than an explicit outcome — invalid/unknown state is not made visible in the type.
- **Recommendation:** Annotate `-> Literal["vcf", "pgen"] | None` and make the final `return None` explicit, or raise for the unrecognized case if callers require a valid type.

### [consistency] `ContigNormalizer` builds its remapper with O(n^2) `list.index` calls
- **Location:** _utils.py:42
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** `{k: self.contigs.index(c) for k, c in self.contig_map.items()}` calls `list.index` once per mapping entry, each an O(n) scan → O(n^2) construction. Also two parallel lookup mechanisms coexist: `norm` uses `contig_map.get` while `c_idxs` uses a hirola `HashTable`. For typical genomes (dozens of contigs) the cost is negligible, but it violates the vectorized/no-Python-scan default and adds surface area.
- **Recommendation:** Precompute `name_to_index = {c: i for i, c in enumerate(self.contigs)}` once and index into it. Low priority given contig counts are small; flag mainly for consistency.

### [consistency] `np_to_pl_dtype` is a long if/elif ladder better expressed as a lookup
- **Location:** _utils.py:175-219
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** ~15-branch `if dtype == ...: return ...` chain mapping numpy dtypes to Polars dtypes. This is a static table; the branching form is more error-prone to extend than data.
- **Recommendation:** Replace with a `dict[np.dtype, type[pl.DataType]]` lookup and a single `raise ValueError` on miss. Behavior-preserving.

### [api-hygiene] Dead public-looking helpers: `is_dtype` and `POLARS_V_IDX_TYPE`
- **Location:** _utils.py:80-95 (`is_dtype`), _types.py:9 (`POLARS_V_IDX_TYPE`)
- **Severity:** med
- **Effort:** S
- **Risk:** none
- **Problem:** `is_dtype` has no callers anywhere in `python/` (only its definition). `POLARS_V_IDX_TYPE` likewise has no usages — `_var_ranges.py` converts index dtypes via `np_to_pl_dtype` instead. Both are underscore-free (read as intended-public) but unused and not re-exported.
- **Recommendation:** Delete both. If `is_dtype` is intended as a public TypeGuard for downstream users, it must be re-exported and documented in `skills/genoray-api/SKILL.md`; otherwise remove it.

### [api-hygiene] `DTYPE` TypeVar defined twice (in `_types.py` and `_utils.py`)
- **Location:** _types.py:11, _utils.py:18
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** The same `DTYPE = TypeVar("DTYPE", bound=np.generic)` is declared in both modules. `_var_ranges.py` imports it from `_utils`. Two identically-named-but-distinct TypeVars can subtly break variance/identity checks in a strict type checker and is plain DRY violation for a shared primitive.
- **Recommendation:** Keep the single definition in `_types.py` (the types module) and import it into `_utils.py`; delete the `_utils.py` copy.

### [api-hygiene] Underscore-free names in `_mutcat.py` read as public but aren't exported
- **Location:** _mutcat.py:158 (`code_ranges`), 167 (`labels`), 212 (`classify_dbs78`), 598 (`build_entry_codes`), 656 (`count_matrix`), plus `SBS96`/`DBS78`/`ID83`/`SBS96_INDEX`/`N_CODES`/`MUTCAT_VERSION`
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** `_mutcat.py` is a private module, but exposes many underscore-free names implying a public API. None are re-exported from `genoray/__init__.py` (only `Reference`, `fit_signatures`, `cosmic_signatures` from this file group are). Consumers (`_svar.py`) import the live ones directly. The mix of dead public-looking names (see first finding) and live-but-internal ones blurs the internal/public boundary.
- **Recommendation:** After deleting the dead classifiers, treat the survivors as package-internal: they are fine unprefixed if a same-package convention is adopted, but confirm none are meant for `import genoray` reach. If any should be public (e.g. `labels`, `code_ranges` for users interpreting `count_matrix` output), re-export and document them in the SKILL; otherwise leave internal and note the convention.

### [structure] `var_ranges` shadows its own function name with a local variable
- **Location:** _var_ranges.py:97 (`var_ranges = np.stack(...)`)
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** Inside `def var_ranges(...)`, the result array is bound to a local named `var_ranges`, shadowing the function within its own body. Harmless at runtime but confusing and defeats any recursion/self-reference and some linters.
- **Recommendation:** Rename the local to `ranges` or `result`.
