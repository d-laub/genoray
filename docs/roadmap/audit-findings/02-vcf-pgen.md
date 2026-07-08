# Audit: _vcf.py + _pgen.py

## Summary
Both backends are functional but carry heavy boilerplate and parallel-but-divergent
structure. The largest problems are (1) five near-identical phantom-type wrapper
blocks per file with copy-paste TypeGuards and `empty` classmethods, (2) the mode
`issubclass(...)` dispatch table repeated 5+ times in `_pgen.py` and twice in
`_vcf.py`, and (3) the dosage-extract-and-validate block duplicated five times in
`_vcf.py`. The two files share a real conceptual core (sample management, contig
normalization, index load/validate, memory-per-variant arithmetic, chunking, and the
"extend range to cover haplotype length" algorithm) but implement it twice with
divergent code and even divergent public surfaces (PGEN exposes `read_ranges`,
`chunk_ranges`, `var_idxs`; VCF exposes none of these and keeps `_var_idxs` private).
The right amount of sharing is *helper extraction* (phantom-type factory, memory math,
empty-result yielding, dosage extraction), not a premature base class — the actual
read engines (cyvcf2 per-variant streaming vs. pgenlib random-access-by-index) are too
different to unify cleanly. There is also at least one latent behavior divergence
(filter not applied on the `out is None` path of `_fill_genos_and_dosages`) that should
be treated as a bug, not just a smell.

## Findings

### [structure] Phantom-type wrapper boilerplate duplicated ~5x per file and across files
- **Location:** _vcf.py:52-173 (Genos8/Genos16/Dosages/Genos8Dosages/Genos16Dosages); _pgen.py:37-147 (Genos/Dosages/Phasing/GenosPhasing/GenosDosages/GenosPhasingDosages)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** Every phantom mode repeats the same three-part pattern: an `_is_*` `TypeGuard` predicate that checks `isinstance(np.ndarray)` + dtype + ndim (+ shape), a `class X(NDArray[...], Phantom, predicate=...)`, and an `empty` classmethod that allocates `np.empty(...)`. This is ~120 lines in `_vcf.py` and ~110 in `_pgen.py`, almost all mechanical. The tuple modes (`Genos8Dosages`, `GenosPhasing`, etc.) additionally repeat an isinstance-per-element predicate. Two full docstrings (`_is_genos8_dosages`, `_is_genos16_dosages`) are copy-pasted verbatim.
- **Recommendation:** Provide small factories in a shared module (e.g. `_modes.py`): one for scalar array modes (`make_array_mode(dtype, ndim, shape_pred)`) and one for tuple-of-modes. Generates the predicate + Phantom class + `empty`. Cuts both files substantially and forces the two backends to agree on the pattern.

### [consistency] `empty` classmethod signature diverges between backends (`phasing` param)
- **Location:** _vcf.py:64-68, 96-101 (`empty(cls, n_samples, ploidy, n_variants, phasing)`); _pgen.py:49-51, 63-65 (`empty(cls, n_samples, ploidy, n_variants)`)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** CLAUDE.md states "All `Phantom` subclasses have `empty(n_samples, ploidy, n_variants)` classmethods," but VCF's variants take a 4th `phasing` argument that PGEN's do not. Callers must branch on backend. Worse, `Dosages.empty` in `_vcf.py` accepts `ploidy` and `phasing` and ignores both (line 99-101) — dead parameters kept only for signature uniformity within the file.
- **Recommendation:** Standardize the signature. Since phasing on VCF changes the ploidy axis length, prefer folding it in as an effective-ploidy computed by the caller (`ploidy + phasing`) so `empty(n_samples, ploidy, n_variants)` is uniform across both backends, matching the documented contract.

### [consistency] Filter NOT applied on the `out is None` path of `_fill_genos_and_dosages` (latent bug)
- **Location:** _vcf.py:1416-1466 (out-is-None branch has no `filter(...)`); compare _fill_genos:1286-1287 and _fill_dosages:1352-1353 which filter before both branches; the guard in _fill_genos_and_dosages sits at line 1471, only reachable when `out is not None`
- **Severity:** high
- **Effort:** S
- **Risk:** med (fixing changes output when a filter is set and no `.gvi` index is loaded)
- **Problem:** `_fill_genos` and `_fill_dosages` apply `self._filter` at the top, before the `out is None` and `out is not None` branches. `_fill_genos_and_dosages` applies it only inside the `out is not None` branch. When `read(..., mode=Genos*Dosages)` is called with no index loaded (so `n_variants is None` → `out is None`), a configured `_filter` is silently ignored for the genos+dosages mode but honored for genos-only and dosages-only modes. This is an inconsistency that reads as a bug.
- **Recommendation:** Hoist the `if self._filter is not None: vcf = filter(self._filter, vcf)` to the top of `_fill_genos_and_dosages`, matching the sibling fill methods. Add a regression test with a filter + no index across all three modes.

### [structure] Mode `issubclass(...)` dispatch table repeated 5+ times in _pgen.py
- **Location:** _pgen.py:510-521 (read), 588-599 (chunk), 665-676 (read_ranges), 769-780 (chunk_ranges), 901-910 (_chunk_ranges_with_length); mirrored in _vcf.py:626-655 and 659-668 (read)
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** The same `if issubclass(mode, Genos): ... elif Dosages ... elif GenosPhasing ...` ladder mapping a mode to its `_read_*` method appears five times in `_pgen.py`. Any new mode requires editing five sites; drift risk is high. VCF has the same problem in `read` where the entire dispatch is duplicated between the `out is None` and `out is not None` branches.
- **Recommendation:** In PGEN, build a `dict[type[Phantom], Callable]` (e.g. `{Genos: self._read_genos, ...}`) once and look up `reader = table[mode]`; the `_chunk_ranges_with_length` variant (which omits `Dosages`) can filter that table. In VCF, collapse the two `read` branches so the mode dispatch is written once and the `out` array is threaded through.

### [structure] Dosage extract-and-validate block duplicated five times in _vcf.py
- **Location:** _vcf.py:757-767 (chunk), 1358-1368 and 1394-1403 (_fill_dosages), 1438-1448 and 1488-1497 (_fill_genos_and_dosages), 1621-1627 (_ext_genos_dosages_with_length)
- **Severity:** med
- **Effort:** S
- **Risk:** low
- **Problem:** The pattern `d = v.format(field); if d is None: raise DosageFieldError(...); if d.shape[1] > 1: raise MultiallelicDosageError(...); ... d.squeeze(1)` is copy-pasted five times with identical error messages.
- **Recommendation:** Extract `_extract_dosage(self, v) -> NDArray[np.float32]` that performs the fetch, both checks, and squeeze, returning the 1-D per-sample dosage. All five sites call it.

### [structure] `_ext_genos_with_length` and `_ext_genos_dosages_with_length` are near-duplicate
- **Location:** _vcf.py:1544-1589 vs 1591-1646
- **Severity:** med
- **Effort:** M
- **Risk:** low
- **Problem:** The two extension methods share the entire control structure (contig coord build, warnings filter, per-variant loop, `v.start < ext_start` skip, indel `hap_lens` update, `_CHECK_LEN_EVERY_N` break, trailing `last_end` fixup). They differ only in whether a dosage array is also collected. `_CHECK_LEN_EVERY_N = 20` is redefined in each.
- **Recommendation:** Merge into one method parameterized by whether dosages are wanted (reuse the extracted `_extract_dosage`), or have the dosage variant delegate to the genos variant for the shared length bookkeeping.

### [consistency] The "extend to haplotype length" algorithm is implemented twice, divergently
- **Location:** _vcf.py:887-972 (`_chunk_with_length_helper`) + 1544-1646 (`_ext_*`) vs _pgen.py:1011-1106 (`_gen_with_length`)
- **Severity:** med
- **Effort:** L
- **Risk:** med
- **Problem:** Both backends implement the same conceptual algorithm — read a chunk, then keep extending past `end` until every haplotype is at least as long as the query range, accounting for indel `ILEN`. VCF drives it by streaming forward from a cyvcf2 coordinate and checking `hap_lens` every 20 records; PGEN drives it by geometrically doubling an index window (`_idx_extension *= 2`) over precomputed `v_starts/v_ends/ilens`. The two contracts even differ in what the final tuple carries (see next finding). This is the deepest divergence in the pair and the hardest to keep correct in tandem.
- **Recommendation:** Do not force a shared implementation (the data access models genuinely differ). Instead, document the shared invariant in one place and add cross-backend property tests asserting both produce haplotypes ≥ query length for the same variants. Longer term, consider expressing VCF's index-backed path (when a `.gvi` index is loaded, `v_starts/v_ends/ilens` are available) via the same windowing helper PGEN uses.

### [consistency] `_chunk_ranges_with_length` third tuple element differs between backends
- **Location:** _vcf.py:800-804 & 968-972 (yields `n_extension_vars: int`) vs _pgen.py:791-795 & 933-942 (yields `chunk_idxs: NDArray[V_IDX_TYPE]`)
- **Severity:** med
- **Effort:** M
- **Risk:** med
- **Problem:** Two methods with the same name and role on sibling classes return structurally different tuples: VCF's third element is a count of extension variants; PGEN's is the array of variant indices in the chunk. A downstream consumer (GenVarLoader) cannot treat these interchangeably, which undermines the point of parallel backends.
- **Recommendation:** Reconcile to a single contract (most likely PGEN's richer `chunk_idxs`, since a count is derivable from it). Update the consumer and `skills/genoray-api/SKILL.md` if this signature is public-facing.

### [consistency] Copy-pasted "not found in VCF file" warning inside PGEN
- **Location:** _pgen.py:566, 651, 734, 861 (all say "not found in VCF file")
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** Four PGEN methods emit `logger.warning("Query contig {contig} not found in VCF file, ...")` — wrong backend name, clearly copied from `_vcf.py`. Misleading in logs.
- **Recommendation:** Say "PGEN file" (or make the message backend-agnostic). If a shared empty-result helper is extracted (below), centralize the message there.

### [structure] Repeated "contig missing → yield/return empty" boilerplate
- **Location:** _pgen.py:563-569, 649-656, 732-739, 859-874 (+ the `tot_variants == 0` twins at 573-575, 660-663, 746-749, 880-891); _vcf.py:608-613, 713-719, 844-854, 857-864
- **Severity:** low
- **Effort:** M
- **Risk:** low
- **Problem:** Each range/chunk method opens with the same `c = norm(contig); if c is None: warn; yield/return empty` block, and most also repeat a `n_variants == 0` empty-result block with the identical single-element generator expression `(mode.empty(...) for _ in range(1))`.
- **Recommendation:** Add small private helpers (`_empty(mode)`, `_empty_gen(mode, end=...)`) and a `_norm_or_warn(contig)` that returns the normalized contig or None with the warning. Removes a dozen near-identical blocks per file.

### [consistency] `_mem_per_variant` logic duplicated and subtly divergent
- **Location:** _vcf.py:1513-1542 vs _pgen.py:944-964
- **Severity:** low
- **Effort:** M
- **Risk:** low
- **Problem:** Both compute bytes-per-variant by summing per-array `n_samples * axis * itemsize`, but VCF reads dtype from `mode._gdtype` while PGEN uses `mode._dtype`/`mode._dtypes`, and only PGEN doubles the estimate when a sample sorter is active (`_s_unsorter` is an ndarray). The doubling asymmetry means chunk sizing differs in kind between backends for the same logical operation.
- **Recommendation:** Have each mode expose a uniform `nbytes_per_variant(n_samples, effective_ploidy)` (natural once the phantom factory exists), so `_mem_per_variant` becomes a one-liner in both files. Decide deliberately whether the sample-copy doubling should apply to VCF too.

### [consistency] Two-filter both-or-neither invariant is a runtime check that could be a type
- **Location:** _vcf.py:242-245, 311-369 (`_check_filter_pair`, `filter` property/setter accepting a 2-tuple or None)
- **Severity:** low
- **Effort:** M
- **Risk:** med
- **Problem:** VCF requires a `(callable, pl.Expr)` pair that must be both-set-or-both-None, enforced by `_check_filter_pair` at three call sites and by a tuple-shape check in the setter. This is a "make invalid states unrepresentable" candidate: the paired filter is really one concept (a record predicate with its index-expression twin).
- **Recommendation:** Introduce a small frozen `Filter(record: Callable, expr: pl.Expr)` value object; `self._filter: Filter | None` then makes the invalid "one set, other None" state unrepresentable and removes `_check_filter_pair`. (Weigh against churn — the public `filter` tuple API would need a compat shim and a SKILL.md update.)

### [api-hygiene] Public range API is asymmetric between backends
- **Location:** _pgen.py exposes `var_idxs` (422), `read_ranges` (603), `chunk_ranges` (680); _vcf.py has only private `_var_idxs` (540) and no `read_ranges`/`chunk_ranges`; both share public `n_vars_in_ranges`
- **Severity:** med
- **Effort:** L
- **Risk:** low
- **Problem:** The two "sibling backends with parallel structure" expose materially different public surfaces. PGEN users can query variant indices and read/chunk multiple ranges at once; VCF users cannot, and VCF keeps the index accessor private. A caller writing backend-generic code hits a wall.
- **Recommendation:** Decide the intended common surface. Either promote `_var_idxs` and add `read_ranges`/`chunk_ranges` to VCF (streaming implementation), or document explicitly (in SKILL.md and docstrings) that multi-range APIs are PGEN-only and why. Don't leave it as accidental divergence.

### [api-hygiene] `read(out=...)` exists on VCF but not PGEN; misleading floating docstring; `TypeGuard` type typo
- **Location:** _vcf.py:570-577 (`read` has `out` param) vs _pgen.py:469-475 (no `out`); _vcf.py:36-37 (bare triple-quoted string about "int64 ... CSI indexes" that documents nothing and contradicts the adjacent `V_IDX_TYPE = np.uint32`); _vcf.py:140 (`_is_genos16_dosages` annotated `TypeGuard[tuple[Genos8, Dosages]]`, should be `Genos16`)
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** Three small hygiene issues: (1) `read` takes an `out=` output buffer on VCF but not PGEN — another silent asymmetry. (2) The floating string literal at the top of `_vcf.py` reads like it should be a `V_IDX_TYPE` docstring but is orphaned above it and describes int64/CSI, while the real constant is uint32 — dead and misleading. (3) `_is_genos16_dosages`'s `TypeGuard` names `Genos8` instead of `Genos16` (copy-paste), weakening the type guarantee.
- **Recommendation:** Remove or correctly attach the floating string; fix the `TypeGuard` to `Genos16`; and decide whether `out=` belongs on both backends or neither (consistency).

### [structure] `_oxbow_reader` misplaced inside the `get_record_info` overload group
- **Location:** _vcf.py:1005-1012 (defined between the last `@overload` stub at 995-1004 and the real implementation at 1014)
- **Severity:** low
- **Effort:** S
- **Risk:** none
- **Problem:** A helper method body is interleaved between the overload stubs and the concrete `get_record_info` implementation. It happens to work (the overloads are contiguous enough) but it reads as a mistake and obscures which `def get_record_info` is the implementation.
- **Recommendation:** Move `_oxbow_reader` out of the overload block (e.g. below `get_record_info`), so the overload stubs sit directly above their implementation.

### [api-hygiene] PGEN `__del__` double-closes the reader when no separate dosage path
- **Location:** _pgen.py:242 (`self._dose_pgen = self._geno_pgen`) and 389-393 (`__del__` closes both)
- **Severity:** low
- **Effort:** S
- **Risk:** low
- **Problem:** When `dosage_path is None`, `_dose_pgen` is the *same object* as `_geno_pgen`; `__del__` then calls `.close()` on it twice. Depending on pgenlib's tolerance this is at best redundant, at worst an error on interpreter shutdown.
- **Recommendation:** Guard with `if self._dose_pgen is not self._geno_pgen:` before the second close.
