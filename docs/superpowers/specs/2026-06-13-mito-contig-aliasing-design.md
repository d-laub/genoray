# Mitochondrial contig aliasing in `ContigNormalizer`

Fixes [#61](https://github.com/d-laub/genoray/issues/61).

## Problem

`ContigNormalizer` bridges the `chr`-prefix difference (`chr1` ↔ `1`) but does not
treat all mitochondrial contig aliases as equivalent. The mito genome appears under
several names in the wild — `MT` (Ensembl/GRCh38), `chrM` (UCSC), `M`, and `chrMT`.

The existing `chr`-prefix logic already makes `M ↔ chrM` work (stripping `chr` from
`chrM` yields `M`), but `MT` and `chrMT` resolve to `None`. Against a UCSC-named
reference (`chrM`), an Ensembl-named `.svar` source (`MT`) crashes partway through
annotation:

```
File ".../genoray/_mutcat.py", line 505, in classify_variants
    five = reference.fetch(c, p - 1, p).tobytes()
File ".../genoray/_reference.py", line 54, in _load_contig
    raise ValueError(f"Contig {contig!r} not found in reference GRCh38.fa.")
ValueError: Contig 'MT' not found in reference GRCh38.fa.
```

This is a common GRCh38 mix: GDC/Ensembl variant sources + a UCSC-named FASTA.

## Fix

Extend `ContigNormalizer.__init__` so the four spellings `{M, MT, chrM, chrMT}` are
mutually equivalent. After building the existing `contig_map`, detect whether the
reference contains a mito contig and, if so, map all four spellings to it.

```python
# module level
_MITO_ALIASES = ("M", "MT", "chrM", "chrMT")

# in __init__, merged AFTER the existing three dicts
mito = next((c for c in self.contigs if c in _MITO_ALIASES), None)
mito_map = {a: mito for a in _MITO_ALIASES} if mito is not None else {}
self.contig_map = (
    {f"{c[3:]}": c for c in contigs if c.startswith("chr")}
    | {f"chr{c}": c for c in contigs if not c.startswith("chr")}
    | {c: c for c in contigs}
    | mito_map
)
```

### Why this is sufficient

`remapper`, the `HashTable` (`_c2dup`), and `dup2i` are all derived from
`contig_map`. Adding entries there flows through to both `norm()` and `c_idxs()`
automatically. No changes are needed in `_reference.py`, `_svar.py`, `_pgen.py`, or
`_vcf.py`.

### Behavior

Reference contains `chrM`:

| query   | current | desired |
|---------|---------|---------|
| `MT`    | `None`  | `chrM`  |
| `chrMT` | `None`  | `chrM`  |
| `M`     | `chrM`  | `chrM`  |
| `chrM`  | `chrM`  | `chrM`  |

Symmetric when the reference uses `MT` (all four → `MT`). If the reference has no
mito contig, nothing changes — all four still resolve to `None`.

### Degenerate input

If a reference somehow contains more than one mito spelling, the aliases bind to the
first one in contig order (`next(...)`). Deterministic and harmless.

## Scope

- **Exact spellings only.** Matching is against the literal set `{M, MT, chrM, chrMT}`,
  mirroring the case-sensitive `chr`-prefix handling already in place. Lowercase
  variants (`m`, `chrm`) are not handled.
- **Out of scope:** coordinate-system differences, pseudo-contigs (`PAR1`/`PAR2`), and
  the optional graceful-skip / `UNCLASSIFIED` handling for genuinely-absent contigs
  (left for a separate issue).

## Testing

Add cases to `tests/test_utils.py` following the existing `pytest_cases`
`contig_*` pattern consumed by `test_normalize_contig_name`:

- `ContigNormalizer(["chr1", "chrM"])`: `MT`, `chrMT`, `M`, `chrM` → all `chrM`.
- `ContigNormalizer(["1", "MT"])`: `M`, `chrM`, `chrMT`, `MT` → all `MT`.
- No-mito reference (`["chr1", "chr2"]`): `MT` → `None`.

Also update the stale docstring note at `_utils.py:21` which currently states the
M/MT equivalence is not handled.

## Documentation

No public-API surface changes (`ContigNormalizer` is internal, `_utils.py`). No
`skills/genoray-api/SKILL.md` update required. The fix only relaxes contig-name
matching, which is already documented behavior.
