# `data/` — large benchmark/measurement inputs (git-ignored)

The variant files here are git-ignored (`data/*`); only this README is tracked.
It records where the inputs and their matching reference FASTAs live on the
Carter HPC so measurement/benchmark work is reproducible.

## Variant inputs

| File | Description | Samples |
| --- | --- | --- |
| `chr21.bcf` | Germline chr21 (1000 Genomes / 1kGP) | >1k |
| `gdc.chr21.bcf` | Somatic chr21 (GDC / TCGA) | >1k |
| `clinvar.vcf.gz` | ClinVar variants | n/a |

## Reference FASTAs

Match each input to the reference build it was called against:

- **Germline / 1kGP** (`chr21.bcf`):
  `/carter/users/dlaub/data/1kGP/GRCh38_full_analysis_set_plus_decoy_hla.fa`
- **Somatic / GDC** (`gdc.chr21.bcf`):
  `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`

Both are GRCh38. Use the matching reference when converting with
`genoray write` / `SparseVar2.from_vcf(reference=...)` (needed for indel REF
validation and left-alignment).
