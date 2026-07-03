Test VCFs are generated from the vcfixture builders in `fixtures.py` — edit
those, not any `.vcf` file. After changes, run `gen_from_vcf.sh` (or
`pixi run gen`) to regenerate the .vcf, .vcf.gz, .pgen, and .svar derivatives.
Note that PLINK 2 does not support multi-allelics and dosages at the same time!
Thus, there is a bi-allelic VCF with dosages and a multi-allelic VCF without.
