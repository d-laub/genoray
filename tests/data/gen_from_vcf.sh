#! /bin/bash

set -e

vcf=$(dirname "$0")/test.vcf

echo "Bgzipping and indexing VCF file..."
bgzip -c "$vcf" >| "$vcf".gz
bcftools index "$vcf".gz

echo "Converting VCF to PLINK format..."
plink2 --make-pgen --vcf "$vcf".gz --out "${vcf%.vcf}"