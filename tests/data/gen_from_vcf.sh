#! /bin/bash

set -e

ddir=$(dirname "$0")
vcf=$ddir/test.vcf

echo "Bgzipping and indexing VCF file..."
bgzip -c "$vcf" >| "$vcf".gz
bcftools index "$vcf".gz

echo "Converting VCF to PLINK format..."
plink2 --make-pgen --vcf "$vcf".gz 'dosage=DS' --out "${vcf%.vcf}"

rm -f "$ddir"/test.log
rm -f "$ddir"/test.pvar.gvi