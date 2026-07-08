# svar2-codec

Encode/decode for the [SVAR 2.0](https://github.com/d-laub/genoray) variant-key bit
layout: the 2-bit SNP code stream and the 32-bit indel key (inline INS/SNP, pure
DEL, and long-allele-bank lookup lanes).

Pure and `std`-only — no I/O, no Python bindings. It is the single source of truth
for the on-disk key layouts, shared by the `genoray` converter and by downstream
Rust consumers (e.g. GenVarLoader) that decode SVAR2 query results in-process.

## License

Licensed under either of Apache-2.0 or MIT at your option.
