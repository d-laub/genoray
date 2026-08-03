//! Optional runtime calibration of the per-contig reader count.
//!
//! The scale bench fitted a knee at w ~ 3-7 that moves with cohort size. That
//! fit was taken on synthetic corpora on one machine, and node speed on this
//! cluster varies by 2.08x; `t_read` and `t_exec` also move with compression
//! ratio, field count, and ploidy, none of which a fitted knee sees. This
//! measures the ratio on the actual input, on the actual machine.

/// Per-chunk timings from the probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// Seconds for ONE shard worker to produce one dense chunk.
    pub t_read_s: f64,
    /// Seconds for `dense2sparse_vk` to consume one dense chunk.
    pub t_exec_s: f64,
}

/// Safety clamp, not a tuning parameter: it bounds the damage from a probe
/// that measured a pathological prefix. The harness never observed a knee
/// above 7, so this sits well clear of any real workload -- reaching it means
/// the probe, not the workload, is what to look at.
pub const W_MAX: usize = 16;

/// Chunks timed per probe. Two is enough to get past first-chunk warmup while
/// staying negligible against a real conversion.
pub const PROBE_CHUNKS: usize = 2;

/// Readers needed to keep one executor fed: `w/t_read >= 1/t_exec`.
///
/// Rounds UP -- rounding down starves the executor, which is the serial stage
/// the whole probe exists to keep busy.
pub fn workers_from_rates(rates: &Rates) -> usize {
    let exec_is_positive = rates.t_exec_s > 0.0;
    if !exec_is_positive || !rates.t_read_s.is_finite() || rates.t_read_s <= 0.0 {
        // A non-positive or non-finite measurement is a broken clock reading
        // on a small chunk, not an infinitely fast stage. Do not divide by it.
        return 1;
    }
    let w = (rates.t_read_s / rates.t_exec_s).ceil();
    if !w.is_finite() {
        return 1;
    }
    (w as usize).clamp(1, W_MAX)
}

/// Measure one reader's chunk-production rate against one executor's
/// chunk-consumption rate, on a bounded prefix of `chrom`.
///
/// Deliberately unsharded and unpooled: `t_read_s` must be ONE worker's rate,
/// or the ratio it feeds means nothing.
pub fn probe_rates(
    vcf_path: &str,
    chrom: &str,
    samples: &[&str],
    chunk_size: usize,
    ploidy: usize,
    fields: &[crate::field::FieldSpec],
) -> Result<Rates, crate::error::ConversionError> {
    use std::time::Instant;

    let src = crate::vcf_reader::VcfRecordSource::new(
        vcf_path,
        chrom,
        samples,
        0, // htslib_threads: inline decode, matching a shard worker
        ploidy,
        fields,
        Vec::new(), // whole contig
        crate::svar2_view::OverlapMode::Pos,
    )?;
    let mut asm = crate::chunk_assembler::ChunkAssembler::new(
        Box::new(src),
        samples.len(),
        ploidy,
        None, // no FASTA: left-alignment cost is not what is being compared
        chrom,
        false, // skip_out_of_scope
        crate::normalize::CheckRef::Error,
        fields,
    )?;

    // The bank is a required sink for dense2sparse_vk, not an output. No
    // writer thread is spawned to drain the channel -- there is nothing to
    // write anywhere, so the probe leaves no user-visible bytes. `_rx` stays
    // alive so `flush_buffer`'s `send` (if a pathological chunk of long
    // alleles ever exceeds the buffer capacity) has somewhere to land rather
    // than panicking on a disconnected channel; the channel is unbounded so
    // that same `send` can never block waiting on a consumer that doesn't
    // exist, and its queued buffers are simply dropped with `_rx` on return.
    let (tx_long, _rx) = crossbeam_channel::unbounded::<Vec<u8>>();
    let mut bank = crate::nrvk::LongAlleleTableWriter::new(tx_long, 8 * 1024 * 1024);

    let mut read_s = 0.0f64;
    let mut exec_s = 0.0f64;
    let mut chunks = 0usize;

    while chunks < PROBE_CHUNKS {
        let t0 = Instant::now();
        let Some(chunk) = asm.read_next_chunk(chunk_size, chunks, None)? else {
            break; // Short contig: fewer than PROBE_CHUNKS chunks exist.
        };
        read_s += t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let _ = crate::rvk::dense2sparse_vk(&chunk, &mut bank, false, fields);
        exec_s += t1.elapsed().as_secs_f64();

        chunks += 1;
    }

    if chunks == 0 {
        return Err(crate::error::ConversionError::Input(format!(
            "probe found no records on contig {chrom}"
        )));
    }
    Ok(Rates {
        t_read_s: read_s / chunks as f64,
        t_exec_s: exec_s / chunks as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_htslib::bcf::record::GenotypeAllele;
    use rust_htslib::bcf::{Format, Header, Writer};

    /// To keep the executor fed, w readers must supply at least as fast as one
    /// executor drains: w/t_read >= 1/t_exec, so w = ceil(t_read/t_exec).
    #[test]
    fn workers_are_the_read_to_exec_ratio_rounded_up() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.30,
                t_exec_s: 0.10
            }),
            3
        );
        // 0.35/0.10 = 3.5 -> 4: rounding DOWN starves the executor, which is
        // the bottleneck this whole probe exists to keep busy.
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.35,
                t_exec_s: 0.10
            }),
            4
        );
    }

    /// A reader faster than the executor still needs one worker, not zero.
    #[test]
    fn floor_is_one_worker() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.01,
                t_exec_s: 0.50
            }),
            1
        );
    }

    /// W_MAX bounds the damage from a probe that hit a pathological prefix --
    /// an all-reference stretch reads far faster than it converts. The harness
    /// never observed a knee above 7.
    #[test]
    fn clamped_at_w_max() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 100.0,
                t_exec_s: 0.001
            }),
            W_MAX
        );
    }

    /// A zero or negative t_exec is a broken measurement (clock granularity on
    /// a tiny chunk), not an infinitely fast executor. Fall back rather than
    /// dividing by it.
    #[test]
    fn degenerate_exec_time_falls_back_to_one() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 1.0,
                t_exec_s: 0.0
            }),
            1
        );
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 1.0,
                t_exec_s: -1.0
            }),
            1
        );
    }

    /// One-contig, CSI-indexed BCF fixture with `n_samples` diploid samples
    /// named `s0..sN`, `n_variants` SNPs on contig "chr1", every sample
    /// het-carrying the ALT -- the probe densifies GT for all samples, so
    /// every record needs a full row.
    fn probe_fixture_vcf(dir: &std::path::Path, n_variants: usize, n_samples: usize) -> String {
        let path = dir.join("probe.bcf");
        let mut header = Header::new();
        header.push_record(
            format!(
                "##contig=<ID=chr1,length={}>",
                (n_variants as u32) * 10 + 100
            )
            .as_bytes(),
        );
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        let sample_names: Vec<String> = (0..n_samples).map(|i| format!("s{i}")).collect();
        for s in &sample_names {
            header.push_sample(s.as_bytes());
        }
        {
            let mut writer =
                Writer::from_path(&path, &header, false, Format::Bcf).expect("open BCF writer");
            for v in 0..n_variants {
                let mut record = writer.empty_record();
                record.set_rid(Some(0));
                record.set_pos((v as i64) * 10);
                record.set_alleles(&[b"A", b"C"]).expect("set alleles");
                let alleles: Vec<GenotypeAllele> = (0..n_samples)
                    .flat_map(|_| [GenotypeAllele::Phased(0), GenotypeAllele::Phased(1)])
                    .collect();
                record.push_genotypes(&alleles).expect("push genotypes");
                writer.write(&record).expect("write record");
            }
        }
        rust_htslib::bcf::index::build(&path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
            .expect("build BCF index");
        path.to_str().expect("utf8 path").to_string()
    }

    /// The probe returns positive timings and a plausible worker count on a
    /// real file. Bounds, not a pinned number: this measures a machine, and
    /// node speed here varies 2.08x.
    #[test]
    fn probe_returns_usable_rates_on_a_fixture() {
        let dir = tempfile::tempdir().unwrap();
        let path = probe_fixture_vcf(dir.path(), 200, 64); // 200 variants, 64 samples
        let samples: Vec<String> = (0..64).map(|i| format!("s{i}")).collect();
        let refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
        let rates = probe_rates(&path, "chr1", &refs, 64, 2, &[]).unwrap();
        assert!(rates.t_read_s > 0.0, "rates = {rates:?}");
        assert!(rates.t_exec_s > 0.0, "rates = {rates:?}");
        let w = workers_from_rates(&rates);
        assert!((1..=W_MAX).contains(&w), "w = {w}");
    }

    /// A contig absent from the file is a caller error, and must surface as
    /// one rather than as a silent w=1 that quietly under-provisions.
    #[test]
    fn probe_errors_on_an_unknown_contig() {
        let dir = tempfile::tempdir().unwrap();
        let path = probe_fixture_vcf(dir.path(), 50, 8);
        let samples: Vec<String> = (0..8).map(|i| format!("s{i}")).collect();
        let refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
        assert!(probe_rates(&path, "chrNope", &refs, 32, 2, &[]).is_err());
    }
}
