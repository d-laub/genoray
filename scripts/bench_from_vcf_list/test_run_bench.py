from run_bench import parse_per_contig_highwater, parse_arena_heaps

# Two contig banners; rss_samples are (elapsed_s, rss_kb) sampled during the run.
_STDERR = (
    "==> Processing 1\n(work)\n==> Processing 2\n(work)\nCohort Processing Complete.\n"
)
_BANNER_TIMES = {"1": 0.0, "2": 5.0}  # the driver stamps each banner's arrival time


def test_per_contig_highwater_buckets_by_banner():
    rss = [(0.5, 41_000_000), (2.0, 41_000_000), (5.5, 80_000_000), (9.0, 80_000_000)]
    hw = parse_per_contig_highwater(_STDERR, rss, banner_times=_BANNER_TIMES)
    assert hw == {"1": 41_000_000, "2": 80_000_000}


def test_arena_heaps_counts_64mb_heaps():
    smaps = "Size:  65536 kB\n...\nSize:  65536 kB\n...\nSize:   4 kB\n"
    assert parse_arena_heaps(smaps) == 2


def test_arena_heaps_absent_is_zero():
    assert parse_arena_heaps("") == 0
