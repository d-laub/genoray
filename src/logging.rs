//! SVAR2 write-path logging & progress events bridged to Python.
use crossbeam_channel::Sender;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Warning,
    Info,
    Debug,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Warning => "warning",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
        }
    }
}

#[derive(Debug)]
pub enum Event {
    ContigStart {
        chrom: String,
        total: Option<u64>,
    },
    Progress {
        chrom: String,
        delta: u64,
    },
    ContigDone {
        chrom: String,
        kept: u64,
        excluded: u64,
        elapsed_ms: u64,
    },
    Log {
        level: LogLevel,
        chrom: Option<String>,
        message: String,
        target: String,
    },
}

#[derive(Clone)]
pub struct EventSink {
    inner: Option<Arc<SinkInner>>,
}

struct SinkInner {
    tx: Sender<Event>,
    // Keyed by chrom so ticks from concurrently-dispatched contigs (rayon
    // fan-out over chroms) never accumulate into a shared counter and get
    // flushed under the wrong chrom's label.
    pending: Mutex<HashMap<String, u64>>,
    flush_every: u64,
}

impl EventSink {
    pub fn new(tx: Sender<Event>, flush_every: u64) -> Self {
        EventSink {
            inner: Some(Arc::new(SinkInner {
                tx,
                pending: Mutex::new(HashMap::new()),
                flush_every: flush_every.max(1),
            })),
        }
    }
    pub fn disabled() -> Self {
        EventSink { inner: None }
    }

    pub fn contig_start(&self, chrom: &str, total: Option<u64>) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::ContigStart {
                chrom: chrom.to_string(),
                total,
            });
        }
    }

    pub fn tick(&self, chrom: &str, n: u64) {
        if let Some(i) = &self.inner {
            let take = {
                let mut map = i.pending.lock().unwrap();
                let entry = map.entry(chrom.to_string()).or_insert(0);
                *entry += n;
                if *entry >= i.flush_every {
                    std::mem::replace(entry, 0)
                } else {
                    0
                }
            };
            if take > 0 {
                let _ = i.tx.send(Event::Progress {
                    chrom: chrom.to_string(),
                    delta: take,
                });
            }
        }
    }

    pub fn flush(&self, chrom: &str) {
        if let Some(i) = &self.inner {
            let take = {
                let mut map = i.pending.lock().unwrap();
                map.remove(chrom).unwrap_or(0)
            };
            if take > 0 {
                let _ = i.tx.send(Event::Progress {
                    chrom: chrom.to_string(),
                    delta: take,
                });
            }
        }
    }

    pub fn contig_done(&self, chrom: &str, kept: u64, excluded: u64, elapsed_ms: u64) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::ContigDone {
                chrom: chrom.to_string(),
                kept,
                excluded,
                elapsed_ms,
            });
        }
    }

    pub fn send_log(&self, level: LogLevel, chrom: Option<&str>, target: &str, message: String) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::Log {
                level,
                chrom: chrom.map(|s| s.to_string()),
                message,
                target: target.to_string(),
            });
        }
    }
}

pub fn level_from_str(s: &str) -> Option<LevelFilter> {
    match s {
        "off" => Some(LevelFilter::OFF),
        "warning" => Some(LevelFilter::WARN),
        "info" => Some(LevelFilter::INFO),
        "debug" => Some(LevelFilter::DEBUG),
        _ => None,
    }
}

fn to_log_level(l: &tracing::Level) -> LogLevel {
    match *l {
        tracing::Level::ERROR | tracing::Level::WARN => LogLevel::Warning,
        tracing::Level::INFO => LogLevel::Info,
        _ => LogLevel::Debug,
    }
}

#[derive(Default)]
struct FieldGrab {
    message: String,
    chrom: Option<String>,
}

impl Visit for FieldGrab {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "chrom" {
            self.chrom = Some(value.to_string());
        } else if field.name() == "message" {
            self.message = value.to_string();
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let v = format!("{value:?}");
        if field.name() == "chrom" {
            self.chrom = Some(v.trim_matches('"').to_string());
        } else if field.name() == "message" {
            self.message = v;
        }
    }
}

pub struct ChannelLayer {
    sink: EventSink,
}
impl ChannelLayer {
    pub fn new(sink: EventSink) -> Self {
        Self { sink }
    }
}

impl<S: tracing::Subscriber> Layer<S> for ChannelLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut g = FieldGrab::default();
        event.record(&mut g);
        self.sink.send_log(
            to_log_level(event.metadata().level()),
            g.chrom.as_deref(),
            event.metadata().target(),
            g.message,
        );
    }
}

pub fn with_channel_subscriber<R>(sink: EventSink, level: &str, f: impl FnOnce() -> R) -> R {
    let filter = level_from_str(level).unwrap_or(LevelFilter::INFO);
    let subscriber =
        tracing_subscriber::registry().with(ChannelLayer::new(sink).with_filter(filter));
    tracing::subscriber::with_default(subscriber, f)
}

static FMT_INIT: std::sync::Once = std::sync::Once::new();

/// Install a global stderr fmt subscriber driven by `GENORAY_LOG`, at most once
/// per process. Called by pure-Rust entry points (bench bin) — NOT by the
/// Python pipeline, which uses `with_channel_subscriber` instead.
pub fn install_fmt_fallback() {
    FMT_INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_env("GENORAY_LOG")
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .compact()
            .with_writer(std::io::stderr)
            .try_init();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    #[test]
    fn tick_flushes_on_threshold_and_remainder() {
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 100);
        sink.contig_start("chr1", Some(250));
        sink.tick("chr1", 60);
        sink.tick("chr1", 60); // crosses 100 -> swap drains buffer, emits Progress{delta:120}
        sink.tick("chr1", 20); // below threshold -> stays buffered as remainder
        sink.flush("chr1"); // emits the buffered remainder: Progress{delta:20}
        sink.contig_done("chr1", 230, 20, 1234);

        let evs: Vec<Event> = rx.try_iter().collect();
        assert!(matches!(
            evs[0],
            Event::ContigStart {
                total: Some(250),
                ..
            }
        ));
        // one Progress of 120 (threshold), then the flushed remainder of 20
        let deltas: Vec<u64> = evs
            .iter()
            .filter_map(|e| match e {
                Event::Progress { delta, .. } => Some(*delta),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec![120, 20]);
        assert_eq!(deltas.iter().sum::<u64>(), 140);
        assert!(matches!(
            evs.last().unwrap(),
            Event::ContigDone {
                kept: 230,
                excluded: 20,
                elapsed_ms: 1234,
                ..
            }
        ));
    }

    #[test]
    fn tick_attributes_per_chrom_under_interleaving() {
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 100);

        sink.tick("chrA", 60);
        sink.tick("chrB", 60);
        sink.tick("chrA", 60); // chrA now 120 >= 100 -> emits Progress{chrom:"chrA", delta:120}
        sink.flush("chrA"); // chrA remainder 0 -> nothing
        sink.flush("chrB"); // chrB 60 -> emits Progress{chrom:"chrB", delta:60}

        let progress: Vec<(String, u64)> = rx
            .try_iter()
            .filter_map(|e| match e {
                Event::Progress { chrom, delta } => Some((chrom, delta)),
                _ => None,
            })
            .collect();

        assert_eq!(progress.len(), 2, "expected exactly two Progress events");
        assert!(progress.contains(&("chrA".to_string(), 120)));
        assert!(progress.contains(&("chrB".to_string(), 60)));

        // No Progress event may mix counts across chroms: totals per chrom
        // must equal exactly what was ticked for that chrom.
        let chr_a_total: u64 = progress
            .iter()
            .filter(|(c, _)| c == "chrA")
            .map(|(_, d)| d)
            .sum();
        let chr_b_total: u64 = progress
            .iter()
            .filter(|(c, _)| c == "chrB")
            .map(|(_, d)| d)
            .sum();
        assert_eq!(chr_a_total, 120);
        assert_eq!(chr_b_total, 60);
    }

    #[test]
    fn disabled_sink_is_silent() {
        let sink = EventSink::disabled();
        sink.tick("chr1", 10);
        sink.contig_done("chr1", 1, 0, 1); // must not panic
    }

    #[test]
    fn channel_layer_routes_events_at_level() {
        use crossbeam_channel::unbounded;
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 1);
        with_channel_subscriber(sink, "info", || {
            tracing::info!(chrom = "chr1", "excluded 12 records");
            tracing::debug!(chrom = "chr1", "chr1:100 REF mismatch"); // filtered out at info
        });
        let logs: Vec<(String, String)> = rx
            .try_iter()
            .filter_map(|e| match e {
                Event::Log { chrom, message, .. } => Some((chrom.unwrap_or_default(), message)),
                _ => None,
            })
            .collect();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].0, "chr1");
        assert!(logs[0].1.contains("excluded 12"));
    }
}
