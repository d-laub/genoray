//! SVAR2 write-path logging & progress events bridged to Python.
use crossbeam_channel::Sender;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    pending: AtomicU64,
    flush_every: u64,
}

impl EventSink {
    pub fn new(tx: Sender<Event>, flush_every: u64) -> Self {
        EventSink {
            inner: Some(Arc::new(SinkInner {
                tx,
                pending: AtomicU64::new(0),
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
            let prev = i.pending.fetch_add(n, Ordering::Relaxed) + n;
            if prev >= i.flush_every {
                // Take whatever is currently buffered and emit it.
                let take = i.pending.swap(0, Ordering::Relaxed);
                if take > 0 {
                    let _ = i.tx.send(Event::Progress {
                        chrom: chrom.to_string(),
                        delta: take,
                    });
                }
            }
        }
    }

    pub fn flush(&self, chrom: &str) {
        if let Some(i) = &self.inner {
            let take = i.pending.swap(0, Ordering::Relaxed);
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
        sink.tick("chr1", 60); // crosses 100 -> emits Progress{delta:120}
        sink.flush("chr1"); // remainder 0 already flushed? remainder=20 -> emit 20
        sink.contig_done("chr1", 230, 20, 1234);

        let evs: Vec<Event> = rx.try_iter().collect();
        assert!(matches!(
            evs[0],
            Event::ContigStart {
                total: Some(250),
                ..
            }
        ));
        // one Progress of 120, then remainder 20
        let deltas: Vec<u64> = evs
            .iter()
            .filter_map(|e| match e {
                Event::Progress { delta, .. } => Some(*delta),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.iter().sum::<u64>(), 120);
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
    fn disabled_sink_is_silent() {
        let sink = EventSink::disabled();
        sink.tick("chr1", 10);
        sink.contig_done("chr1", 1, 0, 1); // must not panic
    }
}
