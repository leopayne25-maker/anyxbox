//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};

const LAZY_EAGER: u8 = 0;
const LAZY_DEFERRED: u8 = 1;
const LAZY_FLUSHING: u8 = 2;
const LAZY_RESYNC: u8 = 3;

fn is_lazy_held(state: u8) -> bool {
    state == LAZY_DEFERRED || state == LAZY_FLUSHING
}

thread_local! {
    static IN_LAZY_VEH: Cell<bool> = const { Cell::new(false) };
}

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Inline-first buffer for coalesced dirty page runs produced by
/// [`CopiedPages::plan_stale`]: avoids heap allocation for up to 8 runs per
/// refresh (the overwhelming majority of refreshes).
struct SmallRuns {
    inline: [(usize, usize); 8],
    inline_len: usize,
    spill: Vec<(usize, usize)>,
}

impl SmallRuns {
    fn new() -> Self {
        Self {
            inline: [(0, 0); 8],
            inline_len: 0,
            spill: Vec::new(),
        }
    }

    fn push(&mut self, from: usize, to: usize) {
        if self.inline_len < self.inline.len() {
            self.inline[self.inline_len] = (from, to);
            self.inline_len += 1;
        } else {
            self.spill.push((from, to));
        }
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.inline[..self.inline_len]
            .iter()
            .copied()
            .chain(self.spill.iter().copied())
    }
}

/// Per-page last copied write-watch sequence, paired with a 64-page-per-word
/// bitmask of uncopied pages so partially-initialized buffers and windows
/// still hit the O(1)/O(|written|) fast path on initialized subranges and skip
/// initialized regions 64 pages at a time when uncopied pages exist.
struct CopiedPages {
    at: Vec<u64>,
    /// Bit `i` (word `i / 64`, bit `i % 64`) is 1 iff `at[i] == 0`.
    /// Unused high bits in the final word are always 0.
    uncopied_bits: Vec<u64>,
    uncopied: usize,
}

impl CopiedPages {
    fn new(pages: usize) -> Self {
        let words = pages.div_ceil(64);
        let mut uncopied_bits = vec![u64::MAX; words];
        let rem = pages % 64;
        if rem != 0 {
            uncopied_bits[words - 1] = (1u64 << rem) - 1;
        }
        Self {
            at: vec![0; pages],
            uncopied_bits,
            uncopied: pages,
        }
    }

    fn get(&self, index: usize) -> Option<u64> {
        self.at.get(index).copied()
    }

    fn set(&mut self, index: usize, now: u64) {
        if let Some(slot) = self.at.get_mut(index) {
            if *slot == 0 && now != 0 {
                self.uncopied -= 1;
                self.uncopied_bits[index / 64] &= !(1u64 << (index % 64));
            } else if *slot != 0 && now == 0 {
                self.uncopied += 1;
                self.uncopied_bits[index / 64] |= 1u64 << (index % 64);
            }
            *slot = now;
        }
    }

    fn all_uncopied(&self) -> bool {
        self.uncopied == self.at.len()
    }

    #[cfg(test)]
    fn all_copied(&self) -> bool {
        self.uncopied == 0
    }

    #[cfg(feature = "diagnostics")]
    fn copied_count(&self) -> usize {
        self.at.len() - self.uncopied
    }

    /// Whether every page index in `lo..=hi` has been copied whole at least once.
    fn range_all_copied(&self, lo: usize, hi: usize) -> bool {
        if self.uncopied == 0 {
            return true;
        }
        let first_w = lo / 64;
        let last_w = hi / 64;
        let lo_mask = u64::MAX << (lo % 64);
        let hi_mask = u64::MAX >> (63 - (hi % 64));
        if first_w == last_w {
            return (self.uncopied_bits[first_w] & lo_mask & hi_mask) == 0;
        }
        if (self.uncopied_bits[first_w] & lo_mask) != 0 {
            return false;
        }
        for &word in &self.uncopied_bits[first_w + 1..last_w] {
            if word != 0 {
                return false;
            }
        }
        (self.uncopied_bits[last_w] & hi_mask) == 0
    }

    /// Lowest page index `>= from_idx` and `<= hi` that has never been copied whole.
    fn next_uncopied(&self, from_idx: usize, hi: usize) -> Option<usize> {
        if from_idx > hi || self.uncopied == 0 {
            return None;
        }
        let mut w = from_idx / 64;
        let last_w = hi / 64;
        let mut word = self.uncopied_bits[w] & (u64::MAX << (from_idx % 64));
        loop {
            if w == last_w {
                word &= u64::MAX >> (63 - (hi % 64));
            }
            if word != 0 {
                return Some(w * 64 + word.trailing_zeros() as usize);
            }
            if w == last_w {
                return None;
            }
            w += 1;
            word = self.uncopied_bits[w];
        }
    }

    /// Plans the coalesced byte ranges `[run_from, run_to)` inside `[from, to)`
    /// that must be copied from title memory, updating whole-page copied
    /// timestamps to `now`. Runs purely on metadata so callers can release the
    /// `copied` mutex before invoking memory copies.
    #[allow(clippy::too_many_arguments)]
    fn plan_stale(
        &mut self,
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
    ) -> SmallRuns {
        self.plan_stale_filtered(base, start, end, from, to, now, written, |_| false)
    }

    /// Like [`Self::plan_stale`], skipping page indices for which `skip_idx`
    /// returns `true` (used by lazy GPU-resident write-back so `DEFERRED`
    /// pages are neither copied from title memory nor coalesced across).
    #[allow(clippy::too_many_arguments)]
    fn plan_stale_filtered(
        &mut self,
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        skip_idx: impl Fn(usize) -> bool,
    ) -> SmallRuns {
        let mut runs = SmallRuns::new();
        if from >= to {
            return runs;
        }
        let first_page = from / PAGE;
        let last = (to - 1) / PAGE;
        let lo = first_page - base;
        let hi = last - base;

        let push_page = |runs: &mut SmallRuns, run: &mut Option<(usize, usize)>, page: usize| {
            match *run {
                Some((_, run_end)) if page < run_end => {}
                Some((run_start, run_end)) if page == run_end => {
                    *run = Some((run_start, page + 1));
                }
                Some((run_start, run_end)) => {
                    runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
                    *run = Some((page, page + 1));
                }
                None => *run = Some((page, page + 1)),
            }
        };

        let mut run: Option<(usize, usize)> = None;
        let w_start = if written.len() > 8 {
            written.partition_point(|&(p, _)| p < first_page)
        } else {
            0
        };

        // Fast path: every page in `[from, to)` has been initialized at least
        // once (even if other pages of the same buffer/window have not).
        if self.range_all_copied(lo, hi) {
            if written.is_empty() {
                return runs;
            }
            for &(page, write) in &written[w_start..] {
                if page < first_page {
                    continue;
                }
                if page > last {
                    break;
                }
                let idx = page - base;
                if skip_idx(idx) || write <= self.at[idx] {
                    continue;
                }
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    self.set(idx, now);
                }
                push_page(&mut runs, &mut run, page);
            }
            if let Some((run_start, run_end)) = run {
                runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
            }
            return runs;
        }

        // Sparse merge of uncopied pages (via 64-page bitmask) and dirty
        // `written` entries in `[first_page, last]`.
        let mut wi = w_start;
        while wi < written.len() && written[wi].0 < first_page {
            wi += 1;
        }
        let mut next_u = self.next_uncopied(lo, hi);
        loop {
            while wi < written.len() && written[wi].0 <= last {
                let (wp, write) = written[wi];
                let idx = wp - base;
                if self.at[idx] == 0 || write > self.at[idx] {
                    break;
                }
                wi += 1;
            }
            let next_w = (wi < written.len() && written[wi].0 <= last).then(|| written[wi].0 - base);
            let idx = match (next_u, next_w) {
                (Some(u), Some(w)) if u < w => {
                    next_u = self.next_uncopied(u + 1, hi);
                    u
                }
                (Some(u), Some(w)) if w < u => {
                    wi += 1;
                    w
                }
                (Some(u), Some(_)) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    wi += 1;
                    u
                }
                (Some(u), None) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    u
                }
                (None, Some(w)) => {
                    wi += 1;
                    w
                }
                (None, None) => break,
            };
            if skip_idx(idx) {
                continue;
            }
            let page = base + idx;
            if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                self.set(idx, now);
            }
            push_page(&mut runs, &mut run, page);
        }
        if let Some((run_start, run_end)) = run {
            runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
        }
        runs
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never), paired with an uncopied bitmask.
    copied: Mutex<CopiedPages>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Lock-free mirror of `!pending.is_empty()` so read-only buffers and
    /// windows skip acquiring `pending` on every refresh and flight check.
    has_pending: AtomicBool,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// Whether a shader write to this transient or rebar window was logged.
    written_said: std::sync::atomic::AtomicBool,
    /// (Neptune) `VICTORY_WINDOW_REBAR=1`: a window whose one copy is `buffer`, in a GPU upload heap
    /// (card memory the CPU maps write-combined) that `data` maps. No CPU twin, no card twin, no
    /// card: refreshes write the title's pages into it and shaders read it. The CPU must never
    /// read `data`.
    rebar: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
    /// Per-64 KiB page state for opt-in lazy GPU-resident write-back
    /// (`VICTORY_LAZY_COPYBACK=1`). Zero overhead when unused.
    lazy_state: Vec<AtomicU8>,
    /// Number of pages currently in `LAZY_DEFERRED` or `LAZY_FLUSHING`.
    deferred_count: AtomicUsize,
    /// Number of pages currently in `LAZY_RESYNC` (flushed to title memory on
    /// demand via VEH; CPU staging/snapshot needs resync on next update).
    resync_count: AtomicUsize,
    /// Highest `Host::fence` value signalled after a GPU submission that
    /// declared shader writes on this shadow (for lock-free cross-queue sync).
    last_write_fence: AtomicU64,
    #[cfg(test)]
    lazy_opted_in: bool,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    /// `None` for a transient window ([`window_upload`]): no CPU side, `chunks` instead.
    staging: Option<ID3D12Resource>,
    /// (Neptune) Transient window: the pages the title changed, each in a chunk of pooled
    /// upload memory (`crate::windowup`), in the order they were copied, until [`card_dirty`]
    /// takes them for the send.
    chunks: Mutex<Vec<(usize, usize, crate::windowup::Chunk)>>,
    /// (Neptune) Transient window: per 4 KiB page of title memory from the window's first on, a
    /// print of the bytes last copied into a chunk for it (0: none, or only a part of the page)
    /// ([`window_skip_same`]); empty until the window first takes title memory. Locked around
    /// the copy into chunks, with `chunks` locked inside it.
    prints: Mutex<Vec<u64>>,
    /// (Neptune) Whether `dirty` took a range since [`card_dirty`] last looked.
    flagged: std::sync::atomic::AtomicBool,
    /// The copy this is part of, once it is registered ([`note_carded`]):
    /// what [`Card::flag`] puts in [`FLAGGED`].
    me: std::sync::OnceLock<std::sync::Weak<Shadow>>,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// Lock-free mirror of `!dirty.is_empty()` updated strictly under `dirty`.
    has_dirty: AtomicBool,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// Lock-free mirror of `!sent.is_empty()` updated strictly under `sent`.
    has_sent: AtomicBool,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` (16) and `VICTORY_CARD_BUFFERS_MB` (512).
/// Run 69's gameplay has ten title buffers of 16 MiB or more, 494 MB, and
/// 1,650 MB of the card's budget free.
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default) << 20
        };
        // From 1 MiB up and 1,024 MB in all by default (it was 16 and 512):
        // the shader writes of the smaller buffers are then found by the
        // card's hashes too, and with that the first room's walls were
        // drawn in every run.
        // (Neptune) `VICTORY_CARD_BUFFERS_MIN_KB` sets the least size below a megabyte (64 KiB at
        // the least, a piece of the card's compare): an experiment on whether the shader writes
        // of the small buffers, found on the CPU otherwise, are what the first room's walls miss.
        let least = std::env::var("VICTORY_CARD_BUFFERS_MIN_KB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(mb("VICTORY_CARD_BUFFERS_MIN_MB", 1).max(1 << 20), |kb| (kb << 10).max(64 << 10));
        (least, mb("VICTORY_CARD_BUFFERS_MB", 1024))
    })
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=1`; off by default).
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame), and
/// [`card_for`] makes one only while the card is under eight tenths of its
/// budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used, and a copy may be made up to nine tenths of the budget: the room
/// it takes was kept for it. After the burst what is kept is what the
/// burst left of the limit (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    card_limits().1.saturating_sub(CARD_BYTES[0].load(Ordering::Relaxed) + CARD_REFUSED[1].load(Ordering::Relaxed))
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// The copies in the card's memory whose [`Card::flagged`] went up since
/// [`card_dirty`] last took them, each once per time it went up.
///
/// [`card_dirty`] went through all of [`CARDED`] at every submission (about
/// 59 a frame in the world) to find the few that changed: in a 30 s profile
/// of the world that pass was 2.5 % of the submitting thread's time, though
/// it took no lock of the copies it passed over.
static FLAGGED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// A copy just made in the card's memory: listed in [`CARDED`], and in
/// [`FLAGGED`] if it is made flagged (its first send takes it whole).
/// Copies that are gone are taken off [`CARDED`] here, each time its length
/// reaches a power of two.
fn note_carded(shadow: &Arc<Shadow>) {
    let Some(card) = &shadow.card else {
        return;
    };
    let weak = Arc::downgrade(shadow);
    let _ = card.me.set(weak.clone());
    {
        let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
        if carded.len() >= 64 && carded.len().is_power_of_two() {
            carded.retain(|w| w.strong_count() > 0);
        }
        carded.push(weak.clone());
    }
    if card.flagged.load(Ordering::Acquire) {
        FLAGGED.lock().unwrap_or_else(|e| e.into_inner()).push(weak);
    }
}

impl Card {
    /// `dirty` or `chunks` took something (after it is in them): the copy is
    /// flagged, and goes into [`FLAGGED`] if it was not flagged already.
    fn flag(&self) {
        if !self.flagged.swap(true, Ordering::AcqRel)
            && let Some(me) = self.me.get()
        {
            FLAGGED.lock().unwrap_or_else(|e| e.into_inner()).push(me.clone());
        }
    }
}

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|w| w.strong_count() > 0).count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// Bytes of the copies in the card's memory (buffers and windows) and of
/// their "before" twins, for the ram line.
pub(crate) fn card_copy_bytes() -> [u64; 2] {
    [
        CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed),
        CARD_BYTES[1].load(Ordering::Relaxed),
    ]
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // (With `card_room_kept_on` the texture queue kept this copy's room
    // free, and the copy may go up to the queue's own line.)
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while a fifth of the card's budget is free, as title buffers do
/// (`card_for`): textures keep their room, and once they have filled the
/// card later windows stay where they are. The CPU's side stays in the PC's
/// memory, so the memory shared with the card does not shrink by this
/// alone; what moves is what shaders read.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        // (Neptune) Unset: 4,096 MB where the system gives the process 14 GB of
        // the card or more (a 16 GB card: the windows then have room beside the
        // textures, and draws stop reading their vertices over the bus; the first
        // launches with it had every wall and body), none on a smaller card.
        let unset = || match host().and_then(|h| h.video_memory()) {
            Some([budget, ..]) if budget >= 14 << 30 => 4096,
            _ => 0,
        };
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or_else(unset) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 10 * 8
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// (Neptune) With `VICTORY_WINDOW_REBAR=1` (off by default): windows outside buffers keep one copy,
/// in a GPU upload heap (`Host::gpu_upload_buffer`): refreshes write title pages straight into
/// the card's memory through the CPU mapping and shaders read the same bytes. Taken before
/// `window_upload` and the card side; a window that gets none of it is made as without.
pub(crate) fn window_rebar() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_REBAR").is_some_and(|v| v == "1"))
}

/// Rebar windows alive, their bytes, bytes written into them since the ram line last looked, windows
/// refused for budget, buffers refused folding into one, windows refused for other reasons (no
/// support, no memory), shader writes declared to one (bug), unordered-access bindings refused.
pub(crate) static REBAR: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A GPU upload buffer for a window of `size` bytes, if it is to have one: within the windows' card
/// budget (`window_card_most`, counted with the card sides in `WINDOW_CARD_BYTES`) and the
/// card's own memory (the same fifth kept free as `window_card_for` keeps).
fn window_rebar_for(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let host = host()?;
    if !host.gpu_upload_supported() {
        REBAR[5].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let most = window_card_most();
    let room = most != 0 && host.video_memory().is_none_or(|[budget, used, ..]| used + size <= budget / 10 * 8);
    if !room || WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        if room {
            WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        }
        REBAR[3].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let made = host.gpu_upload_buffer(size);
    if made.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        REBAR[5].fetch_add(1, Ordering::Relaxed);
    }
    made
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
///
/// (Neptune) The second list is the transient windows ([`window_upload`]): each with the
/// ranges and the chunks they are sent from, in the order they were copied, not joined.
/// Nothing merges them back, so nothing is noted in `sent`.
pub(crate) fn card_dirty(sent_as: u64) -> (Vec<(Arc<Shadow>, Vec<(usize, usize)>)>, Vec<Uploads>) {
    let mut out = Vec::new();
    let mut uploads: Vec<Uploads> = Vec::new();
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    // Only the copies flagged since the last call ([`FLAGGED`]); one flagged
    // again while this runs is in the next call's list.
    let flagged = std::mem::take(&mut *FLAGGED.lock().unwrap_or_else(|e| e.into_inner()));
    for weak in flagged {
        let Some(shadow) = weak.upgrade() else {
            continue;
        };
        // (Neptune) No longer flagged: nothing to take.
        if shadow.card.as_ref().is_some_and(|c| !c.flagged.swap(false, Ordering::AcqRel)) {
            continue;
        }
        if shadow.transient() {
            let taken = shadow.card.as_ref().map(|c| std::mem::take(&mut *c.chunks.lock().unwrap_or_else(|e| e.into_inner())));
            if let Some(taken) = taken.filter(|t| !t.is_empty()) {
                uploads.push((shadow, taken));
            }
            continue;
        }
        let ranges = shadow
            .card
            .as_ref()
            .map(|c| {
                let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let ranges = std::mem::take(&mut *dirty);
                c.has_dirty.store(false, Ordering::Release);
                if relay() {
                    let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.retain(|s| s.2 > fetched);
                    sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
                    c.has_sent.store(!sent.is_empty(), Ordering::Release);
                }
                ranges
            })
            .unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
    }
    (out, uploads)
}

/// A transient window and what to send to its card side: (offset, end, chunk).
pub(crate) type Uploads = (Arc<Shadow>, Vec<(usize, usize, crate::windowup::Chunk)>);

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// The fetch inside flights (`CARD_FETCH` in host.rs) copies each piece
    /// from `inflight_piece_from` on: what the merge takes of the piece lies
    /// in that slot.
    #[test]
    fn a_piece_fetched_in_flight_fits_its_slot() {
        let slot = crate::host::INFLIGHT_SLOT as usize;
        for start in [0usize, 4096, 7, PIECE - 16, PIECE - 3, 3 * PIECE + 0x1f09, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    let at = crate::host::inflight_piece_from(&span, i);
                    assert_eq!(at % 16, 0);
                    assert!(lo >= at && hi - at <= slot, "{start:#x} {from} {to} piece {i}: {lo}..{hi} from {at}");
                }
            }
        }
    }

    /// The slots are a power of two, one per 64 KiB asked, within the
    /// shader's 32-bit offsets.
    #[test]
    fn inflight_slots_are_a_power_of_two() {
        use crate::host::{INFLIGHT_SLOT, inflight_slots};
        assert_eq!(inflight_slots(64 << 20), 1024);
        assert_eq!(inflight_slots(100 << 20), 1024);
        assert_eq!(inflight_slots(1), 1);
        assert_eq!(inflight_slots(1 << 20), 16);
        let most = inflight_slots(u64::MAX);
        assert!(most.is_power_of_two() && u64::from(most) * INFLIGHT_SLOT <= 1 << 32);
    }

    /// Tickets drawn and landed in order, the mark raised as the landing
    /// raises it (every ticket a landed flight drew), through the counter's
    /// wrap with the slots the default gives: no two tickets in the air
    /// that got a slot share one, slots are still given after the wrap, and
    /// the mark follows the counter.
    #[test]
    fn inflight_tickets_never_share_a_slot_across_the_wrap() {
        use crate::host::{inflight_gets_slot, inflight_raise, inflight_slots};
        let slots = inflight_slots(64 << 20);
        let mut landed = u32::MAX - 3000;
        let mut next = landed;
        // Flights of 1 to 700 flagged pieces, up to three in the air:
        // (tickets drawn, those that got a slot).
        let mut air: std::collections::VecDeque<(Vec<u32>, Vec<u32>)> = Default::default();
        let mut after_wrap = 0;
        for flight in 0..40u32 {
            let drawn: Vec<u32> = (0..1 + flight * 37 % 700).map(|i| next.wrapping_add(i)).collect();
            next = next.wrapping_add(drawn.len() as u32);
            let slotted: Vec<u32> = drawn.iter().copied().filter(|&k| inflight_gets_slot(k, landed, slots)).collect();
            after_wrap += slotted.iter().filter(|&&k| k < 1 << 20).count();
            air.push_back((drawn, slotted));
            let mut used: Vec<u32> = air.iter().flat_map(|a| a.1.iter()).map(|k| k % slots).collect();
            let n = used.len();
            used.sort_unstable();
            used.dedup();
            assert_eq!(used.len(), n, "flight {flight}: two tickets in the air share a slot");
            if air.len() == 3
                && let Some((drawn, _)) = air.pop_front()
            {
                for k in drawn {
                    landed = inflight_raise(landed, k);
                }
            }
        }
        while let Some((drawn, _)) = air.pop_front() {
            for k in drawn {
                landed = inflight_raise(landed, k);
            }
        }
        assert!(next < 1 << 20, "the counter wrapped");
        assert!(after_wrap > 0, "slots after the wrap");
        assert_eq!(landed, next);
        // Raised in wrapping order, never back.
        assert_eq!(inflight_raise(u32::MAX - 1, u32::MAX), 0);
        assert_eq!(inflight_raise(5, 2), 5);
        assert_eq!(inflight_raise(u32::MAX - 1, 3), 4);
        assert!(!inflight_gets_slot(3, 4, 8));
    }

    /// Flights 5, 6, 7 in the air with the fetch inside flights: 5 lands
    /// the old way with the queue's count at 7. The title's byte that went
    /// up in send 7 must still be told apart when 6 lands from its slot
    /// (its merge leaves out sends after 6), so send 7's note must outlive
    /// the mark (`card_dirty` keeps notes above it, `card_merge_relay`
    /// leaves out those above its "fetched after").
    #[test]
    fn a_fetch_the_old_way_keeps_the_notes_flights_in_the_air_need() {
        use crate::host::fallback_fetched_mark;
        let notes = [5u64, 6, 7];
        let mark = fallback_fetched_mark(true, 7, Some(5));
        let kept: Vec<u64> = notes.iter().copied().filter(|&s| s > mark).collect();
        assert!(kept.iter().any(|&s| s > 6), "send 7's note was dropped before flight 6 merged");
        assert_eq!(mark, 5);
        // Without the switch every later merge fetches after 7: as before.
        assert_eq!(fallback_fetched_mark(false, 7, Some(5)), 7);
        assert_eq!(fallback_fetched_mark(true, 7, None), 0);
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

/// Drains write-combining buffers after sequential CPU stores into a ReBAR
/// (`D3D12_HEAP_TYPE_GPU_UPLOAD`) mapping so subsequent reads or GPU work
/// observe the completed writes in order.
#[inline]
fn wc_sfence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SFENCE is part of baseline SSE2 on every x86-64 CPU.
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

/// Copies `len` bytes from write-combined (USWC) memory (`src`) into cached
/// system memory (`dst`) using AVX2 `VMOVNTDQA` streaming loads on 32-byte
/// aligned addresses bracketed by `_mm_mfence()`.
///
/// # Safety
/// `src` must be valid for reads of `len` bytes, `dst` valid for writes of
/// `len` bytes, and the two ranges must not overlap.
#[cfg_attr(not(test), allow(dead_code))]
unsafe fn copy_from_wc(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; pointer bounds guaranteed by caller.
        unsafe { copy_from_wc_avx2(src, dst, len) };
        return;
    }
    // SAFETY: caller's bounds and non-overlap guarantee.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_from_wc_avx2(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::{__m256i, _mm_mfence, _mm256_storeu_si256, _mm256_stream_load_si256};
    struct MfenceOnExit;
    impl Drop for MfenceOnExit {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: MFENCE is part of baseline SSE2 on every x86-64 CPU.
            unsafe { _mm_mfence() };
        }
    }
    _mm_mfence();
    let _exit_fence = MfenceOnExit;

    let prefix = src.align_offset(32).min(len);
    if prefix > 0 {
        // SAFETY: `prefix <= len`.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, prefix) };
    }
    let mut at = prefix;
    while at + 256 <= len {
        // SAFETY: `src.add(at)` is 32-byte aligned and `at + 256 <= len`.
        unsafe {
            let s = src.add(at);
            let d = dst.add(at);
            let v0 = _mm256_stream_load_si256(s.cast::<__m256i>());
            let v1 = _mm256_stream_load_si256(s.add(32).cast::<__m256i>());
            let v2 = _mm256_stream_load_si256(s.add(64).cast::<__m256i>());
            let v3 = _mm256_stream_load_si256(s.add(96).cast::<__m256i>());
            let v4 = _mm256_stream_load_si256(s.add(128).cast::<__m256i>());
            let v5 = _mm256_stream_load_si256(s.add(160).cast::<__m256i>());
            let v6 = _mm256_stream_load_si256(s.add(192).cast::<__m256i>());
            let v7 = _mm256_stream_load_si256(s.add(224).cast::<__m256i>());
            _mm256_storeu_si256(d.cast::<__m256i>(), v0);
            _mm256_storeu_si256(d.add(32).cast::<__m256i>(), v1);
            _mm256_storeu_si256(d.add(64).cast::<__m256i>(), v2);
            _mm256_storeu_si256(d.add(96).cast::<__m256i>(), v3);
            _mm256_storeu_si256(d.add(128).cast::<__m256i>(), v4);
            _mm256_storeu_si256(d.add(160).cast::<__m256i>(), v5);
            _mm256_storeu_si256(d.add(192).cast::<__m256i>(), v6);
            _mm256_storeu_si256(d.add(224).cast::<__m256i>(), v7);
        }
        at += 256;
    }
    while at + 32 <= len {
        // SAFETY: `src.add(at)` is 32-byte aligned and `at + 32 <= len`.
        unsafe {
            let v = _mm256_stream_load_si256(src.add(at).cast::<__m256i>());
            _mm256_storeu_si256(dst.add(at).cast::<__m256i>(), v);
        }
        at += 32;
    }
    if at < len {
        // SAFETY: remaining tail within `len`.
        unsafe { std::ptr::copy_nonoverlapping(src.add(at), dst.add(at), len - at) };
    }
}

/// Compares `len` bytes of write-combined memory (`wc_src`) against cached
/// memory (`cached`) using AVX2 `VMOVNTDQA` streaming loads bracketed by
/// `_mm_mfence()`.
///
/// # Safety
/// Both `wc_src` and `cached` must be valid for reads of `len` bytes.
#[cfg_attr(not(test), allow(dead_code))]
unsafe fn slices_equal_wc(wc_src: *const u8, cached: *const u8, len: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; pointer bounds guaranteed by caller.
        return unsafe { slices_equal_wc_avx2(wc_src, cached, len) };
    }
    // SAFETY: caller's bounds guarantee.
    unsafe {
        std::slice::from_raw_parts(wc_src, len) == std::slice::from_raw_parts(cached, len)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn slices_equal_wc_avx2(wc_src: *const u8, cached: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm_mfence, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_set1_epi8, _mm256_stream_load_si256,
    };
    struct MfenceOnExit;
    impl Drop for MfenceOnExit {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: MFENCE is part of baseline SSE2 on every x86-64 CPU.
            unsafe { _mm_mfence() };
        }
    }
    _mm_mfence();
    let _exit_fence = MfenceOnExit;

    let prefix = wc_src.align_offset(32).min(len);
    for i in 0..prefix {
        // SAFETY: `i < prefix <= len`.
        if unsafe { *wc_src.add(i) != *cached.add(i) } {
            return false;
        }
    }
    let mut at = prefix;
    const BLOCK: usize = 4096;
    while at + 256 <= len {
        let block_end = at.saturating_add(BLOCK).min(len - (len - at) % 256);
        let mut acc0 = _mm256_set1_epi8(-1);
        let mut acc1 = _mm256_set1_epi8(-1);
        while at < block_end {
            // SAFETY: `wc_src.add(at)` is 32-byte aligned and `at + 256 <= len`.
            unsafe {
                let w = wc_src.add(at);
                let c = cached.add(at);
                let w0 = _mm256_stream_load_si256(w.cast::<__m256i>());
                let w1 = _mm256_stream_load_si256(w.add(32).cast::<__m256i>());
                let w2 = _mm256_stream_load_si256(w.add(64).cast::<__m256i>());
                let w3 = _mm256_stream_load_si256(w.add(96).cast::<__m256i>());
                let w4 = _mm256_stream_load_si256(w.add(128).cast::<__m256i>());
                let w5 = _mm256_stream_load_si256(w.add(160).cast::<__m256i>());
                let w6 = _mm256_stream_load_si256(w.add(192).cast::<__m256i>());
                let w7 = _mm256_stream_load_si256(w.add(224).cast::<__m256i>());
                let e0 = _mm256_cmpeq_epi8(w0, _mm256_loadu_si256(c.cast::<__m256i>()));
                let e1 = _mm256_cmpeq_epi8(w1, _mm256_loadu_si256(c.add(32).cast::<__m256i>()));
                let e2 = _mm256_cmpeq_epi8(w2, _mm256_loadu_si256(c.add(64).cast::<__m256i>()));
                let e3 = _mm256_cmpeq_epi8(w3, _mm256_loadu_si256(c.add(96).cast::<__m256i>()));
                let e4 = _mm256_cmpeq_epi8(w4, _mm256_loadu_si256(c.add(128).cast::<__m256i>()));
                let e5 = _mm256_cmpeq_epi8(w5, _mm256_loadu_si256(c.add(160).cast::<__m256i>()));
                let e6 = _mm256_cmpeq_epi8(w6, _mm256_loadu_si256(c.add(192).cast::<__m256i>()));
                let e7 = _mm256_cmpeq_epi8(w7, _mm256_loadu_si256(c.add(224).cast::<__m256i>()));
                acc0 = _mm256_and_si256(acc0, _mm256_and_si256(_mm256_and_si256(e0, e1), _mm256_and_si256(e2, e3)));
                acc1 = _mm256_and_si256(acc1, _mm256_and_si256(_mm256_and_si256(e4, e5), _mm256_and_si256(e6, e7)));
            }
            at += 256;
        }
        if _mm256_movemask_epi8(_mm256_and_si256(acc0, acc1)) != -1i32 {
            return false;
        }
    }
    while at + 32 <= len {
        // SAFETY: `wc_src.add(at)` is 32-byte aligned and `at + 32 <= len`.
        let eq = unsafe {
            let w = _mm256_stream_load_si256(wc_src.add(at).cast::<__m256i>());
            let c = _mm256_loadu_si256(cached.add(at).cast::<__m256i>());
            _mm256_cmpeq_epi8(w, c)
        };
        if _mm256_movemask_epi8(eq) != -1i32 {
            return false;
        }
        at += 32;
    }
    for i in at..len {
        // SAFETY: `i < len`.
        if unsafe { *wc_src.add(i) != *cached.add(i) } {
            return false;
        }
    }
    true
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let n = now.len().min(held.len()).min(title.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_piece_avx2(&now[..n], &mut held[..n], &mut title[..n]) };
    }
    merge_piece_scalar(&now[..n], &mut held[..n], &mut title[..n])
}

fn merge_piece_scalar(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((&n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        if n != *h {
            differs |= n ^ *t;
            *t = n;
            *h = n;
        }
    }
    differs != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_piece_avx2(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
        _mm256_or_si256, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };
    let n = now.len();
    let (np, hp, tp) = (now.as_ptr(), held.as_mut_ptr(), title.as_mut_ptr());
    let mut vdiff = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        unsafe {
            let vn = _mm256_loadu_si256(np.add(i).cast::<__m256i>());
            let vh = _mm256_loadu_si256(hp.add(i).cast::<__m256i>());
            let eq = _mm256_cmpeq_epi8(vn, vh);
            if _mm256_movemask_epi8(eq) != -1i32 {
                let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
                let merged = _mm256_blendv_epi8(vn, vt, eq);
                vdiff = _mm256_or_si256(vdiff, _mm256_xor_si256(merged, vt));
                _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                _mm256_storeu_si256(hp.add(i).cast::<__m256i>(), vn);
            }
        }
        i += 32;
    }
    let differs = _mm256_testz_si256(vdiff, vdiff) == 0;
    merge_piece_scalar(&now[i..], &mut held[i..], &mut title[i..]) || differs
}

/// Synchronous card-buffer copy-back merge for [`Shadow::card_merge`]: takes
/// `gpu` where `gpu != old`, else `title`, writing merged bytes into both
/// `title` and `old` without dirtying unchanged 32-byte vectors. Returns
/// whether `title` changed.
fn merge_sync_card(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    let n = title.len().min(gpu.len()).min(old.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_sync_card_avx2(&mut title[..n], &gpu[..n], &mut old[..n]) };
    }
    merge_sync_card_scalar(&mut title[..n], &gpu[..n], &mut old[..n])
}

fn merge_sync_card_scalar(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((t, &g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
        if g != *o {
            differs |= g ^ *t;
            *t = g;
            *o = g;
        } else if *o != *t {
            *o = *t;
        }
    }
    differs != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_sync_card_avx2(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_testz_si256, _mm256_xor_si256,
    };
    let n = title.len();
    let (tp, gp, op) = (title.as_mut_ptr(), gpu.as_ptr(), old.as_mut_ptr());
    let mut vdiff = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        unsafe {
            let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
            let vg = _mm256_loadu_si256(gp.add(i).cast::<__m256i>());
            let vo = _mm256_loadu_si256(op.add(i).cast::<__m256i>());
            let eq_go = _mm256_cmpeq_epi8(vg, vo);
            let eq_to = _mm256_cmpeq_epi8(vt, vo);
            if _mm256_movemask_epi8(_mm256_and_si256(eq_go, eq_to)) != -1i32 {
                let merged = _mm256_blendv_epi8(vg, vt, eq_go);
                let diff_t = _mm256_xor_si256(merged, vt);
                if _mm256_testz_si256(diff_t, diff_t) == 0 {
                    vdiff = _mm256_or_si256(vdiff, diff_t);
                    _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                }
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(merged, vo)) != -1i32 {
                    _mm256_storeu_si256(op.add(i).cast::<__m256i>(), merged);
                }
            }
        }
        i += 32;
    }
    let differs = _mm256_testz_si256(vdiff, vdiff) == 0;
    merge_sync_card_scalar(&mut title[i..], &gpu[i..], &mut old[i..]) || differs
}

/// `[from, to)` without the ranges of `skip`, in order.
pub(crate) fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

/// The runs of 4-byte words of title memory (cut on its addresses;
/// `read` and `merged` are of the memory at `address`) that hold a byte in
/// which `merged` differs from `read`, as (offset, bytes) in order to
/// `put`. A shader stores words, so these are what the card wrote; every
/// other byte of `merged` is the title's as it was read.
fn word_runs(address: usize, read: &[u8], merged: &[u8], mut put: impl FnMut(usize, usize)) {
    let n = read.len().min(merged.len());
    let mut i = 0;
    while i < n {
        // Equal bytes are passed over, 32 at a time while they last.
        match first_difference(&read[i..n], &merged[i..n]) {
            Some(d) => i += d,
            None => break,
        }
        // From the word that holds this byte (or the stretch's first
        // byte) to the end of the last word in a row that holds one.
        let first = i - ((address + i) & 3).min(i);
        let mut end = first;
        while end < n {
            let word = (((address + end) | 3) + 1 - address).min(n);
            if read[end..word] == merged[end..word] {
                break;
            }
            end = word;
        }
        put(first, end - first);
        i = end;
    }
}

/// Whether the merges that write a stretch of title memory back
/// ([`Shadow::relay_merge`], [`Shadow::card_merge`],
/// [`Shadow::card_merge_relay`]) look, right before that write, for title
/// bytes it is about to put back to older values (`VICTORY_MERGE_RACE=1`;
/// off by default). Logging only: [`merge_race`].
fn merge_race_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_RACE").is_some_and(|v| v == "1"))
}

/// [`merge_race`]'s sums. Of frame `frame`, and in `total` since the
/// start: stretches written back, their bytes, title bytes put back to
/// older values, bytes the title and the card both changed, stretches with
/// either, the runs a write of the card's words alone makes
/// ([`word_runs`]) and their bytes. `said`: the frame's first stretches
/// with either. `quiet`: frames since the last line.
struct MergeRace {
    frame: u64,
    now: [u64; 7],
    said: Vec<String>,
    total: [u64; 7],
    quiet: u64,
}

static MERGE_RACE: Mutex<MergeRace> =
    Mutex::new(MergeRace { frame: 0, now: [0; 7], said: Vec::new(), total: [0; 7], quiet: 0 });

/// With [`merge_race_on`], right before a merge writes `merged` over title
/// memory at `address` (a stretch of the copy of the title buffer `copy`:
/// start and bytes): title memory is read once more and compared with
/// `read`, what the merge read of it before. A byte that changed in
/// between and that the write-back stores with its value as first read
/// (the card did not change it) is a title byte put back to an older
/// value; one the write-back stores with the card's byte is one both
/// changed. Only bytes the write-back stores are looked at. Summed per
/// frame; a line for every frame that has any, and one every 100 frames
/// whatever it holds, so that "none" is said, not assumed.
///
/// A lower bound: a title write that lands after this read (while the
/// write-back itself runs) is not seen, nor one that wrote a byte and then
/// its old value again. Costs one more read and two compares of every
/// stretch written back, on the thread that merges.
fn merge_race(read: Option<&[u8]>, copy: (u64, u64), address: usize, merged: &[u8], which: &'static str) {
    if !merge_race_on() {
        return;
    }
    let Some(read) = read else {
        return;
    };
    let n = read.len().min(merged.len());
    let (read, merged) = (&read[..n], &merged[..n]);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    word_runs(address, read, merged, |off, len| runs.push((off, len)));
    let whole = [(0, n)];
    let looked: &[(usize, usize)] = &whole;
    let mut again = vec![0u8; n];
    let (mut put_back, mut both, mut first) = (0u64, 0u64, None);
    if crate::host::read_into(address as *const u8, &mut again) {
        for &(off, len) in looked {
            for i in off..off + len {
                if again[i] == read[i] || again[i] == merged[i] {
                    continue;
                }
                if merged[i] == read[i] {
                    put_back += 1;
                    first.get_or_insert(i);
                } else {
                    both += 1;
                }
            }
        }
    }
    let frame = crate::frame::current();
    let mut race = MERGE_RACE.lock().unwrap_or_else(|e| e.into_inner());
    if race.frame != frame {
        race.quiet += 1;
        if race.now[4] > 0 || race.quiet >= 100 {
            tracing::info!(
                frame = race.frame,
                stretches = race.now[0],
                stretch_bytes = race.now[1],
                put_back = race.now[2],
                both_changed = race.now[3],
                stretches_hit = race.now[4],
                word_runs = race.now[5],
                word_run_bytes = race.now[6],
                stretches_since_start = race.total[0],
                stretch_bytes_since_start = race.total[1],
                put_back_since_start = race.total[2],
                both_changed_since_start = race.total[3],
                stretches_hit_since_start = race.total[4],
                word_runs_since_start = race.total[5],
                word_run_bytes_since_start = race.total[6],
                first = ?race.said,
                "merge race: title bytes that changed between a merge's read of title memory and its write-back of what shaders wrote, in this frame (VICTORY_MERGE_RACE=1)"
            );
            race.quiet = 0;
        }
        race.frame = frame;
        race.now = [0; 7];
        race.said.clear();
    }
    let hit = u64::from(put_back + both > 0);
    let run_bytes: u64 = runs.iter().map(|r| r.1 as u64).sum();
    let add = [1, n as u64, put_back, both, hit, runs.len() as u64, run_bytes];
    for (i, v) in add.iter().enumerate() {
        race.now[i] += v;
        race.total[i] += v;
    }
    if hit > 0 && race.said.len() < 8 {
        race.said.push(format!(
            "{which}: copy {:#x}+{:#x} stretch {address:#x}+{n:#x}: {put_back} put back (first at {:#x}), {both} both changed",
            copy.0,
            copy.1,
            address + first.unwrap_or(0)
        ));
    }
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length (AVX2 32 bytes at
/// a time, falling back to chunked scalar).
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= len`.
        return unsafe { first_difference_avx2(&a[..len], &b[..len]) };
    }
    first_difference_scalar(&a[..len], &b[..len])
}

fn first_difference_scalar(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn first_difference_avx2(a: &[u8], b: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
    let n = a.len();
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within both slices.
        let mask = unsafe {
            let va = _mm256_loadu_si256(ap.add(i).cast::<__m256i>());
            let vb = _mm256_loadu_si256(bp.add(i).cast::<__m256i>());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)) as u32
        };
        if mask != u32::MAX {
            return Some(i + (!mask).trailing_zeros() as usize);
        }
        i += 32;
    }
    first_difference_scalar(&a[i..], &b[i..]).map(|off| i + off)
}

/// Where `a` and `b` last differ within their common length (AVX2 32 bytes at
/// a time, falling back to chunked scalar).
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= len`.
        return unsafe { last_difference_avx2(&a[..len], &b[..len]) };
    }
    last_difference_scalar(&a[..len], &b[..len])
}

fn last_difference_scalar(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn last_difference_avx2(a: &[u8], b: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
    let n = a.len();
    let whole = n & !31;
    if let Some(i) = a[whole..].iter().zip(&b[whole..]).rposition(|(x, y)| x != y) {
        return Some(whole + i);
    }
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut i = whole;
    while i >= 32 {
        i -= 32;
        // SAFETY: `i + 32 <= whole <= n` within both slices.
        let mask = unsafe {
            let va = _mm256_loadu_si256(ap.add(i).cast::<__m256i>());
            let vb = _mm256_loadu_si256(bp.add(i).cast::<__m256i>());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)) as u32
        };
        if mask != u32::MAX {
            return Some(i + 31 - (!mask).leading_zeros() as usize);
        }
    }
    None
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
}

thread_local! {
    /// (Neptune) Whether this thread is the one that copies back (`crate::host`'s relay):
    /// its jobs go to kept threads of its own.
    static BACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// (Neptune) This thread is the one that copies back: see [`workers`].
pub(crate) fn wide_for_back() {
    BACK.with(|back| back.set(true));
}

/// The kept threads for this thread's jobs. (Neptune) Two sets: the submitting thread's
/// refreshes and merges under shader writes, and the copy-back's merges. In one line of
/// jobs the submitting thread's waited behind the copy-back's, and it stood waiting for the
/// kept threads an eighth of its time.
fn workers() -> &'static Workers {
    static WORKERS: [std::sync::OnceLock<Workers>; 2] = [const { std::sync::OnceLock::new() }; 2];
    static STARTED: [std::sync::Once; 2] = [const { std::sync::Once::new() }; 2];
    let which = usize::from(BACK.with(std::cell::Cell::get));
    let workers = WORKERS[which].get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED[which].call_once(|| {
        // `VICTORY_WIDE_THREADS=<n>` sets the submitting thread's number (2-24); the
        // copy-back gets half as many.
        let n = std::env::var("VICTORY_WIDE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(2, 6), |n| n.clamp(2, 24));
        let n = if which == 1 { (n / 2).max(2) } else { n };
        for i in 0..n {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{which}-{i}")).spawn(move || {
                let workers = WORKERS[which].get().expect("made above");
                loop {
                    let job = {
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                break job;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    job();
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let left = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            queue.push_back(Box::new(move || {
                job();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    workers.more.notify_all();
    loop {
        let job = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(job) => job(),
            None => break,
        }
    }
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while *n > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs(snaps));
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn.
    // Without a branch, so the loop runs many bytes at a time: the GPU's
    // byte where the copy's byte is not what it was before the submission
    // and the title left its byte as it was, else the title's. (With the
    // two searches above this was 44 ms a frame for 970 pieces.)
    // Where the shader and the title both changed a byte while the GPU ran,
    // the title's stays: the shader computed its byte from the copy's older
    // data (one cleared a bit of a record the title had already replaced,
    // and copying it back broke the title's object: an access violation
    // through 0xffffffff00000000; PalmVIDev e0b70476, VIctory f39c6c2).
    mask.clear();
    mask.resize(n, 0);
    let mut differs = 0u8;
    for (((t, h), o), m) in title.iter_mut().zip(host.iter_mut()).zip(old).zip(mask.iter_mut()) {
        let gpu = if (*h != *o) & (*t == *o) { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        *m = merged ^ *t;
        differs |= *m;
        *t = merged;
        *h = merged;
    }
    if differs == 0 {
        return None;
    }
    let mut i = 0;
    while i < n {
        while i < n && mask[i] == 0 {
            i += 1;
        }
        let start = i;
        while i < n && mask[i] != 0 {
            i += 1;
        }
        if i > start {
            runs.push((start, i - start));
        }
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
///
/// (Neptune) Then the times it told the caller to settle the flights, by
/// why: memory not write-watched, the journal could not say what was
/// written, more changed runs than the submission may send up, no snapshot.
static MERGED: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 7] {
    [0, 1, 2, 3, 4, 5, 6].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// (Neptune) Whether a refresh of a copy in the card's memory sends up only the bytes that
/// differ from what the copy holds (`VICTORY_COPY_CHANGED=1`; off by default).
///
/// A page counts as written when the kernel says so, and what our own copy-back writes into
/// title memory counts too: some 1,500 pieces of 64 KiB a frame came back from the card,
/// were written into title memory, were found written by the next refresh, were copied into
/// the copy that already held them, were sent up to the card that had made them, and were
/// hashed there again. With the switch the title's bytes are read beside the copy and only
/// the stretches that differ go in and up. What also changes: title bytes written again
/// with the value the copy holds do not go up over what a shader has written there since.
pub(crate) fn copy_changed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_CHANGED").is_some_and(|v| v == "1"))
}

/// (fps-neptune) `VICTORY_COPY_COALESCE=<bytes>` (0, off, by default; 256 suggested): changed
/// runs of one refresh that lie fewer bytes apart than this are sent up as one range. The bytes
/// between them are the same on the CPU side and the card (nothing of the stretch is being written
/// by shaders: `update` cut those ranges out), so sending them again changes nothing; what it saves
/// is ranges (27,661 a frame for 51 MB in run main-neptune-2258).
fn copy_coalesce_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("VICTORY_COPY_COALESCE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// `runs` (offset, bytes; in order) with those at most `gap` bytes apart joined.
fn coalesce_runs(runs: &[(usize, usize)], gap: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for &(r, m) in runs {
        match out.last_mut() {
            Some(last) if gap > 0 && r <= last.0 + last.1 + gap => last.1 = (r + m).max(last.0 + last.1) - last.0,
            _ => out.push((r, m)),
        }
    }
    out
}

#[cfg(test)]
mod coalesce_tests {
    use super::coalesce_runs;

    #[test]
    fn zero_gap_changes_nothing() {
        let runs = [(0, 4), (8, 4), (100, 4)];
        assert_eq!(coalesce_runs(&runs, 0), runs.to_vec());
    }

    #[test]
    fn near_runs_join_far_ones_stay() {
        let runs = [(0, 4), (8, 4), (100, 4), (300, 4)];
        assert_eq!(coalesce_runs(&runs, 16), vec![(0, 12), (100, 4), (300, 4)]);
        assert_eq!(coalesce_runs(&runs, 100), vec![(0, 104), (300, 4)]);
    }

    #[test]
    fn touching_and_contained_runs() {
        assert_eq!(coalesce_runs(&[(0, 8), (8, 4), (2, 2)], 1), vec![(0, 12)]);
    }
}

/// (Neptune) Whether windows outside buffers with a card side keep no CPU side
/// (`VICTORY_CARD_WINDOW_UPLOAD=1`; off by default): the pages the title changed are copied
/// at refresh into pooled upload memory (`crate::windowup`) and sent from there. Such a
/// window ("transient") cannot be written by shaders: a buffer is not folded into one, and a
/// declared shader write to one is counted and logged as an error.
pub(crate) fn window_upload() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_WINDOW_UPLOAD").is_some_and(|v| v == "1"))
}

/// Transient windows alive, buffers not folded into one, shader writes declared to one.
pub(crate) static TRANSIENT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// (Neptune) What a transient window does with a 4 KiB page of title memory it takes that holds
/// the bytes it last sent for the page (`VICTORY_WINDOW_SKIP_SAME`, [`window_skip_same`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SkipSame {
    /// `0`: sends it again and compares nothing, the code path as before the switch.
    Off,
    /// Unset, the default: sends it again; one page in 16, picked by its address
    /// ([`sampled_page`]), is read again and compared, for the frame-rate line.
    Sample,
    /// `1`: leaves it out. It is neither copied nor sent: the card side holds those bytes.
    Skip,
}

/// (Neptune) `VICTORY_WINDOW_SKIP_SAME` ([`SkipSame`]; read once).
///
/// A transient window ([`window_upload`]) has no CPU side to compare with, so every page the
/// write watch or the guards name is copied into a chunk and sent up whole: some 72 MB a frame
/// in the world (Series S, 20 fps), against 12 MB that differed of the 50 MB the copies in the
/// card's memory compared. A page written need not hold other bytes: the title rewrites
/// constant data. With `1` each window keeps a print of the bytes it last sent for
/// each 4 KiB page ([`page_print`], 8 bytes a page: 8 KiB for a window of 4 MiB), and a page
/// whose bytes give the same print is neither copied into a chunk nor sent.
///
/// What it rests on: the card side holds what the window last sent for each page. Nothing but
/// its sends writes it, and `Host::card_before` copies them in the order they were copied (a
/// send that replaces an earlier one before it went up comes after it). A buffer is not folded
/// into a transient window; a declared write of shaders to one (unordered access, a title buffer
/// copy on the card, title memory by address) is a bug, counted and logged
/// ([`Shadow::transient_written`]), and from then on the window compares nothing and sends all
/// it takes. A window let go takes its prints with it, so one made again for the same memory
/// starts with none; a page sent only in part has none; a chunk is not read after its send.
/// shortcut: two pages that differ and give the same print (about once in 2^64) leave the card
/// side with the older bytes until the title writes the page again.
pub(crate) fn window_skip_same() -> SkipSame {
    static MODE: std::sync::OnceLock<SkipSame> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_WINDOW_SKIP_SAME").ok().as_deref().map(str::trim) {
        Some("1") => SkipSame::Skip,
        Some("0") => SkipSame::Off,
        _ => SkipSame::Sample,
    })
}

/// (Neptune) The pages transient windows keep a print of ([`window_skip_same`]).
const PRINTED: usize = 4 << 10;

/// (Neptune) Bytes of the 4 KiB pages transient windows took since the last frame-rate line, by
/// what they were ([`Sent`]): the bytes the window last sent there, other bytes, nothing to
/// compare with (never sent, a page taken in part, or a window shaders write). With
/// [`SkipSame::Sample`] the pages compared, 16 times over: an estimate.
static WINDOW_SAME: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// (Neptune) Bytes of title memory transient windows took since the last frame-rate line: sent
/// up (copied into chunks), and left out ([`SkipSame::Skip`]).
static WINDOW_TAKEN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// (Neptune) With the frame line, while there are transient windows ([`window_upload`]): what
/// they took of title memory per frame over `frames`, sent and left out, and against what they
/// last sent ([`WINDOW_SAME`], [`window_skip_same`]). Taken and reset.
pub(crate) fn window_same_line(frames: f64) {
    if !window_upload() {
        return;
    }
    let mb = |n: &AtomicU64| n.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames;
    let [sent, left] = WINDOW_TAKEN.each_ref().map(mb);
    let [same, other, unknown] = WINDOW_SAME.each_ref().map(mb);
    let skip = match window_skip_same() {
        SkipSame::Skip => "1: left out",
        SkipSame::Sample => "unset: sent again, one 4 KiB page in 16 compared",
        SkipSame::Off => "0: sent again, nothing compared",
    };
    tracing::info!(
        "title memory transient windows took per frame: {sent:.1} MB sent up, {left:.1} MB left out; of what they \
         took, {same:.1} MB held the bytes the window last sent for their 4 KiB page, {other:.1} MB other bytes, \
         {unknown:.1} MB nothing to compare with (VICTORY_WINDOW_SKIP_SAME={skip})"
    );
}

/// (Neptune) Bytes of the transient windows' prints alive (`Card::prints`), for the ram line.
static WINDOW_PRINT_BYTES: AtomicU64 = AtomicU64::new(0);

/// (Neptune) [`WINDOW_PRINT_BYTES`].
pub(crate) fn window_print_bytes() -> u64 {
    WINDOW_PRINT_BYTES.load(Ordering::Relaxed)
}

/// (Neptune) What the bytes of a 4 KiB page a transient window takes are against its print of
/// what it last sent for the page: the index into [`WINDOW_SAME`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sent {
    Same = 0,
    Other = 1,
    Unknown = 2,
}

/// (Neptune) Keys of [`page_print`]: 24 for the words of the 64-byte stripes (stripe `n` of a
/// 1 KiB block takes keys `n` to `n + 7`), then 8 to start the lanes with and to mix them by.
/// SplitMix64's numbers from 0.
const PRINT_KEYS: [u64; 32] = {
    let mut keys = [0u64; 32];
    let (mut x, mut i) = (0u64, 0);
    while i < 32 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        keys[i] = z ^ (z >> 31);
        i += 1;
    }
    keys
};

/// (Neptune) A 64-bit print of a page's bytes ([`window_skip_same`]): equal bytes give equal
/// prints, and two different pages the same print about once in 2^64. Eight lanes take the
/// 64-bit words of each 64-byte stripe as XXH3 accumulates them: a word is added to the next
/// lane, and the product of its halves, keyed by its place in the stripe and the stripe's in
/// its 1 KiB block, to its own; the lanes are mixed after each block and folded into one at the
/// end, with the bytes left over, as `record_print` folds a descriptor's words. Some 40 GB/s on
/// one core. Not a hash anyone can attack: the title does not aim at it.
fn page_print(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes = [0u64; 8];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = PRINT_KEYS[24 + i] ^ bytes.len() as u64;
    }
    let mut stripes = bytes.chunks_exact(64);
    for (n, stripe) in (&mut stripes).enumerate() {
        let key = n % 16;
        let mut words = [0u64; 8];
        for (word, b) in words.iter_mut().zip(stripe.chunks_exact(8)) {
            *word = u64::from_le_bytes(b.try_into().unwrap_or([0; 8]));
        }
        for i in 0..8 {
            let x = words[i] ^ PRINT_KEYS[key + i];
            lanes[i ^ 1] = lanes[i ^ 1].wrapping_add(words[i]);
            lanes[i] = lanes[i].wrapping_add((x & 0xffff_ffff).wrapping_mul(x >> 32));
        }
        if key == 15 {
            for (i, lane) in lanes.iter_mut().enumerate() {
                *lane = (*lane ^ (*lane >> 47) ^ PRINT_KEYS[24 + i]).wrapping_mul(0x9e37_79b1);
            }
        }
    }
    let mut rest = stripes.remainder().chunks_exact(8);
    let words = rest.by_ref().map(|w| u64::from_le_bytes(w.try_into().unwrap_or([0; 8])));
    let mut h = K;
    for word in lanes.into_iter().chain(words) {
        h = (h ^ word).wrapping_mul(K);
        h ^= h >> 29;
    }
    for &b in rest.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(K);
    }
    h ^ (h >> 32)
}

/// (Neptune) Of 4 KiB page `index` of a transient window, `bytes` are taken (`whole`: all of the
/// page that lies in the window): what they are against the page's print, which becomes theirs.
/// A part of a page leaves it none: the card side then holds bytes of two takings.
fn page_sent(prints: &mut [u64], index: usize, whole: bool, bytes: &[u8]) -> Sent {
    let Some(print) = prints.get_mut(index) else {
        return Sent::Unknown;
    };
    if !whole {
        *print = 0;
        return Sent::Unknown;
    }
    // 0 is no print.
    let now = page_print(bytes).max(1);
    match std::mem::replace(print, now) {
        was if was == now => Sent::Same,
        0 => Sent::Unknown,
        _ => Sent::Other,
    }
}

/// (Neptune) The prints of a transient window over `[start, end)`: one per 4 KiB page that lies
/// in it.
fn prints_for(start: usize, end: usize) -> usize {
    (end - 1) / PRINTED - start / PRINTED + 1
}

/// (Neptune) `VICTORY_WINDOW_SKIP_SAME=1`: title memory `[from, to)` of a transient window over
/// `[start, end)`, read by `read` (address, bytes to fill) into `scratch` 256 KiB at a time; the
/// runs of 4 KiB pages that do not hold the bytes the window last sent for them go to `send`
/// (address, bytes), in order, and the others are left out. `prints`: the window's, one per page
/// from `start`'s on ([`prints_for`]). What each page was goes to `tally`, in bytes. Returns the
/// bytes sent.
fn window_copy(
    (start, end): (usize, usize),
    (from, to): (usize, usize),
    prints: &mut [u64],
    scratch: &mut Vec<u8>,
    tally: &mut [u64; 3],
    mut read: impl FnMut(usize, &mut [u8]),
    mut send: impl FnMut(usize, &[u8]),
) -> usize {
    // Pages never cross a block: what is read stays in the core's cache while it is printed
    // and copied into chunks.
    const BLOCK: usize = 256 << 10;
    if scratch.len() < BLOCK {
        scratch.resize(BLOCK, 0);
    }
    let first = start / PRINTED;
    let (mut sent, mut at) = (0, from);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    while at < to {
        let stop = ((at / BLOCK + 1) * BLOCK).min(to);
        let bytes = &mut scratch[..stop - at];
        read(at, bytes);
        runs.clear();
        for page in at / PRINTED..stop.div_ceil(PRINTED) {
            let (a, b) = ((page * PRINTED).max(at), ((page + 1) * PRINTED).min(stop));
            let whole = a == (page * PRINTED).max(start) && b == ((page + 1) * PRINTED).min(end);
            let what = page_sent(prints, page - first, whole, &bytes[a - at..b - at]);
            tally[what as usize] += (b - a) as u64;
            if what == Sent::Same {
                continue;
            }
            match runs.last_mut() {
                Some(last) if last.1 == a => last.1 = b,
                _ => runs.push((a, b)),
            }
        }
        for &(a, b) in &runs {
            send(a, &bytes[a - at..b - at]);
            sent += b - a;
        }
        at = stop;
    }
    sent
}

/// (Neptune) Whether 4 KiB page `page` of title memory (its number) is one of the one in 16 that
/// [`SkipSame::Sample`] compares: picked by a multiply of its number, so the same pages each
/// time, spread over the 64 KiB pages and the windows.
fn sampled_page(page: usize) -> bool {
    (page as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 60 == 0
}

/// (Neptune) [`SkipSame::Sample`]: of title memory `[from, to)` a transient window over
/// `[start, end)` has just sent, the 4 KiB pages [`sampled_page`] picks are read by `read`
/// (address, bytes to fill; false: not all there) and compared with the window's `prints`, which
/// take them; what they were goes to `tally`, 16 times their bytes. Nothing is left out.
fn window_sample(
    (start, end): (usize, usize),
    (from, to): (usize, usize),
    prints: &mut [u64],
    tally: &mut [u64; 3],
    mut read: impl FnMut(usize, &mut [u8]) -> bool,
) {
    let first = start / PRINTED;
    let mut page_bytes = [0u8; PRINTED];
    for page in (from / PRINTED..to.div_ceil(PRINTED)).filter(|&p| sampled_page(p)) {
        let (a, b) = ((page * PRINTED).max(from), ((page + 1) * PRINTED).min(to));
        let whole = a == (page * PRINTED).max(start) && b == ((page + 1) * PRINTED).min(end);
        let bytes = &mut page_bytes[..b - a];
        // A page not all there now is as one taken in part.
        let what = page_sent(prints, page - first, whole && read(a, bytes), bytes);
        tally[what as usize] += 16 * (b - a) as u64;
    }
}

/// (Neptune) [`WINDOW_SAME`] takes `tally`.
fn window_same_add(tally: &[u64; 3]) {
    for (n, &bytes) in WINDOW_SAME.iter().zip(tally) {
        if bytes != 0 {
            n.fetch_add(bytes, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod window_same_tests {
    use super::{PRINTED, Sent, page_print, page_sent, prints_for, sampled_page, window_copy, window_sample};

    /// Title memory for the tests: bytes that differ from page to page and within each.
    fn memory(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    /// Equal bytes give equal prints; a page with one bit flipped, two words or two stripes
    /// swapped, a word moved by a byte, or one byte longer gives another.
    #[test]
    fn a_print_changes_with_any_change_of_the_page() {
        let page = memory(PRINTED, 7);
        let print = page_print(&page);
        assert_eq!(page_print(&page.clone()), print);
        for bit in 0..PRINTED * 8 {
            let mut other = page.clone();
            other[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(page_print(&other), print, "bit {bit}");
        }
        let swapped = |a: usize, b: usize, n: usize| {
            let mut other = page.clone();
            for i in 0..n {
                other.swap(a + i, b + i);
            }
            page_print(&other)
        };
        // Words in one stripe, in two stripes of a block, in two blocks; whole stripes the
        // same keys meet in other blocks; the halves of a word.
        let swaps = [(0, 8, 8), (8, 72, 8), (64, 1088, 8), (0, 64, 64), (64, 1088, 64), (0, 1024, 1024), (0, 4, 4)];
        for (a, b, n) in swaps {
            assert_ne!(swapped(a, b, n), print, "{a} {b} {n}");
        }
        // Pages of zeros and of one repeated word, their sign bits flipped two at a time.
        let zeros = vec![0u8; PRINTED];
        let mut two = zeros.clone();
        two[7] ^= 0x80;
        two[15] ^= 0x80;
        assert_ne!(page_print(&two), page_print(&zeros));
        let mut ones = vec![0u8; PRINTED];
        ones.iter_mut().step_by(8).for_each(|b| *b = 1);
        assert_ne!(page_print(&ones), page_print(&zeros));
        let mut shifted = page.clone();
        shifted.rotate_right(1);
        assert_ne!(page_print(&shifted), print);
        assert_ne!(page_print(&page[..PRINTED - 1]), page_print(&page[..PRINTED - 2]));
        assert_ne!(page_print(&page[..13]), page_print(&page[..12]));
    }

    /// A page is the same when its bytes give the print last taken, unknown the first time and
    /// when taken in part (which leaves it no print), other when its bytes changed.
    #[test]
    fn a_page_is_the_same_only_after_the_same_bytes_were_taken_whole() {
        let mut prints = vec![0u64; 3];
        let (a, b) = (memory(PRINTED, 1), memory(PRINTED, 2));
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Unknown);
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Same);
        assert_eq!(page_sent(&mut prints, 1, true, &b), Sent::Other);
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Other);
        assert_eq!(page_sent(&mut prints, 1, false, &a[..100]), Sent::Unknown);
        assert_eq!(prints[1], 0, "a page taken in part keeps no print");
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Unknown);
        assert_eq!(page_sent(&mut prints, 1, true, &a), Sent::Same);
        assert_eq!((prints[0], prints[2]), (0, 0), "the other pages are not touched");
        // Past the window: nothing to compare with.
        assert_eq!(page_sent(&mut prints, 3, true, &a), Sent::Unknown);
    }

    /// A window over title memory, refreshed again and again over the ranges asked while the
    /// title writes some pages with other bytes and rewrites others with the bytes they hold:
    /// with the pages that hold what was last sent left out, the card side holds the title's
    /// bytes over every range refreshed, and nothing is sent of a page the title only rewrote.
    #[test]
    fn the_card_side_holds_the_titles_bytes_with_the_same_pages_left_out() {
        const START: usize = 0x40_0000;
        const LEN: usize = 1 << 20;
        let mut title = memory(LEN, 3);
        let mut card = vec![0xeeu8; LEN];
        let mut prints = vec![0u64; prints_for(START, START + LEN)];
        let mut scratch = Vec::new();
        let mut refresh = |title: &[u8], card: &mut [u8], prints: &mut [u64], from: usize, to: usize| {
            let mut tally = [0u64; 3];
            let mut sends = Vec::new();
            let sent = window_copy(
                (START, START + LEN),
                (from, to),
                prints,
                &mut scratch,
                &mut tally,
                |at, into| into.copy_from_slice(&title[at - START..at - START + into.len()]),
                |at, bytes| {
                    card[at - START..at - START + bytes.len()].copy_from_slice(bytes);
                    sends.push((at, at + bytes.len()));
                },
            );
            assert_eq!(&card[from - START..to - START], &title[from - START..to - START], "{from:#x}..{to:#x}");
            assert_eq!(tally.iter().sum::<u64>(), (to - from) as u64);
            assert_eq!(sent as u64, tally[Sent::Other as usize] + tally[Sent::Unknown as usize]);
            // In order; two touch only where a block read ends.
            assert!(
                sends.windows(2).all(|w| w[0].1 < w[1].0 || (w[0].1 == w[1].0 && w[0].1 % (256 << 10) == 0)),
                "{sends:?}"
            );
            (tally, sends)
        };
        // The first time, all of it is sent: nothing to compare with.
        let (tally, sends) = refresh(&title, &mut card, &mut prints, START, START + LEN);
        assert_eq!(tally, [0, 0, LEN as u64]);
        assert_eq!(sends.len(), (LEN / (256 << 10)), "one run per block read");
        // Nothing changed: nothing sent.
        let (tally, sends) = refresh(&title, &mut card, &mut prints, START, START + LEN);
        assert_eq!((tally, sends.len()), ([LEN as u64, 0, 0], 0));
        // Pages 3 and 9 written with other bytes, 4 to 8 rewritten with their own.
        let page = |p: usize| p * PRINTED..(p + 1) * PRINTED;
        title[page(3)].iter_mut().for_each(|b| *b ^= 0x55);
        title[page(9).start + 4095] ^= 1;
        let (tally, sends) = refresh(&title, &mut card, &mut prints, START, START + 16 * PRINTED);
        assert_eq!(tally, [14 * PRINTED as u64, 2 * PRINTED as u64, 0]);
        let at = |p: usize| START + p * PRINTED;
        assert_eq!(sends, [(at(3), at(4)), (at(9), at(10))]);
        // A range that begins and ends inside pages: those are sent in part and keep no print,
        // so they are sent whole the next time even though nothing changed. Page 21 changed
        // and goes with the part before it; page 22 is the same and left out.
        let (from, to) = (at(20) + 100, at(23) + 7);
        title[page(21)].iter_mut().for_each(|b| *b = b.wrapping_add(1));
        let (tally, sends) = refresh(&title, &mut card, &mut prints, from, to);
        assert_eq!(tally, [PRINTED as u64, PRINTED as u64, (PRINTED - 100 + 7) as u64]);
        assert_eq!(sends, [(from, at(22)), (at(23), to)]);
        let (tally, sends) = refresh(&title, &mut card, &mut prints, at(20), at(24));
        assert_eq!(tally, [2 * PRINTED as u64, 0, 2 * PRINTED as u64]);
        assert_eq!(sends, [(at(20), at(21)), (at(23), at(24))]);
        // Many rounds of writes, of bytes set back to what they held before the last refresh,
        // and of ranges, the card side checked each time.
        let mut x = 0x1234_5678_9abc_def1u64;
        let mut next = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        let mut undo: Vec<(usize, u8)> = Vec::new();
        for _ in 0..300 {
            for _ in 0..next(6) {
                let i = next(LEN);
                match next(3) {
                    0 => title[i] = title[i].wrapping_add(1 + next(255) as u8),
                    1 => {
                        undo.push((i, title[i]));
                        title[i] ^= 0x10;
                    }
                    _ => {
                        if let Some((i, was)) = undo.pop() {
                            title[i] = was;
                        }
                    }
                }
            }
            let from = START + next(LEN);
            let to = (from + 1 + next(300 << 10)).min(START + LEN);
            refresh(&title, &mut card, &mut prints, from, to);
        }
    }

    /// The pages compared without the switch are about one in 16 of all, the same each time;
    /// their bytes count 16 times, and nothing compared changes what is sent.
    #[test]
    fn one_page_in_sixteen_is_compared_and_counted_sixteen_times() {
        let picked = (0..1 << 16).filter(|&p| sampled_page(p)).count();
        assert!((3600..=4600).contains(&picked), "{picked} of 65536");
        assert_eq!((0..1 << 16).filter(|&p| sampled_page(p)).count(), picked);
        const START: usize = 0x80_0000;
        const LEN: usize = 4 << 20;
        let title = memory(LEN, 9);
        let mut prints = vec![0u64; prints_for(START, START + LEN)];
        let read = |at: usize, into: &mut [u8]| {
            into.copy_from_slice(&title[at - START..at - START + into.len()]);
            true
        };
        let mut tally = [0u64; 3];
        window_sample((START, START + LEN), (START, START + LEN), &mut prints, &mut tally, read);
        let picked = (START / PRINTED..(START + LEN) / PRINTED).filter(|&p| sampled_page(p)).count() as u64;
        assert_eq!(tally, [0, 0, 16 * PRINTED as u64 * picked]);
        let mut tally = [0u64; 3];
        window_sample((START, START + LEN), (START, START + LEN), &mut prints, &mut tally, read);
        assert_eq!(tally, [16 * PRINTED as u64 * picked, 0, 0]);
        // A page not all there is not compared, and keeps no print.
        let p = (START / PRINTED..).find(|&p| sampled_page(p)).unwrap_or(0);
        let mut tally = [0u64; 3];
        let (from, to) = (p * PRINTED, (p + 1) * PRINTED);
        window_sample((START, START + LEN), (from, to), &mut prints, &mut tally, |_, _| false);
        assert_eq!(tally, [0, 0, 16 * PRINTED as u64]);
        assert_eq!(prints[p - START / PRINTED], 0);
    }
}

/// Pages the copy-back told their copy it holds (`crate::watch::quiet_back`).
static QUIET_BACK: AtomicU64 = AtomicU64::new(0);

/// Writes of the copy-back that went through a view of our own.
static QUIET_BACK_VIEWS: AtomicU64 = AtomicU64::new(0);

/// [`QUIET_BACK`] and [`QUIET_BACK_VIEWS`], taken.
pub(crate) fn quiet_back_stats() -> [u64; 2] {
    [QUIET_BACK.swap(0, Ordering::Relaxed), QUIET_BACK_VIEWS.swap(0, Ordering::Relaxed)]
}

/// Bytes read for refreshes with [`copy_changed`], and those of them that differed.
static COPY_CHANGED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// [`COPY_CHANGED`], taken.
pub(crate) fn copy_changed_stats() -> [u64; 2] {
    [0, 1].map(|i| COPY_CHANGED[i].swap(0, Ordering::Relaxed))
}

/// Adds the 4 KiB pages of `page` named by `mask` to the byte run being built (`run`), which
/// grows where they follow it and goes to `done` where they do not.
fn extend_runs(run: &mut Option<(usize, usize)>, done: &mut Vec<(usize, usize)>, page: usize, mask: u16) {
    const SUB: usize = 4096;
    for k in (0..16).filter(|k| mask >> k & 1 == 1) {
        let (a, b) = (page * PAGE + k * SUB, page * PAGE + (k + 1) * SUB);
        match run {
            Some((_, e)) if *e == a => *e = b,
            _ => done.extend(run.replace((a, b))),
        }
    }
}

/// Where `now` differs from `was`, as runs (offset, bytes) in steps of 256 bytes; runs
/// less than 1 KiB apart are one.
fn changed_spans(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    const STEP: usize = 256;
    const JOIN: usize = 1024;
    let n = now.len().min(was.len());
    let mut runs: Vec<(usize, usize)> = Vec::new();
    if now[..n] == was[..n] {
        return runs;
    }
    let mut i = 0;
    while i < n {
        let e = (i + STEP).min(n);
        if now[i..e] != was[i..e] {
            match runs.last_mut() {
                Some(last) if i <= last.0 + last.1 + JOIN => last.1 = e - last.0,
                _ => runs.push((i, e - i)),
            }
        }
        i = e;
    }
    runs
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    // (Neptune) Nothing differs, the usual answer: one comparison of the whole.
    if now[..n] == was[..n] {
        return runs;
    }
    let mut i = 0;
    while i < n {
        // Equal bytes, eight at a time while they last.
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// (Neptune) Whether the title's changed bytes under flights' shader writes
/// go up to a copy in the card's memory as whole 32-bit words
/// (`VICTORY_UNDER_WRITES_WORDS=1`; off by default).
///
/// The first scene's title rewrites 26,000-33,000 runs a frame there, about
/// two bytes each and some 400 to a page (runs ve44, ve45): numbers that
/// move a little, of which the low bytes change and the high ones stay.
/// Each run was a copy of its own on the card, and with more than
/// `VICTORY_UNDER_WRITES_RUNS` of them in a submission the flights were
/// settled. Widened to the words they lie in, runs of neighbouring numbers
/// touch and are one.
///
/// What it changes: the bytes a run is widened by are the title's, as the
/// copy's CPU side holds them (the title did not change them), and they go
/// up with the run. A shader's write to them is undone only if a flight in
/// the air wrote the very word the title also wrote, and a word that is
/// half the title's and half a shader's is no value either way.
pub(crate) fn under_writes_words() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UNDER_WRITES_WORDS").is_some_and(|v| v == "1"))
}

/// (Neptune) `runs` (offset, bytes; in order, as [`changed_runs`] gives
/// them) of a stretch of `n` bytes of title memory from `address` on,
/// widened to the 32-bit words of title memory they lie in (cut to the
/// stretch) and joined where they then touch or overlap.
fn widen_to_words(address: usize, n: usize, runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (at, bytes) in runs {
        let from = ((address + at) & !3).max(address) - address;
        let to = ((address + at + bytes + 3) & !3).min(address + n) - address;
        match out.last_mut() {
            Some(last) if from <= last.0 + last.1 => last.1 = last.1.max(to - last.0),
            _ => out.push((from, to - from)),
        }
    }
    out
}

#[cfg(test)]
mod merge_tests {
    use super::{PIECE, back_piece, changed_runs, widen_to_words};

    /// Runs widened to words hold every byte of the runs given, lie in the
    /// stretch, begin and end on word lines of title memory (or at the
    /// stretch's ends), and do not touch one another.
    #[test]
    fn word_runs_hold_the_runs_and_end_on_word_lines() {
        for address in [0x1000usize, 0x1001, 0x1002, 0x1003] {
            let n = 64;
            let runs = vec![(0, 1), (5, 2), (9, 2), (13, 1), (30, 3), (40, 1), (62, 2)];
            let words = widen_to_words(address, n, runs.clone());
            for &(at, bytes) in &runs {
                assert!(words.iter().any(|&(w, m)| w <= at && at + bytes <= w + m), "{address:#x} {at}");
            }
            for &(w, m) in &words {
                assert!(m > 0 && w + m <= n);
                assert!(w == 0 || (address + w) % 4 == 0, "{address:#x} {w}");
                assert!(w + m == n || (address + w + m) % 4 == 0, "{address:#x} {w} {m}");
            }
            assert!(words.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{address:#x} {words:?}");
        }
        // Neighbouring numbers of which two bytes each changed: one run.
        assert_eq!(widen_to_words(0x2000, 32, vec![(0, 2), (4, 2), (8, 2), (12, 2), (20, 2)]), [(0, 16), (20, 4)]);
        assert!(widen_to_words(0x2000, 32, Vec::new()).is_empty());
    }

    /// A piece put back takes the GPU's bytes where the copy changed and
    /// the title's everywhere else, reads and writes title memory only from
    /// the first change to the last, and its scratch holds that stretch and
    /// no more: not the 64 KiB per range it once was.
    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.deferred_count.load(Ordering::Acquire) > 0 || self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_flush_all_on_drop();
        }
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        if self.rebar {
            REBAR[0].fetch_sub(1, Ordering::Relaxed);
            REBAR[1].fetch_sub(self.len, Ordering::Relaxed);
            WINDOW_CARD_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        } else if self.transient() {
            TRANSIENT[0].fetch_sub(1, Ordering::Relaxed);
        } else {
            LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        }
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
            let prints = card.prints.get_mut().unwrap_or_else(|e| e.into_inner()).len();
            if prints != 0 {
                WINDOW_PRINT_BYTES.fetch_sub(prints as u64 * 8, Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = loose_write();
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = loose_write();
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.copied_count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = loose_write();
        if map.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        let idle: Vec<(u64, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        idle.into_iter().filter_map(|key| map.remove(&key)).collect()
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

/// For the resident line: the CPU sides of the copies of title buffers and of the windows
/// outside buffers, as (address, length) each. Copies in place have no CPU side.
pub(crate) fn cpu_ranges() -> [Vec<(usize, usize)>; 2] {
    let sides = |shadows: Vec<Arc<Shadow>>| -> Vec<(usize, usize)> {
        let mut out: Vec<_> = shadows
            .iter()
            .filter(|s| !s.data.is_null() && s.place.is_none() && !s.transient() && !s.rebar)
            .map(|s| (s.data as usize, s.len as usize))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    };
    let loose: Vec<_> = LOOSE.read().unwrap_or_else(|e| e.into_inner()).values().cloned().collect();
    let kept: Vec<_> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    [sides(kept), sides(loose)]
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = loose_write();
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .filter(|s| !s.transient() && !s.rebar)
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.len() as u64, pool.iter().map(|v| v.capacity() as u64).sum::<u64>())
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = ((start + len).div_ceil(PAGE as u64) - start / PAGE as u64) as usize;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(CopiedPages::new(pages)),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            has_pending: AtomicBool::new(false),
            flights: Mutex::default(),
            before: Mutex::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            written_said: std::sync::atomic::AtomicBool::new(false),
            rebar: false,
            place: None,
            lazy_state: (0..pages).map(|_| AtomicU8::new(LAZY_EAGER)).collect(),
            deferred_count: AtomicUsize::new(0),
            resync_count: AtomicUsize::new(0),
            last_write_fence: AtomicU64::new(0),
            #[cfg(test)]
            lazy_opted_in: lazy_copyback(),
        }
    }

    /// (Neptune) A window of title memory outside buffers with a card side and no CPU side
    /// ([`window_upload`]).
    pub(crate) fn transient(&self) -> bool {
        self.window && self.card.as_ref().is_some_and(|c| c.staging.is_none())
    }

    /// Whether this is a transient window that shaders are said to write: a bug (it keeps no
    /// CPU side to merge into), logged once per window. (Neptune) From then on the window
    /// compares nothing with what it last sent ([`window_skip_same`]): its card side may hold
    /// the shaders' bytes.
    fn transient_written(&self, address: usize) -> bool {
        if self.rebar {
            // Nothing to merge into or compare with: write-combined memory is never read.
            REBAR[6].fetch_add(1, Ordering::Relaxed);
            if !self.written_said.swap(true, Ordering::Relaxed) {
                tracing::error!(
                    address = format_args!("{address:#x}"),
                    copy_start = format_args!("{:#x}", self.start),
                    copy_bytes = self.len,
                    "shaders write a window in a GPU upload heap (VICTORY_WINDOW_REBAR=1): its writes are not merged back"
                );
            }
            return true;
        }
        if !self.transient() {
            return false;
        }
        TRANSIENT[2].fetch_add(1, Ordering::Relaxed);
        if !self.written_said.swap(true, Ordering::Relaxed) {
            tracing::error!(
                address = format_args!("{address:#x}"),
                copy_start = format_args!("{:#x}", self.start),
                copy_bytes = self.len,
                "shaders write a transient window (VICTORY_CARD_WINDOW_UPLOAD=1): its writes are not merged back"
            );
        }
        true
    }

    /// (Neptune) Whether this is a window in a GPU upload heap ([`window_rebar`]).
    pub(crate) fn is_rebar(&self) -> bool {
        self.rebar
    }

    /// (Neptune) Of a rebar window: whether [`Shadow::update`] of `[address, address + len)` would
    /// overwrite bytes a flight in the air may still be reading, i.e. a page the title wrote since
    /// the copy last took it (as [`Shadow::stale_under_writes`] decides, from the journal; where it
    /// cannot say: yes). A window never refreshed has nothing in the air.
    pub(crate) fn rebar_stale(&self, address: usize, len: usize) -> bool {
        if !self.rebar {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        if from >= to || copied.all_uncopied() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        match crate::watch::asked(from, to) {
            Some((_, written)) => written
                .into_iter()
                .any(|(page, write)| page >= base && copied.get(page - base).is_none_or(|at| at == 0 || write > at)),
            None => true,
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        // (Neptune) Every declaration, before the cases below that return: the sanitizer must see
        // them all (`VICTORY_CLAMP_AFTER_WRITER`).
        if crate::indirect::after_writer() {
            let end = address.saturating_add(len.min(self.left(address)));
            list.with_san(|s| s.wrote(address as usize, end as usize));
        }
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() || self.transient_written(address as usize) {
            return;
        }
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() || self.transient_written(address) {
            return;
        }
        // Flush overlapping OTHER shadows' deferred pages and resync any
        // `LAZY_RESYNC` pages ONLY on subranges not already in `self.pending`:
        // ranges in `self.pending` may be concurrently read/merged by the relay
        // thread and still need their `FlushedBaseline` intact.
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        let need_other = LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0;
        let need_resync = self.resync_count.load(Ordering::Acquire) > 0;
        if from < to && (need_other || need_resync) {
            let busy_raw: Vec<(usize, usize)> = if self.has_pending.load(Ordering::Acquire) {
                self.pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .copied()
                    .filter(|&(a, e)| a < to && e > from)
                    .collect()
            } else {
                Vec::new()
            };
            let free: Vec<(usize, usize)> = if busy_raw.is_empty() {
                vec![(from, to)]
            } else {
                let busy_pages: Vec<(usize, usize)> = busy_raw
                    .iter()
                    .map(|&(a, e)| (a & !(PAGE - 1), e.div_ceil(PAGE).saturating_mul(PAGE)))
                    .collect();
                outside(from, to, busy_pages)
            };
            for (a, e) in free {
                if need_other && LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0 {
                    let flushed = lazy_flush_other_shadows(self.id, a, e);
                    if !flushed.is_empty() {
                        self.resync_pages_from_title(&flushed, &busy_raw);
                    }
                }
                if self.resync_count.load(Ordering::Acquire) > 0 {
                    self.lazy_resync_range(a, e, &busy_raw);
                }
            }
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            self.has_pending.store(true, Ordering::Release);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        self.has_pending.store(true, Ordering::Release);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from);
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(self: &Arc<Self>, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(self: &Arc<Self>, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before_vec) = snapshot.as_mut()
            && from < to
        {
            if self.resync_count.load(Ordering::Acquire) > 0 {
                self.lazy_apply_flushed_baseline(from, &mut before_vec[from..to]);
            }
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before_vec.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            let deferred_mask: Vec<bool> = if lazy_copyback() && !differing.is_empty() {
                let offsets: Vec<(usize, usize)> = differing.iter().map(|&(at, n)| (at, at + n)).collect();
                self.try_defer_shadow_pieces(&offsets, None)
            } else {
                Vec::new()
            };
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (idx, (at, n)) in differing.into_iter().enumerate() {
                if deferred_mask.get(idx).copied().unwrap_or(false) {
                    continue;
                }
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                let at = at + first;
                crate::watch::writing(start + at, n);
                // Only the GPU's bytes (upstream #2009): the rest of the
                // stretch is the title's as read, and a title write landing
                // there since stays.
                for &(off, len) in &runs {
                    crate::host::write_mapped((start + at + off) as *mut u8, title[off..].as_ptr(), len);
                    written += len;
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || self.rebar || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        if self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_apply_flushed_baseline(at, held);
        }
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = merge_race_on().then(|| title.clone());
        if !merge_piece(now, held, title) {
            return 0;
        }
        crate::watch::writing(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "pc relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            (at - (start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        self.has_pending.store(!pending.is_empty(), Ordering::Release);
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        self.has_pending.load(Ordering::Acquire)
    }

    /// Stamps the `Host::fence` value signalled after a GPU submission that
    /// declared shader writes on this shadow (monotonic max).
    pub(crate) fn note_write_fence(&self, fence: u64) {
        self.last_write_fence.fetch_max(fence, Ordering::Release);
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        if !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        if self.resync_count.load(Ordering::Acquire) > 0 {
            for &(a, e) in &busy {
                for page in a / PAGE..e.div_ceil(PAGE) {
                    if page >= base
                        && (page - base) < self.lazy_state.len()
                        && self.lazy_state[page - base].load(Ordering::Acquire) == LAZY_RESYNC
                    {
                        return true;
                    }
                }
            }
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => stale_pages_in(base, &copied.at, a, e, &written)
                .into_iter()
                .any(|p| !has_deferred || !is_lazy_held(self.lazy_state[p - base].load(Ordering::Acquire))),
            None => true,
        })
    }

    /// Whether `[address, address + len)` overlaps both `self.pending` and at
    /// least one `LAZY_RESYNC` page (used by `host::stale_under_flights` for
    /// UAV-only `Step::GpuWrite` commands without a preceding `Step::Refresh`).
    pub(crate) fn has_lazy_resync_under_writes(&self, address: usize, len: usize) -> bool {
        if self.resync_count.load(Ordering::Acquire) == 0 || !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        let base = start / PAGE;
        for (a, e) in busy {
            for page in a / PAGE..e.div_ceil(PAGE) {
                if page >= base
                    && (page - base) < self.lazy_state.len()
                    && self.lazy_state[page - base].load(Ordering::Acquire) == LAZY_RESYNC
                {
                    return true;
                }
            }
        }
        false
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        if !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            MERGED[3].fetch_add(1, Ordering::Relaxed);
            return true;
        }
        let base = start / PAGE;
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        if self.resync_count.load(Ordering::Acquire) > 0 {
            for &(a, e) in &busy {
                for page in a / PAGE..e.div_ceil(PAGE) {
                    if page >= base
                        && (page - base) < self.lazy_state.len()
                        && self.lazy_state[page - base].load(Ordering::Acquire) == LAZY_RESYNC
                    {
                        return true;
                    }
                }
            }
        }
        let mut title = vec![0u8; PAGE];
        for (a, e) in busy {
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards.
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                MERGED[4].fetch_add(1, Ordering::Relaxed);
                return true;
            };
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_pages_in(base, &copied.at, a, e, &written)
                    .into_iter()
                    .filter(|&p| !has_deferred || !is_lazy_held(self.lazy_state[p - base].load(Ordering::Acquire)))
                    .collect()
            };
            for page in stale {
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                if lo >= hi {
                    continue;
                }
                let st_before = self.lazy_state[page - base].load(Ordering::Acquire);
                if is_lazy_held(st_before) {
                    continue;
                }
                if st_before == LAZY_RESYNC {
                    return true;
                }
                let n = hi - lo;
                let title = &mut title[..n];
                let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
                if !crate::host::read_into(lo as *const u8, title) {
                    continue;
                }
                // Re-check after `read_into`: if a concurrent VEH fault
                // transitioned this page to `LAZY_RESYNC`, settle in-flight
                // flights instead of misclassifying prior GPU bytes as CPU writes.
                if self.lazy_state[page - base].load(Ordering::Acquire) == LAZY_RESYNC {
                    return true;
                }
                MERGED[0].fetch_add(1, Ordering::Relaxed);
                let at = lo - start;
                if self.card.is_some() {
                    debug_assert!(!self.transient(), "no shader writes reach a transient window");
                    // SAFETY: inside the mapping of the copy's CPU side,
                    // which only the submitting thread writes.
                    let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let runs = changed_runs(title, held);
                    // (Neptune) As whole 32-bit words of title memory, joined
                    // where they touch (`under_writes_words`).
                    let runs = if under_writes_words() { widen_to_words(lo, n, runs) } else { runs };
                    if runs.len() > *runs_left {
                        MERGED[5].fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                    *runs_left -= runs.len();
                    for &(r, m) in &runs {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                    }
                    MERGED[1].fetch_add(runs.len() as u64, Ordering::Relaxed);
                    // (Neptune) One taking of the lock for the page's runs:
                    // there are hundreds to a page.
                    self.card_changed_runs(at, &runs);
                } else {
                    let Some(snapshot) = before.as_mut() else {
                        MERGED[6].fetch_add(1, Ordering::Relaxed);
                        return true;
                    };
                    // SAFETY: the snapshot's allocation holds the copy's
                    // bytes at the copy's offsets (`writing`), and these
                    // are in a declared range.
                    let held = unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                    for (r, m) in changed_runs(title, held) {
                        // SAFETY: inside the copy's mapping. The card may be
                        // writing the copy beside these bytes; they are
                        // written here and nothing around them.
                        unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        MERGED[1].fetch_add(1, Ordering::Relaxed);
                        MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                    }
                }
                drop(before);
                // Taken, if the range holds all of the page that is in the
                // copy; else the rest of it is the refresh's, which notes
                // only whole pages, and the page is looked at again.
                if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    copied.set(page - base, now);
                }
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        if !self.has_pending.load(Ordering::Acquire) {
            return self.update_free(from, to, &[]);
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for &(a, e) in &busy {
                if a > at {
                    bytes += self.update_free(at, a, &busy);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to, &busy);
            }
            return bytes;
        }
        self.update_free(from, to, &[])
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) -> usize {
        // Detect any overlapping OTHER shadow's deferred pages WITHOUT flushing
        // or unprotecting them: marking those pages stale (`copied = 0`) makes
        // `copy_stale` -> `copy` -> `host::copy_mapped` read their fresh GPU
        // bytes directly from the owning shadow via self-bypass while keeping
        // the owning shadow's pages GPU-resident (`PAGE_NOACCESS`, zero VEH
        // faults, zero `LAZY_RESYNC` ping-pong).
        let other_deferred = if LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0 {
            lazy_other_deferred_pages(self.id, from, to)
        } else {
            Vec::new()
        };
        if !other_deferred.is_empty() {
            let base = self.start as usize / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for &page in &other_deferred {
                if page >= base {
                    copied.set(page - base, 0);
                }
            }
        }
        if self.deferred_count.load(Ordering::Acquire) > 0 || self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_prepare_update(from, to, skip_busy);
        }
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                // Keep pages still deferred in another shadow marked `0` so
                // future GPU writes on that owning shadow remain visible to
                // this overlapping shadow on subsequent refreshes.
                if !other_deferred.is_empty() {
                    let base = self.start as usize / PAGE;
                    let mut c = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    for &page in &other_deferred {
                        if page >= base {
                            c.set(page - base, 0);
                        }
                    }
                }
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let bytes = if self.deferred_count.load(Ordering::Acquire) > 0 {
            self.copy_unwatched_skipping_deferred(from, to)
        } else {
            self.copy(from, to)
        };
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::WholeBytes, bytes as u64);
            if bytes > 0 {
                crate::copystat::whole(from, bytes, started.elapsed().as_micros() as u64);
            }
        }
        bytes
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")) {
            return;
        }
        // A transient window has no CPU side to look at; a rebar window's is write-combined
        // memory, which the CPU does not read.
        if self.data.is_null() || self.rebar {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let base = start / PAGE;
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let idx = (at / PAGE) - base;
            if has_deferred && is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire)) {
                at = (at / PAGE + 1) * PAGE;
                continue;
            }
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        let four_k = crate::watch::watch_4k();
        if !four_k {
            let runs = {
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                if has_deferred {
                    copied.plan_stale_filtered(base, start, end, from, to, now, written, |idx| {
                        is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire))
                    })
                } else {
                    copied.plan_stale(base, start, end, from, to, now, written)
                }
            };
            let mut bytes = 0;
            for (run_from, run_to) in runs.iter() {
                bytes += self.copy(run_from, run_to);
            }
            return bytes;
        }
        // (Neptune) VICTORY_WATCH_4K=1 sub-page path preserved with CopiedPages
        // bitset consistency and lazy-deferred page skipping.
        let bytes_runs = {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            let mut writes = written.iter().peekable();
            let (mut run, mut bytes_runs) = (None::<(usize, usize)>, Vec::new());
            let last = (to - 1) / PAGE;
            for page in from / PAGE..=last {
                let write = match writes.peek() {
                    Some(&&(p, w)) if p == page => {
                        writes.next();
                        w
                    }
                    _ => 0,
                };
                let idx = page - base;
                if has_deferred && is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire)) {
                    continue;
                }
                let prev = copied.get(idx).unwrap_or(0);
                if prev == 0 || write > prev {
                    let mask = crate::watch::stale_mask(page, prev);
                    crate::watch::note_4k(mask);
                    if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                        copied.set(idx, now);
                    }
                    extend_runs(&mut run, &mut bytes_runs, page, mask.unwrap_or(u16::MAX));
                }
            }
            bytes_runs.extend(run);
            bytes_runs
        };
        let mut bytes = 0;
        for (a, b) in bytes_runs {
            let (a, b) = (a.max(from), b.min(to));
            if a < b {
                bytes += self.copy(a, b);
            }
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // (Neptune) A transient window has no CPU side: the whole pages go into a chunk of
        // pooled upload memory, to be sent as they are (no byte compare, no snapshot), but for
        // the 4 KiB pages that hold what the window last sent with VICTORY_WINDOW_SKIP_SAME=1.
        if self.transient() {
            let n = self.copy_to_chunk(from, to);
            let left = to.saturating_sub(from).saturating_sub(n);
            for (count, bytes) in WINDOW_TAKEN.iter().zip([n, left]) {
                if bytes != 0 {
                    count.fetch_add(bytes as u64, Ordering::Relaxed);
                }
            }
            return n;
        }
        // (Neptune) A rebar window: the whole pages go straight into the card's memory. No byte
        // compare (it would read write-combined memory), no dirty ranges, no send.
        if self.rebar {
            if to > from {
                // SAFETY: inside the mapping (`from` and `to` are within the buffer).
                let at = unsafe { self.data.add(from - start) };
                crate::host::copy_mapped(at, from as *const u8, to - from, false);
                wc_sfence();
                REBAR[2].fetch_add((to - from) as u64, Ordering::Relaxed);
            }
            return to.saturating_sub(from);
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        // (Neptune) A copy in the card's memory takes only what differs (`copy_changed`).
        if self.card.is_some() && to > from && to - from <= 16 << 20 && copy_changed() {
            let n = to - from;
            // SAFETY: `n` bytes of the mapping, which this refresh alone writes (the ranges
            // shaders are writing were cut out by `update`), as the plain copy below does.
            let held = unsafe { std::slice::from_raw_parts_mut(at, n) };
            thread_local! {
                static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            TITLE.with(|title| {
                let mut title = title.borrow_mut();
                // What a read that fails leaves alone stays as the copy has it.
                title.clear();
                title.extend_from_slice(held);
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, n, false);
                let runs = changed_spans(&title, held);
                COPY_CHANGED[0].fetch_add(n as u64, Ordering::Relaxed);
                for &(r, m) in &runs {
                    held[r..r + m].copy_from_slice(&title[r..r + m]);
                    COPY_CHANGED[1].fetch_add(m as u64, Ordering::Relaxed);
                }
                if let (Some(first), Some(last)) = (runs.first(), runs.last()) {
                    self.changed(from - start + first.0, from - start + last.0 + last.1);
                    self.card_changed_runs(from - start, &runs);
                }
            });
            return n;
        }
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        to - from
    }

    /// [`Shadow::copy`] of a transient window: title memory `[from, to)` into a chunk, noted
    /// for the send. What a read that fails leaves alone is whatever the chunk held.
    /// (Neptune) With `VICTORY_WINDOW_SKIP_SAME=1` only the 4 KiB pages that do not hold what
    /// the window last sent for them ([`window_skip_same`]). Returns the bytes sent.
    fn copy_to_chunk(&self, from: usize, to: usize) -> usize {
        let Some(card) = &self.card else {
            return 0;
        };
        if from >= to {
            return 0;
        }
        let mode = window_skip_same();
        // A window shaders are said to write may hold their bytes: it compares nothing.
        let written = self.written_said.load(Ordering::Relaxed);
        if written && mode != SkipSame::Off {
            WINDOW_SAME[Sent::Unknown as usize].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
        if mode == SkipSame::Skip && !written {
            return self.copy_to_chunks_skipping(card, from, to);
        }
        let n = to - from;
        let Some(chunk) = crate::windowup::alloc(n as u64) else {
            crate::host::fail(format_args!("no upload memory for a window of title memory ({n} bytes)"));
        };
        crate::host::copy_mapped(chunk.ptr(), from as *const u8, n, false);
        if mode == SkipSame::Sample && !written {
            let mut tally = [0u64; 3];
            let window = (self.start as usize, (self.start + self.len) as usize);
            window_sample(window, (from, to), &mut self.window_prints(card), &mut tally, |at, into| {
                crate::host::read_into(at as *const u8, into)
            });
            window_same_add(&tally);
        }
        let start = self.start as usize;
        Self::chunk_sent(card, from - start, to - start, chunk);
        n
    }

    /// (Neptune) Notes `chunk` for the send of offsets `[a, b)` of a transient window.
    fn chunk_sent(card: &Card, a: usize, b: usize, chunk: crate::windowup::Chunk) {
        let mut chunks = card.chunks.lock().unwrap_or_else(|e| e.into_inner());
        // Pages sent again before a send replace the earlier ones (their chunks go back at once).
        chunks.retain(|c| !(a <= c.0 && c.1 <= b));
        chunks.push((a, b, chunk));
        drop(chunks);
        card.flag();
    }

    /// (Neptune) A transient window's prints ([`Card::prints`]), made at the first ask.
    fn window_prints<'a>(&self, card: &'a Card) -> std::sync::MutexGuard<'a, Vec<u64>> {
        let mut prints = card.prints.lock().unwrap_or_else(|e| e.into_inner());
        if prints.is_empty() {
            let n = prints_for(self.start as usize, (self.start + self.len) as usize);
            prints.resize(n, 0);
            WINDOW_PRINT_BYTES.fetch_add(n as u64 * 8, Ordering::Relaxed);
        }
        prints
    }

    /// (Neptune) [`Shadow::copy_to_chunk`] with `VICTORY_WINDOW_SKIP_SAME=1`: title memory
    /// `[from, to)` is read into memory of this thread's, and each run of 4 KiB pages whose
    /// bytes are not what the window last sent goes into a chunk of its own ([`window_copy`]).
    /// The prints stay locked until the chunks are noted: a print is always that of the last
    /// chunk noted for its page.
    fn copy_to_chunks_skipping(&self, card: &Card, from: usize, to: usize) -> usize {
        thread_local! {
            static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let start = self.start as usize;
        let mut prints = self.window_prints(card);
        let mut tally = [0u64; 3];
        let sent = READ.with(|read| {
            window_copy(
                (start, (self.start + self.len) as usize),
                (from, to),
                &mut prints,
                &mut read.borrow_mut(),
                &mut tally,
                |at, into| crate::host::copy_mapped(into.as_mut_ptr(), at as *const u8, into.len(), false),
                |at, bytes| {
                    let n = bytes.len();
                    let Some(chunk) = crate::windowup::alloc(n as u64) else {
                        crate::host::fail(format_args!("no upload memory for a window of title memory ({n} bytes)"));
                    };
                    // SAFETY: the chunk's mapping holds `n` bytes (`alloc`), and nothing else
                    // has the chunk yet.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), chunk.ptr(), n) };
                    Self::chunk_sent(card, at - start, at - start + n, chunk);
                },
            )
        });
        drop(prints);
        window_same_add(&tally);
        sent
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
            _ => dirty.push((from, to)),
        }
        card.has_dirty.store(true, Ordering::Release);
        drop(dirty);
        card.flag();
    }

    /// (Neptune) [`Shadow::card_changed`] for `runs` (offset, bytes; in
    /// order) of the stretch that begins at offset `at`.
    fn card_changed_runs(&self, at: usize, runs: &[(usize, usize)]) {
        let Some(card) = &self.card else {
            return;
        };
        let merged = coalesce_runs(runs, copy_coalesce_gap());
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        for &(r, m) in &merged {
            let (from, to) = (at + r, at + r + m);
            match dirty.last_mut() {
                Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
                _ => dirty.push((from, to)),
            }
        }
        if !dirty.is_empty() {
            card.has_dirty.store(true, Ordering::Release);
        }
        drop(dirty);
        card.flag();
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(Option<&ID3D12Resource>, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((card.staging.as_ref(), before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
        drop(seen);
        crate::host::walls_landed(crate::host::CARD_COPY, self.start as usize, self.len as usize, fetched, written, 0, None);
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || self.transient() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        if self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_apply_flushed_baseline(at, old);
        }
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            let as_read = merge_race_on().then(|| title.clone());
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            if merge_sync_card(title, gpu, old) {
                crate::watch::writing(start + at, n);
                merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card");
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                written = n;
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if self.transient() || at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        if self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_apply_flushed_baseline(at, held);
        }
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            dirty
                .iter()
                .copied()
                .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                .filter(|&(a, b)| a < at + n && b > at)
                .collect()
        };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = (mapped && merge_race_on()).then(|| title.clone());
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        // (Neptune) Our write is not the title's (`crate::watch::quiet_back`).
        if crate::watch::quiet_back() {
            // View memory that is guarded and clean: through a view of our own of the same
            // pages. No guard is opened for it, so none has to be set again, and the copy
            // stays clean for the refreshes.
            if crate::watch::guarded_clean(start + at, n)
                && let Some(alias) = crate::inplace::host_alias(start + at, n)
                && crate::host::write_alias(alias, title.as_ptr(), n)
            {
                crate::descriptor::host::remapped(start + at, n);
                let marks = crate::watch::ours_through(start + at, n);
                self.holds(&marks, &[]);
                QUIET_BACK_VIEWS.fetch_add(1, Ordering::Relaxed);
                return n;
            }
            // Write-watched memory: the kernel forgets our write. Only with
            // `VICTORY_QUIET_BACK_WATCHED=1`: the kernel's note of our write cannot be told
            // from one of the title's into the same 4 KiB page, the title rewrites pages
            // under shader writes all the time, and a write of its own that fell between
            // our asking and our reset never went up to the card (runs came up with lights
            // and colours missing). Without it our write counts as written and the copy
            // reads the page again and compares.
            if crate::watch::quiet_back_watched()
                && let Some(theirs) = crate::watch::ours_before(start + at, n)
            {
                crate::watch::writing(start + at, n);
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                let marks = crate::watch::ours_after(start + at, n);
                self.holds(&marks, &theirs);
                return n;
            }
        }
        crate::watch::writing(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }

    /// (Neptune) The copy-back wrote the journal pages of `marks` (page, the asking that had
    /// last found it written, this one) into title memory from this copy, which holds
    /// those bytes: they are not read again for it. Not the pages of `theirs`, which the
    /// title wrote too, nor one this copy had not taken the last write of.
    fn holds(&self, marks: &[(usize, u64, u64)], theirs: &[usize]) {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        for &(page, before, now) in marks {
            if theirs.contains(&page) {
                continue;
            }
            if let Some(i) = page.checked_sub(base)
                && let Some(held) = copied.get(i)
                && held != 0
                && before <= held
            {
                copied.set(i, now);
                QUIET_BACK.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
// (Neptune) Read by many at once: the title's recording threads look a copy up for every
// buffer view a draw names, and stood in line at one lock for it (a tenth of their working
// time in a sample).
static BUFFERS: std::sync::RwLock<BTreeMap<(u64, usize), Entry>> = std::sync::RwLock::new(BTreeMap::new());

fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    SHAPE.fetch_add(1, Ordering::Release);
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

/// (Neptune) Counts the takings of the two tables' locks to write (title buffers, windows):
/// what a thread found there before the count moved it need not look up again ([`lookup`]).
static SHAPE: AtomicU64 = AtomicU64::new(0);

/// The windows' table, to change it.
fn loose_write() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, u64), Arc<Shadow>>> {
    SHAPE.fetch_add(1, Ordering::Release);
    LOOSE.write().unwrap_or_else(|e| e.into_inner())
}

thread_local! {
    /// (Neptune) What this thread's look-ups found, while [`SHAPE`] stands: the count, and
    /// the copy by (address, length, any).
    static FOUND: std::cell::RefCell<(u64, std::collections::HashMap<(u64, u64, bool), Arc<Shadow>>)> =
        std::cell::RefCell::new((u64::MAX, std::collections::HashMap::new()));
}

/// (Neptune) Whether a thread remembers what its look-ups found
/// (`VICTORY_LOOKUP_CACHE=0`: every look-up goes to the tables, as before). A draw names the
/// same few buffers as the draws before it, and a look-up was a tenth of the recording
/// threads' working time.
fn lookup_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LOOKUP_CACHE").is_none_or(|v| v != "0"))
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    if !lookup_cache() {
        return lookup_tables(address, len, any);
    }
    let shape = SHAPE.load(Ordering::Acquire);
    let hit = FOUND.with(|found| {
        let found = found.borrow();
        (found.0 == shape).then(|| found.1.get(&(address, len, any)).cloned()).flatten()
    });
    if let Some(shadow) = hit {
        // Still in use (what lets idle copies go looks at this).
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow);
    }
    let found = lookup_tables(address, len, any);
    if let Some(shadow) = &found {
        // The count as it stands now: a copy made by this very look-up moved it.
        let shape = SHAPE.load(Ordering::Acquire);
        FOUND.with(|kept| {
            let mut kept = kept.borrow_mut();
            if kept.0 != shape || kept.1.len() >= 8192 {
                kept.1.clear();
                kept.0 = shape;
            }
            kept.1.insert((address, len, any), shadow.clone());
        });
    }
    found
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let removed = buffers().remove(&(start, resource)).and_then(|e| e.shadow);
    if let Some(shadow) = removed
        && shadow.deferred_count.load(Ordering::Acquire) > 0
    {
        // Keep `save_baseline = true` because in-flight lists may still hold
        // `Arc<Shadow>` and settle via `card_merge_relay` / `relay_merge`.
        shadow.lazy_flush_all(true);
    }
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = loose_write();
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = loose_write();
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

fn lookup_tables(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    {
        // (Neptune) The buffer's copy is there already, the usual case: found beside the
        // other readers.
        let map = BUFFERS.read().unwrap_or_else(|e| e.into_inner());
        match map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end) {
            Some((_, entry)) => {
                if let Some(shadow) = &entry.shadow {
                    return Some(shadow.clone());
                }
            }
            None => {
                drop(map);
                return loose(address, end, len <= 1, any);
            }
        }
    }
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && size >= card_limits().0;
        // Folded (`VICTORY_FOLD_BUFFERS=1`): a small buffer inside one
        // window cell takes the cell's window as its copy, so that the
        // buffer and the memory around it, which the title also reaches by
        // address, are one copy. Two copies of the same bytes made every
        // submission that refreshed one while a flight's shaders wrote the
        // other wait for the card to drain: ~2.8 times a frame in the
        // world, for 10-40 KB each (a 6.4 KB ring and a 465 KB buffer
        // inside 4 MiB windows). One copy's own pending writes are left
        // alone by its refresh and need no such wait.
        if !big && fold_buffers() && !fine() && key.0 & !(WINDOW - 1) == (entry.end - 1) & !(WINDOW - 1) {
            // (Neptune) Not into a transient window: shaders write a buffer, and such a window
            // has no CPU side to merge into (`window_upload`).
            let window = loose(key.0, entry.end, false, false).filter(|w| {
                let refuse = w.transient() || w.rebar;
                if w.rebar {
                    REBAR[4].fetch_add(1, Ordering::Relaxed);
                } else if refuse && TRANSIENT[1].fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::info!(start = format_args!("{:#x}", key.0), len, "a buffer is not folded into a transient window: it has a copy of its own");
                }
                !refuse
            });
            if let Some(window) = window {
                if FOLDED.fetch_add(1, Ordering::Relaxed) < 24 {
                    tracing::info!(
                        start = format_args!("{:#x}", key.0),
                        len,
                        "a buffer's copy is its window's (VICTORY_FOLD_BUFFERS)"
                    );
                }
                entry.shadow = Some(window);
                return entry.shadow.clone();
            }
        }
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        let (buffer, data) = host()?.shared_buffer(size)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the one just made is the CPU's side.
        let (buffer, card) = match card_for(size) {
            Some(on_card) => {
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging: Some(buffer),
                    chunks: Mutex::default(),
                    prints: Mutex::default(),
                    flagged: std::sync::atomic::AtomicBool::new(true),
                    me: std::sync::OnceLock::new(),
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    has_dirty: AtomicBool::new(false),
                    sent: Mutex::default(),
                    has_sent: AtomicBool::new(false),
                    size,
                };
                (on_card, Some(card))
            }
            None => (buffer, None),
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        let shadow = Arc::new(shadow);
        note_carded(&shadow);
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Whether small title buffers inside one window cell share the cell's
/// window as their copy (on by default; `VICTORY_FOLD_BUFFERS=0` gives
/// each its own copy as before). Measured in the world (4090, same load):
/// 179.2 -> 160.1 ms a frame, settles 4.5 -> 1.0 a frame.
fn fold_buffers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FOLD_BUFFERS").is_none_or(|v| v != "0"))
}

/// Buffers folded into windows so far ([`fold_buffers`]).
pub(crate) static FOLDED: AtomicU64 = AtomicU64::new(0);

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = loose_write();
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: std::sync::RwLock<BTreeMap<(u64, u64), Arc<Shadow>>> = std::sync::RwLock::new(BTreeMap::new());

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    {
        // (Neptune) The window is there already, the usual case: found and noted used beside
        // the other readers (what lets windows go takes the lock alone).
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let (cell, _) = window_of(address, end, WINDOW);
        let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
        let (start, len) = if unsized_ask && tails() {
            let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
            longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
        } else if any && fine() && !unsized_ask {
            let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
            holder(there, address, end).unwrap_or(named)
        } else {
            named
        };
        if let Some(shadow) = map.get(&(start, len)) {
            shadow.used.store(crate::host::moment(), Ordering::Relaxed);
            shadow.asked[0].fetch_min(address, Ordering::Relaxed);
            shadow.asked[1].fetch_max(end, Ordering::Relaxed);
            if unsized_ask {
                shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
            }
            return Some(shadow.clone());
        }
    }
    let mut map = loose_write();
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    let (start, len) = if unsized_ask && tails() {
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
    } else if any && fine() && !unsized_ask {
        // The windows that begin in the ask's cell at or below the ask (one
        // that begins in a cell below and reaches this far is passed over:
        // the ask then makes its own).
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        holder(there, address, end).unwrap_or(named)
    } else {
        named
    };
    // Under the lock, with the handing out: see `let_go_idle`.
    let note = |shadow: &Shadow| {
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow);
        return Some(shadow.clone());
    }
    // In place, if asked for and the window's memory can be opened: the
    // window is the title's own memory, with nothing to copy.
    if let Some(placed) = crate::inplace::place_for(start, len, true) {
        let shadow = Arc::new(Shadow::placed(placed, start, len, true));
        note(&shadow);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    // (Neptune) With `window_upload`, a window that gets a card side has no CPU side
    // (a transient window); one that gets none is made as before.
    // (Neptune) With `window_rebar`: one copy in a GPU upload heap, before either.
    let rebar = if window_rebar() { window_rebar_for(len) } else { None };
    let rebar_made = rebar.is_some();
    let transient = if window_upload() && rebar.is_none() { window_card_for(len) } else { None };
    let (buffer, data, card) = if let Some((buffer, data)) = rebar {
        crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} in a GPU upload heap"));
        (buffer, data, None)
    } else if let Some(on_card) = transient {
        crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card (transient)"));
        let card = Card {
            staging: None,
            chunks: Mutex::default(),
            prints: Mutex::default(),
            flagged: std::sync::atomic::AtomicBool::new(true),
            me: std::sync::OnceLock::new(),
            before: Mutex::default(),
            dirty: Mutex::default(),
            has_dirty: AtomicBool::new(false),
            sent: Mutex::default(),
            has_sent: AtomicBool::new(false),
            size: len,
        };
        (on_card, std::ptr::null_mut(), Some(card))
    } else {
        let (buffer, data) = host()?.shared_buffer(len)?;
        crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
        // With room for windows on the card (`window_card_most`): shaders get a
        // buffer there, and the one just made is the CPU's side.
        match window_card_for(len) {
            Some(on_card) => {
                crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
                let card = Card {
                    staging: Some(buffer),
                    chunks: Mutex::default(),
                    prints: Mutex::default(),
                    flagged: std::sync::atomic::AtomicBool::new(true),
                    me: std::sync::OnceLock::new(),
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    has_dirty: AtomicBool::new(false),
                    sent: Mutex::default(),
                    has_sent: AtomicBool::new(false),
                    size: len,
                };
                (on_card, data, Some(card))
            }
            None => (buffer, data, None),
        }
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.card = card;
    shadow.rebar = rebar_made;
    if shadow.rebar {
        // No CPU side: not counted with the copies' bytes in the PC's memory.
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        REBAR[0].fetch_add(1, Ordering::Relaxed);
        REBAR[1].fetch_add(len, Ordering::Relaxed);
    }
    if shadow.transient() {
        // No CPU side: not counted with the copies' bytes in the PC's memory.
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        TRANSIENT[0].fetch_add(1, Ordering::Relaxed);
    }
    let shadow = Arc::new(shadow);
    note_carded(&shadow);
    note(&shadow);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `own_bytes`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if len == 0 {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, address, len));
    map.insert((address, len), (shadow.clone(), now));
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let kept = EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.len();
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_OWN_COPY=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept",
        asked as f64 / frames,
        made as f64 / frames
    );
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, cover, grid, holder, longest_tail, tail_of, window_of};

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }
}

#[cfg(test)]
mod card_merge_tests {
    use super::{Card, Shadow};
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;

    /// A plain buffer on the software adapter: what a copy's `buffer` and
    /// its card's `staging` are; the merge never touches them.
    fn warp_buffer() -> Option<ID3D12Resource> {
        // SAFETY: plain DXGI and D3D12 creation calls on the software adapter.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).ok()?;
            let device = device?;
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 64 << 10,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .ok()?;
            buffer
        }
    }

    /// The mixed landing the fetch inside flights must survive, on a real
    /// copy in the card's memory and its merge (`Shadow::card_merge_relay`):
    /// flights 5, 6, 7 in the air; 5 lands the old way with the queue's
    /// count at 7; the title wrote a byte that went up in send 7 into a
    /// piece flight 6's shaders write; 6 then lands from its own slot, which
    /// holds that byte as it was before send 7. The notes of sends are
    /// dropped as `card_dirty` drops them, below the mark the old-way fetch
    /// sets (`crate::host::fallback_fetched_mark`): with the mark at the
    /// landed plans' send the title's byte survives and the shader's write
    /// lands; with the mark at the fetch's count (as before the fix) the
    /// title's byte is put back to the card's older one.
    #[test]
    fn a_title_byte_sent_after_a_flight_survives_its_slot() {
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let (title_byte, shader_byte) = (100usize, 200usize);
        let content: Vec<u8> = (0..LEN).map(|i| (i * 7 + 3) as u8).collect();
        let land = |mark: u64| -> Vec<u8> {
            let mut title = content.clone();
            let mut side = content.clone();
            // The title's byte, refreshed into the CPU's side and sent up in send 7.
            title[title_byte] ^= 0x5a;
            side[title_byte] = title[title_byte];
            let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
            shadow.card = Some(Card {
                staging: Some(staging.clone()),
                chunks: Mutex::default(),
                prints: Mutex::default(),
                flagged: std::sync::atomic::AtomicBool::new(false),
                me: std::sync::OnceLock::new(),
                before: Mutex::new(None),
                dirty: Mutex::default(),
                has_dirty: std::sync::atomic::AtomicBool::new(false),
                sent: Mutex::new(vec![(title_byte, title_byte + 1, 7)]),
                has_sent: std::sync::atomic::AtomicBool::new(true),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
            });
            // As `card_dirty` with the mark set by the old-way fetch.
            if let Some(card) = &shadow.card {
                card.sent.lock().unwrap().retain(|s| s.2 > mark);
            }
            // Flight 6's slot: the card's bytes behind its lists, before send 7.
            let mut slot = content.clone();
            slot[shader_byte] ^= 0xc3;
            let written = shadow.card_merge_relay(0, &slot, 6, &mut Vec::new());
            assert!(written > 0);
            drop(shadow);
            title
        };
        let fixed = land(crate::host::fallback_fetched_mark(true, 7, Some(5)));
        assert_eq!(fixed[title_byte], content[title_byte] ^ 0x5a, "the title's byte was lost");
        assert_eq!(fixed[shader_byte], content[shader_byte] ^ 0xc3, "the shader's write did not land");
        // The mark at the fetch's count: the note of send 7 is gone and the
        // older card byte is taken for a shader's write.
        let before = land(7);
        assert_eq!(before[title_byte], content[title_byte]);
    }

    /// `card_dirty` takes the copies flagged since its last call, from the
    /// list they went into when flagged ([`super::FLAGGED`]), each once, and
    /// looks at no other copy in the card's memory. (The only test that
    /// calls it: the list is the process's.)
    #[test]
    fn card_dirty_takes_only_the_copies_flagged_since() {
        use super::{card_dirty, note_carded};
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let title = vec![0u8; LEN];
        let (mut side_a, mut side_b, mut side_c) = (vec![0u8; LEN], vec![0u8; LEN], vec![0u8; LEN]);
        let make = |side: &mut Vec<u8>, flagged: bool| {
            let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
            shadow.card = Some(Card {
                staging: Some(staging.clone()),
                chunks: Mutex::default(),
                prints: Mutex::default(),
                flagged: std::sync::atomic::AtomicBool::new(flagged),
                me: std::sync::OnceLock::new(),
                before: Mutex::new(None),
                dirty: Mutex::default(),
                has_dirty: std::sync::atomic::AtomicBool::new(false),
                sent: Mutex::default(),
                has_sent: std::sync::atomic::AtomicBool::new(false),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
            });
            let shadow = Arc::new(shadow);
            note_carded(&shadow);
            shadow
        };
        let taken = |out: &[(Arc<Shadow>, Vec<(usize, usize)>)]| -> Vec<(*const Shadow, Vec<(usize, usize)>)> {
            out.iter().map(|(s, r)| (Arc::as_ptr(s), r.clone())).collect()
        };
        // Made flagged, as copies are: its first call takes it (nothing in
        // `dirty` yet, so nothing to send).
        let a = make(&mut side_a, true);
        let b = make(&mut side_b, false);
        let c = make(&mut side_c, false);
        let (out, uploads) = card_dirty(1);
        assert!(out.is_empty() && uploads.is_empty());
        assert!(!a.card.as_ref().unwrap().flagged.load(Ordering::Acquire), "the copy made flagged was not taken");
        // b changes. c holds a range and the flag without having been put
        // in the list: only the listed copies are looked at.
        b.card_changed(0, 16);
        if let Some(card) = &c.card {
            card.dirty.lock().unwrap().push((0, 8));
            card.flagged.store(true, Ordering::Release);
        }
        let (out, _) = card_dirty(2);
        assert_eq!(taken(&out), vec![(Arc::as_ptr(&b), vec![(0, 16)])]);
        drop(out);
        // Nothing changed since: nothing taken.
        assert!(card_dirty(3).0.is_empty());
        // Changed twice before the next call: taken once, both ranges.
        b.card_changed(16, 32);
        b.card_changed(64, 80);
        let (out, _) = card_dirty(4);
        assert_eq!(taken(&out), vec![(Arc::as_ptr(&b), vec![(16, 32), (64, 80)])]);
        drop(out);
        // A copy let go before the call is passed over (the list holds it
        // weakly).
        b.card_changed(0, 4);
        drop(b);
        assert!(card_dirty(5).0.is_empty());
        drop((a, c));
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        loose_write()
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod run_tests {
    use super::{PAGE, extend_runs};

    #[test]
    fn masks_become_merged_byte_runs() {
        let (mut run, mut done) = (None, Vec::new());
        extend_runs(&mut run, &mut done, 1, 0b1000_0000_0000_0110); // sub-pages 1, 2 and 15
        extend_runs(&mut run, &mut done, 2, 0b11); // 15 of page 1 joins 0 and 1 of page 2
        extend_runs(&mut run, &mut done, 4, u16::MAX);
        done.extend(run);
        let p = PAGE;
        assert_eq!(done, [(p + 4096, p + 3 * 4096), (p + 15 * 4096, 2 * p + 2 * 4096), (4 * p, 5 * p)]);
    }
}

// ============================================================================
// Opt-in Lazy GPU-Resident Write-Back (`VICTORY_LAZY_COPYBACK=1`, default OFF)
// ============================================================================
//
// Strict Acyclic Lock Hierarchy (deadlock-free with Windows VEH):
//   Level 0: `host.submitted`, `CARD_WORK`, `BUFFERS`, `LOOSE`, `EXACT`,
//            `CARDED`, `Shadow::{copied, pending, before, kept, card.*}`,
//            `watch::{JOURNAL, MAPPING, GUARDS}`.
//   Level 1: `LAZY_TABLE` (metadata-only leaf table: armed/busy pages, pinned
//            fence pages, active remap barriers, and flushed 64 KiB baselines).
//            NEVER acquires Level 0 or Level 2 while held, and NEVER calls
//            D3D12 or waits on GPU work.
//   Level 2: `LAZY_GPU` (dedicated D3D12 allocator, command list, fence, event,
//            and persistent 64 KiB `D3D12_HEAP_TYPE_READBACK` buffer). NEVER
//            acquires Level 0 or Level 1 while held.
//   Level 3: `FlushWait` (`Mutex<Option<FlushOutcome>>` + `Condvar`). Leaf only.
//
// VEH (`lazy_on_fault`) ONLY acquires Level 1 -> drops -> Level 2 -> drops ->
// `VirtualProtect` + `memcpy` -> Level 1 -> drops -> Level 3. It NEVER touches
// any Level 0 lock (`watch`, `Shadow`, `CARD_WORK`, `host.submitted`).

/// Whether a `VICTORY_BACK_READS` environment value enables copy-back read
/// diagnostics (`"1"`, `"2"`, `"alt"`, or any non-empty value other than `"0"`).
/// Whether opt-in Lazy GPU-Resident Write-Back is enabled
/// (`VICTORY_LAZY_COPYBACK=1`; default OFF).
pub(crate) fn lazy_copyback() -> bool {
    #[cfg(test)]
    {
        match LAZY_TEST_OVERRIDE.load(Ordering::Acquire) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_COPYBACK").is_some_and(|v| v == "1"))
}

#[cfg(test)]
static LAZY_TEST_OVERRIDE: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
type PostTransitionHook = Box<dyn FnOnce(usize) + Send>;

#[cfg(test)]
static TRY_DEFER_POST_TRANSITION_HOOK: Mutex<Option<PostTransitionHook>> = Mutex::new(None);

#[cfg(test)]
static LAZY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) struct LazyTestGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prev: u8,
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn test_disable_lazy_copyback() -> LazyTestGuard {
    LazyTestGuard::disable()
}

#[cfg(test)]
impl LazyTestGuard {
    fn enable() -> Self {
        let lock = LAZY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = LAZY_TEST_OVERRIDE.swap(2, Ordering::AcqRel);
        Self { _lock: lock, prev }
    }

    fn disable() -> Self {
        let lock = LAZY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = LAZY_TEST_OVERRIDE.swap(1, Ordering::AcqRel);
        Self { _lock: lock, prev }
    }
}

#[cfg(test)]
impl Drop for LazyTestGuard {
    fn drop(&mut self) {
        LAZY_TEST_OVERRIDE.store(self.prev, Ordering::Release);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlushOutcome {
    /// Page has been flushed/unprotected (`PAGE_READWRITE`) and is accessible.
    Accessible,
    /// Arming committed (`PAGE_NOACCESS`): waiting faults must retry lookup.
    ArmedRetry,
}

struct FlushWait {
    state: Mutex<Option<FlushOutcome>>,
    ready: Condvar,
}

impl FlushWait {
    fn new() -> Self {
        Self {
            state: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn wait(&self) -> FlushOutcome {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(outcome) = *guard {
                return outcome;
            }
            guard = self.ready.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn finish(&self, outcome: FlushOutcome) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(outcome);
        self.ready.notify_all();
    }
}

struct ArmedPage {
    shadow: Weak<Shadow>,
    shadow_id: u64,
    offset: usize,
    page_idx: usize,
    /// When the armed page is a single-view `MEM_MAPPED` section (`inplace::Cover::Run`),
    /// stores `(section_handle, section_offset, section_bytes)` so VEH / flush can
    /// write all 64 KiB through our persistent writable `inplace::OWN` alias view
    /// BEFORE flipping the title view from `PAGE_NOACCESS` to `PAGE_READWRITE` —
    /// zero torn-read window and zero per-page `MapViewOfFile`/`UnmapViewOfFile`.
    section_view: Option<(usize, u64, u64)>,
    /// `true` iff this page was armed via Path A (`MEM_MAPPED` clean `watch` view),
    /// `false` for Path B (`MEM_PRIVATE | MEM_WRITE_WATCH`). Used by `watch::known_view`
    /// and `watch::guard` to keep steady-state `MEM_MAPPED` refreshes lock-free without
    /// ever misclassifying `MEM_PRIVATE` pages as view memory.
    is_mapped_view: bool,
}

enum LazyEntry {
    Armed(ArmedPage),
    Busy {
        shadow_id: u64,
        wait: Arc<FlushWait>,
    },
}

/// Per-page baseline flushed from GPU into title memory, paired with the
/// unapplied byte intervals inside `[0, PAGE)` so `lazy_apply_flushed_baseline`
/// applies each byte at most once (idempotent across consecutive relay merges
/// or overlapping ranges) while keeping `LAZY_RESYNC` active until the next
/// `Shadow::update` / `Shadow::writing`.
struct FlushedBaseline {
    bytes: Vec<u8>,
    remaining: Vec<(usize, usize)>,
}

impl FlushedBaseline {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            remaining: vec![(0, PAGE)],
        }
    }

    /// Copies any unapplied subranges intersecting `[src_lo, src_hi)` into `dst`
    /// (where `dst[0]` corresponds to `src_lo`), removing them from `remaining`.
    /// Returns `true` when every byte in `[0, PAGE)` has been applied.
    fn apply_once(&mut self, src_lo: usize, src_hi: usize, dst: &mut [u8]) -> bool {
        let mut next = Vec::with_capacity(self.remaining.len() + 1);
        for &(r_lo, r_hi) in &self.remaining {
            if r_hi <= src_lo || r_lo >= src_hi {
                next.push((r_lo, r_hi));
                continue;
            }
            if r_lo < src_lo {
                next.push((r_lo, src_lo));
            }
            let o_lo = r_lo.max(src_lo);
            let o_hi = r_hi.min(src_hi);
            let dst_off = o_lo - src_lo;
            let n = o_hi - o_lo;
            dst[dst_off..dst_off + n].copy_from_slice(&self.bytes[o_lo..o_hi]);
            if r_hi > src_hi {
                next.push((src_hi, r_hi));
            }
        }
        self.remaining = next;
        self.remaining.is_empty()
    }
}

struct LazyTable {
    pages: BTreeMap<usize, LazyEntry>,
    pinned: BTreeSet<usize>,
    /// Active `watch::title_memory_remapping` / `lazy_flush_and_hold` page
    /// intervals `(id, first_page, last_page)`.
    remapping: Vec<(u64, usize, usize)>,
    next_remap_id: u64,
    flushed_baselines: BTreeMap<(u64, usize), FlushedBaseline>,
}

impl LazyTable {
    const fn new() -> Self {
        Self {
            pages: BTreeMap::new(),
            pinned: BTreeSet::new(),
            remapping: Vec::new(),
            next_remap_id: 1,
            flushed_baselines: BTreeMap::new(),
        }
    }

    fn is_remapping(&self, page: usize) -> bool {
        self.remapping.iter().any(|&(_, lo, hi)| lo <= page && page < hi)
    }
}

/// Level 1 metadata table for lazy GPU-resident pages.
static LAZY_TABLE: Mutex<LazyTable> = Mutex::new(LazyTable::new());

/// Lock-free count of currently armed or busy lazy pages (0 when inactive).
static LAZY_ARMED_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Number of lock-free hash stripes for O(1) contention-free deferred-page
/// queries and Dekker-synchronized host access holds.
const LAZY_STRIPES: usize = 1024;
const LAZY_STRIPE_MASK: usize = LAZY_STRIPES - 1;

/// Exact count of `LAZY_TABLE.pages` entries hashing to `page & LAZY_STRIPE_MASK`.
/// Maintained under `LAZY_TABLE` via [`lazy_table_insert_busy`] and
/// [`lazy_table_remove_page`].
static LAZY_ARMED_STRIPES: [std::sync::atomic::AtomicU32; LAZY_STRIPES] =
    [const { std::sync::atomic::AtomicU32::new(0) }; LAZY_STRIPES];

/// Lock-free host access hold counters (per-stripe for ranges `<= 8` pages,
/// global for larger ranges). Pairs with `LAZY_ARMED_STRIPES` via `SeqCst`
/// Dekker ordering so `host::read_into`/`copy_mapped`/`write_mapped`/`copy_fill`
/// never acquire `LAZY_TABLE` on the common non-deferred path while still
/// strictly blocking concurrent relay `try_defer_piece_relay` arming.
static LAZY_HOLD_STRIPES: [std::sync::atomic::AtomicU32; LAZY_STRIPES] =
    [const { std::sync::atomic::AtomicU32::new(0) }; LAZY_STRIPES];
static LAZY_HOLD_GLOBAL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Monotonic count of completed lazy flushes (for stale-fault disambiguation).
static LAZY_FLUSH_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Monotonic counters: `[0]=newly_armed, [1]=steady_state_hits,
/// [2]=veh_read_faults, [3]=veh_write_faults, [4]=explicit_flushes,
/// [5]=bytes_flushed`.
static LAZY_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Additional monotonic diagnostic totals: `[0]=self_bypass_hits,
/// [1]=materialize_us, [2]=protect_us, [3]=protect_calls`.
static LAZY_EXTRA_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Nanosecond accumulator so sub-microsecond `VirtualProtect` calls are never
/// truncated away from `lazy_protect_ms`.
static LAZY_PROTECT_NS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Per-frame-window counters swapped in [`take_lazy_interval`]:
/// `[0]=lazy_deferred_pages, [1]=lazy_veh_faults, [2]=lazy_self_bypass_hits,
/// [3]=lazy_bytes_readback, [4]=lazy_materialize_us, [5]=lazy_protect_us`.
static LAZY_METER_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Per-frame lazy copy-back diagnostic figures for `frametime::Row` and logging.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LazyTelemetryInterval {
    pub deferred_pages: f64,
    pub veh_faults: f64,
    pub self_bypass_hits: f64,
    pub bytes_readback: f64,
    pub materialize_ms: f64,
    pub protect_ms: f64,
}

fn lazy_record_deferred(pages: u64) {
    if pages == 0 {
        return;
    }
    LAZY_METER_STATS[0].fetch_add(pages, Ordering::Relaxed);
}

fn lazy_record_veh_fault(is_write: bool) {
    LAZY_STATS[if is_write { 3 } else { 2 }].fetch_add(1, Ordering::Relaxed);
    LAZY_METER_STATS[1].fetch_add(1, Ordering::Relaxed);
}

fn lazy_record_self_bypass(pages: u64) {
    if pages == 0 {
        return;
    }
    LAZY_EXTRA_STATS[0].fetch_add(pages, Ordering::Relaxed);
    LAZY_METER_STATS[2].fetch_add(pages, Ordering::Relaxed);
}

fn lazy_record_bytes_readback(bytes: u64) {
    if bytes == 0 {
        return;
    }
    LAZY_STATS[5].fetch_add(bytes, Ordering::Relaxed);
    LAZY_METER_STATS[3].fetch_add(bytes, Ordering::Relaxed);
}

fn lazy_record_materialize_us(us: u64) {
    if us == 0 {
        return;
    }
    LAZY_EXTRA_STATS[1].fetch_add(us, Ordering::Relaxed);
    LAZY_METER_STATS[4].fetch_add(us, Ordering::Relaxed);
}

/// Unconditionally records one actual `VirtualProtect` syscall and accumulates
/// its duration in nanoseconds (flushing whole microseconds without sub-us loss).
pub(crate) fn lazy_record_protect_ns(ns: u64) {
    LAZY_EXTRA_STATS[3].fetch_add(1, Ordering::Relaxed);
    let prev = LAZY_PROTECT_NS_TOTAL.fetch_add(ns, Ordering::Relaxed);
    let us_delta = (prev.saturating_add(ns) / 1_000).saturating_sub(prev / 1_000);
    if us_delta > 0 {
        LAZY_EXTRA_STATS[2].fetch_add(us_delta, Ordering::Relaxed);
        LAZY_METER_STATS[5].fetch_add(us_delta, Ordering::Relaxed);
    }
}

/// Takes and resets per-frame lazy copy-back diagnostic figures over `frames`.
pub(crate) fn take_lazy_interval(frames: f64) -> LazyTelemetryInterval {
    let f = frames.max(1.0);
    let s: [u64; 6] = std::array::from_fn(|i| LAZY_METER_STATS[i].swap(0, Ordering::Relaxed));
    LazyTelemetryInterval {
        deferred_pages: s[0] as f64 / f,
        veh_faults: s[1] as f64 / f,
        self_bypass_hits: s[2] as f64 / f,
        bytes_readback: s[3] as f64 / f,
        materialize_ms: s[4] as f64 / 1e3 / f,
        protect_ms: s[5] as f64 / 1e3 / f,
    }
}

/// Logs per-frame lazy copy-back diagnostics (`Meter::frame`) whenever
/// `VICTORY_LAZY_COPYBACK=1` is active or any lazy work occurred in the window.
pub(crate) fn lazy_line(_frames: f64, iv: &LazyTelemetryInterval) {
    let resident = LAZY_ARMED_COUNT.load(Ordering::Relaxed);
    let any = iv.deferred_pages > 0.0
        || iv.veh_faults > 0.0
        || iv.self_bypass_hits > 0.0
        || iv.bytes_readback > 0.0
        || iv.materialize_ms > 0.0
        || iv.protect_ms > 0.0;
    if !lazy_copyback() && resident == 0 && !any {
        return;
    }
    tracing::info!(
        "lazy copy-back per frame: lazy_deferred_pages={:.0}, lazy_veh_faults={:.1}, \
         lazy_self_bypass_hits={:.1}, lazy_bytes_readback={:.0}, lazy_materialize_ms={:.2}, \
         lazy_protect_ms={:.2} (resident_pages={resident})",
        iv.deferred_pages,
        iv.veh_faults,
        iv.self_bypass_hits,
        iv.bytes_readback,
        iv.materialize_ms,
        iv.protect_ms,
    );
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct LazyTelemetrySnapshot {
    deferred_pages: u64,
    veh_faults: u64,
    self_bypass_hits: u64,
    materialize_us: u64,
    protect_us: u64,
    protect_calls: u64,
    bytes_flushed: u64,
}

#[cfg(test)]
fn lazy_telemetry_snapshot() -> LazyTelemetrySnapshot {
    LazyTelemetrySnapshot {
        deferred_pages: LAZY_STATS[0].load(Ordering::Relaxed) + LAZY_STATS[1].load(Ordering::Relaxed),
        veh_faults: LAZY_STATS[2].load(Ordering::Relaxed) + LAZY_STATS[3].load(Ordering::Relaxed),
        self_bypass_hits: LAZY_EXTRA_STATS[0].load(Ordering::Relaxed),
        materialize_us: LAZY_EXTRA_STATS[1].load(Ordering::Relaxed),
        protect_us: LAZY_EXTRA_STATS[2].load(Ordering::Relaxed),
        protect_calls: LAZY_EXTRA_STATS[3].load(Ordering::Relaxed),
        bytes_flushed: LAZY_STATS[5].load(Ordering::Relaxed),
    }
}

/// Lock-free 2-level bitset of currently armed `MEM_MAPPED` (`is_mapped_view`)
/// 64 KiB pages for O(1) contention-free `watch::known_view` and `watch::guard`
/// steady-state checks (47-bit Windows user VA = `2^31` 64 KiB pages = `2^15`
/// parts of `1024 * 64` bits each; zero heap allocation until a `MEM_MAPPED`
/// page is armed).
const LAZY_MAPPED_PART_WORDS: usize = 1024;
const LAZY_MAPPED_PART_PAGES: usize = LAZY_MAPPED_PART_WORDS * 64;
struct LazyMappedPart([AtomicU64; LAZY_MAPPED_PART_WORDS]);
static LAZY_MAPPED_VIEW: [std::sync::atomic::AtomicPtr<LazyMappedPart>; 1 << 15] =
    [const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; 1 << 15];
static LAZY_MAPPED_COUNT: AtomicUsize = AtomicUsize::new(0);

fn lazy_mapped_part(page: usize, make: bool) -> Option<&'static LazyMappedPart> {
    let slot = LAZY_MAPPED_VIEW.get(page / LAZY_MAPPED_PART_PAGES)?;
    let mut ptr = slot.load(Ordering::Acquire);
    if ptr.is_null() {
        if !make {
            return None;
        }
        let layout = std::alloc::Layout::new::<LazyMappedPart>();
        // SAFETY: non-zero layout; zeroed memory is a valid `[AtomicU64; 1024]`.
        let fresh = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<LazyMappedPart>();
        if fresh.is_null() {
            return None;
        }
        match slot.compare_exchange(std::ptr::null_mut(), fresh, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => ptr = fresh,
            Err(other) => {
                // SAFETY: ours alone, just allocated with `layout`.
                unsafe { std::alloc::dealloc(fresh.cast(), layout) };
                ptr = other;
            }
        }
    }
    // SAFETY: parts live for `'static` once published.
    Some(unsafe { &*ptr })
}

fn lazy_mapped_view_set(page: usize, on: bool) {
    let Some(part) = lazy_mapped_part(page, on) else {
        return;
    };
    let word = (page % LAZY_MAPPED_PART_PAGES) / 64;
    let bit = 1u64 << (page & 63);
    if on {
        let prev = part.0[word].fetch_or(bit, Ordering::Release);
        if prev & bit == 0 {
            LAZY_MAPPED_COUNT.fetch_add(1, Ordering::Release);
        }
    } else {
        let prev = part.0[word].fetch_and(!bit, Ordering::Release);
        if prev & bit != 0 {
            LAZY_MAPPED_COUNT.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Lock-free check: whether any `MEM_MAPPED` lazy page is currently armed.
#[inline]
pub(crate) fn has_lazy_mapped_view_pages() -> bool {
    LAZY_MAPPED_COUNT.load(Ordering::Acquire) > 0
}

/// Lock-free O(1) check: whether 64 KiB `page` is currently armed as a
/// `MEM_MAPPED` lazy view page (always `false` for `MEM_PRIVATE` pages).
#[inline]
pub(crate) fn is_lazy_mapped_view_page(page: usize) -> bool {
    if LAZY_MAPPED_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let Some(part) = lazy_mapped_part(page, false) else {
        return false;
    };
    let word = (page % LAZY_MAPPED_PART_PAGES) / 64;
    let bit = 1u64 << (page & 63);
    part.0[word].load(Ordering::Acquire) & bit != 0
}

fn lazy_table_insert_busy(table: &mut LazyTable, page: usize, shadow_id: u64, wait: Arc<FlushWait>) {
    let prev = table.pages.insert(page, LazyEntry::Busy { shadow_id, wait });
    match prev {
        None => {
            // Both `LAZY_ARMED_COUNT` and `LAZY_ARMED_STRIPES` use `SeqCst` so the
            // Dekker pair with `LAZY_HOLD_GLOBAL`/`LAZY_HOLD_STRIPES` (`SeqCst`) is
            // strictly in the global `SeqCst` total order for all range sizes.
            LAZY_ARMED_COUNT.fetch_add(1, Ordering::SeqCst);
            LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].fetch_add(1, Ordering::SeqCst);
        }
        Some(LazyEntry::Armed(a)) if a.is_mapped_view => {
            lazy_mapped_view_set(page, false);
        }
        Some(_) => {}
    }
}

fn lazy_table_insert_armed(table: &mut LazyTable, page: usize, armed: ArmedPage) {
    let is_mapped = armed.is_mapped_view;
    let prev = table.pages.insert(page, LazyEntry::Armed(armed));
    if prev.is_none() {
        LAZY_ARMED_COUNT.fetch_add(1, Ordering::SeqCst);
        LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].fetch_add(1, Ordering::SeqCst);
    }
    if is_mapped {
        lazy_mapped_view_set(page, true);
    } else if matches!(prev, Some(LazyEntry::Armed(a)) if a.is_mapped_view) {
        lazy_mapped_view_set(page, false);
    }
}

fn lazy_table_remove_page(table: &mut LazyTable, page: usize) -> Option<LazyEntry> {
    let prev = table.pages.remove(&page);
    if let Some(ref entry) = prev {
        if matches!(entry, LazyEntry::Armed(a) if a.is_mapped_view) {
            lazy_mapped_view_set(page, false);
        }
        LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].fetch_sub(1, Ordering::Release);
        LAZY_ARMED_COUNT.fetch_sub(1, Ordering::AcqRel);
    }
    prev
}

#[inline]
fn lazy_is_held(page: usize) -> bool {
    LAZY_HOLD_GLOBAL.load(Ordering::SeqCst) > 0
        || LAZY_HOLD_STRIPES[page & LAZY_STRIPE_MASK].load(Ordering::SeqCst) > 0
}

#[inline]
fn lazy_range_may_have_armed(first: usize, last: usize) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::SeqCst) == 0 || first >= last {
        return false;
    }
    if last - first <= 16 {
        (first..last).any(|p| LAZY_ARMED_STRIPES[p & LAZY_STRIPE_MASK].load(Ordering::SeqCst) > 0)
    } else {
        true
    }
}

enum HoldKind {
    None,
    Small(usize, usize),
    Global,
}

/// Zero-allocation, lock-free host access hold guard preventing concurrent
/// relay `try_defer_piece_relay` from arming `PAGE_NOACCESS` on pages covered
/// by `[at, at + len)`.
pub struct LazyAccessHold {
    kind: HoldKind,
}

impl Drop for LazyAccessHold {
    fn drop(&mut self) {
        match self.kind {
            HoldKind::None => {}
            HoldKind::Small(first, last) => {
                for p in first..last {
                    LAZY_HOLD_STRIPES[p & LAZY_STRIPE_MASK].fetch_sub(1, Ordering::Release);
                }
            }
            HoldKind::Global => {
                LAZY_HOLD_GLOBAL.fetch_sub(1, Ordering::Release);
            }
        }
    }
}

/// Acquires a lock-free host access hold for `[at, at + len)`. When
/// `VICTORY_LAZY_COPYBACK` is OFF, executes zero atomics and zero locks.
pub(crate) fn lazy_hold_access(at: usize, len: usize) -> LazyAccessHold {
    if len == 0 || !lazy_copyback() {
        return LazyAccessHold { kind: HoldKind::None };
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if last - first <= 8 {
        for p in first..last {
            LAZY_HOLD_STRIPES[p & LAZY_STRIPE_MASK].fetch_add(1, Ordering::SeqCst);
        }
        LazyAccessHold {
            kind: HoldKind::Small(first, last),
        }
    } else {
        LAZY_HOLD_GLOBAL.fetch_add(1, Ordering::SeqCst);
        LazyAccessHold { kind: HoldKind::Global }
    }
}

/// Number of 64 KiB pages (4 MiB) batched per single `LazyReadback` GPU submit.
const LAZY_READBACK_BATCH_PAGES: usize = 64;

/// Dedicated GPU readback context (Level 2), pre-allocated before any
/// `PAGE_NOACCESS` page is ever armed so VEH never allocates D3D12 objects.
///
/// Uses a dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue (falling back to
/// `host.queue` only if the device refuses a copy queue) so on-demand
/// materialization executes immediately on the hardware copy engine instead of
/// stalling behind queued render flights on the direct queue, and batches up to
/// 4 MiB (`LAZY_READBACK_BATCH_PAGES` pages) into a single GPU submit.
struct LazyReadback {
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    host_wait_event: windows::Win32::Foundation::HANDLE,
    fence_value: u64,
    readback: ID3D12Resource,
    mapped: *const u8,
}

// SAFETY: accessed exclusively under `LAZY_GPU` mutex; D3D12 objects are
// free-threaded and `event`/`host_wait_event` are only waited on under `LAZY_GPU`.
unsafe impl Send for LazyReadback {}

impl Drop for LazyReadback {
    fn drop(&mut self) {
        // SAFETY: `readback` was mapped in `LazyReadback::new` and events are ours.
        unsafe {
            self.readback.Unmap(0, None);
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
            let _ = windows::Win32::Foundation::CloseHandle(self.host_wait_event);
        }
    }
}

impl LazyReadback {
    fn new() -> Option<Self> {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::System::Threading::CreateEventW;

        let host = host()?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: (LAZY_READBACK_BATCH_PAGES * PAGE) as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: plain D3D12 object creation on `host.device`.
        unsafe {
            let copy_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            };
            let (queue, list_type) = match host.device.CreateCommandQueue::<ID3D12CommandQueue>(&copy_desc) {
                Ok(q) => (q, D3D12_COMMAND_LIST_TYPE_COPY),
                Err(_) => (host.queue().clone(), D3D12_COMMAND_LIST_TYPE_DIRECT),
            };
            let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(list_type).ok()?;
            let list: ID3D12GraphicsCommandList =
                host.device.CreateCommandList(0, list_type, &allocator, None).ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            let host_wait_event = match CreateEventW(None, false, false, None) {
                Ok(ev) => ev,
                Err(_) => {
                    let _ = windows::Win32::Foundation::CloseHandle(event);
                    return None;
                }
            };
            let mut readback: Option<ID3D12Resource> = None;
            if host
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .is_err()
            {
                let _ = windows::Win32::Foundation::CloseHandle(event);
                let _ = windows::Win32::Foundation::CloseHandle(host_wait_event);
                return None;
            }
            let readback = match readback {
                Some(r) => r,
                None => {
                    let _ = windows::Win32::Foundation::CloseHandle(event);
                    let _ = windows::Win32::Foundation::CloseHandle(host_wait_event);
                    return None;
                }
            };
            let mut mapped = std::ptr::null_mut();
            if readback.Map(0, None, Some(&mut mapped)).is_err() {
                let _ = windows::Win32::Foundation::CloseHandle(event);
                let _ = windows::Win32::Foundation::CloseHandle(host_wait_event);
                return None;
            }
            crate::host::name(&readback, || "lazy copy-back 4 MiB readback".into());
            Some(Self {
                queue,
                allocator,
                list,
                fence,
                event,
                host_wait_event,
                fence_value: 0,
                readback,
                mapped: mapped.cast(),
            })
        }
    }

    /// Reads `reqs.len()` 64 KiB pages into `dst` (`dst.len() == reqs.len() * PAGE`),
    /// strictly synchronized against any in-flight async writes on `host.queue`
    /// (lock-free via `host.submitted_fence()` + `shadow.last_write_fence` +
    /// `host.fence.GetCompletedValue()`, never locking `host.submitted`):
    /// - When `host.fence` has already passed every required write fence: ZERO wait.
    /// - When a non-carded shadow has an in-flight write on `host.queue`: waits on
    ///   `host.fence` before CPU-copying `shadow.data`.
    /// - When a carded shadow has an in-flight write on `host.queue`: issues
    ///   cross-queue `self.queue.Wait(&host.fence, req_fence)` before
    ///   `ExecuteCommandLists` so D3D12 multi-engine ordering and Frame N+1 data
    ///   coherence hold with zero debug layer errors.
    fn read_pages(&mut self, reqs: &[(&Shadow, usize)], dst: &mut [u8]) -> bool {
        use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};
        use windows::core::Interface;

        if reqs.is_empty() {
            return true;
        }
        if dst.len() != reqs.len() * PAGE {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let latest_host_fence = host.submitted_fence();
        let mut non_carded_req = 0u64;
        let mut carded_req = 0u64;
        let mut carded_indices: Vec<usize> = Vec::new();
        let mut non_carded_indices: Vec<usize> = Vec::new();
        for (i, &(shadow, offset)) in reqs.iter().enumerate() {
            if offset.saturating_add(PAGE) > shadow.len as usize {
                return false;
            }
            let wf = shadow.last_write_fence.load(Ordering::Acquire);
            let need = if shadow.writes_pending() {
                wf.max(latest_host_fence)
            } else {
                wf
            };
            if shadow.card.is_some() {
                carded_req = carded_req.max(need);
                carded_indices.push(i);
            } else {
                non_carded_req = non_carded_req.max(need);
                non_carded_indices.push(i);
            }
        }

        if !non_carded_indices.is_empty() {
            // SAFETY: free-threaded query/wait on `host.fence` using our exclusive
            // `host_wait_event` under `LAZY_GPU`; never locks `host.submitted`.
            let completed = unsafe { host.fence.GetCompletedValue() };
            if non_carded_req > completed {
                let waited = unsafe {
                    (|| -> windows::core::Result<()> {
                        host.fence
                            .SetEventOnCompletion(non_carded_req, self.host_wait_event)?;
                        WaitForSingleObject(self.host_wait_event, INFINITE);
                        Ok(())
                    })()
                };
                if waited.is_err() {
                    return false;
                }
            }
            for i in non_carded_indices {
                let (shadow, offset) = reqs[i];
                // SAFETY: `offset + PAGE <= shadow.len` checked above and any
                // in-flight write on `host.queue` has completed.
                unsafe {
                    let dst_ptr = dst.as_mut_ptr().add(i * PAGE);
                    if shadow.data.is_null() {
                        continue;
                    }
                    if shadow.rebar {
                        copy_from_wc(shadow.data.add(offset), dst_ptr, PAGE);
                    } else {
                        std::ptr::copy_nonoverlapping(shadow.data.add(offset), dst_ptr, PAGE);
                    }
                }
            }
        }
        if carded_indices.is_empty() {
            return true;
        }
        let mut waited_cross_queue = false;
        for chunk in carded_indices.chunks(LAZY_READBACK_BATCH_PAGES) {
            // SAFETY: exclusive ownership under `LAZY_GPU`; `self.queue.Wait`
            // strictly orders behind any in-flight `host.queue` write so every
            // `shadow.buffer` is in `COMMON` before `CopyBufferRegion` promotes it.
            let ok = unsafe {
                (|| -> windows::core::Result<()> {
                    if !waited_cross_queue && carded_req > host.fence.GetCompletedValue() {
                        self.queue.Wait(&host.fence, carded_req)?;
                        waited_cross_queue = true;
                    }
                    self.allocator.Reset()?;
                    self.list.Reset(&self.allocator, None)?;
                    let mut c = 0usize;
                    while c < chunk.len() {
                        let (shadow, start_off) = reqs[chunk[c]];
                        let mut run_pages = 1usize;
                        while c + run_pages < chunk.len() {
                            let (next_shadow, next_off) = reqs[chunk[c + run_pages]];
                            if std::ptr::eq(shadow, next_shadow)
                                && next_off == start_off + run_pages * PAGE
                            {
                                run_pages += 1;
                            } else {
                                break;
                            }
                        }
                        self.list.CopyBufferRegion(
                            &self.readback,
                            (c * PAGE) as u64,
                            &shadow.buffer,
                            start_off as u64,
                            (run_pages * PAGE) as u64,
                        );
                        c += run_pages;
                    }
                    self.list.Close()?;
                    let cmd: ID3D12CommandList = self.list.cast()?;
                    self.queue.ExecuteCommandLists(&[Some(cmd)]);
                    self.fence_value += 1;
                    self.queue.Signal(&self.fence, self.fence_value)?;
                    self.fence.SetEventOnCompletion(self.fence_value, self.event)?;
                    WaitForSingleObject(self.event, INFINITE);
                    for (slot, &req_idx) in chunk.iter().enumerate() {
                        std::ptr::copy_nonoverlapping(
                            self.mapped.add(slot * PAGE),
                            dst.as_mut_ptr().add(req_idx * PAGE),
                            PAGE,
                        );
                    }
                    Ok(())
                })()
            };
            if ok.is_err() {
                return false;
            }
        }
        true
    }

    fn read_page(&mut self, shadow: &Shadow, offset: usize, dst: &mut [u8]) -> bool {
        self.read_pages(&[(shadow, offset)], dst)
    }
}

/// Level 2 GPU readback serializer.
static LAZY_GPU: Mutex<Option<LazyReadback>> = Mutex::new(None);
/// Lock-free readiness flag so `ensure_lazy_readback()` never blocks behind an
/// active GPU readback after one-time initialization.
static LAZY_GPU_READY: AtomicBool = AtomicBool::new(false);
/// Count of threads actively requesting or holding `LAZY_GPU` for a `COPY`
/// queue readback. Pairs via `SeqCst` with [`lazy_submit_gpu_guard`] so
/// `Host::submit` (`DIRECT`) and `LazyReadback::read_pages` (`COPY`) never
/// overlap on `shadow.buffer` or miss a newly-submitted write fence.
static LAZY_READBACK_ACTIVE: AtomicUsize = AtomicUsize::new(0);

struct LazyReadbackGuard {
    guard: std::sync::MutexGuard<'static, Option<LazyReadback>>,
}

impl Drop for LazyReadbackGuard {
    fn drop(&mut self) {
        LAZY_READBACK_ACTIVE.fetch_sub(1, Ordering::Release);
    }
}

fn lock_lazy_gpu_for_readback() -> LazyReadbackGuard {
    LAZY_READBACK_ACTIVE.fetch_add(1, Ordering::SeqCst);
    LazyReadbackGuard {
        guard: LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner()),
    }
}

/// Scoped `LAZY_GPU` guard acquired by `Host::submit` around `ExecuteCommandLists`,
/// `Signal`, `submitted_fence.store`, and `note_write_fence` (dropped before any
/// CPU `WaitForSingleObject`) ONLY when `VICTORY_LAZY_COPYBACK=1` has armed pages
/// or an active readback. Strictly follows Level 0 (`host.submitted`) -> Level 2
/// (`LAZY_GPU`), preventing D3D12 multi-engine `COPY`/`DIRECT` hazards with zero
/// cost when lazy copy-back is inactive.
pub(crate) struct LazySubmitGpuGuard {
    _guard: Option<std::sync::MutexGuard<'static, Option<LazyReadback>>>,
}

pub(crate) fn lazy_submit_gpu_guard() -> LazySubmitGpuGuard {
    if !lazy_copyback()
        || (LAZY_ARMED_COUNT.load(Ordering::SeqCst) == 0
            && LAZY_READBACK_ACTIVE.load(Ordering::SeqCst) == 0)
    {
        return LazySubmitGpuGuard { _guard: None };
    }
    LazySubmitGpuGuard {
        _guard: Some(LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner())),
    }
}

fn ensure_lazy_readback() -> bool {
    if LAZY_GPU_READY.load(Ordering::Acquire) {
        return true;
    }
    let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
    if gpu.is_none() {
        *gpu = LazyReadback::new();
    }
    let ok = gpu.is_some();
    if ok {
        LAZY_GPU_READY.store(true, Ordering::Release);
    }
    ok
}

/// Whether `addr` is currently committed and accessible for a `READ` or `WRITE`.
/// Used in `lazy_on_fault` only after `LAZY_FLUSH_TOTAL > 0` to distinguish a
/// benign concurrent stale fault (already flushed by another thread) from a
/// genuine access violation or a `watch.rs` `PAGE_READONLY` write guard.
fn is_accessible(addr: usize, is_write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
        PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0 || mbi.State != MEM_COMMIT {
        return false;
    }
    if is_write {
        matches!(mbi.Protect, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
    } else {
        matches!(
            mbi.Protect,
            PAGE_READWRITE | PAGE_READONLY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_READ
        )
    }
}

/// Coalesced `VirtualProtect(..., PAGE_READWRITE)` over a contiguous run of
/// committed `PAGE_NOACCESS` pages `[run_first, run_last)`. Binary-splits on
/// VAD/uncommitted boundaries so raw `VirtualFree(MEM_DECOMMIT)` never crashes
/// and a single view/allocation still takes ONE `VirtualProtect` syscall.
fn restore_rw_run(run_first: usize, run_last: usize, restored: &mut [bool], base_first: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect,
        VirtualQuery,
    };
    if run_first >= run_last {
        return;
    }
    let addr = run_first * PAGE;
    let bytes = (run_last - run_first) * PAGE;
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n != 0
        && mbi.State == MEM_COMMIT
        && mbi.Protect == PAGE_NOACCESS
        && (mbi.BaseAddress as usize) <= addr
        && (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) >= addr.saturating_add(bytes)
    {
        let mut old = 0u32;
        let t0 = std::time::Instant::now();
        // SAFETY: verified committed `PAGE_NOACCESS` contiguous run we armed.
        let ok = unsafe { VirtualProtect(addr as *const _, bytes, PAGE_READWRITE, &mut old) != 0 };
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        if ok {
            for p in run_first..run_last {
                restored[p - base_first] = true;
            }
            return;
        }
    }
    if run_last - run_first == 1 {
        return;
    }
    let mid = run_first + (run_last - run_first) / 2;
    restore_rw_run(run_first, mid, restored, base_first);
    restore_rw_run(mid, run_last, restored, base_first);
}

/// Restores `PAGE_READWRITE` on `[page_addr, page_addr + PAGE)` only if it is
/// still a single committed `PAGE_NOACCESS` region (defending against raw
/// unnotified `VirtualFree` + `VirtualAlloc` reuse), and writes `data` into it.
///
/// For section-backed `MEM_MAPPED` pages (`section_view = Some((section, off, sec_bytes))`,
/// used by physical console GPU heaps), writes all 64 KiB through our persistent
/// writable `inplace::OWN` alias + `Release` fence WHILE `page_addr` is STILL
/// `PAGE_NOACCESS`, then flips `page_addr` to `PAGE_READWRITE` in one atomic
/// PTE transition (zero torn-read window, zero per-page `MapViewOfFile`/`UnmapViewOfFile`).
fn unprotect_and_write_page(
    page_addr: usize,
    data: Option<&[u8]>,
    section_view: Option<(usize, u64, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualProtect, VirtualQuery,
    };
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0
        || mbi.State != MEM_COMMIT
        || (mbi.BaseAddress as usize) > page_addr
        || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        || mbi.Protect != PAGE_NOACCESS
    {
        return false;
    }
    if let (Some(bytes), Some((section, off, sec_bytes))) = (data, section_view)
        && mbi.Type == MEM_MAPPED
    {
        debug_assert_eq!(bytes.len(), PAGE);
        if crate::inplace::write_section_own(section, sec_bytes, off, bytes) {
            std::sync::atomic::fence(Ordering::Release);
            let mut old = 0u32;
            let t0 = std::time::Instant::now();
            // SAFETY: atomic PTE flip revealing fully populated physical page.
            let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
            lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
            return ok;
        }
    }
    let mut old = 0u32;
    let t0 = std::time::Instant::now();
    // SAFETY: verified committed `PAGE_NOACCESS` 64 KiB page we armed.
    let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
    if !ok {
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        return false;
    }
    if let Some(bytes) = data {
        debug_assert_eq!(bytes.len(), PAGE);
        // SAFETY: `page_addr` was just verified committed and set `PAGE_READWRITE`
        // for `PAGE` bytes, and `bytes` is our disjoint heap buffer. Copy
        // immediately before querying the clock or recording telemetry to keep
        // the `MEM_PRIVATE` post-`VirtualProtect` window as narrow as possible.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), page_addr as *mut u8, PAGE) };
        std::sync::atomic::fence(Ordering::Release);
    }
    lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
    true
}

/// Writes GPU bytes ONLY to the unwritten complement `[0..rel_lo)` and
/// `[rel_hi..PAGE)` of a partially overwritten deferred page `[page_addr, page_addr + PAGE)`
/// (used by [`lazy_release_written_range`] after DirectStorage writes into
/// backing physical pages), then restores `PAGE_READWRITE`. Never touches
/// `[rel_lo..rel_hi)`.
fn unprotect_and_write_complement(
    page_addr: usize,
    gpu_bytes: &[u8],
    rel_lo: usize,
    rel_hi: usize,
    section_view: Option<(usize, u64, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualProtect, VirtualQuery,
    };
    debug_assert_eq!(gpu_bytes.len(), PAGE);
    debug_assert!(rel_lo <= rel_hi && rel_hi <= PAGE);
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0
        || mbi.State != MEM_COMMIT
        || (mbi.BaseAddress as usize) > page_addr
        || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        || mbi.Protect != PAGE_NOACCESS
    {
        return false;
    }
    if let Some((section, off, sec_bytes)) = section_view
        && mbi.Type == MEM_MAPPED
    {
        let mut own_ok = true;
        if rel_lo > 0 {
            own_ok &= crate::inplace::write_section_own(section, sec_bytes, off, &gpu_bytes[..rel_lo]);
        }
        if own_ok && rel_hi < PAGE {
            own_ok &= crate::inplace::write_section_own(
                section,
                sec_bytes,
                off + rel_hi as u64,
                &gpu_bytes[rel_hi..],
            );
        }
        if own_ok {
            std::sync::atomic::fence(Ordering::Release);
            let mut old = 0u32;
            let t0 = std::time::Instant::now();
            // SAFETY: atomic PTE flip revealing fully coherent page.
            let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
            lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
            return ok;
        }
    }
    let mut old = 0u32;
    let t0 = std::time::Instant::now();
    // SAFETY: verified committed `PAGE_NOACCESS` 64 KiB page we armed.
    let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
    if !ok {
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        return false;
    }
    // SAFETY: `page_addr` is committed and `PAGE_READWRITE`; copy only complement
    // immediately before querying the clock or recording telemetry.
    unsafe {
        let base = page_addr as *mut u8;
        if rel_lo > 0 {
            std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr(), base, rel_lo);
        }
        if rel_hi < PAGE {
            std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr().add(rel_hi), base.add(rel_hi), PAGE - rel_hi);
        }
        std::sync::atomic::fence(Ordering::Release);
    }
    lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
    true
}

/// Flushes one 64 KiB page that has already been transitioned to `LazyEntry::Busy`
/// in `LAZY_TABLE` and `LAZY_FLUSHING` in `shadow.lazy_state`.
#[allow(clippy::too_many_arguments)]
fn flush_armed_page(
    shadow: &Shadow,
    page: usize,
    offset: usize,
    page_idx: usize,
    section_view: Option<(usize, u64, u64)>,
    wait: &FlushWait,
    is_veh: bool,
    is_write: bool,
    save_baseline: bool,
) {
    // Heap allocation (never 64 KiB on a VEH stack).
    let mut buf = vec![0u8; PAGE];
    let t_mat = std::time::Instant::now();
    let ok = {
        let mut gpu = lock_lazy_gpu_for_readback();
        gpu.guard.as_mut().is_some_and(|rb| rb.read_page(shadow, offset, &mut buf))
    };
    let wrote = unprotect_and_write_page(page * PAGE, ok.then_some(buf.as_slice()), section_view);
    lazy_record_materialize_us(t_mat.elapsed().as_micros() as u64);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        lazy_table_remove_page(&mut table, page);
        LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
        if save_baseline {
            if ok && wrote {
                table
                    .flushed_baselines
                    .insert((shadow.id, page_idx), FlushedBaseline::new(buf));
            }
            shadow.lazy_state[page_idx].store(LAZY_RESYNC, Ordering::Release);
            shadow.resync_count.fetch_add(1, Ordering::AcqRel);
        } else {
            shadow.lazy_state[page_idx].store(LAZY_EAGER, Ordering::Release);
        }
        shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
    }
    if is_veh {
        lazy_record_veh_fault(is_write);
    } else {
        LAZY_STATS[4].fetch_add(1, Ordering::Relaxed);
    }
    if ok {
        lazy_record_bytes_readback(PAGE as u64);
    }
    wait.finish(FlushOutcome::Accessible);
}

struct ClaimedFlushPage<'a> {
    page: usize,
    shadow: &'a Shadow,
    offset: usize,
    page_idx: usize,
    section_view: Option<(usize, u64, u64)>,
    wait: Arc<FlushWait>,
}

/// Coalesced batch flush of claimed `Busy` pages: ONE `LazyReadback::read_pages`
/// (up to 4 MiB on the copy queue), persistent `OWN` section writes for
/// `MEM_MAPPED` pages, ONE `VirtualProtect(..., PAGE_READWRITE)` per contiguous
/// run, and ONE `LAZY_TABLE` commit lock.
fn flush_claimed_batch(batch: &[ClaimedFlushPage<'_>], save_baseline: bool) {
    if batch.is_empty() {
        return;
    }
    let n = batch.len();
    let mut bufs = vec![0u8; n * PAGE];
    let t_mat = std::time::Instant::now();
    let reqs: Vec<(&Shadow, usize)> = batch.iter().map(|c| (c.shadow, c.offset)).collect();
    let ok = {
        let mut gpu = lock_lazy_gpu_for_readback();
        gpu.guard.as_mut().is_some_and(|rb| rb.read_pages(&reqs, &mut bufs))
    };

    // Pre-write section-backed pages through persistent `inplace::OWN` ONLY
    // after verifying via region-cached `VirtualQuery` that the VA is still
    // committed `MEM_MAPPED | PAGE_NOACCESS` (never writing into an unnotified
    // unmapped/replaced section).
    let mut prewritten = vec![false; n];
    if ok {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
        };
        let mut cached: Option<(usize, usize, u32, u32, u32)> = None;
        let mut any_prewritten = false;
        for (i, c) in batch.iter().enumerate() {
            let Some((section, off, sec_bytes)) = c.section_view else {
                continue;
            };
            let page_addr = c.page * PAGE;
            let (base_a, end_a, state, ty, prot) = match cached {
                Some(m) if page_addr >= m.0 && page_addr + PAGE <= m.1 => m,
                _ => {
                    // SAFETY: plain struct filled by `VirtualQuery`.
                    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                    // SAFETY: valid out pointer.
                    let qn = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
                    if qn == 0 {
                        continue;
                    }
                    let m = (
                        mbi.BaseAddress as usize,
                        (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize),
                        mbi.State,
                        mbi.Type,
                        mbi.Protect,
                    );
                    cached = Some(m);
                    m
                }
            };
            if state == MEM_COMMIT
                && ty == MEM_MAPPED
                && prot == PAGE_NOACCESS
                && page_addr >= base_a
                && page_addr + PAGE <= end_a
                && crate::inplace::write_section_own(
                    section,
                    sec_bytes,
                    off,
                    &bufs[i * PAGE..(i + 1) * PAGE],
                )
            {
                prewritten[i] = true;
                any_prewritten = true;
            }
        }
        if any_prewritten {
            std::sync::atomic::fence(Ordering::Release);
        }
    }

    // Coalesce contiguous pages in `batch` (already sorted by `page`) into runs
    // for `VirtualProtect(..., PAGE_READWRITE)`.
    let mut restored = vec![false; n];
    let mut idx = 0usize;
    while idx < n {
        let run_start_idx = idx;
        let run_first = batch[idx].page;
        let mut run_last = run_first + 1;
        idx += 1;
        while idx < n && batch[idx].page == run_last {
            run_last += 1;
            idx += 1;
        }
        restore_rw_run(run_first, run_last, &mut restored[run_start_idx..idx], run_first);
    }

    // Copy bytes for any restored pages that were not prewritten via `OWN`.
    if ok {
        let mut copied_any = false;
        for (i, c) in batch.iter().enumerate() {
            if restored[i] && !prewritten[i] {
                // SAFETY: `restore_rw_run` verified committed `PAGE_READWRITE`.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bufs.as_ptr().add(i * PAGE),
                        (c.page * PAGE) as *mut u8,
                        PAGE,
                    );
                }
                copied_any = true;
            }
        }
        if copied_any {
            std::sync::atomic::fence(Ordering::Release);
        }
    }
    lazy_record_materialize_us(t_mat.elapsed().as_micros() as u64);

    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for (i, c) in batch.iter().enumerate() {
            lazy_table_remove_page(&mut table, c.page);
            if save_baseline {
                if ok && restored[i] {
                    table.flushed_baselines.insert(
                        (c.shadow.id, c.page_idx),
                        FlushedBaseline::new(bufs[i * PAGE..(i + 1) * PAGE].to_vec()),
                    );
                }
                c.shadow.lazy_state[c.page_idx].store(LAZY_RESYNC, Ordering::Release);
                c.shadow.resync_count.fetch_add(1, Ordering::AcqRel);
            } else {
                c.shadow.lazy_state[c.page_idx].store(LAZY_EAGER, Ordering::Release);
            }
            c.shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
        }
        LAZY_FLUSH_TOTAL.fetch_add(n as u64, Ordering::Release);
    }
    LAZY_STATS[4].fetch_add(n as u64, Ordering::Relaxed);
    if ok {
        lazy_record_bytes_readback((n * PAGE) as u64);
    }
    for c in batch {
        c.wait.finish(FlushOutcome::Accessible);
    }
}

/// Permanently pins title pages overlapping `[at, at + len)` (e.g., GPU fence
/// polling and `defer_write` targets) so they are NEVER protected `PAGE_NOACCESS`.
/// Any currently deferred page in the range is flushed first.
pub(crate) fn lazy_pin_range(at: usize, len: usize) {
    if len == 0 || !lazy_copyback() {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for page in first..last {
            table.pinned.insert(page);
        }
    }
    lazy_flush_range(at, len);
}

/// Returns all currently armed/busy lazy page numbers in `[first, last)` in
/// sorted order under AT MOST ONE `LAZY_TABLE` lock (zero locks when
/// `LAZY_ARMED_COUNT == 0` or small range has empty stripes).
pub(crate) fn lazy_deferred_pages_in_range(first: usize, last: usize) -> Vec<usize> {
    if !lazy_range_may_have_armed(first, last) {
        return Vec::new();
    }
    let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
    table.pages.range(first..last).map(|(&p, _)| p).collect()
}

/// Returns deferred page numbers in `[from, to)` owned by ANY shadow other than
/// `my_id` without flushing or unprotecting them.
fn lazy_other_deferred_pages(my_id: u64, from: usize, to: usize) -> Vec<usize> {
    if from >= to {
        return Vec::new();
    }
    let first = from / PAGE;
    let last = to.div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return Vec::new();
    }
    let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
    table
        .pages
        .range(first..last)
        .filter_map(|(&p, entry)| match entry {
            LazyEntry::Armed(a) if a.shadow_id != my_id => Some(p),
            LazyEntry::Busy { shadow_id, .. } if *shadow_id != my_id => Some(p),
            _ => None,
        })
        .collect()
}

/// Internal layer self-bypass reader: services `[src, src + dst.len())` into
/// host slice `dst` while caller holds [`LazyAccessHold`].
/// - Non-deferred ranges take the lock-free zero-mutex fast path directly to
///   `fallback_read`.
/// - Any `LAZY_DEFERRED` page in the range is read directly from its owning
///   shadow (batched on the copy queue for carded VRAM, direct copy for shared/
///   ReBAR) WITHOUT unprotecting title memory, WITHOUT triggering VEH, and
///   WITHOUT clearing `LAZY_DEFERRED` or setting `LAZY_RESYNC`!
pub(crate) fn lazy_read_bypass(
    src: usize,
    dst: &mut [u8],
    mut fallback_read: impl FnMut(usize, &mut [u8]) -> bool,
) -> bool {
    let len = dst.len();
    if len == 0 {
        return true;
    }
    let end = src.saturating_add(len);
    let first = src / PAGE;
    let last = end.div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return fallback_read(src, dst);
    }

    // Snapshot all `Armed` pages overlapping `[first, last)`, waiting on any
    // transiently `Busy` entry. Because the caller holds `LazyAccessHold`, no
    // new page in `[first, last)` can transition into `LAZY_TABLE` concurrently.
    let armed: Vec<(usize, Arc<Shadow>, usize)> = loop {
        enum Snapshot {
            None,
            Wait(Arc<FlushWait>),
            Yield,
            Pages(Vec<(usize, Arc<Shadow>, usize)>),
        }
        let snap = {
            let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let mut found = Vec::new();
            let mut wait = None;
            let mut yield_drop = false;
            for (&page, entry) in table.pages.range(first..last) {
                match entry {
                    LazyEntry::Busy { wait: w, .. } => {
                        wait = Some(w.clone());
                        break;
                    }
                    LazyEntry::Armed(a) => match a.shadow.upgrade() {
                        Some(shadow) => found.push((page, shadow, a.offset)),
                        None => {
                            yield_drop = true;
                            break;
                        }
                    },
                }
            }
            if let Some(w) = wait {
                Snapshot::Wait(w)
            } else if yield_drop {
                Snapshot::Yield
            } else if found.is_empty() {
                Snapshot::None
            } else {
                Snapshot::Pages(found)
            }
        };
        match snap {
            Snapshot::None => return fallback_read(src, dst),
            Snapshot::Wait(w) => {
                let _ = w.wait();
            }
            Snapshot::Yield => std::thread::yield_now(),
            Snapshot::Pages(pages) => break pages,
        }
    };

    // 1. Batch-read all deferred pages directly from their owning shadow(s).
    let t_mat = std::time::Instant::now();
    let mut page_bufs = vec![0u8; armed.len() * PAGE];
    let reqs: Vec<(&Shadow, usize)> = armed.iter().map(|(_, s, off)| (s.as_ref(), *off)).collect();
    let read_ok = {
        let mut gpu = lock_lazy_gpu_for_readback();
        gpu.guard.as_mut().is_some_and(|rb| rb.read_pages(&reqs, &mut page_bufs))
    };
    lazy_record_materialize_us(t_mat.elapsed().as_micros() as u64);
    if !read_ok {
        // Fallback: explicitly flush and read via normal kernel/plain path.
        lazy_flush_range(src, len);
        return fallback_read(src, dst);
    }
    lazy_record_self_bypass(armed.len() as u64);

    // 2. Copy deferred page slices into `dst` and service non-deferred gaps via `fallback_read`.
    let mut cursor = src;
    for (i, &(page, _, _)) in armed.iter().enumerate() {
        let p_lo = (page * PAGE).max(src);
        let p_hi = ((page + 1) * PAGE).min(end);
        if cursor < p_lo {
            let run_dst = &mut dst[cursor - src..p_lo - src];
            if !fallback_read(cursor, run_dst) {
                return false;
            }
        }
        if p_lo < p_hi {
            let in_page_lo = p_lo - page * PAGE;
            let in_page_hi = p_hi - page * PAGE;
            let page_slice = &page_bufs[i * PAGE + in_page_lo..i * PAGE + in_page_hi];
            dst[p_lo - src..p_hi - src].copy_from_slice(page_slice);
        }
        cursor = p_hi;
    }
    cursor >= end || fallback_read(cursor, &mut dst[cursor - src..end - src])
}

/// Deadlock-free VEH demand-fault handler called from `watch.rs::on_fault` on
/// both READ and WRITE access violations before `watch` acquires any lock.
/// Returns `true` iff the fault was handled and execution may resume.
pub(crate) fn lazy_on_fault(addr: usize, is_write: bool) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 && LAZY_FLUSH_TOTAL.load(Ordering::Acquire) == 0 {
        return false;
    }
    if IN_LAZY_VEH.with(Cell::get) {
        return false;
    }
    struct InVehGuard;
    impl Drop for InVehGuard {
        fn drop(&mut self) {
            IN_LAZY_VEH.with(|c| c.set(false));
        }
    }
    IN_LAZY_VEH.with(|c| c.set(true));
    let _in_veh = InVehGuard;

    let page = addr / PAGE;
    // Fast lock-free exit when this stripe has no armed pages and the fault is
    // not a stale post-flush race.
    if LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].load(Ordering::Acquire) == 0 {
        return LAZY_FLUSH_TOTAL.load(Ordering::Acquire) > 0 && is_accessible(addr, is_write);
    }
    loop {
        let action = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            match table.pages.get(&page) {
                Some(LazyEntry::Busy { wait, .. }) => Ok(Err(wait.clone())),
                Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                    Some(shadow) => {
                        let (shadow_id, offset, page_idx, section_view) =
                            (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                        let wait = Arc::new(FlushWait::new());
                        shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                        lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                        Ok(Ok((shadow, offset, page_idx, section_view, wait)))
                    }
                    // `Shadow::drop` is actively executing and will acquire
                    // `LAZY_TABLE` momentarily to transition to `Busy` and flush.
                    None => Err(true),
                },
                None => Err(false),
            }
        };
        match action {
            Ok(Ok((shadow, offset, page_idx, section_view, wait))) => {
                flush_armed_page(&shadow, page, offset, page_idx, section_view, &wait, true, is_write, true);
                return true;
            }
            Ok(Err(wait)) => {
                if wait.wait() == FlushOutcome::Accessible {
                    return true;
                }
                continue;
            }
            Err(true) => {
                std::thread::yield_now();
                continue;
            }
            Err(false) => {
                return LAZY_FLUSH_TOTAL.load(Ordering::Acquire) > 0 && is_accessible(addr, is_write);
            }
        }
    }
}

/// Flushes every deferred GPU-resident page overlapping `[at, at + len)` into
/// title memory and restores `PAGE_READWRITE` in coalesced batches (up to 4 MiB
/// per copy-queue submit and ONE `VirtualProtect` per contiguous run), waiting
/// on any in-flight flush. Safe to call with zero Level 0 locks held.
pub(crate) fn lazy_flush_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return;
    }
    loop {
        let mut claimed: Vec<(usize, Arc<Shadow>, usize, usize, Option<(usize, u64, u64)>, Arc<FlushWait>)> =
            Vec::new();
        let mut wait_on: Option<Arc<FlushWait>> = None;
        let mut yield_drop = false;
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let keys: Vec<usize> = table
                .pages
                .range(first..last)
                .take(LAZY_READBACK_BATCH_PAGES)
                .map(|(&p, _)| p)
                .collect();
            for page in keys {
                match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => {
                        if claimed.is_empty() {
                            wait_on = Some(wait.clone());
                            break;
                        }
                    }
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                            claimed.push((page, shadow, offset, page_idx, section_view, wait));
                        }
                        None => {
                            if claimed.is_empty() {
                                yield_drop = true;
                                break;
                            }
                        }
                    },
                    None => {}
                }
            }
        }
        if !claimed.is_empty() {
            let batch: Vec<ClaimedFlushPage<'_>> = claimed
                .iter()
                .map(|(page, shadow, offset, page_idx, section_view, wait)| ClaimedFlushPage {
                    page: *page,
                    shadow: shadow.as_ref(),
                    offset: *offset,
                    page_idx: *page_idx,
                    section_view: *section_view,
                    wait: wait.clone(),
                })
                .collect();
            flush_claimed_batch(&batch, true);
            continue;
        }
        if let Some(wait) = wait_on {
            let _ = wait.wait();
            continue;
        }
        if yield_drop {
            std::thread::yield_now();
            continue;
        }
        return;
    }
}

/// Scoped active-remap guard held inside `watch::Remapping` from entry of
/// `watch::title_memory_remapping` until `Remapping::drop`: prevents any
/// concurrent relay/submit thread from re-arming `PAGE_NOACCESS` while views
/// are being unmapped or replaced.
pub(crate) struct LazyRemapGuard {
    id: Option<u64>,
}

impl Drop for LazyRemapGuard {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            table.remapping.retain(|&(rid, _, _)| rid != id);
        }
    }
}

/// Registers an active remap barrier in `LAZY_TABLE` for the duration of
/// `watch::title_memory_remapping()` without flushing unrelated deferred pages
/// (exact remapped ranges are flushed by `watch::title_memory_remapped(at, len)`
/// and `watch::views_mapped` before unmapping/aliasing).
/// Safe to call with zero Level 0 locks held.
pub(crate) fn lazy_begin_remapping_barrier() -> LazyRemapGuard {
    if !lazy_copyback() {
        return LazyRemapGuard { id: None };
    }
    let last = usize::MAX.div_ceil(PAGE);
    let id = {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let id = table.next_remap_id;
        table.next_remap_id = table.next_remap_id.wrapping_add(1).max(1);
        table.remapping.push((id, 0, last));
        id
    };
    LazyRemapGuard { id: Some(id) }
}

/// Registers a scoped non-arming range barrier for `[at, at + len)` and flushes
/// any currently armed or in-flight-arming deferred pages in that range before
/// returning. Used for title-destination writes and explicit tests; host read
/// paths use lock-free [`lazy_hold_access`] + [`lazy_read_bypass`].
pub(crate) fn lazy_flush_and_hold(at: usize, len: usize) -> LazyAccessHold {
    let hold = lazy_hold_access(at, len);
    if len > 0 && lazy_copyback() {
        lazy_flush_range(at, len);
    }
    hold
}

/// Releases deferred `PAGE_NOACCESS` pages overlapping `[at, at + len)` AFTER
/// external/DirectStorage bytes have ALREADY been written directly into the
/// backing physical pages (`watch::title_memory_written`).
pub(crate) fn lazy_release_written_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let first = at / PAGE;
    let last = end.div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return;
    }
    loop {
        enum Next {
            Done,
            Wait(Arc<FlushWait>),
            Yield,
            Release(usize, Arc<Shadow>, usize, usize, Option<(usize, u64, u64)>, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table.pages.range(first..last).next().map(|(&p, _)| p);
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                            Next::Release(page, shadow, offset, page_idx, section_view, wait)
                        }
                        None => Next::Yield,
                    },
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return,
            Next::Wait(wait) => {
                let _ = wait.wait();
            }
            Next::Yield => std::thread::yield_now(),
            Next::Release(page, shadow, offset, page_idx, section_view, wait) => {
                let page_addr = page * PAGE;
                let page_end = page_addr.saturating_add(PAGE);
                let w_lo = at.max(page_addr);
                let w_hi = end.min(page_end);
                let full_cover = w_lo == page_addr && w_hi == page_end;
                let t_mat = std::time::Instant::now();
                let mut baseline = if full_cover {
                    let captured = if shadow.writes_pending() {
                        let mut buf = vec![0u8; PAGE];
                        let ok = {
                            let mut gpu = lock_lazy_gpu_for_readback();
                            gpu.guard.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                        };
                        if ok {
                            lazy_record_bytes_readback(PAGE as u64);
                        }
                        ok.then_some(buf)
                    } else {
                        None
                    };
                    let _ = unprotect_and_write_page(page_addr, None, None);
                    captured
                } else {
                    let mut buf = vec![0u8; PAGE];
                    let ok = {
                        let mut gpu = lock_lazy_gpu_for_readback();
                        gpu.guard.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                    };
                    let wrote = if ok {
                        unprotect_and_write_complement(
                            page_addr,
                            &buf,
                            w_lo - page_addr,
                            w_hi - page_addr,
                            section_view,
                        )
                    } else {
                        unprotect_and_write_page(page_addr, None, None)
                    };
                    if ok {
                        lazy_record_bytes_readback(PAGE as u64);
                    }
                    (ok && wrote).then_some(buf)
                };
                if full_cover && baseline.is_none() && shadow.writes_pending() {
                    let mut buf = vec![0u8; PAGE];
                    let ok = {
                        let mut gpu = lock_lazy_gpu_for_readback();
                        gpu.guard.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                    };
                    if ok {
                        lazy_record_bytes_readback(PAGE as u64);
                        baseline = Some(buf);
                    }
                }
                lazy_record_materialize_us(t_mat.elapsed().as_micros() as u64);
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    lazy_table_remove_page(&mut table, page);
                    LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
                    if let Some(buf) = baseline {
                        table
                            .flushed_baselines
                            .insert((shadow.id, page_idx), FlushedBaseline::new(buf));
                    }
                    shadow.lazy_state[page_idx].store(LAZY_RESYNC, Ordering::Release);
                    shadow.resync_count.fetch_add(1, Ordering::AcqRel);
                    shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
                }
                LAZY_STATS[4].fetch_add(1, Ordering::Relaxed);
                wait.finish(FlushOutcome::Accessible);
            }
        }
    }
}

/// Flushes deferred pages in `[from, to)` owned by ANY shadow other than `my_id`
/// in coalesced batches (used by `Shadow::writing` when another shadow takes
/// GPU write ownership of an overlapping address range).
fn lazy_flush_other_shadows(my_id: u64, from: usize, to: usize) -> Vec<usize> {
    if from >= to {
        return Vec::new();
    }
    let first = from / PAGE;
    let last = to.div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return Vec::new();
    }
    let mut flushed = Vec::new();
    loop {
        let mut claimed: Vec<(usize, Arc<Shadow>, usize, usize, Option<(usize, u64, u64)>, Arc<FlushWait>)> =
            Vec::new();
        let mut wait_on: Option<(usize, Arc<FlushWait>)> = None;
        let mut yield_drop = false;
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let keys: Vec<usize> = table
                .pages
                .range(first..last)
                .filter_map(|(&p, entry)| match entry {
                    LazyEntry::Armed(a) if a.shadow_id != my_id => Some(p),
                    LazyEntry::Busy { shadow_id, .. } if *shadow_id != my_id => Some(p),
                    _ => None,
                })
                .take(LAZY_READBACK_BATCH_PAGES)
                .collect();
            for page in keys {
                match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => {
                        if claimed.is_empty() {
                            wait_on = Some((page, wait.clone()));
                            break;
                        }
                    }
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                            claimed.push((page, shadow, offset, page_idx, section_view, wait));
                        }
                        None => {
                            if claimed.is_empty() {
                                yield_drop = true;
                                break;
                            }
                        }
                    },
                    None => {}
                }
            }
        }
        if !claimed.is_empty() {
            let batch: Vec<ClaimedFlushPage<'_>> = claimed
                .iter()
                .map(|(page, shadow, offset, page_idx, section_view, wait)| ClaimedFlushPage {
                    page: *page,
                    shadow: shadow.as_ref(),
                    offset: *offset,
                    page_idx: *page_idx,
                    section_view: *section_view,
                    wait: wait.clone(),
                })
                .collect();
            flush_claimed_batch(&batch, true);
            flushed.extend(claimed.iter().map(|c| c.0));
            continue;
        }
        if let Some((page, wait)) = wait_on {
            let _ = wait.wait();
            flushed.push(page);
            continue;
        }
        if yield_drop {
            std::thread::yield_now();
            continue;
        }
        return flushed;
    }
}

/// Drops deferred `PAGE_NOACCESS` protections overlapping `[at, at + len)` before
/// `VirtualFree` without wasting a GPU readback on memory being freed, restoring
/// `PAGE_READWRITE` in coalesced contiguous runs and waiting on any in-flight flush.
pub(crate) fn lazy_release_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if lazy_copyback() || LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0 {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        if !table.pinned.is_empty() {
            let stale_pins: Vec<usize> = table.pinned.range(first..last).copied().collect();
            for p in stale_pins {
                table.pinned.remove(&p);
            }
        }
    }
    if !lazy_range_may_have_armed(first, last) {
        return;
    }
    loop {
        let mut claimed: Vec<(usize, Weak<Shadow>, usize, Arc<FlushWait>)> = Vec::new();
        let mut wait_on: Option<Arc<FlushWait>> = None;
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let keys: Vec<usize> = table.pages.range(first..last).map(|(&p, _)| p).collect();
            for page in keys {
                match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => {
                        if claimed.is_empty() {
                            wait_on = Some(wait.clone());
                            break;
                        }
                    }
                    Some(LazyEntry::Armed(armed)) => {
                        let (shadow, shadow_id, page_idx) =
                            (armed.shadow.clone(), armed.shadow_id, armed.page_idx);
                        let wait = Arc::new(FlushWait::new());
                        if let Some(s) = shadow.upgrade() {
                            s.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                        }
                        lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                        claimed.push((page, shadow, page_idx, wait));
                    }
                    None => {}
                }
            }
        }
        if !claimed.is_empty() {
            let n = claimed.len();
            let mut restored = vec![false; n];
            let mut idx = 0usize;
            while idx < n {
                let run_start = idx;
                let run_first = claimed[idx].0;
                let mut run_last = run_first + 1;
                idx += 1;
                while idx < n && claimed[idx].0 == run_last {
                    run_last += 1;
                    idx += 1;
                }
                restore_rw_run(run_first, run_last, &mut restored[run_start..idx], run_first);
            }
            {
                let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                for (page, shadow, page_idx, _) in &claimed {
                    lazy_table_remove_page(&mut table, *page);
                    if let Some(s) = shadow.upgrade() {
                        s.lazy_state[*page_idx].store(LAZY_EAGER, Ordering::Release);
                        s.deferred_count.fetch_sub(1, Ordering::AcqRel);
                    }
                }
                LAZY_FLUSH_TOTAL.fetch_add(n as u64, Ordering::Release);
            }
            for (_, _, _, wait) in claimed {
                wait.finish(FlushOutcome::Accessible);
            }
            continue;
        }
        if let Some(wait) = wait_on {
            let _ = wait.wait();
            continue;
        }
        return;
    }
}

/// Rolls back contiguous runs of pages that were temporarily set to
/// `PAGE_NOACCESS` back to `PAGE_READWRITE` with coalesced `VirtualProtect`.
fn rollback_noaccess_pages_coalesced(pages: &[usize]) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if pages.is_empty() {
        return;
    }
    fn rollback_subrun(from: usize, to: usize) {
        let count = to - from;
        if count == 0 {
            return;
        }
        let mut _rev = 0u32;
        let t0 = std::time::Instant::now();
        // SAFETY: restoring `PAGE_READWRITE` on pages we just set `PAGE_NOACCESS`.
        let ok = unsafe {
            VirtualProtect((from * PAGE) as *const _, count * PAGE, PAGE_READWRITE, &mut _rev) != 0
        };
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        if ok || count == 1 {
            return;
        }
        let mid = from + count / 2;
        rollback_subrun(from, mid);
        rollback_subrun(mid, to);
    }
    let mut i = 0;
    while i < pages.len() {
        let run_first = pages[i];
        let mut run_last = run_first + 1;
        i += 1;
        while i < pages.len() && pages[i] == run_last {
            run_last += 1;
            i += 1;
        }
        rollback_subrun(run_first, run_last);
    }
}

/// Coalesced `VirtualProtect(..., PAGE_NOACCESS)` over `[from, to)`: clips each
/// sub-run to a single uniform `VirtualQuery` region (`MEM_COMMIT && MEM_PRIVATE
/// && PAGE_READWRITE`) BEFORE calling multi-page `VirtualProtect` so interior
/// non-matching protections are never clobbered, never overwrites `cur+1..sub_to`
/// with `old` on unexpected prior protection, and records every syscall.
fn protect_private_run_noaccess(from: usize, to: usize, protected: &mut Vec<usize>) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualProtect, VirtualQuery,
    };
    let mut cur = from;
    while cur < to {
        let addr = cur * PAGE;
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
        if n == 0 || mbi.State != MEM_COMMIT || mbi.Type != MEM_PRIVATE || mbi.Protect != PAGE_READWRITE {
            let skip_end = if n != 0 {
                (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) / PAGE
            } else {
                cur + 1
            };
            cur = skip_end.max(cur + 1);
            continue;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) / PAGE;
        let sub_to = to.min(region_end);
        if sub_to <= cur {
            cur += 1;
            continue;
        }
        let count = sub_to - cur;
        let mut old = 0u32;
        let t0 = std::time::Instant::now();
        // SAFETY: `VirtualQuery` verified `[cur, sub_to)` is one uniform
        // committed `MEM_PRIVATE` `PAGE_READWRITE` region.
        let ok = unsafe {
            VirtualProtect((cur * PAGE) as *const _, count * PAGE, PAGE_NOACCESS, &mut old) != 0
        };
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        if ok && old == PAGE_READWRITE {
            protected.extend(cur..sub_to);
        } else if ok {
            let mut _rev = 0u32;
            let tr = std::time::Instant::now();
            // SAFETY: restoring page `cur` to its reported prior protection.
            unsafe { VirtualProtect((cur * PAGE) as *const _, PAGE, old, &mut _rev) };
            lazy_record_protect_ns(tr.elapsed().as_nanos() as u64);
            if count > 1 {
                let tr2 = std::time::Instant::now();
                // SAFETY: restoring remaining verified `PAGE_READWRITE` pages.
                unsafe {
                    VirtualProtect(
                        ((cur + 1) * PAGE) as *const _,
                        (count - 1) * PAGE,
                        PAGE_READWRITE,
                        &mut _rev,
                    )
                };
                lazy_record_protect_ns(tr2.elapsed().as_nanos() as u64);
            }
        }
        cur = sub_to;
    }
}

/// Batch-defers candidate `(shadow, lo, hi, fetched_after)` pieces across one
/// or more shadows (used by `host::relay_card_back`, `host::relay_card_inflight`,
/// and `host::card_back`). Groups pieces by `(shadow, fetched_after)` and arms
/// contiguous runs with coalesced `VirtualProtect` syscalls.
pub(crate) fn try_defer_pieces_batch_with_after(
    pieces: &[(Arc<Shadow>, usize, usize, Option<u64>)],
) -> Vec<bool> {
    if pieces.is_empty() || !lazy_copyback() {
        return vec![false; pieces.len()];
    }
    let mut out = vec![false; pieces.len()];
    let mut by_key: Vec<((usize, Option<u64>), Arc<Shadow>, Vec<(usize, usize, usize)>)> = Vec::new();
    for (i, (shadow, lo, hi, after)) in pieces.iter().enumerate() {
        let key = (Arc::as_ptr(shadow) as usize, *after);
        if let Some(entry) = by_key.iter_mut().find(|e| e.0 == key) {
            entry.2.push((i, *lo, *hi));
        } else {
            by_key.push((key, shadow.clone(), vec![(i, *lo, *hi)]));
        }
    }
    for ((_, after), shadow, list) in by_key {
        let offsets: Vec<(usize, usize)> = list.iter().map(|&(_, lo, hi)| (lo, hi)).collect();
        let res = shadow.try_defer_shadow_pieces(&offsets, after);
        for ((orig_idx, _, _), deferred) in list.into_iter().zip(res) {
            out[orig_idx] = deferred;
        }
    }
    out
}

impl Shadow {
    /// Coalesced batch arming for a shadow's GPU-written pieces `offsets` (`(lo, hi)`):
    /// - O(1) lock-free fast path for already-`LAZY_DEFERRED` pieces.
    /// - Amortizes `copied`, `card.dirty`/`card.sent`, `VIEWS`, `JOURNAL`,
    ///   `GUARDS`, and `LAZY_TABLE` locks from O(N) to O(1).
    /// - Zero `VirtualQuery` on `MEM_MAPPED` view pages and cached region-level
    ///   `VirtualQuery` on `MEM_PRIVATE` pages.
    /// - Coalesces contiguous runs into ONE `VirtualProtect(..., PAGE_NOACCESS)`
    ///   syscall (one TLB shootdown) per contiguous run.
    pub(crate) fn try_defer_shadow_pieces(
        self: &Arc<Self>,
        offsets: &[(usize, usize)],
        fetched_after: Option<u64>,
    ) -> Vec<bool> {
        use windows_sys::Win32::System::Memory::{
            GetWriteWatch, MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE,
            VirtualQuery,
        };

        let mut out = vec![false; offsets.len()];
        if offsets.is_empty() || !lazy_copyback() || self.place.is_some() || self.window || self.rebar || self.transient() || self.data.is_null() {
            return out;
        }
        #[cfg(test)]
        if !self.lazy_opted_in && std::env::var_os("VICTORY_LAZY_COPYBACK").is_none_or(|v| v != "1") {
            return out;
        }

        let start = self.start as usize;
        let len = self.len as usize;
        let base = start / PAGE;
        let watched_ok = self.watched.load(Ordering::Relaxed) == 1;

        struct Cand {
            out_idx: usize,
            lo: usize,
            page: usize,
            page_idx: usize,
            copied_at: u64,
            section_view: Option<(usize, u64, u64)>,
        }

        let mut steady_hits = 0u64;
        let mut cands: Vec<Cand> = Vec::new();
        for (i, &(lo, hi)) in offsets.iter().enumerate() {
            let page_addr = start.saturating_add(lo);
            if hi.saturating_sub(lo) != PAGE || !page_addr.is_multiple_of(PAGE) || hi > len {
                continue;
            }
            let page = page_addr / PAGE;
            let idx = page - base;
            if idx >= self.lazy_state.len() {
                continue;
            }
            let state = self.lazy_state[idx].load(Ordering::Acquire);
            if state == LAZY_DEFERRED {
                out[i] = true;
                steady_hits += 1;
                continue;
            }
            if state == LAZY_EAGER && watched_ok {
                cands.push(Cand {
                    out_idx: i,
                    lo,
                    page,
                    page_idx: idx,
                    copied_at: 0,
                    section_view: None,
                });
            }
        }
        if steady_hits > 0 {
            LAZY_STATS[1].fetch_add(steady_hits, Ordering::Relaxed);
            lazy_record_deferred(steady_hits);
        }
        if cands.is_empty() {
            return out;
        }

        // Filter 1: whole-page initialized (`copied_at != 0`) under ONE `self.copied` lock.
        {
            let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            cands.retain_mut(|c| {
                let at = copied.get(c.page_idx).unwrap_or(0);
                c.copied_at = at;
                at != 0
            });
        }
        if cands.is_empty() {
            return out;
        }

        // Filter 2: no un-uploaded CPU dirty / newer sent bytes under ONE lock.
        if let Some(card) = &self.card
            && (card.has_dirty.load(Ordering::Acquire)
                || (fetched_after.is_some() && card.has_sent.load(Ordering::Acquire)))
        {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = fetched_after.map(|_| card.sent.lock().unwrap_or_else(|e| e.into_inner()));
            cands.retain(|c| {
                let (lo, hi) = (c.lo, c.lo + PAGE);
                if dirty.iter().any(|&(a, b)| a < hi && b > lo) {
                    return false;
                }
                if let (Some(fa), Some(s)) = (fetched_after, sent.as_deref())
                    && s.iter().any(|&(a, b, seq)| seq > fa && a < hi && b > lo)
                {
                    return false;
                }
                true
            });
        }
        if cands.is_empty() {
            return out;
        }

        // Filter 3: journal check under ONE `JOURNAL` lock.
        {
            let queries: Vec<(usize, u64)> = cands.iter().map(|c| (c.page, c.copied_at)).collect();
            let written = crate::watch::was_written_since_batch(&queries);
            let mut k = 0;
            cands.retain(|_| {
                let ok = !written[k];
                k += 1;
                ok
            });
        }
        if cands.is_empty() {
            return out;
        }

        // Install VEH and pre-allocate readback channel ONCE before any `PAGE_NOACCESS`.
        crate::watch::install_veh();
        if !ensure_lazy_readback() {
            return out;
        }

        // Classify `MEM_MAPPED` (clean `watch`-guarded) vs `MEM_PRIVATE` under
        // ONE `GUARDS` lock + ONE `VIEWS` lock (zero `VirtualQuery` syscalls for
        // `MEM_MAPPED` views, whether registered or unregistered, and immediate
        // rejection of aliased or open-guard `MEM_MAPPED` pages!).
        let cand_pages: Vec<usize> = cands.iter().map(|c| c.page).collect();
        let clean_guarded = crate::watch::clean_guarded_mask(&cand_pages);
        let addrs: Vec<usize> = cands.iter().map(|c| c.page * PAGE).collect();
        let views_info = crate::inplace::single_view_pages_batch(&addrs);
        let mut mapped_cands: Vec<Cand> = Vec::new();
        let mut private_cands: Vec<Cand> = Vec::new();
        for ((mut c, is_clean_guarded), vinfo) in cands
            .into_iter()
            .zip(clean_guarded)
            .zip(views_info)
        {
            if is_clean_guarded {
                match vinfo {
                    crate::inplace::PageViewInfo::Aliased => {}
                    crate::inplace::PageViewInfo::Single(s, o, len) => {
                        c.section_view = Some((s, o, len));
                        mapped_cands.push(c);
                    }
                    crate::inplace::PageViewInfo::Unknown => {
                        c.section_view = None;
                        mapped_cands.push(c);
                    }
                }
            } else if vinfo == crate::inplace::PageViewInfo::Unknown {
                private_cands.push(c);
            }
        }

        mapped_cands.sort_unstable_by_key(|c| c.page);
        private_cands.sort_unstable_by_key(|c| c.page);
        // Preserve duplicate `out_idx` entries for identical pages so all
        // duplicate offsets receive `out[idx] = true` if their page arms.
        let mut dup_out: Vec<(usize, usize)> = Vec::new();
        let dedup_preserve = |v: &mut Vec<Cand>, dups: &mut Vec<(usize, usize)>| {
            let mut w = 0usize;
            for r in 0..v.len() {
                if w > 0 && v[w - 1].page == v[r].page {
                    dups.push((v[r].page, v[r].out_idx));
                } else {
                    if w != r {
                        v.swap(w, r);
                    }
                    w += 1;
                }
            }
            v.truncate(w);
        };
        dedup_preserve(&mut mapped_cands, &mut dup_out);
        dedup_preserve(&mut private_cands, &mut dup_out);

        let mut newly_armed = 0u64;
        let mut armed_pages_set: BTreeSet<usize> = BTreeSet::new();

        // ---- Path A: `MEM_MAPPED` single-view clean `watch`-guarded runs ----
        if !mapped_cands.is_empty() {
            let mut reserved: Vec<(Cand, Arc<FlushWait>)> = Vec::with_capacity(mapped_cands.len());
            {
                let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                for c in mapped_cands {
                    if table.pinned.contains(&c.page)
                        || table.is_remapping(c.page)
                        || table.pages.contains_key(&c.page)
                    {
                        continue;
                    }
                    let wait = Arc::new(FlushWait::new());
                    lazy_table_insert_busy(&mut table, c.page, self.id, wait.clone());
                    if lazy_is_held(c.page) {
                        lazy_table_remove_page(&mut table, c.page);
                        continue;
                    }
                    reserved.push((c, wait));
                }
            }
            if !reserved.is_empty() {
                let mut transitioned_set: BTreeSet<usize> = BTreeSet::new();
                let mut i = 0;
                while i < reserved.len() {
                    let run_first = reserved[i].0.page;
                    let mut run_last = run_first + 1;
                    i += 1;
                    while i < reserved.len() && reserved[i].0.page == run_last {
                        run_last += 1;
                        i += 1;
                    }
                    let ok_blocks = crate::watch::transition_clean_guards_run_to_noaccess(run_first, run_last);
                    transitioned_set.extend(ok_blocks);
                }

                #[cfg(test)]
                if !transitioned_set.is_empty()
                    && let Some(hook) = TRY_DEFER_POST_TRANSITION_HOOK
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take()
                    && let Some(&first_page) = transitioned_set.iter().next()
                {
                    hook(first_page);
                }

                let transitioned_indices: Vec<usize> = reserved
                    .iter()
                    .enumerate()
                    .filter(|(_, (c, _))| transitioned_set.contains(&c.page))
                    .map(|(idx, _)| idx)
                    .collect();
                let post_queries: Vec<(usize, u64)> = transitioned_indices
                    .iter()
                    .map(|&idx| (reserved[idx].0.page, reserved[idx].0.copied_at))
                    .collect();
                let post_written = if post_queries.is_empty() {
                    Vec::new()
                } else {
                    crate::watch::was_written_since_batch(&post_queries)
                };
                let mut surviving = vec![false; reserved.len()];
                for (&r_idx, &w) in transitioned_indices.iter().zip(post_written.iter()) {
                    if !w {
                        surviving[r_idx] = true;
                    }
                }

                let mut to_rollback_vp: Vec<usize> = Vec::new();
                let mut to_remove: Vec<(usize, Arc<FlushWait>)> = Vec::new();
                let mut committed_waits: Vec<Arc<FlushWait>> = Vec::new();
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    for (r_idx, (c, wait)) in reserved.into_iter().enumerate() {
                        if surviving[r_idx]
                            && !table.pinned.contains(&c.page)
                            && !table.is_remapping(c.page)
                            && !lazy_is_held(c.page)
                        {
                            lazy_table_insert_armed(
                                &mut table,
                                c.page,
                                ArmedPage {
                                    shadow: Arc::downgrade(self),
                                    shadow_id: self.id,
                                    offset: c.lo,
                                    page_idx: c.page_idx,
                                    section_view: c.section_view,
                                    is_mapped_view: true,
                                },
                            );
                            self.lazy_state[c.page_idx].store(LAZY_DEFERRED, Ordering::Release);
                            self.deferred_count.fetch_add(1, Ordering::AcqRel);
                            out[c.out_idx] = true;
                            if !dup_out.is_empty() {
                                armed_pages_set.insert(c.page);
                            }
                            newly_armed += 1;
                            committed_waits.push(wait);
                        } else {
                            if transitioned_set.contains(&c.page) {
                                to_rollback_vp.push(c.page);
                            }
                            to_remove.push((c.page, wait));
                        }
                    }
                }
                if !to_rollback_vp.is_empty() {
                    rollback_noaccess_pages_coalesced(&to_rollback_vp);
                }
                if !to_remove.is_empty() {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    for &(page, _) in &to_remove {
                        lazy_table_remove_page(&mut table, page);
                    }
                    drop(table);
                    for (_, wait) in to_remove {
                        wait.finish(FlushOutcome::Accessible);
                    }
                }
                for wait in committed_waits {
                    wait.finish(FlushOutcome::ArmedRetry);
                }
            }
        }

        // ---- Path B: `MEM_PRIVATE | MEM_WRITE_WATCH` runs (cached `VirtualQuery`) ----
        if !private_cands.is_empty() {
            let mut cached: Option<(usize, usize, u32, u32, u32)> = None;
            private_cands.retain(|c| {
                let page_addr = c.page * PAGE;
                let (base_a, end_a, state, ty, prot) = match cached {
                    Some(m) if page_addr >= m.0 && page_addr + PAGE <= m.1 => m,
                    _ => {
                        // SAFETY: plain struct filled by `VirtualQuery`.
                        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                        // SAFETY: valid out pointer.
                        let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
                        if n == 0 {
                            return false;
                        }
                        let m = (
                            mbi.BaseAddress as usize,
                            (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize),
                            mbi.State,
                            mbi.Type,
                            mbi.Protect,
                        );
                        cached = Some(m);
                        m
                    }
                };
                state == MEM_COMMIT
                    && ty == MEM_PRIVATE
                    && prot == PAGE_READWRITE
                    && page_addr >= base_a
                    && page_addr + PAGE <= end_a
            });

            if !private_cands.is_empty() {
                let mut reserved: Vec<(Cand, Arc<FlushWait>)> = Vec::with_capacity(private_cands.len());
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    for c in private_cands {
                        if table.pinned.contains(&c.page)
                            || table.is_remapping(c.page)
                            || table.pages.contains_key(&c.page)
                        {
                            continue;
                        }
                        let wait = Arc::new(FlushWait::new());
                        lazy_table_insert_busy(&mut table, c.page, self.id, wait.clone());
                        if lazy_is_held(c.page) {
                            lazy_table_remove_page(&mut table, c.page);
                            continue;
                        }
                        reserved.push((c, wait));
                    }
                }

                if !reserved.is_empty() {
                    let mut protected_set: BTreeSet<usize> = BTreeSet::new();
                    let mut ww_clean_set: BTreeSet<usize> = BTreeSet::new();
                    let mut touched_buf: Vec<usize> = Vec::new();
                    let mut i = 0;
                    while i < reserved.len() {
                        let run_first = reserved[i].0.page;
                        let mut run_last = run_first + 1;
                        i += 1;
                        while i < reserved.len() && reserved[i].0.page == run_last {
                            run_last += 1;
                            i += 1;
                        }
                        let mut run_prot: Vec<usize> = Vec::with_capacity(run_last - run_first);
                        protect_private_run_noaccess(run_first, run_last, &mut run_prot);
                        if run_prot.is_empty() {
                            continue;
                        }
                        protected_set.extend(run_prot.iter().copied());

                        let mut j = 0;
                        while j < run_prot.len() {
                            let sub_first = run_prot[j];
                            let mut sub_last = sub_first + 1;
                            j += 1;
                            while j < run_prot.len() && run_prot[j] == sub_last {
                                sub_last += 1;
                                j += 1;
                            }
                            let sub_pages = sub_last - sub_first;
                            let need = sub_pages * (PAGE / 4096);
                            if touched_buf.len() < need {
                                touched_buf.resize(need, 0);
                            }
                            let mut count = need;
                            let mut granularity = 0u32;
                            // SAFETY: non-resetting write-watch query over the contiguous run.
                            let ww_ok = unsafe {
                                GetWriteWatch(
                                    0,
                                    (sub_first * PAGE) as *const _,
                                    sub_pages * PAGE,
                                    touched_buf.as_mut_ptr().cast(),
                                    &mut count,
                                    &mut granularity,
                                ) == 0
                            };
                            if !ww_ok {
                                continue;
                            }
                            if count == 0 {
                                ww_clean_set.extend(sub_first..sub_last);
                            } else {
                                let mut dirty_pages: BTreeSet<usize> = BTreeSet::new();
                                for &addr in &touched_buf[..count] {
                                    dirty_pages.insert(addr / PAGE);
                                }
                                for p in sub_first..sub_last {
                                    if !dirty_pages.contains(&p) {
                                        ww_clean_set.insert(p);
                                    }
                                }
                            }
                        }
                    }

                    let clean_indices: Vec<usize> = reserved
                        .iter()
                        .enumerate()
                        .filter(|(_, (c, _))| ww_clean_set.contains(&c.page))
                        .map(|(idx, _)| idx)
                        .collect();
                    let post_queries: Vec<(usize, u64)> = clean_indices
                        .iter()
                        .map(|&idx| (reserved[idx].0.page, reserved[idx].0.copied_at))
                        .collect();
                    let post_written = if post_queries.is_empty() {
                        Vec::new()
                    } else {
                        crate::watch::was_written_since_batch(&post_queries)
                    };
                    let mut surviving = vec![false; reserved.len()];
                    for (&r_idx, &w) in clean_indices.iter().zip(post_written.iter()) {
                        if !w {
                            surviving[r_idx] = true;
                        }
                    }

                    let mut to_rollback_vp: Vec<usize> = Vec::new();
                    let mut to_remove: Vec<(usize, Arc<FlushWait>)> = Vec::new();
                    let mut committed_waits: Vec<Arc<FlushWait>> = Vec::new();
                    {
                        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                        for (r_idx, (c, wait)) in reserved.into_iter().enumerate() {
                            if surviving[r_idx]
                                && !table.pinned.contains(&c.page)
                                && !table.is_remapping(c.page)
                                && !lazy_is_held(c.page)
                            {
                                lazy_table_insert_armed(
                                    &mut table,
                                    c.page,
                                    ArmedPage {
                                        shadow: Arc::downgrade(self),
                                        shadow_id: self.id,
                                        offset: c.lo,
                                        page_idx: c.page_idx,
                                        section_view: None,
                                        is_mapped_view: false,
                                    },
                                );
                                self.lazy_state[c.page_idx].store(LAZY_DEFERRED, Ordering::Release);
                                self.deferred_count.fetch_add(1, Ordering::AcqRel);
                                out[c.out_idx] = true;
                                if !dup_out.is_empty() {
                                    armed_pages_set.insert(c.page);
                                }
                                newly_armed += 1;
                                committed_waits.push(wait);
                            } else {
                                if protected_set.contains(&c.page) {
                                    to_rollback_vp.push(c.page);
                                }
                                to_remove.push((c.page, wait));
                            }
                        }
                    }
                    if !to_rollback_vp.is_empty() {
                        rollback_noaccess_pages_coalesced(&to_rollback_vp);
                    }
                    if !to_remove.is_empty() {
                        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                        for &(page, _) in &to_remove {
                            lazy_table_remove_page(&mut table, page);
                        }
                        drop(table);
                        for (_, wait) in to_remove {
                            wait.finish(FlushOutcome::Accessible);
                        }
                    }
                    for wait in committed_waits {
                        wait.finish(FlushOutcome::ArmedRetry);
                    }
                }
            }
        }

        if !dup_out.is_empty() {
            for (page, extra_idx) in dup_out {
                if armed_pages_set.contains(&page) {
                    out[extra_idx] = true;
                }
            }
        }
        if newly_armed > 0 {
            LAZY_STATS[0].fetch_add(newly_armed, Ordering::Relaxed);
            lazy_record_deferred(newly_armed);
        }
        out
    }

    /// Single-piece wrapper around [`Self::try_defer_shadow_pieces`].
    #[cfg(test)]
    pub(crate) fn try_defer_piece(self: &Arc<Self>, lo: usize, hi: usize) -> bool {
        self.try_defer_shadow_pieces(&[(lo, hi)], None)[0]
    }

    /// Relay variant of [`Self::try_defer_piece`]: also refuses to defer any
    /// newly-armed piece that has pending/later CPU uploads (`dirty` / `sent`).
    #[cfg(test)]
    pub(crate) fn try_defer_piece_relay(self: &Arc<Self>, lo: usize, hi: usize, fetched_after: u64) -> bool {
        self.try_defer_shadow_pieces(&[(lo, hi)], Some(fetched_after))[0]
    }

    /// Applies any unapplied bytes of a VEH-flushed 64 KiB baseline into `dst`
    /// (covering shadow offsets `[at, at + dst.len())`) before eager 3-way merge
    /// so CPU writes made after a mid-flight VEH flush are never overwritten by
    /// stale pre-GPU staging/snapshot bytes. Consumes applied byte subranges so
    /// back-to-back merges (`card_merge_relay` / `relay_merge`) are idempotent
    /// and never rewind `self.data`/`before` to an older baseline.
    fn lazy_apply_flushed_baseline(&self, at: usize, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }
        let start = self.start as usize;
        let base = start / PAGE;
        let first = (start + at) / PAGE;
        let last = (start + at + dst.len()).div_ceil(PAGE);
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for page in first..last {
            let idx = page - base;
            if idx >= self.lazy_state.len() || self.lazy_state[idx].load(Ordering::Acquire) != LAZY_RESYNC {
                continue;
            }
            let page_addr = page * PAGE;
            let lo = page_addr.max(start + at);
            let hi = (page_addr + PAGE).min(start + at + dst.len());
            if lo >= hi {
                continue;
            }
            let src_lo = lo - page_addr;
            let src_hi = hi - page_addr;
            let dst_off = lo - (start + at);
            let n = hi - lo;
            let remove = if let Some(baseline) = table.flushed_baselines.get_mut(&(self.id, idx)) {
                baseline.apply_once(src_lo, src_hi, &mut dst[dst_off..dst_off + n])
            } else {
                false
            };
            if remove {
                table.flushed_baselines.remove(&(self.id, idx));
            }
        }
    }

    /// Copies specific flushed pages from title memory into this shadow (used by
    /// [`Shadow::writing`] after [`lazy_flush_other_shadows`] materializes another
    /// overlapping shadow's deferred pages), skipping any page intersecting
    /// `skip_busy` (`self.pending`).
    fn resync_pages_from_title(&self, pages: &[usize], skip_busy: &[(usize, usize)]) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let watched = self.watched.load(Ordering::Relaxed) == 1;
        for &page in pages {
            if page < base {
                continue;
            }
            let idx = page - base;
            if idx >= self.lazy_state.len() {
                continue;
            }
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let p_lo = page_lo.max(start);
            let p_hi = page_hi.min(end);
            if p_lo >= p_hi {
                continue;
            }
            let whole = p_lo == page_lo && p_hi == page_hi;
            let now = if watched {
                crate::watch::writes(p_lo, p_hi, true).map_or(0, |(now, _)| now)
            } else {
                0
            };
            self.copy(p_lo, p_hi);
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            copied.set(idx, if whole { now } else { 0 });
        }
    }

    /// Transitions `LAZY_RESYNC` pages in `[from, to)` back to `LAZY_EAGER`
    /// (skipping any page overlapping `skip_busy`) and clears their `copied`
    /// entry so [`Shadow::update_free`] copies fresh title memory (including
    /// any CPU writes made after VEH flush) into staging and schedules VRAM
    /// upload (`card_changed`).
    fn lazy_prepare_update(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) {
        if self.resync_count.load(Ordering::Acquire) == 0 || from >= to {
            return;
        }
        let base = self.start as usize / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let mut resynced = Vec::new();
        for page in first..last {
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let idx = page - base;
            if idx < self.lazy_state.len()
                && self.lazy_state[idx]
                    .compare_exchange(LAZY_RESYNC, LAZY_EAGER, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                self.resync_count.fetch_sub(1, Ordering::AcqRel);
                resynced.push(idx);
            }
        }
        if resynced.is_empty() {
            return;
        }
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for &idx in &resynced {
                copied.set(idx, 0);
            }
        }
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for idx in resynced {
            table.flushed_baselines.remove(&(self.id, idx));
        }
    }

    /// Resyncs `LAZY_RESYNC` pages in `[from, to)` (skipping any page
    /// overlapping `skip_busy`) directly from title memory into `self.data`
    /// (and queues `card_changed`) before a UAV-only submission declared via
    /// [`Shadow::writing`] runs without a preceding `refresh`.
    fn lazy_resync_range(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return;
        }
        let base = start / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let mut pages = Vec::new();
        for page in first..last {
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let idx = page - base;
            if idx < self.lazy_state.len()
                && self.lazy_state[idx]
                    .compare_exchange(LAZY_RESYNC, LAZY_EAGER, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                self.resync_count.fetch_sub(1, Ordering::AcqRel);
                pages.push((page, idx));
            }
        }
        if pages.is_empty() {
            return;
        }
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            for &(_, idx) in &pages {
                table.flushed_baselines.remove(&(self.id, idx));
            }
        }
        let watched = self.watched.load(Ordering::Relaxed) == 1;
        for (page, idx) in pages {
            let p_lo = (page * PAGE).max(start);
            let p_hi = ((page + 1) * PAGE).min(end);
            if p_lo >= p_hi {
                continue;
            }
            let whole = p_lo == page * PAGE && p_hi == (page + 1) * PAGE;
            let now = if watched {
                crate::watch::writes(p_lo, p_hi, true).map_or(0, |(now, _)| now)
            } else {
                u64::from(whole)
            };
            self.copy(p_lo, p_hi);
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            copied.set(idx, if whole { now } else { 0 });
        }
    }

    /// Unwatched copy of `[from, to)` that skips `LAZY_DEFERRED` / `LAZY_FLUSHING`
    /// pages so GPU-resident pages are never spuriously flushed or overwritten.
    fn copy_unwatched_skipping_deferred(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        let base = start / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let mut bytes = 0;
        let mut seg_start: Option<usize> = None;
        for page in first..last {
            let idx = page - base;
            let skip = idx < self.lazy_state.len() && is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire));
            let p_lo = (page * PAGE).max(from);
            let p_hi = ((page + 1) * PAGE).min(to);
            if skip {
                if let Some(s) = seg_start.take() {
                    bytes += self.copy(s, p_lo);
                }
            } else {
                seg_start.get_or_insert(p_lo);
                if page + 1 == last
                    && let Some(s) = seg_start.take()
                {
                    bytes += self.copy(s, p_hi);
                }
            }
        }
        bytes
    }

    /// Flushes every deferred page owned by this `Shadow` in coalesced batches.
    /// Pass `save_baseline = true` from `unregister` (where in-flight lists may
    /// still hold `Arc<Shadow>` and settle via `card_merge_relay`), or `false`
    /// from `Drop::drop` (which also removes any remaining `flushed_baselines`).
    fn lazy_flush_all(&self, save_baseline: bool) {
        let first = self.start as usize / PAGE;
        let last = (self.start.saturating_add(self.len) as usize).div_ceil(PAGE);
        if lazy_range_may_have_armed(first, last) {
            loop {
                let mut claimed: Vec<(usize, usize, usize, Option<(usize, u64, u64)>, Arc<FlushWait>)> =
                    Vec::new();
                let mut wait_on: Option<Arc<FlushWait>> = None;
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    let keys: Vec<usize> = table
                        .pages
                        .range(first..last)
                        .filter_map(|(&page, entry)| match entry {
                            LazyEntry::Armed(a) if a.shadow_id == self.id => Some(page),
                            LazyEntry::Busy { shadow_id, .. } if *shadow_id == self.id => Some(page),
                            _ => None,
                        })
                        .take(LAZY_READBACK_BATCH_PAGES)
                        .collect();
                    for page in keys {
                        match table.pages.get(&page) {
                            Some(LazyEntry::Busy { wait, .. }) => {
                                if claimed.is_empty() {
                                    wait_on = Some(wait.clone());
                                    break;
                                }
                            }
                            Some(LazyEntry::Armed(armed)) => {
                                let (offset, page_idx, section_view) =
                                    (armed.offset, armed.page_idx, armed.section_view);
                                let wait = Arc::new(FlushWait::new());
                                self.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                                lazy_table_insert_busy(&mut table, page, self.id, wait.clone());
                                claimed.push((page, offset, page_idx, section_view, wait));
                            }
                            None => {}
                        }
                    }
                }
                if !claimed.is_empty() {
                    let batch: Vec<ClaimedFlushPage<'_>> = claimed
                        .iter()
                        .map(|(page, offset, page_idx, section_view, wait)| ClaimedFlushPage {
                            page: *page,
                            shadow: self,
                            offset: *offset,
                            page_idx: *page_idx,
                            section_view: *section_view,
                            wait: wait.clone(),
                        })
                        .collect();
                    flush_claimed_batch(&batch, save_baseline);
                    continue;
                }
                if let Some(wait) = wait_on {
                    let _ = wait.wait();
                    continue;
                }
                break;
            }
        }
        if !save_baseline {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            table.flushed_baselines.retain(|&(sid, _), _| sid != self.id);
        }
    }

    /// Flushes every deferred page owned by this `Shadow` and removes all of its
    /// `flushed_baselines` during `Drop::drop` without relying on `Weak::upgrade()`.
    fn lazy_flush_all_on_drop(&mut self) {
        self.lazy_flush_all(false);
    }
}


#[cfg(test)]
#[path = "shadow_lazy_tests.rs"]
mod lazy_and_simd_tests;
