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

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
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
    /// let go ([`let_go_idle`]); also tracks LRU age for card-buffer demotion.
    used: AtomicU64,
    /// Monotonic access tick for deterministic LRU ordering among card-backed
    /// title buffers accessed within the same frame/tick.
    access_tick: AtomicU64,
    /// Latest host queue fence value signalled behind a submission that
    /// referenced this copy; demotion refuses to drop a card placement while
    /// `last_fence > completed_fence`.
    last_fence: AtomicU64,
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
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
    /// Whether shaders/UAVs may have written directly to the card-side buffer
    /// (requires readback of non-dirty ranges before demotion).
    gpu_written: AtomicBool,
    /// Whether `size` is accounted in the global `CARD_BUDGET` / `CARD_BYTES`.
    counted: bool,
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

/// Threshold separating small card buffers (`< 1 MiB`) from large hot card
/// buffers (`>= 1 MiB`) for size-tiered headroom reservation and demotion.
pub(crate) const LARGE_CARD_MIN: u64 = 1 << 20;

/// Maximum default headroom reserved inside `VICTORY_CARD_BUFFERS_MB` for
/// `>= 1 MiB` buffers when `VICTORY_CARD_BUFFERS_MIN_KB < 1024`.
const DEFAULT_LARGE_RESERVE_MAX: u64 = 256 << 20;

/// Maximum DXGI video-memory percentage at which new `< 1 MiB` card buffers
/// are admitted, preserving headroom on 8-12 GB cards.
const DXGI_SMALL_ADMIT_PCT: u64 = 85;

/// Default DXGI video-memory pressure percentage above which cold,
/// fence-completed small card buffers are demoted back to host RAM.
const DEFAULT_DXGI_PRESSURE_PCT: u64 = 88;

/// Maximum bytes read back from the GPU in a single demotion batch.
const DEMOTE_BATCH_MAX_BYTES: u64 = 16 << 20;

/// Maximum bounded readback batches attempted when making room for a single
/// `>= 1 MiB` buffer (at most `4 * 16 MiB = 64 MiB`).
const DEMOTE_MAX_PASSES: usize = 4;

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Card bytes surrendered back to the texture/streaming VRAM pool by
/// DXGI-pressure demotion (`demote_under_pressure`). Subtracted from
/// `card_room_kept()` alongside `CARD_BYTES[0]` and `CARD_REFUSED[1]` so
/// demoting cold small buffers genuinely lowers `used + card_room_kept()` in
/// `resource::room()` instead of shifting the freed bytes into reserved room.
static CARD_PRESSURE_RELEASED: AtomicU64 = AtomicU64::new(0);

/// Diagnostics counters for card-buffer tiering and demotion:
/// - `[0]`: `card_demoted_count`
/// - `[1]`: `card_demoted_bytes`
/// - `[2]`: `card_reserved_large_rejects`
pub(crate) static CARD_TIER_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Snapshot of card-buffer tiering & demotion diagnostic counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CardTierStats {
    pub(crate) card_demoted_count: u64,
    pub(crate) card_demoted_bytes: u64,
    pub(crate) card_reserved_large_rejects: u64,
}

/// Returns cumulative card-buffer tiering and demotion counters.
pub(crate) fn card_tier_stats() -> CardTierStats {
    CardTierStats {
        card_demoted_count: CARD_TIER_STATS[0].load(Ordering::Relaxed),
        card_demoted_bytes: CARD_TIER_STATS[1].load(Ordering::Relaxed),
        card_reserved_large_rejects: CARD_TIER_STATS[2].load(Ordering::Relaxed),
    }
}

/// Admission outcome for a title buffer requesting card-memory placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CardAdmit {
    Admitted,
    /// Small `< 1 MiB` buffer rejected to preserve reserved headroom for
    /// `>= 1 MiB` buffers (`card_reserved_large_rejects`).
    ReservedForLarge,
    /// Total `VICTORY_CARD_BUFFERS_MB` cap reached.
    CapFull,
    /// DXGI local video-memory budget threshold exceeded.
    DxgiPressure,
    /// Size outside `[least, 1 GiB)` or cap is zero.
    OutOfRange,
}

/// Atomic accounting state for card-resident title buffers, splitting total
/// bytes from `< 1 MiB` small-buffer bytes so small-buffer floods cannot
/// starve `>= 1 MiB` buffers.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CardBudget {
    pub(crate) total_bytes: u64,
    pub(crate) small_bytes: u64,
}

impl CardBudget {
    /// Checks admission policy for a title buffer of `size` bytes and, if
    /// admitted, accounts its bytes immediately.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_admit(
        &mut self,
        size: u64,
        least: u64,
        most: u64,
        reserve: u64,
        video_memory: Option<[u64; 4]>,
        card_room_kept: bool,
        pressure_pct: u64,
    ) -> CardAdmit {
        if most == 0 || size < least || size >= (1 << 30) {
            return CardAdmit::OutOfRange;
        }
        let is_small = size < LARGE_CARD_MIN;
        if let Some([budget, used, ..]) = video_memory
            && budget > 0
        {
            let large_pct = if card_room_kept { 97 } else { 80 };
            let limit_pct = if is_small {
                large_pct.min(pressure_pct).min(DXGI_SMALL_ADMIT_PCT)
            } else {
                large_pct
            };
            let limit_bytes = ((u128::from(budget) * u128::from(limit_pct)) / 100) as u64;
            if used.saturating_add(size) > limit_bytes {
                return CardAdmit::DxgiPressure;
            }
        }
        if is_small && reserve > 0 {
            let small_cap = most.saturating_sub(reserve);
            if self.small_bytes.saturating_add(size) > small_cap {
                return CardAdmit::ReservedForLarge;
            }
        }
        if self.total_bytes.saturating_add(size) > most {
            return CardAdmit::CapFull;
        }
        self.total_bytes += size;
        if is_small {
            self.small_bytes += size;
        }
        CardAdmit::Admitted
    }

    /// Releases `size` previously admitted bytes back to the budget.
    pub(crate) fn release(&mut self, size: u64) {
        self.total_bytes = self.total_bytes.saturating_sub(size);
        if size < LARGE_CARD_MIN {
            self.small_bytes = self.small_bytes.saturating_sub(size);
        }
    }
}

static CARD_BUDGET: Mutex<CardBudget> = Mutex::new(CardBudget {
    total_bytes: 0,
    small_bytes: 0,
});

/// Computes headroom in bytes reserved inside `most` for `>= 1 MiB` buffers
/// whenever `least < 1 MiB` (e.g. `VICTORY_CARD_BUFFERS_MIN_KB=64`).
/// Default: `min(256 MiB, most / 3)`. `override_mb` (`VICTORY_CARD_LARGE_RESERVE_MB`)
/// overrides the reservation (`Some(0)` disables it; clamped to `most`).
pub(crate) fn card_large_reserve_for(least: u64, most: u64, override_mb: Option<u64>) -> u64 {
    if least >= LARGE_CARD_MIN || most == 0 {
        return 0;
    }
    match override_mb {
        Some(mb) => mb.saturating_mul(1 << 20).min(most),
        None => DEFAULT_LARGE_RESERVE_MAX.min(most / 3),
    }
}

/// Active headroom reservation in bytes for `>= 1 MiB` card buffers.
fn card_large_reserve() -> u64 {
    static RESERVE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *RESERVE.get_or_init(|| {
        let (least, most) = card_limits();
        let override_mb = std::env::var("VICTORY_CARD_LARGE_RESERVE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok());
        card_large_reserve_for(least, most, override_mb)
    })
}

/// Whether fence-safe LRU demotion of cold small card buffers is enabled
/// (on by default; `VICTORY_CARD_DEMOTE=0` turns it off).
fn card_demote_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_DEMOTE").is_none_or(|v| v != "0"))
}

/// DXGI video-memory usage percentage above which cold small card buffers are
/// demoted (`VICTORY_CARD_PRESSURE_PCT`, default 88, clamped to 50..=98).
fn card_pressure_pct() -> u64 {
    static PCT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PCT.get_or_init(|| {
        std::env::var("VICTORY_CARD_PRESSURE_PCT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(DEFAULT_DXGI_PRESSURE_PCT, |pct| pct.clamp(50, 98))
    })
}

/// Computes how many bytes of cold `< 1 MiB` card buffers should be demoted
/// under DXGI local video-memory pressure (`video_memory = [budget, used, ..]`).
/// Triggers at `used >= budget * pressure_pct / 100` and aims down to
/// `(pressure_pct - 4)%` (clamped to `small_card_bytes`) to create hysteresis
/// headroom for streaming allocations on 8-12 GB GPUs.
pub(crate) fn dxgi_demote_target_bytes(
    video_memory: Option<[u64; 4]>,
    small_card_bytes: u64,
    pressure_pct: u64,
) -> u64 {
    if small_card_bytes == 0 {
        return 0;
    }
    let Some([budget, used, ..]) = video_memory else {
        return 0;
    };
    if budget == 0 {
        return 0;
    }
    let high_mark = ((u128::from(budget) * u128::from(pressure_pct)) / 100) as u64;
    if used < high_mark {
        return 0;
    }
    let low_pct = pressure_pct.saturating_sub(4);
    let low_mark = ((u128::from(budget) * u128::from(low_pct)) / 100) as u64;
    used.saturating_sub(low_mark).min(small_card_bytes)
}

/// Current card bytes held by `< 1 MiB` title buffers.
fn card_small_bytes() -> u64 {
    CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner()).small_bytes
}

/// Current total card bytes held by title buffers in [`CARD_BUDGET`].
fn card_total_bytes() -> u64 {
    CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner()).total_bytes
}

/// Computes the exact shortfall in bytes that must be freed from existing small
/// card buffers so a `>= 1 MiB` buffer of `size` bytes can pass both the total
/// `most` cap and the DXGI local video-memory budget limit.
pub(crate) fn card_admit_shortfall(
    size: u64,
    total_bytes: u64,
    most: u64,
    video_memory: Option<[u64; 4]>,
    card_room_kept: bool,
) -> u64 {
    let cap_shortfall = total_bytes.saturating_add(size).saturating_sub(most);
    let dxgi_shortfall = match video_memory {
        Some([budget, used, ..]) if budget > 0 => {
            let limit_pct: u64 = if card_room_kept { 97 } else { 80 };
            let limit_bytes = ((u128::from(budget) * u128::from(limit_pct)) / 100) as u64;
            used.saturating_add(size).saturating_sub(limit_bytes)
        }
        _ => 0,
    };
    cap_shortfall.max(dxgi_shortfall)
}

/// Pure calculation for [`card_room_kept`]: bytes reserved for future card
/// copies after subtracting live card bytes, refused `>= 1 MiB` bytes, and
/// headroom explicitly surrendered by DXGI-pressure demotion.
pub(crate) fn card_room_kept_for(
    most: u64,
    card_bytes: u64,
    refused_bytes: u64,
    pressure_released: u64,
) -> u64 {
    most.saturating_sub(
        card_bytes
            .saturating_add(refused_bytes)
            .saturating_add(pressure_released),
    )
}

/// Releases a title buffer's card-side bytes from both [`CARD_BUDGET`] and
/// [`CARD_BYTES`] in lockstep.
fn release_card_bytes(size: u64, had_before: bool) {
    if size > 0 {
        let mut budget = CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
        budget.release(size);
        CARD_BYTES[0].store(budget.total_bytes, Ordering::Relaxed);
    }
    if had_before {
        CARD_BYTES[1].fetch_sub(card_hash_bytes(size), Ordering::Relaxed);
    }
}

/// Attempts to reserve `size` bytes in the global [`CARD_BUDGET`], syncing
/// [`CARD_BYTES`]`[0]` atomically on admission.
fn try_admit_global_card(size: u64, video_memory: Option<[u64; 4]>) -> CardAdmit {
    let (least, most) = card_limits();
    let reserve = card_large_reserve();
    let mut budget = CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    let res = budget.try_admit(
        size,
        least,
        most,
        reserve,
        video_memory,
        card_room_kept_on(),
        card_pressure_pct(),
    );
    if res == CardAdmit::Admitted {
        CARD_BYTES[0].store(budget.total_bytes, Ordering::Relaxed);
    }
    res
}

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
    card_room_kept_for(
        card_limits().1,
        CARD_BYTES[0].load(Ordering::Relaxed),
        CARD_REFUSED[1].load(Ordering::Relaxed),
        CARD_PRESSURE_RELEASED.load(Ordering::Relaxed),
    )
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

/// Records and periodically logs a DXGI video-memory refusal of a `>= 1 MiB`
/// card copy using the exact `limit_pct` threshold that rejected it.
fn note_card_refused(size: u64, video_memory: Option<[u64; 4]>, limit_pct: u64) {
    let (_, most) = card_limits();
    let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
    let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
    if (n <= 64 || n.is_multiple_of(32))
        && let Some([budget, used, ..]) = video_memory
    {
        let limit_bytes = ((u128::from(budget) * u128::from(limit_pct)) / 100) as u64;
        tracing::info!(
            "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
             {} MB ({limit_pct}%); the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
            size >> 10,
            used >> 20,
            budget >> 20,
            limit_bytes >> 20,
            CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
            most >> 20,
            n,
            bytes >> 20,
            crate::frame::current()
        );
    }
}

/// Checks whether `shadow` is a cold, fence-completed small (`< 1 MiB`) title
/// buffer card copy that can safely be demoted back to host RAM:
/// - must be a non-window, non-rebar card-backed copy with a CPU staging side
///   and `card.size < LARGE_CARD_MIN` (hot `>= 1 MiB` buffers and transient
///   windows are never demoted);
/// - `Arc::strong_count == 1` (no command list, `CardPlan`, or lookup cache
///   holds it);
/// - `host::references(&shadow.buffer) == 1` (no descriptor heap slot points
///   to the VRAM buffer);
/// - `pending` is empty (no un-merged shader write range);
/// - `last_fence <= completed_fence` (all GPU submissions touching this buffer
///   have completed on the host queue fence).
fn small_card_demotable(shadow: &Arc<Shadow>, completed_fence: u64) -> Option<(u64, u64, bool)> {
    if shadow.window || shadow.rebar {
        return None;
    }
    let card = shadow.card.as_ref()?;
    if card.staging.is_none() || card.size == 0 || card.size >= LARGE_CARD_MIN {
        return None;
    }
    if Arc::strong_count(shadow) != 1 || crate::host::references(&shadow.buffer) != 1 {
        return None;
    }
    if shadow.last_fence() > completed_fence {
        return None;
    }
    if !shadow.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty() {
        return None;
    }
    let gpu_touched = shadow.last_fence() > 0 || card.gpu_written.load(Ordering::Acquire);
    let needs_readback = if !gpu_touched {
        false
    } else {
        let dirty = joined(card.dirty.lock().unwrap_or_else(|e| e.into_inner()).clone());
        !outside(0, card.size as usize, dirty).is_empty()
    };
    Some((card.size, shadow.last_fence(), needs_readback))
}

/// Demotes a single verified `Entry`'s card-backed `Shadow` back to host RAM in
/// place by reusing `card.staging` (which is already an L0 write-back
/// `ALLOW_UNORDERED_ACCESS` buffer persistently mapped at `shadow.data`).
///
/// If `readback` is `Some(gpu_bytes)`, copies every byte OUTSIDE `card.dirty`
/// from `gpu_bytes` into `card.staging` before dropping the VRAM buffer,
/// preserving both GPU shader writes and pending CPU dirty writes byte-for-byte.
/// Old VRAM resources are pushed onto `to_drop` so COM `Release()` runs outside
/// the table lock.
#[allow(clippy::too_many_arguments)]
fn demote_entry_in_place(
    entry: &mut Entry,
    expected_id: u64,
    expected_fence: Option<u64>,
    completed_fence: u64,
    readback: Option<&[u8]>,
    budget: &mut CardBudget,
    sync_global_bytes: bool,
    to_drop: &mut Vec<ID3D12Resource>,
) -> u64 {
    let Some(shadow) = entry.shadow.as_ref() else {
        return 0;
    };
    if shadow.id != expected_id {
        return 0;
    }
    let Some((_, fence, needs_readback)) = small_card_demotable(shadow, completed_fence) else {
        return 0;
    };
    if let Some(recorded) = expected_fence
        && fence != recorded
    {
        return 0;
    }
    if needs_readback && readback.is_none() {
        return 0;
    }
    let old_arc = entry.shadow.take().unwrap();
    let mut old = match Arc::try_unwrap(old_arc) {
        Ok(inner) => inner,
        Err(shared) => {
            entry.shadow = Some(shared);
            return 0;
        }
    };
    let mut card = old.card.take().expect("verified carded shadow");
    let Some(staging) = card.staging.take() else {
        old.card = Some(card);
        entry.shadow = Some(Arc::new(old));
        return 0;
    };
    let size = card.size;
    if let Some(gpu_bytes) = readback {
        let copy_len = (size as usize).min(gpu_bytes.len());
        let dirty = joined(card.dirty.into_inner().unwrap_or_else(|e| e.into_inner()));
        for (a, b) in outside(0, copy_len, dirty) {
            // SAFETY: `old.data` is the persistent mapping of `card.staging`
            // (allocated for `size` bytes), `b <= copy_len <= size`, and we
            // hold unique ownership of `old` via `Arc::try_unwrap`.
            unsafe {
                std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr().add(a), old.data.add(a), b - a);
            }
        }
    }
    let old_vram = std::mem::replace(&mut old.buffer, staging);
    // SAFETY: `old.buffer` is the live host-RAM staging buffer.
    old.gpu = unsafe { old.buffer.GetGPUVirtualAddress() };
    old.id = next_shadow_id();
    let old_before = card.before.into_inner().unwrap_or_else(|e| e.into_inner());
    let had_before = old_before.is_some();
    budget.release(size);
    if sync_global_bytes {
        CARD_BYTES[0].store(budget.total_bytes, Ordering::Relaxed);
        if had_before {
            CARD_BYTES[1].fetch_sub(card_hash_bytes(size), Ordering::Relaxed);
        }
    }
    CARD_TIER_STATS[0].fetch_add(1, Ordering::Relaxed);
    CARD_TIER_STATS[1].fetch_add(size, Ordering::Relaxed);
    to_drop.push(old_vram);
    if let Some(before) = old_before {
        to_drop.push(before);
    }
    entry.shadow = Some(Arc::new(old));
    size
}

/// Phase 1 (fast, no GPU wait): demotes coldest fence-completed small card
/// buffers that require no GPU readback directly inside `map`, returning
/// `(demoted_count, demoted_bytes)`. Caller must drop `to_drop` after releasing
/// `map`.
fn demote_fast_small_in_map(
    map: &mut BTreeMap<(u64, usize), Entry>,
    target_bytes: u64,
    completed_fence: u64,
    budget: &mut CardBudget,
    sync_global_bytes: bool,
    to_drop: &mut Vec<ID3D12Resource>,
) -> (u64, u64) {
    if target_bytes == 0 {
        return (0, 0);
    }
    let mut candidates: Vec<(u64, u64, (u64, usize), u64)> = map
        .iter()
        .filter_map(|(&key, entry)| {
            let shadow = entry.shadow.as_ref()?;
            let (_, _, needs_readback) = small_card_demotable(shadow, completed_fence)?;
            if needs_readback {
                return None;
            }
            Some((
                shadow.used.load(Ordering::Relaxed),
                shadow.access_tick.load(Ordering::Relaxed),
                key,
                shadow.id,
            ))
        })
        .collect();
    if candidates.is_empty() {
        return (0, 0);
    }
    candidates.sort_unstable_by_key(|&(used, tick, _, _)| (used, tick));
    SHAPE.fetch_add(1, Ordering::SeqCst);
    let (mut count, mut bytes) = (0u64, 0u64);
    for (_, _, key, id) in candidates {
        if bytes >= target_bytes {
            break;
        }
        if let Some(entry) = map.get_mut(&key) {
            let freed = demote_entry_in_place(
                entry,
                id,
                None,
                completed_fence,
                None,
                budget,
                sync_global_bytes,
                to_drop,
            );
            if freed > 0 {
                count += 1;
                bytes += freed;
            }
        }
    }
    (count, bytes)
}

/// Candidate collected for Phase 2 GPU readback outside the table lock.
struct ReadbackCandidate {
    key: (u64, usize),
    id: u64,
    fence: u64,
    size: u64,
}

/// Collects coldest fence-completed small card buffers requiring GPU readback,
/// capped at `target_bytes.min(DEMOTE_BATCH_MAX_BYTES)`.
fn collect_readback_candidates(
    map: &BTreeMap<(u64, usize), Entry>,
    target_bytes: u64,
    completed_fence: u64,
) -> (Vec<ReadbackCandidate>, Vec<(ID3D12Resource, u64)>) {
    if target_bytes == 0 {
        return (Vec::new(), Vec::new());
    }
    let cap = target_bytes.min(DEMOTE_BATCH_MAX_BYTES);
    let mut ranked: Vec<(u64, u64, (u64, usize), u64, u64, u64)> = map
        .iter()
        .filter_map(|(&key, entry)| {
            let shadow = entry.shadow.as_ref()?;
            let (size, fence, needs_readback) = small_card_demotable(shadow, completed_fence)?;
            if !needs_readback {
                return None;
            }
            Some((
                shadow.used.load(Ordering::Relaxed),
                shadow.access_tick.load(Ordering::Relaxed),
                key,
                shadow.id,
                fence,
                size,
            ))
        })
        .collect();
    ranked.sort_unstable_by_key(|&(used, tick, _, _, _, _)| (used, tick));
    let mut meta = Vec::new();
    let mut srcs = Vec::new();
    let mut planned = 0u64;
    for (_, _, key, id, fence, size) in ranked {
        if planned >= cap {
            break;
        }
        if let Some(shadow) = map.get(&key).and_then(|e| e.shadow.as_ref())
            && shadow.id == id
        {
            meta.push(ReadbackCandidate { key, id, fence, size });
            srcs.push((shadow.buffer.clone(), size));
            planned += size;
        }
    }
    (meta, srcs)
}

/// Applies Phase 2 readback bytes into `map` after `srcs` have been dropped
/// (so COM refcounts are restored to 1) and `SHAPE` is bumped.
fn apply_readback_demotions(
    map: &mut BTreeMap<(u64, usize), Entry>,
    meta: &[ReadbackCandidate],
    readbacks: &[Vec<u8>],
    completed_fence: u64,
    budget: &mut CardBudget,
    sync_global_bytes: bool,
    to_drop: &mut Vec<ID3D12Resource>,
) -> (u64, u64) {
    if meta.is_empty() {
        return (0, 0);
    }
    SHAPE.fetch_add(1, Ordering::SeqCst);
    let (mut count, mut bytes) = (0u64, 0u64);
    for (cand, gpu_bytes) in meta.iter().zip(readbacks.iter()) {
        if gpu_bytes.len() < cand.size as usize {
            continue;
        }
        if let Some(entry) = map.get_mut(&cand.key) {
            let freed = demote_entry_in_place(
                entry,
                cand.id,
                Some(cand.fence),
                completed_fence,
                Some(gpu_bytes),
                budget,
                sync_global_bytes,
                to_drop,
            );
            if freed > 0 {
                count += 1;
                bytes += freed;
            }
        }
    }
    (count, bytes)
}

/// Runs both Phase 1 (fast) and Phase 2 (GPU-readback) fence-safe LRU demotion
/// over cold `< 1 MiB` card buffers in `BUFFERS`, never holding `BUFFERS` lock
/// across GPU command execution or COM resource destruction.
fn demote_cold_small_global(target_bytes: u64, host: &crate::host::Host) -> (u64, u64) {
    if target_bytes == 0 || !card_demote_on() {
        return (0, 0);
    }
    let completed = host.completed_fence();
    let mut to_drop = Vec::new();
    let (mut total_count, mut total_bytes) = (0u64, 0u64);
    let (meta, srcs) = {
        let mut map = buffers();
        let (c1, b1) = {
            let mut budget = CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
            demote_fast_small_in_map(&mut map, target_bytes, completed, &mut budget, true, &mut to_drop)
        };
        total_count += c1;
        total_bytes += b1;
        let rem = target_bytes.saturating_sub(total_bytes);
        collect_readback_candidates(&map, rem, completed)
    };
    drop(to_drop);
    if meta.is_empty() {
        return (total_count, total_bytes);
    }
    let readbacks = host.demote_readback(&srcs);
    drop(srcs);
    let Some(readbacks) = readbacks else {
        return (total_count, total_bytes);
    };
    let mut to_drop2 = Vec::new();
    {
        let mut map = buffers();
        let completed_now = host.completed_fence();
        let mut budget = CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
        let (c2, b2) = apply_readback_demotions(
            &mut map,
            &meta,
            &readbacks,
            completed_now,
            &mut budget,
            true,
            &mut to_drop2,
        );
        total_count += c2;
        total_bytes += b2;
    }
    drop(to_drop2);
    if total_count > 0 {
        crate::host::trail(|| {
            format!(
                "card demotion: {total_count} cold small card buffers ({} KiB) demoted to host RAM",
                total_bytes >> 10
            )
        });
    }
    (total_count, total_bytes)
}

/// Called periodically from `Host::sweep`: when DXGI local video-memory usage
/// exceeds the pressure threshold (default 88%), safely demotes cold,
/// fence-completed `< 1 MiB` card buffers back to host RAM to relieve VRAM
/// pressure on 8-12 GB cards without disturbing hot `>= 1 MiB` buffers.
pub(crate) fn demote_under_pressure(video_memory: Option<[u64; 4]>) -> (u64, u64) {
    if !cards() || !card_demote_on() {
        return (0, 0);
    }
    let target = dxgi_demote_target_bytes(video_memory, card_small_bytes(), card_pressure_pct());
    if target == 0 {
        return (0, 0);
    }
    let Some(host) = host() else {
        return (0, 0);
    };
    let (count, bytes) = demote_cold_small_global(target, host);
    if bytes > 0 {
        // Surrender this headroom from `card_room_kept()` so `used + card_room_kept()`
        // in `resource::room()` genuinely drops by `bytes` for streaming textures.
        CARD_PRESSURE_RELEASED.fetch_add(bytes, Ordering::Relaxed);
    }
    (count, bytes)
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
                if relay() {
                    let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.retain(|s| s.2 > fetched);
                    sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
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
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
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

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
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
            let had_before = card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some();
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
                if had_before {
                    CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
                }
            } else if card.counted {
                release_card_bytes(card.size, had_before);
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
        let pages = copied.iter().filter(|&&at| at != 0).count() as u64;
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

/// Allocates a unique monotonic copy ID (re-used when a shadow is demoted so
/// caches keyed by `Shadow::id` do not alias the old VRAM resource).
fn next_shadow_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Monotonic access tick for deterministic LRU ordering of card-backed copies.
fn next_access_tick() -> u64 {
    static TICK: AtomicU64 = AtomicU64::new(1);
    TICK.fetch_add(1, Ordering::Relaxed)
}

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

    /// Records a host access (lookup, refresh, or shader write) for LRU
    /// ordering.
    pub(crate) fn note_access(&self) {
        self.used.store(crate::host::moment(), Ordering::Relaxed);
        self.access_tick.store(next_access_tick(), Ordering::Relaxed);
    }

    /// Stamps the host queue fence value signalled behind a submission that
    /// uses this copy, preventing demotion while the GPU flight is in air.
    pub(crate) fn note_gpu_fence(&self, fence: u64) {
        self.last_fence.fetch_max(fence, Ordering::Release);
    }

    /// Latest host queue fence value recorded for GPU work on this copy.
    pub(crate) fn last_fence(&self) -> u64 {
        self.last_fence.load(Ordering::Acquire)
    }

    /// Notes that a shader/UAV may write directly to this copy's card buffer.
    pub(crate) fn note_gpu_written(&self) {
        if let Some(card) = &self.card {
            card.gpu_written.store(true, Ordering::Release);
        }
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.note_access();
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: Mutex::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            access_tick: AtomicU64::new(next_access_tick()),
            last_fence: AtomicU64::new(0),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: next_shadow_id(),
            card: None,
            window: false,
            written_said: std::sync::atomic::AtomicBool::new(false),
            rebar: false,
            place: None,
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
        if from >= to || copied.iter().all(|&at| at == 0) {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        match crate::watch::asked(from, to) {
            Some((_, written)) => written
                .into_iter()
                .any(|(page, write)| page >= base && copied.get(page - base).is_none_or(|&at| at == 0 || write > at)),
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
        self.note_access();
        self.note_gpu_written();
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
        if let Some(card) = &self.card {
            card.gpu_written.store(true, Ordering::Release);
            pending.push(range);
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
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
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
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
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
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
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
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => !stale_pages_in(base, &copied, a, e, &written).is_empty(),
            None => true,
        })
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
        let mut title = vec![0u8; PAGE];
        for (a, e) in busy {
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards. The journal alone said nothing for copies
            // of view memory, which is guarded and not write-watched, and
            // those were all of run 99's cases: the flights were settled
            // 4-5 times a frame for "0 pages looked at".
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                MERGED[4].fetch_add(1, Ordering::Relaxed);
                return true;
            };
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_pages_in(base, &copied, a, e, &written)
            };
            for page in stale {
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                if lo >= hi {
                    continue;
                }
                let n = hi - lo;
                let title = &mut title[..n];
                // The lock of the copy's reference (the snapshot, or the
                // CPU's side of a copy in the card's memory), from before
                // title memory is read until the title's bytes are in: with
                // the copy-back on its own thread ([`relay`]) that thread
                // puts what the card wrote into title memory and into the
                // reference under the same lock, and a page read before
                // such a merge and compared after it would take the card's
                // bytes for the title's.
                let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
                // A page that is not there is left as it is, and looked at
                // again next time.
                if !crate::host::read_into(lo as *const u8, title) {
                    continue;
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
                    if let Some(at) = copied.get_mut(page - base) {
                        *at = now;
                    }
                }
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        self.note_access();
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
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
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
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
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        to - from
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
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
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
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run, mut bytes_runs) = (0, None::<(usize, usize)>, Vec::new());
        let last = (to - 1) / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let four_k = crate::watch::watch_4k();
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // (Neptune) With VICTORY_WATCH_4K=1: only the 4 KiB pages written since.
                let mask = crate::watch::stale_mask(page, *at);
                if four_k {
                    crate::watch::note_4k(mask);
                }
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                extend_runs(&mut run, &mut bytes_runs, page, mask.unwrap_or(u16::MAX));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        bytes_runs.extend(run);
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
        card.gpu_written.store(true, Ordering::Release);
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
            let mut differs = 0u8;
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
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
            if let Some(held) = page.checked_sub(base).and_then(|i| copied.get_mut(i))
                && *held != 0
                && before <= *held
            {
                *held = now;
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
        // Still in use (what lets idle copies go and card LRU demotion look at this).
        shadow.note_access();
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
    buffers().remove(&(start, resource));
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
                    shadow.note_access();
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
    if let Some(existing) = map.get(&key)?.shadow.as_ref() {
        existing.note_access();
        return Some(existing.clone());
    }
    let entry_end = map.get(&key)?.end;
    let mut buf_len = entry_end - key.0;
    // Rounded up so that views ending in the buffer's last 256 bytes fit.
    // Shaders write some buffers (unordered access), so not upload memory.
    let mut size = buf_len.next_multiple_of(1 << 16);
    // In place, if asked for and the memory can be opened
    // (`crate::inplace::place_for`): the buffer is the title's own
    // memory. Not the big ones `cards` puts in the card's memory: those
    // are read too often to stay in the PC's.
    let mut big = cards() && size >= card_limits().0;
    // Folded (`VICTORY_FOLD_BUFFERS=1`): a small buffer inside one
    // window cell takes the cell's window as its copy, so that the
    // buffer and the memory around it, which the title also reaches by
    // address, are one copy. Two copies of the same bytes made every
    // submission that refreshed one while a flight's shaders wrote the
    // other wait for the card to drain: ~2.8 times a frame in the
    // world, for 10-40 KB each (a 6.4 KB ring and a 465 KB buffer
    // inside 4 MiB windows). One copy's own pending writes are left
    // alone by its refresh and need no such wait.
    if !big && fold_buffers() && !fine() && key.0 & !(WINDOW - 1) == (entry_end - 1) & !(WINDOW - 1) {
        // (Neptune) Not into a transient window: shaders write a buffer, and such a window
        // has no CPU side to merge into (`window_upload`).
        let window = loose(key.0, entry_end, false, false).filter(|w| {
            let refuse = w.transient() || w.rebar;
            if w.rebar {
                REBAR[4].fetch_add(1, Ordering::Relaxed);
            } else if refuse && TRANSIENT[1].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(start = format_args!("{:#x}", key.0), len = buf_len, "a buffer is not folded into a transient window: it has a copy of its own");
            }
            !refuse
        });
        if let Some(window) = window {
            if FOLDED.fetch_add(1, Ordering::Relaxed) < 24 {
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len = buf_len,
                    "a buffer's copy is its window's (VICTORY_FOLD_BUFFERS)"
                );
            }
            map.get_mut(&key)?.shadow = Some(window.clone());
            return Some(window);
        }
    }
    if !big && let Some(placed) = crate::inplace::place_for(key.0, buf_len, false) {
        let shadow = Arc::new(Shadow::placed(placed, key.0, buf_len, false));
        map.get_mut(&key)?.shadow = Some(shadow.clone());
        return Some(shadow);
    }
    let host = host()?;
    // Decide card placement with size-tiered headroom reservation and, for
    // hot >= 1 MiB buffers, bounded fence-safe LRU demotion of cold small card
    // buffers when the exact shortfall is satisfiable.
    let mut admitted_card = false;
    let mut to_drop: Vec<ID3D12Resource> = Vec::new();
    if big && size < (1 << 30) && host.card_ready() {
        let mut vm = host.video_memory();
        let mut res = try_admit_global_card(size, vm);
        if matches!(res, CardAdmit::CapFull | CardAdmit::DxgiPressure)
            && size >= LARGE_CARD_MIN
            && card_demote_on()
        {
            let (_, most) = card_limits();
            let max_reclaimable = (DEMOTE_MAX_PASSES as u64) * DEMOTE_BATCH_MAX_BYTES;
            for _ in 0..DEMOTE_MAX_PASSES {
                let shortfall = card_admit_shortfall(size, card_total_bytes(), most, vm, card_room_kept_on());
                if shortfall == 0
                    || shortfall > max_reclaimable
                    || card_small_bytes() < shortfall
                {
                    break;
                }
                let completed = host.completed_fence();
                let (_, freed1) = {
                    let mut budget = CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
                    demote_fast_small_in_map(&mut map, shortfall, completed, &mut budget, true, &mut to_drop)
                };
                let rem = shortfall.saturating_sub(freed1);
                let (meta, srcs) = if rem > 0 {
                    collect_readback_candidates(&map, rem, completed)
                } else {
                    (Vec::new(), Vec::new())
                };
                if freed1 == 0 && meta.is_empty() {
                    break;
                }
                // Drop `map` before releasing VRAM resources or running a GPU
                // readback wait so other threads are never blocked and DXGI
                // immediately observes the freed VRAM.
                drop(map);
                drop(to_drop);
                to_drop = Vec::new();
                let mut freed2 = 0u64;
                if !meta.is_empty() {
                    let readbacks = host.demote_readback(&srcs);
                    drop(srcs);
                    if let Some(readbacks) = readbacks {
                        let mut to_drop2 = Vec::new();
                        {
                            let mut m = buffers();
                            let completed_now = host.completed_fence();
                            let mut budget = CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
                            let (_, b2) = apply_readback_demotions(
                                &mut m,
                                &meta,
                                &readbacks,
                                completed_now,
                                &mut budget,
                                true,
                                &mut to_drop2,
                            );
                            freed2 = b2;
                        }
                        drop(to_drop2);
                    }
                }
                map = buffers();
                // Re-validate `entry` after re-acquiring `map`: another thread
                // may have created its shadow, unregistered it, or re-registered
                // it with a different length.
                let entry = map.get(&key)?;
                if entry.end < end {
                    return None;
                }
                if let Some(existing) = entry.shadow.as_ref() {
                    existing.note_access();
                    return Some(existing.clone());
                }
                buf_len = entry.end - key.0;
                size = buf_len.next_multiple_of(1 << 16);
                big = cards() && size >= card_limits().0;
                if !big || size >= (1 << 30) {
                    res = CardAdmit::OutOfRange;
                    break;
                }
                vm = host.video_memory();
                res = try_admit_global_card(size, vm);
                if res == CardAdmit::Admitted || size < LARGE_CARD_MIN || (freed1 == 0 && freed2 == 0) {
                    break;
                }
            }
        }
        match res {
            CardAdmit::Admitted => admitted_card = true,
            CardAdmit::ReservedForLarge => {
                CARD_TIER_STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            CardAdmit::DxgiPressure => {
                // Only `>= 1 MiB` rejections reduce `card_room_kept()` via
                // `CARD_REFUSED[1]`; `< 1 MiB` rejections at `DXGI_SMALL_ADMIT_PCT`
                // must NOT collapse `card_room_kept()` and starve large buffers.
                if size >= LARGE_CARD_MIN {
                    let large_pct = if card_room_kept_on() { 97 } else { 80 };
                    note_card_refused(size, vm, large_pct);
                }
            }
            CardAdmit::CapFull | CardAdmit::OutOfRange => {}
        }
    }
    let Some((buffer, data)) = host.shared_buffer(size) else {
        if admitted_card {
            release_card_bytes(size, false);
        }
        drop(map);
        drop(to_drop);
        return None;
    };
    crate::host::name(&buffer, || format!("title buffer {:#x}+{buf_len:#x}", key.0));
    // A big one, with `cards`: shaders get a buffer in the card's own
    // memory, and the one just made is the CPU's side.
    let (buffer, card) = if admitted_card {
        match host.uav_buffer(size) {
            Some(on_card) => {
                crate::host::name(&on_card, || format!("title buffer {:#x}+{buf_len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len = buf_len,
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
                    sent: Mutex::default(),
                    size,
                    gpu_written: AtomicBool::new(false),
                    counted: true,
                };
                (on_card, Some(card))
            }
            None => {
                release_card_bytes(size, false);
                (buffer, None)
            }
        }
    } else {
        (buffer, None)
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    if buf_len >= 16 << 20 {
        tracing::info!(
            start = format_args!("{:#x}", key.0),
            len = buf_len,
            "large buffer copied to the host"
        );
    } else {
        tracing::debug!(
            start = format_args!("{:#x}", key.0),
            len = buf_len,
            "buffer copied to the host"
        );
    }
    let mut shadow = Shadow::new(buffer, data, gpu, key.0, buf_len);
    shadow.card = card;
    let shadow = Arc::new(shadow);
    note_carded(&shadow);
    map.get_mut(&key)?.shadow = Some(shadow.clone());
    drop(map);
    drop(to_drop);
    Some(shadow)
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
            sent: Mutex::default(),
            size: len,
            gpu_written: AtomicBool::new(false),
            counted: false,
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
                    sent: Mutex::default(),
                    size: len,
                    gpu_written: AtomicBool::new(false),
                    counted: false,
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
                sent: Mutex::new(vec![(title_byte, title_byte + 1, 7)]),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
                gpu_written: std::sync::atomic::AtomicBool::new(false),
                counted: false,
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
                sent: Mutex::default(),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
                gpu_written: std::sync::atomic::AtomicBool::new(false),
                counted: false,
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

#[cfg(test)]
mod vram_tier_tests {
    use super::*;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;

    const WRITE_PATTERN_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), UAV(u0)"
cbuffer Params : register(b0) { uint byte_offset; uint count; uint base_val; };
RWByteAddressBuffer buf : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < count) {
        buf.Store(byte_offset + tid.x * 4u, base_val + tid.x);
    }
}
"#;

    struct WarpRig {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        event: HANDLE,
        signal: u64,
        readback_alloc: ID3D12CommandAllocator,
        readback_list: ID3D12GraphicsCommandList,
        readback_fence: ID3D12Fence,
        readback_event: HANDLE,
        readback_signal: u64,
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
    }

    impl Drop for WarpRig {
        fn drop(&mut self) {
            // SAFETY: handles created by `CreateEventW` in `WarpRig::new`.
            unsafe {
                let _ = CloseHandle(self.event);
                let _ = CloseHandle(self.readback_event);
            }
        }
    }

    impl WarpRig {
        fn new() -> Option<Self> {
            let adapter = crate::host::warp_adapter().ok()?;
            let mut device: Option<ID3D12Device> = None;
            // SAFETY: creating a software WARP device and standard D3D12 objects.
            unsafe {
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
                list.Close().ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let readback_alloc: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let readback_list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &readback_alloc, None)
                    .ok()?;
                readback_list.Close().ok()?;
                let readback_fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let readback_event = CreateEventW(None, false, false, None).ok()?;

                let code = crate::host::compile(
                    WRITE_PATTERN_CS,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                )
                .ok()?;
                let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).ok()?;
                let pso: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    event,
                    signal: 0,
                    readback_alloc,
                    readback_list,
                    readback_fence,
                    readback_event,
                    readback_signal: 0,
                    root,
                    pso,
                })
            }
        }

        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(1),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buf: Option<ID3D12Resource> = None;
            let mut ptr = std::ptr::null_mut();
            // SAFETY: valid D3D12 device and buffer description.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buf,
                    )
                    .expect("shared_buffer");
                let buf = buf.unwrap();
                buf.Map(0, None, Some(&mut ptr)).expect("map shared_buffer");
                (buf, ptr.cast())
            }
        }

        fn default_uav_buffer(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(1),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buf: Option<ID3D12Resource> = None;
            // SAFETY: valid D3D12 device and buffer description.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buf,
                    )
                    .expect("default_uav_buffer");
                buf.unwrap()
            }
        }

        fn make_shadow(&self, start: u64, size: u64, on_card: bool) -> Arc<Shadow> {
            let (staging, data) = self.shared_buffer(size);
            let (buffer, card) = if on_card {
                let card_buf = self.default_uav_buffer(size);
                let card = Card {
                    staging: Some(staging),
                    chunks: Mutex::default(),
                    prints: Mutex::default(),
                    flagged: AtomicBool::new(false),
                    me: std::sync::OnceLock::new(),
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                    size,
                    gpu_written: AtomicBool::new(false),
                    counted: false,
                };
                (card_buf, Some(card))
            } else {
                (staging, None)
            };
            // SAFETY: live D3D12 buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let mut shadow = Shadow::new(buffer, data, gpu, start, size);
            shadow.card = card;
            Arc::new(shadow)
        }

        fn completed_fence(&self) -> u64 {
            // SAFETY: plain fence query.
            unsafe { self.fence.GetCompletedValue() }
        }

        fn run_and_wait(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> u64 {
            self.signal += 1;
            let target = self.signal;
            // SAFETY: allocator/list are idle between waits.
            unsafe {
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
                record(&self.list);
                self.list.Close().unwrap();
                let lists = [Some(self.list.cast::<ID3D12CommandList>().unwrap())];
                self.queue.ExecuteCommandLists(&lists);
                self.queue.Signal(&self.fence, target).unwrap();
                self.fence.SetEventOnCompletion(target, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
            }
            target
        }

        fn upload_staging_to_card(&mut self, shadow: &Shadow) -> u64 {
            let card = shadow.card.as_ref().expect("carded shadow");
            let staging = card.staging.as_ref().expect("staging buffer");
            let len = card.size;
            let fence = self.run_and_wait(|list| unsafe {
                list.CopyBufferRegion(&shadow.buffer, 0, staging, 0, len);
            });
            card.dirty.lock().unwrap().clear();
            shadow.note_gpu_fence(fence);
            fence
        }

        fn dispatch_uav_write(&mut self, shadow: &Shadow, byte_offset: u32, count: u32, base_val: u32) -> u64 {
            let va = unsafe { shadow.buffer.GetGPUVirtualAddress() };
            let root = self.root.clone();
            let pso = self.pso.clone();
            let fence = self.run_and_wait(|list| unsafe {
                list.SetComputeRootSignature(&root);
                list.SetPipelineState(&pso);
                let constants = [byte_offset, count, base_val];
                list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
                list.SetComputeRootUnorderedAccessView(1, va);
                list.Dispatch(count.div_ceil(64), 1, 1);
            });
            shadow.note_gpu_fence(fence);
            fence
        }

        fn read_gpu_buffer(&mut self, src: &ID3D12Resource, len: u64) -> Vec<u8> {
            let (rb, ptr) = self.shared_buffer(len);
            self.run_and_wait(|list| unsafe {
                list.CopyBufferRegion(&rb, 0, src, 0, len);
            });
            // SAFETY: copy finished and `rb` is persistently mapped at `ptr`.
            unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec()
        }

        fn demote_readback(&mut self, srcs: &[(ID3D12Resource, u64)]) -> Option<Vec<Vec<u8>>> {
            let total: u64 = srcs.iter().map(|&(_, l)| l).sum();
            if total == 0 {
                return Some(Vec::new());
            }
            let (staging, ptr) = self.shared_buffer(total.next_multiple_of(64 * KIB));
            self.readback_signal += 1;
            crate::host::demote_readback_run(
                &self.readback_alloc,
                &self.readback_list,
                &self.queue,
                &self.readback_fence,
                self.readback_event,
                self.readback_signal,
                &staging,
                ptr,
                srcs,
                None,
                None,
            )
        }
    }

    #[test]
    fn small_buffer_flood_cannot_starve_large_buffers_with_tiered_reserve() {
        assert_eq!(card_large_reserve_for(64 * KIB, 1024 * MIB, None), 256 * MIB);
        assert_eq!(card_large_reserve_for(64 * KIB, 512 * MIB, None), (512 * MIB) / 3);
        assert_eq!(card_large_reserve_for(MIB, 1024 * MIB, None), 0);
        assert_eq!(card_large_reserve_for(64 * KIB, 1024 * MIB, Some(0)), 0);
        assert_eq!(card_large_reserve_for(64 * KIB, 1024 * MIB, Some(2048)), 1024 * MIB);

        let (least, most) = (64 * KIB, 1024 * MIB);
        let reserve = card_large_reserve_for(least, most, None);
        let mut budget = CardBudget::default();
        let mut small_admitted = 0u64;
        let mut reserved_rejects = 0u64;
        for _ in 0..16_384 {
            match budget.try_admit(64 * KIB, least, most, reserve, None, true, DEFAULT_DXGI_PRESSURE_PCT) {
                CardAdmit::Admitted => small_admitted += 1,
                CardAdmit::ReservedForLarge => reserved_rejects += 1,
                other => panic!("unexpected outcome during small flood: {other:?}"),
            }
        }
        assert_eq!(small_admitted * 64 * KIB, 768 * MIB);
        assert_eq!(reserved_rejects, 4_096);
        assert_eq!(budget.small_bytes, 768 * MIB);
        assert_eq!(budget.total_bytes, 768 * MIB);

        for i in 0..16 {
            assert_eq!(
                budget.try_admit(16 * MIB, least, most, reserve, None, true, DEFAULT_DXGI_PRESSURE_PCT),
                CardAdmit::Admitted,
                "large buffer {i} must not be starved by small-buffer flood"
            );
        }
        assert_eq!(budget.total_bytes, 1024 * MIB);
        assert_eq!(budget.small_bytes, 768 * MIB);
        assert_eq!(
            budget.try_admit(16 * MIB, least, most, reserve, None, true, DEFAULT_DXGI_PRESSURE_PCT),
            CardAdmit::CapFull
        );

        let mut b2 = CardBudget::default();
        assert_eq!(b2.try_admit(800 * MIB, least, most, reserve, None, true, DEFAULT_DXGI_PRESSURE_PCT), CardAdmit::Admitted);
        for _ in 0..(224 * 16) {
            assert_eq!(b2.try_admit(64 * KIB, least, most, reserve, None, true, DEFAULT_DXGI_PRESSURE_PCT), CardAdmit::Admitted);
        }
        assert_eq!(b2.total_bytes, 1024 * MIB);
        assert_eq!(b2.try_admit(64 * KIB, least, most, reserve, None, true, DEFAULT_DXGI_PRESSURE_PCT), CardAdmit::CapFull);
    }

    #[test]
    fn dxgi_pressure_target_reclaims_headroom_on_8gb_and_12gb_cards() {
        for (label, budget_mb) in [("8GB", 7_200u64), ("12GB", 11_200u64)] {
            let vram = budget_mb * MIB;
            let small_resident = 512 * MIB;

            assert_eq!(
                dxgi_demote_target_bytes(Some([vram, vram * 80 / 100, 0, 0]), small_resident, 88),
                0,
                "{label} at 80%"
            );
            assert_eq!(
                dxgi_demote_target_bytes(Some([vram, vram * 87 / 100, 0, 0]), small_resident, 88),
                0,
                "{label} at 87%"
            );
            let at_88 = dxgi_demote_target_bytes(Some([vram, vram * 88 / 100, 0, 0]), small_resident, 88);
            assert_eq!(at_88, (vram * 4 / 100).min(small_resident), "{label} at 88%");
            let at_92 = dxgi_demote_target_bytes(Some([vram, vram * 92 / 100, 0, 0]), small_resident, 88);
            assert_eq!(at_92, (vram * 8 / 100).min(small_resident), "{label} at 92%");

            let mut cb = CardBudget::default();
            let vm_86 = Some([vram, vram * 86 / 100, 0, 0]);
            assert_eq!(
                cb.try_admit(64 * KIB, 64 * KIB, 1024 * MIB, 256 * MIB, vm_86, true, 88),
                CardAdmit::DxgiPressure,
                "{label}: small buffer must not admit at 86% DXGI usage"
            );
            assert_eq!(
                cb.try_admit(16 * MIB, 64 * KIB, 1024 * MIB, 256 * MIB, vm_86, true, 88),
                CardAdmit::Admitted,
                "{label}: hot >=1 MiB buffer still admits at 86% DXGI usage"
            );
        }
    }

    #[test]
    fn warp_small_buffer_flood_reserves_headroom_and_demotes_for_large_buffers() {
        let Some(rig) = WarpRig::new() else {
            eprintln!("vram_tier_tests: WARP adapter unavailable, skipping");
            return;
        };
        let (least, most) = (64 * KIB, 3 * MIB);
        let reserve = card_large_reserve_for(least, most, None);
        assert_eq!(reserve, MIB);

        let mut map: BTreeMap<(u64, usize), Entry> = BTreeMap::new();
        let mut budget = CardBudget::default();
        let before_stats = card_tier_stats();

        let mut local_reserved_rejects = 0u64;
        for i in 0..48u64 {
            let start = 0x1000_0000 + i * (64 * KIB);
            let admit = budget.try_admit(64 * KIB, least, most, reserve, None, true, 88);
            let on_card = match admit {
                CardAdmit::Admitted => true,
                CardAdmit::ReservedForLarge => {
                    local_reserved_rejects += 1;
                    CARD_TIER_STATS[2].fetch_add(1, Ordering::Relaxed);
                    false
                }
                other => panic!("unexpected small admit #{i}: {other:?}"),
            };
            let shadow = rig.make_shadow(start, 64 * KIB, on_card);
            map.insert(
                (start, i as usize + 1),
                Entry {
                    end: start + 64 * KIB,
                    shadow: Some(shadow),
                },
            );
        }
        assert_eq!(local_reserved_rejects, 16);
        assert_eq!(budget.small_bytes, 2 * MIB);
        assert_eq!(budget.total_bytes, 2 * MIB);
        assert!(card_tier_stats().card_reserved_large_rejects >= before_stats.card_reserved_large_rejects + 16);

        for i in 0..16u64 {
            let key = (0x1000_0000 + i * (64 * KIB), i as usize + 1);
            map.get(&key).unwrap().shadow.as_ref().unwrap().note_access();
        }

        let large1_start = 0x2000_0000u64;
        assert_eq!(budget.try_admit(MIB, least, most, reserve, None, true, 88), CardAdmit::Admitted);
        map.insert(
            (large1_start, 1001),
            Entry {
                end: large1_start + MIB,
                shadow: Some(rig.make_shadow(large1_start, MIB, true)),
            },
        );
        assert_eq!(budget.total_bytes, 3 * MIB);

        let large2_start = 0x2100_0000u64;
        assert_eq!(budget.try_admit(MIB, least, most, reserve, None, true, 88), CardAdmit::CapFull);
        let mut to_drop = Vec::new();
        let (demoted_n, demoted_b) =
            demote_fast_small_in_map(&mut map, MIB, rig.completed_fence(), &mut budget, false, &mut to_drop);
        drop(to_drop);
        assert_eq!(demoted_n, 16);
        assert_eq!(demoted_b, MIB);
        assert_eq!(budget.small_bytes, MIB);
        assert_eq!(budget.total_bytes, 2 * MIB);

        for i in 0..16u64 {
            let key = (0x1000_0000 + i * (64 * KIB), i as usize + 1);
            assert!(map.get(&key).unwrap().shadow.as_ref().unwrap().carded());
        }
        for i in 16..32u64 {
            let key = (0x1000_0000 + i * (64 * KIB), i as usize + 1);
            assert!(!map.get(&key).unwrap().shadow.as_ref().unwrap().carded());
        }

        assert_eq!(budget.try_admit(MIB, least, most, reserve, None, true, 88), CardAdmit::Admitted);
        map.insert(
            (large2_start, 1002),
            Entry {
                end: large2_start + MIB,
                shadow: Some(rig.make_shadow(large2_start, MIB, true)),
            },
        );
        assert!(map.get(&(large1_start, 1001)).unwrap().shadow.as_ref().unwrap().carded());
        assert!(map.get(&(large2_start, 1002)).unwrap().shadow.as_ref().unwrap().carded());
    }

    #[test]
    fn warp_demotion_preserves_exact_bytes_across_gpu_writes_and_cpu_dirty_ranges() {
        let Some(mut rig) = WarpRig::new() else {
            eprintln!("vram_tier_tests: WARP adapter unavailable, skipping");
            return;
        };
        const LEN: u64 = 64 * KIB;
        let key = (0x3000_0000u64, 3001usize);
        let shadow = rig.make_shadow(key.0, LEN, true);
        let old_id = shadow.id();

        let mut expected = vec![0u32; (LEN / 4) as usize];
        for (i, word) in expected.iter_mut().enumerate() {
            *word = 0x5A00_0000 | (i as u32);
        }
        // SAFETY: `shadow.data` maps `LEN` bytes of `card.staging`.
        unsafe {
            std::ptr::copy_nonoverlapping(expected.as_ptr().cast::<u8>(), shadow.data, LEN as usize);
        }
        let f1 = rig.upload_staging_to_card(&shadow);
        assert!(rig.completed_fence() >= f1);

        const GPU_OFF: usize = 0x2000;
        const GPU_WORDS: usize = 256;
        const GPU_BASE: u32 = 0xCAFE_1000;
        let f2 = rig.dispatch_uav_write(&shadow, GPU_OFF as u32, GPU_WORDS as u32, GPU_BASE);
        assert!(rig.completed_fence() >= f2);
        for i in 0..GPU_WORDS {
            expected[GPU_OFF / 4 + i] = GPU_BASE + i as u32;
        }

        const CPU_OFF: usize = 0x8000;
        const CPU_WORDS: usize = 256;
        const CPU_BASE: u32 = 0xBEEF_2000;
        for i in 0..CPU_WORDS {
            let val = CPU_BASE + i as u32;
            expected[CPU_OFF / 4 + i] = val;
            // SAFETY: inside `LEN`-byte staging mapping.
            unsafe {
                shadow.data.add(CPU_OFF + i * 4).cast::<u32>().write(val);
            }
        }
        shadow.card_changed(CPU_OFF, CPU_OFF + CPU_WORDS * 4);

        let mut map: BTreeMap<(u64, usize), Entry> = BTreeMap::new();
        map.insert(
            key,
            Entry {
                end: key.0 + LEN,
                shadow: Some(shadow),
            },
        );
        let mut budget = CardBudget {
            total_bytes: LEN,
            small_bytes: LEN,
        };
        let stats_before = card_tier_stats();

        let mut to_drop = Vec::new();
        let (c1, b1) = demote_fast_small_in_map(&mut map, LEN, rig.completed_fence(), &mut budget, false, &mut to_drop);
        assert_eq!((c1, b1), (0, 0));

        let (meta, srcs) = collect_readback_candidates(&map, LEN, rig.completed_fence());
        assert_eq!(meta.len(), 1);
        let readbacks = rig.demote_readback(&srcs).expect("WARP demote readback");
        drop(srcs);
        let (c2, b2) = apply_readback_demotions(
            &mut map,
            &meta,
            &readbacks,
            rig.completed_fence(),
            &mut budget,
            false,
            &mut to_drop,
        );
        drop(to_drop);

        assert_eq!((c2, b2), (1, LEN));
        assert_eq!(budget, CardBudget { total_bytes: 0, small_bytes: 0 });
        let stats_after = card_tier_stats();
        assert!(stats_after.card_demoted_count > stats_before.card_demoted_count);
        assert!(stats_after.card_demoted_bytes >= stats_before.card_demoted_bytes + LEN);

        let demoted = map.get(&key).unwrap().shadow.as_ref().unwrap();
        assert!(!demoted.carded());
        assert_ne!(demoted.id(), old_id);

        // SAFETY: `demoted.data` maps `LEN` bytes of the now-primary host buffer.
        let cpu_words = unsafe { std::slice::from_raw_parts(demoted.data.cast::<u32>(), (LEN / 4) as usize) };
        assert_eq!(cpu_words, expected.as_slice());

        let gpu_bytes = rig.read_gpu_buffer(&demoted.buffer, LEN);
        let gpu_words: Vec<u32> = gpu_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(gpu_words, expected);
    }

    #[test]
    fn warp_demotion_never_evicts_inflight_or_referenced_card_buffers() {
        let Some(mut rig) = WarpRig::new() else {
            eprintln!("vram_tier_tests: WARP adapter unavailable, skipping");
            return;
        };
        const LEN: u64 = 64 * KIB;
        let key = (0x4000_0000u64, 4001usize);
        let shadow = rig.make_shadow(key.0, LEN, true);
        let f_done = rig.dispatch_uav_write(&shadow, 0x100, 16, 0x7777_0000);
        shadow.note_gpu_fence(f_done + 1);

        let mut map: BTreeMap<(u64, usize), Entry> = BTreeMap::new();
        map.insert(
            key,
            Entry {
                end: key.0 + LEN,
                shadow: Some(shadow),
            },
        );

        let (meta, _) = collect_readback_candidates(&map, LEN, f_done);
        assert!(meta.is_empty());

        let held_arc = map.get(&key).unwrap().shadow.clone().unwrap();
        let (meta, _) = collect_readback_candidates(&map, LEN, f_done + 1);
        assert!(meta.is_empty());
        drop(held_arc);

        let held_com = map.get(&key).unwrap().shadow.as_ref().unwrap().buffer.clone();
        let (meta, _) = collect_readback_candidates(&map, LEN, f_done + 1);
        assert!(meta.is_empty());
        drop(held_com);

        map.get(&key)
            .unwrap()
            .shadow
            .as_ref()
            .unwrap()
            .pending
            .lock()
            .unwrap()
            .push((key.0 as usize, (key.0 + LEN) as usize));
        let (meta, _) = collect_readback_candidates(&map, LEN, f_done + 1);
        assert!(meta.is_empty());
        map.get(&key).unwrap().shadow.as_ref().unwrap().pending.lock().unwrap().clear();

        let completed = f_done + 1;
        let (meta, srcs) = collect_readback_candidates(&map, LEN, completed);
        assert_eq!(meta.len(), 1);
        let readbacks = rig.demote_readback(&srcs).expect("WARP readback");
        drop(srcs);
        let mut budget = CardBudget {
            total_bytes: LEN,
            small_bytes: LEN,
        };
        let mut to_drop = Vec::new();
        let (c, b) = apply_readback_demotions(&mut map, &meta, &readbacks, completed, &mut budget, false, &mut to_drop);
        drop(to_drop);
        assert_eq!((c, b), (1, LEN));
        let demoted = map.get(&key).unwrap().shadow.as_ref().unwrap();
        assert!(!demoted.carded());
        for i in 0..16usize {
            // SAFETY: inside `LEN` mapping.
            let got = unsafe { demoted.data.add(0x100 + i * 4).cast::<u32>().read() };
            assert_eq!(got, 0x7777_0000 + i as u32);
        }
    }

    #[test]
    fn warp_dxgi_8gb_and_12gb_pressure_demotes_cold_small_and_keeps_hot_large() {
        let Some(mut rig) = WarpRig::new() else {
            eprintln!("vram_tier_tests: WARP adapter unavailable, skipping");
            return;
        };
        let mut map: BTreeMap<(u64, usize), Entry> = BTreeMap::new();
        let mut budget = CardBudget::default();

        for i in 0..12u64 {
            let start = 0x5000_0000 + i * (64 * KIB);
            let shadow = rig.make_shadow(start, 64 * KIB, true);
            rig.dispatch_uav_write(&shadow, 0, 4, 0xA000_0000 + (i as u32) * 0x100);
            budget.total_bytes += 64 * KIB;
            budget.small_bytes += 64 * KIB;
            map.insert(
                (start, 5000 + i as usize),
                Entry {
                    end: start + 64 * KIB,
                    shadow: Some(shadow),
                },
            );
        }
        for i in 6..12u64 {
            let key = (0x5000_0000 + i * (64 * KIB), 5000 + i as usize);
            map.get(&key).unwrap().shadow.as_ref().unwrap().note_access();
        }
        for j in 0..2u64 {
            let start = 0x6000_0000 + j * (2 * MIB);
            let shadow = rig.make_shadow(start, 2 * MIB, true);
            budget.total_bytes += 2 * MIB;
            map.insert(
                (start, 6000 + j as usize),
                Entry {
                    end: start + 2 * MIB,
                    shadow: Some(shadow),
                },
            );
        }

        let vram_8gb = 7_200 * MIB;
        assert_eq!(dxgi_demote_target_bytes(Some([vram_8gb, vram_8gb * 80 / 100, 0, 0]), budget.small_bytes, 88), 0);

        let target_8gb = dxgi_demote_target_bytes(Some([vram_8gb, vram_8gb * 90 / 100, 0, 0]), budget.small_bytes, 88);
        assert_eq!(target_8gb, 12 * 64 * KIB);
        let completed = rig.completed_fence();
        let (meta, srcs) = collect_readback_candidates(&map, 6 * 64 * KIB, completed);
        assert_eq!(meta.len(), 6);
        let readbacks = rig.demote_readback(&srcs).expect("8GB pressure readback");
        drop(srcs);
        let mut to_drop = Vec::new();
        let (c1, b1) = apply_readback_demotions(&mut map, &meta, &readbacks, completed, &mut budget, false, &mut to_drop);
        drop(to_drop);
        assert_eq!((c1, b1), (6, 6 * 64 * KIB));
        for i in 0..6u64 {
            let key = (0x5000_0000 + i * (64 * KIB), 5000 + i as usize);
            let s = map.get(&key).unwrap().shadow.as_ref().unwrap();
            assert!(!s.carded());
            // SAFETY: mapped host staging buffer.
            let marker = unsafe { s.data.cast::<u32>().read() };
            assert_eq!(marker, 0xA000_0000 + (i as u32) * 0x100);
        }
        for i in 6..12u64 {
            let key = (0x5000_0000 + i * (64 * KIB), 5000 + i as usize);
            assert!(map.get(&key).unwrap().shadow.as_ref().unwrap().carded());
        }

        let vram_12gb = 11_200 * MIB;
        let target_12gb = dxgi_demote_target_bytes(Some([vram_12gb, vram_12gb * 90 / 100, 0, 0]), budget.small_bytes, 88);
        assert_eq!(target_12gb, 6 * 64 * KIB);
        let (meta2, srcs2) = collect_readback_candidates(&map, target_12gb, completed);
        assert_eq!(meta2.len(), 6);
        let readbacks2 = rig.demote_readback(&srcs2).expect("12GB pressure readback");
        drop(srcs2);
        let mut to_drop2 = Vec::new();
        let (c2, b2) = apply_readback_demotions(&mut map, &meta2, &readbacks2, completed, &mut budget, false, &mut to_drop2);
        drop(to_drop2);
        assert_eq!((c2, b2), (6, 6 * 64 * KIB));
        assert_eq!(budget.small_bytes, 0);
        assert_eq!(budget.total_bytes, 4 * MIB);
        for j in 0..2u64 {
            let key = (0x6000_0000 + j * (2 * MIB), 6000 + j as usize);
            assert!(map.get(&key).unwrap().shadow.as_ref().unwrap().carded());
        }
        for i in 6..12u64 {
            let key = (0x5000_0000 + i * (64 * KIB), 5000 + i as usize);
            let s = map.get(&key).unwrap().shadow.as_ref().unwrap();
            assert!(!s.carded());
            // SAFETY: mapped host staging buffer.
            let marker = unsafe { s.data.cast::<u32>().read() };
            assert_eq!(marker, 0xA000_0000 + (i as u32) * 0x100);
        }
    }

    #[test]
    fn card_room_kept_and_admit_shortfall_preserve_vram_accounting_invariants() {
        let most = 1024 * MIB;
        let mut card_bytes = 500 * MIB;
        let refused_large = 0u64;
        let mut pressure_released = 0u64;
        let mut used = 6_480 * MIB;

        let initial_kept = card_room_kept_for(most, card_bytes, refused_large, pressure_released);
        assert_eq!(initial_kept, 524 * MIB);
        assert_eq!(card_room_kept_for(most, card_bytes, refused_large, pressure_released), 524 * MIB);

        let before_effective = used + initial_kept;
        let demoted = 200 * MIB;
        card_bytes -= demoted;
        used -= demoted;
        pressure_released += demoted;
        let after_kept = card_room_kept_for(most, card_bytes, refused_large, pressure_released);
        assert_eq!(after_kept, initial_kept);
        assert_eq!(used + after_kept, before_effective - demoted);

        assert_eq!(card_admit_shortfall(32 * MIB, 1000 * MIB, 1024 * MIB, None, true), 8 * MIB);
        assert_eq!(card_admit_shortfall(32 * MIB, 1024 * MIB, 1024 * MIB, None, true), 32 * MIB);
        let vram_8gb = 7_200 * MIB;
        let limit_97 = (vram_8gb * 97) / 100;
        let used_98 = limit_97 + 10 * MIB;
        assert_eq!(
            card_admit_shortfall(4 * MIB, 500 * MIB, 1024 * MIB, Some([vram_8gb, used_98, 0, 0]), true),
            14 * MIB
        );
    }
}
