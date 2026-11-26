//! Host System RAM pressure tracking, safe `VirtualAlloc`/`MEM_DECOMMIT`
//! backing blocks, and cold-pool / shader-blob reclamation policies.
//!
//! Safety invariants on Windows virtual memory:
//! - [`VmBlock`] exclusively owns private `VirtualAlloc(..., MEM_RESERVE, ...)`
//!   allocations (`MEM_PRIVATE`, never `MEM_WRITE_WATCH`, never `MapViewOfFile`,
//!   never D3D12 driver-mapped pointers).
//! - Any page decommitted with `MEM_DECOMMIT` retains its `MEM_RESERVE` virtual
//!   address range and is transparently re-committed via `VirtualAlloc(...,
//!   MEM_COMMIT, PAGE_READWRITE)` before callers touch it again.
//! - [`verify_safe_for_decommit`] checks `VirtualQuery` (`MEM_PRIVATE` and
//!   `MEM_COMMIT`) and runs a non-resetting `GetWriteWatch` probe (`flags = 0`)
//!   to reject section views (`MEM_MAPPED`), image mappings (`MEM_IMAGE`), and
//!   write-watched title allocations before decommitting.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub(crate) const KIB: usize = 1024;
pub(crate) const MIB: usize = 1024 * 1024;
pub(crate) const GIB: u64 = 1024 * 1024 * 1024;

/// Backing piece and reservation alignment for [`VmBlock`] (`64 KiB`, matching
/// `shadow::PIECE` and Windows allocation granularity).
pub(crate) const VM_PIECE: usize = 64 * KIB;

/// Minimum buffer length upgraded from heap `Vec<u8>` to a [`VmBlock`] so small
/// constant/descriptor shadows avoid `VirtualAlloc` syscall overhead while all
/// large shadows and 4 MiB+ windows support page-level `MEM_DECOMMIT`.
pub(crate) const VM_THRESHOLD: usize = 256 * KIB;

/// Host RAM capacity tier derived from `GlobalMemoryStatusEx::ullTotalPhys`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostTier {
    /// Hosts with <= 16 GiB physical RAM (typically reported ~15.7–16.0 GiB).
    Tight16,
    /// Hosts with <= 32 GiB physical RAM (typically reported ~31.7–32.0 GiB).
    Standard32,
    /// Hosts with > 32 GiB physical RAM.
    Large,
}

impl HostTier {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Tight16 => "<=16GB",
            Self::Standard32 => "<=32GB",
            Self::Large => ">32GB",
        }
    }
}

/// Snapshot of Windows physical and commit memory status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct HostRamStatus {
    pub(crate) total_phys: u64,
    pub(crate) avail_phys: u64,
    pub(crate) avail_commit: u64,
    pub(crate) process_working_set: u64,
    pub(crate) process_private_bytes: u64,
}

/// Global byte and safety-event counters for RAM reclamation diagnostics.
pub(crate) struct RamTrimCounters {
    pub(crate) vm_reserved_bytes: AtomicU64,
    pub(crate) vm_committed_bytes: AtomicU64,
    pub(crate) vm_decommitted_bytes: AtomicU64,
    pub(crate) vm_zero_decommitted_bytes: AtomicU64,
    pub(crate) vm_recommitted_bytes: AtomicU64,
    pub(crate) pool_trimmed_bytes: AtomicU64,
    pub(crate) list_trimmed_count: AtomicU64,
    pub(crate) shader_dxil_dropped_bytes: AtomicU64,
    pub(crate) shader_dxil_reloaded_bytes: AtomicU64,
    pub(crate) shader_container_orig_bytes: AtomicU64,
    pub(crate) shader_container_retained_bytes: AtomicU64,
    pub(crate) unsafe_decommit_blocked: AtomicU64,
}

impl RamTrimCounters {
    const fn new() -> Self {
        Self {
            vm_reserved_bytes: AtomicU64::new(0),
            vm_committed_bytes: AtomicU64::new(0),
            vm_decommitted_bytes: AtomicU64::new(0),
            vm_zero_decommitted_bytes: AtomicU64::new(0),
            vm_recommitted_bytes: AtomicU64::new(0),
            pool_trimmed_bytes: AtomicU64::new(0),
            list_trimmed_count: AtomicU64::new(0),
            shader_dxil_dropped_bytes: AtomicU64::new(0),
            shader_dxil_reloaded_bytes: AtomicU64::new(0),
            shader_container_orig_bytes: AtomicU64::new(0),
            shader_container_retained_bytes: AtomicU64::new(0),
            unsafe_decommit_blocked: AtomicU64::new(0),
        }
    }
}

pub(crate) static COUNTERS: RamTrimCounters = RamTrimCounters::new();

/// Plain snapshot of reclamation metrics and current OS commit/RAM status.
#[cfg(any(feature = "diagnostics", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RamTrimSnapshot {
    pub(crate) aggressive: bool,
    pub(crate) tier: HostTier,
    pub(crate) host: HostRamStatus,
    pub(crate) vm_reserved_bytes: u64,
    pub(crate) vm_committed_bytes: u64,
    pub(crate) vm_decommitted_bytes: u64,
    pub(crate) vm_zero_decommitted_bytes: u64,
    pub(crate) vm_recommitted_bytes: u64,
    pub(crate) pool_trimmed_bytes: u64,
    pub(crate) list_trimmed_count: u64,
    pub(crate) shader_dxil_dropped_bytes: u64,
    pub(crate) shader_dxil_reloaded_bytes: u64,
    pub(crate) shader_container_saved_bytes: u64,
    pub(crate) unsafe_decommit_blocked: u64,
}

/// Whether opt-in aggressive host RAM trimming (`VICTORY_RAM_TRIM=1`) is enabled.
///
/// Default is `false` (default-safe mode: decommits strictly dead unmapped /
/// card-only snapshot ranges, trims cold pools above a soft warm floor, and
/// releases cold post-PSO DXIL blobs while compressing cold containers).
pub(crate) fn aggressive() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_RAM_TRIM").is_some_and(|v| v == "1");
        if on {
            let status = host_status();
            let tier = tier_for_total_bytes(status.total_phys);
            tracing::info!(
                tier = tier.label(),
                total_mb = status.total_phys >> 20,
                avail_mb = status.avail_phys >> 20,
                "aggressive host RAM trimming enabled (VICTORY_RAM_TRIM=1)"
            );
        }
        on
    })
}

/// Classifies a host by physical memory capacity, allowing slight firmware/BIOS
/// carve-outs below nominal 16 GiB and 32 GiB boundaries.
pub(crate) const fn tier_for_total_bytes(total_phys: u64) -> HostTier {
    if total_phys == 0 {
        return HostTier::Standard32;
    }
    if total_phys <= 17 * GIB {
        HostTier::Tight16
    } else if total_phys <= 33 * GIB {
        HostTier::Standard32
    } else {
        HostTier::Large
    }
}

/// Queries OS physical memory, OS commit headroom, and current process private
/// bytes / working set.
#[cfg(windows)]
pub(crate) fn host_status() -> HostRamStatus {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    // SAFETY: plain zero-initialized Win32 struct with `dwLength` set.
    let mut mem: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    mem.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: valid sized `MEMORYSTATUSEX` pointer.
    let mem_ok = unsafe { GlobalMemoryStatusEx(&mut mem) } != 0;

    let mut proc = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: current process handle and sized `PROCESS_MEMORY_COUNTERS`.
    let proc_ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut proc, proc.cb) } != 0;

    HostRamStatus {
        total_phys: if mem_ok { mem.ullTotalPhys } else { 0 },
        avail_phys: if mem_ok { mem.ullAvailPhys } else { 0 },
        avail_commit: if mem_ok { mem.ullAvailPageFile } else { 0 },
        process_working_set: if proc_ok { proc.WorkingSetSize as u64 } else { 0 },
        process_private_bytes: if proc_ok { proc.PagefileUsage as u64 } else { 0 },
    }
}

#[cfg(not(windows))]
pub(crate) fn host_status() -> HostRamStatus {
    HostRamStatus::default()
}

/// Derived trimming thresholds for default-safe vs aggressive (`VICTORY_RAM_TRIM=1`)
/// operation, scaled by host tier and real-time physical/commit pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrimPolicy {
    pub(crate) aggressive: bool,
    pub(crate) tier: HostTier,
    pub(crate) low_memory: bool,
    pub(crate) panic_memory: bool,
    pub(crate) pool_idle_ms: u64,
    pub(crate) shadow_pool_warm_bytes: u64,
    pub(crate) shadow_pool_reserve_slots: usize,
    pub(crate) scratch_spare_warm_bytes: u64,
    pub(crate) texture_spare_warm_bytes: u64,
    pub(crate) list_idle_warm_count: usize,
    pub(crate) list_idle_ms: u64,
    pub(crate) shader_dxil_idle_ms: u64,
    pub(crate) shader_container_idle_ms: u64,
    pub(crate) decommit_cold_reference_ms: Option<u64>,
    pub(crate) fetched_warm_bytes: u64,
}

impl TrimPolicy {
    /// Computes the active trim policy for explicit host parameters (used both in
    /// production and deterministically in unit tests).
    pub(crate) fn for_host(
        aggressive: bool,
        total_phys: u64,
        avail_phys: u64,
        avail_commit: u64,
    ) -> Self {
        let tier = tier_for_total_bytes(total_phys);
        let low_memory = (avail_phys > 0 && avail_phys < 4 * GIB)
            || (avail_commit > 0 && avail_commit < 4 * GIB);
        let panic_memory = (avail_phys > 0 && avail_phys < 2 * GIB)
            || (avail_commit > 0 && avail_commit < 2 * GIB);

        if panic_memory {
            return Self {
                aggressive,
                tier,
                low_memory: true,
                panic_memory: true,
                pool_idle_ms: 1_000,
                shadow_pool_warm_bytes: 0,
                shadow_pool_reserve_slots: 8,
                scratch_spare_warm_bytes: 0,
                texture_spare_warm_bytes: 0,
                list_idle_warm_count: 16,
                list_idle_ms: 2_000,
                shader_dxil_idle_ms: 0,
                shader_container_idle_ms: 0,
                decommit_cold_reference_ms: aggressive.then_some(1_000),
                fetched_warm_bytes: 4 * (MIB as u64),
            };
        }

        if aggressive {
            return match tier {
                HostTier::Tight16 => Self {
                    aggressive: true,
                    tier,
                    low_memory,
                    panic_memory: false,
                    pool_idle_ms: 2_000,
                    shadow_pool_warm_bytes: 0,
                    shadow_pool_reserve_slots: 8,
                    scratch_spare_warm_bytes: 0,
                    texture_spare_warm_bytes: 0,
                    list_idle_warm_count: 24,
                    list_idle_ms: 3_000,
                    shader_dxil_idle_ms: 0,
                    shader_container_idle_ms: 1_000,
                    decommit_cold_reference_ms: Some(2_000),
                    fetched_warm_bytes: 4 * (MIB as u64),
                },
                HostTier::Standard32 => Self {
                    aggressive: true,
                    tier,
                    low_memory,
                    panic_memory: false,
                    pool_idle_ms: if low_memory { 2_000 } else { 4_000 },
                    shadow_pool_warm_bytes: if low_memory { 0 } else { 32 * (MIB as u64) },
                    shadow_pool_reserve_slots: 8,
                    scratch_spare_warm_bytes: if low_memory { 0 } else { 8 * (MIB as u64) },
                    texture_spare_warm_bytes: if low_memory { 0 } else { 16 * (MIB as u64) },
                    list_idle_warm_count: if low_memory { 24 } else { 48 },
                    list_idle_ms: if low_memory { 3_000 } else { 5_000 },
                    shader_dxil_idle_ms: 0,
                    shader_container_idle_ms: 2_000,
                    decommit_cold_reference_ms: Some(if low_memory { 2_000 } else { 4_000 }),
                    fetched_warm_bytes: 8 * (MIB as u64),
                },
                HostTier::Large => Self {
                    aggressive: true,
                    tier,
                    low_memory,
                    panic_memory: false,
                    pool_idle_ms: if low_memory { 2_500 } else { 6_000 },
                    shadow_pool_warm_bytes: if low_memory { 16 * (MIB as u64) } else { 64 * (MIB as u64) },
                    shadow_pool_reserve_slots: 12,
                    scratch_spare_warm_bytes: if low_memory { 8 * (MIB as u64) } else { 16 * (MIB as u64) },
                    texture_spare_warm_bytes: if low_memory { 16 * (MIB as u64) } else { 32 * (MIB as u64) },
                    list_idle_warm_count: if low_memory { 32 } else { 64 },
                    list_idle_ms: 6_000,
                    shader_dxil_idle_ms: 0,
                    shader_container_idle_ms: 4_000,
                    decommit_cold_reference_ms: Some(6_000),
                    fetched_warm_bytes: 16 * (MIB as u64),
                },
            };
        }

        // Default-safe mode: keeps warm caches responsive while still trimming cold
        // startup/streaming spikes above soft high-water ceilings.
        let pool_idle_ms = if low_memory {
            6_000
        } else if tier == HostTier::Tight16 {
            10_000
        } else {
            15_000
        };
        Self {
            aggressive: false,
            tier,
            low_memory,
            panic_memory: false,
            pool_idle_ms,
            shadow_pool_warm_bytes: if low_memory { 32 * (MIB as u64) } else { 128 * (MIB as u64) },
            shadow_pool_reserve_slots: 16,
            scratch_spare_warm_bytes: if low_memory { 8 * (MIB as u64) } else { 32 * (MIB as u64) },
            texture_spare_warm_bytes: if low_memory { 16 * (MIB as u64) } else { 64 * (MIB as u64) },
            list_idle_warm_count: if low_memory || tier == HostTier::Tight16 { 64 } else { 128 },
            list_idle_ms: if low_memory { 8_000 } else { 15_000 },
            shader_dxil_idle_ms: if low_memory { 2_000 } else { 5_000 },
            shader_container_idle_ms: if low_memory { 5_000 } else { 15_000 },
            decommit_cold_reference_ms: None,
            fetched_warm_bytes: 16 * (MIB as u64),
        }
    }

    /// Cached variant (`250 ms` TTL) used on hot paths (`sweep`, `give_back`,
    /// `DxilPin::drop`) to avoid repeating Win32 memory queries every frame.
    pub(crate) fn cached() -> Self {
        let s = host_status_cached();
        Self::for_host(aggressive(), s.total_phys, s.avail_phys, s.avail_commit)
    }
}

/// Returns a short-lived (`250 ms`) cached host status snapshot so hot callers
/// do not invoke `GlobalMemoryStatusEx` / `K32GetProcessMemoryInfo` repeatedly.
pub(crate) fn host_status_cached() -> HostRamStatus {
    const TTL: std::time::Duration = std::time::Duration::from_millis(250);
    static CACHE: std::sync::Mutex<Option<(std::time::Instant, HostRamStatus)>> =
        std::sync::Mutex::new(None);
    let now = std::time::Instant::now();
    {
        let guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, status)) = *guard
            && now.saturating_duration_since(at) < TTL
        {
            return status;
        }
    }
    let fresh = host_status();
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some((now, fresh));
    fresh
}

/// Captures current reclamation counters and live host RAM status.
#[cfg(any(feature = "diagnostics", test))]
pub(crate) fn snapshot() -> RamTrimSnapshot {
    let host = host_status();
    let orig = COUNTERS.shader_container_orig_bytes.load(Relaxed);
    let retained = COUNTERS.shader_container_retained_bytes.load(Relaxed);
    RamTrimSnapshot {
        aggressive: aggressive(),
        tier: tier_for_total_bytes(host.total_phys),
        host,
        vm_reserved_bytes: COUNTERS.vm_reserved_bytes.load(Relaxed),
        vm_committed_bytes: COUNTERS.vm_committed_bytes.load(Relaxed),
        vm_decommitted_bytes: COUNTERS.vm_decommitted_bytes.load(Relaxed),
        vm_zero_decommitted_bytes: COUNTERS.vm_zero_decommitted_bytes.load(Relaxed),
        vm_recommitted_bytes: COUNTERS.vm_recommitted_bytes.load(Relaxed),
        pool_trimmed_bytes: COUNTERS.pool_trimmed_bytes.load(Relaxed),
        list_trimmed_count: COUNTERS.list_trimmed_count.load(Relaxed),
        shader_dxil_dropped_bytes: COUNTERS.shader_dxil_dropped_bytes.load(Relaxed),
        shader_dxil_reloaded_bytes: COUNTERS.shader_dxil_reloaded_bytes.load(Relaxed),
        shader_container_saved_bytes: orig.saturating_sub(retained),
        unsafe_decommit_blocked: COUNTERS.unsafe_decommit_blocked.load(Relaxed),
    }
}

/// Formats a concise diagnostics summary for memory logging (`copystat::log_memory`).
#[cfg(any(feature = "diagnostics", test))]
pub(crate) fn diagnostics_line() -> String {
    let s = snapshot();
    let mb = |b: u64| b as f64 / (MIB as f64);
    format!(
        "ram trim ({}, mode={}): vm reserved {:.1} MB / committed {:.1} MB, \
         decommitted {:.1} MB (zero-pages {:.1} MB, recommitted {:.1} MB), \
         pools trimmed {:.1} MB ({} lists), shader dxil dropped {:.1} MB (reloaded {:.1} MB), \
         shader containers saved {:.1} MB, blocked unsafe decommits {}",
        s.tier.label(),
        if s.aggressive { "aggressive" } else { "default" },
        mb(s.vm_reserved_bytes),
        mb(s.vm_committed_bytes),
        mb(s.vm_decommitted_bytes),
        mb(s.vm_zero_decommitted_bytes),
        mb(s.vm_recommitted_bytes),
        mb(s.pool_trimmed_bytes),
        s.list_trimmed_count,
        mb(s.shader_dxil_dropped_bytes),
        mb(s.shader_dxil_reloaded_bytes),
        mb(s.shader_container_saved_bytes),
        s.unsafe_decommit_blocked,
    )
}

/// Records `bytes` trimmed from idle D3D12/heap/scratch pools.
pub(crate) fn note_pool_trimmed(bytes: u64) {
    if bytes > 0 {
        COUNTERS.pool_trimmed_bytes.fetch_add(bytes, Relaxed);
    }
}

/// Records idle command lists released back to the OS/driver.
pub(crate) fn note_lists_trimmed(count: u64) {
    if count > 0 {
        COUNTERS.list_trimmed_count.fetch_add(count, Relaxed);
    }
}

/// Shrinks a reusable `Vec<u8>` buffer when its capacity far exceeds recent use,
/// keeping small buffers untouched to prevent allocator churn.
pub(crate) fn maybe_shrink_scratch_vec(buf: &mut Vec<u8>, needed: usize, cap_ceiling: usize) {
    buf.clear();
    if buf.capacity() > cap_ceiling && buf.capacity() > needed.saturating_mul(4) {
        let before = buf.capacity();
        let target = needed.max(cap_ceiling / 2);
        buf.shrink_to(target);
        let after = buf.capacity();
        if before > after {
            note_pool_trimmed((before - after) as u64);
        }
    }
}

/// Verifies that `[ptr, ptr + len)` is strictly private, committed virtual
/// memory allocated without `MEM_WRITE_WATCH` (never a mapped section view,
/// image mapping, or write-watched range).
///
/// Uses a non-resetting `GetWriteWatch(0, ...)` probe so existing write-watch
/// dirty state on external regions is never disturbed.
#[cfg(windows)]
pub(crate) fn verify_safe_for_decommit(ptr: *const std::ffi::c_void, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };

    if ptr.is_null() || len == 0 {
        return false;
    }
    let start = ptr as usize;
    let Some(end) = start.checked_add(len) else {
        return false;
    };

    let mut cursor = start;
    while cursor < end {
        // SAFETY: plain zeroed MBI passed to `VirtualQuery`.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer with exact byte length.
        let queried = unsafe { VirtualQuery(cursor as *const _, &mut info, size_of_val(&info)) };
        if queried == 0 || info.RegionSize == 0 {
            COUNTERS.unsafe_decommit_blocked.fetch_add(1, Relaxed);
            return false;
        }
        if info.Type != MEM_PRIVATE || info.State != MEM_COMMIT {
            COUNTERS.unsafe_decommit_blocked.fetch_add(1, Relaxed);
            return false;
        }

        // Non-resetting (`dwFlags = 0`) probe: succeeds (`0`) ONLY if this
        // allocation has `MEM_WRITE_WATCH`, and fails (`ERROR_INVALID_PARAMETER`)
        // on normal non-write-watched `VirtualAlloc` blocks.
        let mut watch_addrs = [std::ptr::null_mut()];
        let mut watch_count: usize = watch_addrs.len();
        let mut granularity: u32 = 0;
        // SAFETY: `info.BaseAddress` was verified committed `MEM_PRIVATE` above
        // and output pointers are stack locals of exact size. `dwFlags = 0` never resets.
        let is_write_watch = unsafe {
            GetWriteWatch(
                0,
                info.BaseAddress,
                4096,
                watch_addrs.as_mut_ptr(),
                &mut watch_count,
                &mut granularity,
            ) == 0
        };
        if is_write_watch {
            COUNTERS.unsafe_decommit_blocked.fetch_add(1, Relaxed);
            return false;
        }

        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if region_end <= cursor {
            COUNTERS.unsafe_decommit_blocked.fetch_add(1, Relaxed);
            return false;
        }
        cursor = region_end;
    }
    true
}

#[cfg(not(windows))]
pub(crate) fn verify_safe_for_decommit(_ptr: *const std::ffi::c_void, _len: usize) -> bool {
    false
}

/// Safe decommit helper that refuses anything failing [`verify_safe_for_decommit`].
#[cfg(windows)]
pub(crate) unsafe fn safe_decommit(ptr: *mut std::ffi::c_void, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
    if !verify_safe_for_decommit(ptr.cast_const(), len) {
        return false;
    }
    // SAFETY: caller ensures no active slice references `[ptr, ptr + len)` and
    // `verify_safe_for_decommit` verified committed non-write-watch `MEM_PRIVATE`.
    unsafe { VirtualFree(ptr, len, MEM_DECOMMIT) != 0 }
}

/// Page-backed host buffer holding an exclusive `MEM_RESERVE` virtual address
/// reservation whose 64 KiB pieces can be individually `MEM_DECOMMIT`ted when
/// dead/zero/idle and cleanly re-committed (`MEM_COMMIT`) without changing VA.
#[cfg(windows)]
pub(crate) struct VmBlock {
    base: std::ptr::NonNull<u8>,
    reserve_len: usize,
    len: usize,
    committed: Vec<bool>,
    committed_bytes: usize,
}

// SAFETY: `VmBlock` owns an exclusive private OS allocation with unique access.
#[cfg(windows)]
unsafe impl Send for VmBlock {}

#[cfg(windows)]
impl VmBlock {
    /// Reserves `len.max(1).next_multiple_of(VM_PIECE)` bytes and commits the
    /// active prefix (`len`) as zero-filled read-write memory.
    pub(crate) fn new_zeroed(len: usize) -> Option<Self> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
            MEM_RELEASE,
        };

        let reserve_len = len.max(1).checked_next_multiple_of(VM_PIECE)?;
        // SAFETY: reserving an anonymous private region of `reserve_len` bytes
        // without `MEM_WRITE_WATCH`.
        let raw = unsafe { VirtualAlloc(std::ptr::null(), reserve_len, MEM_RESERVE, PAGE_NOACCESS) };
        let base = std::ptr::NonNull::new(raw.cast::<u8>())?;

        let active_pieces = len.div_ceil(VM_PIECE);
        let commit_bytes = active_pieces * VM_PIECE;
        let total_pieces = reserve_len / VM_PIECE;
        let mut committed = vec![false; total_pieces];

        if commit_bytes > 0 {
            // SAFETY: committing `[base, base + commit_bytes)` inside our own reservation.
            let ptr = unsafe {
                VirtualAlloc(base.as_ptr().cast(), commit_bytes, MEM_COMMIT, PAGE_READWRITE)
            };
            if ptr.is_null() {
                // SAFETY: releasing the failed reservation we just allocated.
                unsafe { VirtualFree(base.as_ptr().cast(), 0, MEM_RELEASE) };
                return None;
            }
            committed[..active_pieces].fill(true);
        }

        COUNTERS.vm_reserved_bytes.fetch_add(reserve_len as u64, Relaxed);
        COUNTERS.vm_committed_bytes.fetch_add(commit_bytes as u64, Relaxed);

        Some(Self {
            base,
            reserve_len,
            len,
            committed,
            committed_bytes: commit_bytes,
        })
    }

    /// Logical byte length visible to callers.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Total reserved virtual address range in bytes.
    pub(crate) fn capacity(&self) -> usize {
        self.reserve_len
    }

    /// Reservation base address as an integer (for VA identity verification).
    #[cfg(test)]
    pub(crate) fn base_addr(&self) -> usize {
        self.base.as_ptr() as usize
    }

    /// Total bytes currently backed by `MEM_COMMIT` inside this block.
    pub(crate) fn committed_bytes(&self) -> usize {
        self.committed_bytes
    }

    /// Number of logical 64 KiB pieces covering `self.len`.
    pub(crate) fn piece_count(&self) -> usize {
        self.len.div_ceil(VM_PIECE)
    }

    /// Ensures all pieces overlapping `[start, end)` (clamped to `self.reserve_len`)
    /// are committed.
    pub(crate) fn ensure_range_committed(&mut self, start: usize, end: usize) -> bool {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
        if start >= end {
            return true;
        }
        let first = start / VM_PIECE;
        let last = end.min(self.reserve_len).div_ceil(VM_PIECE);
        let mut p = first;
        while p < last {
            if self.committed[p] {
                p += 1;
                continue;
            }
            let run_start = p;
            while p < last && !self.committed[p] {
                p += 1;
            }
            let bytes = (p - run_start) * VM_PIECE;
            // SAFETY: `[run_start * VM_PIECE, p * VM_PIECE)` lies inside our reservation.
            let ptr = unsafe {
                VirtualAlloc(
                    self.base.as_ptr().add(run_start * VM_PIECE).cast(),
                    bytes,
                    MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            if ptr.is_null() {
                return false;
            }
            self.committed[run_start..p].fill(true);
            self.committed_bytes += bytes;
            COUNTERS.vm_committed_bytes.fetch_add(bytes as u64, Relaxed);
            COUNTERS.vm_recommitted_bytes.fetch_add(bytes as u64, Relaxed);
        }
        true
    }

    /// Commits every active byte `[0, self.len)` and returns it as `&[u8]`.
    #[cfg(test)]
    pub(crate) fn as_slice(&mut self) -> &[u8] {
        let len = self.len;
        assert!(self.ensure_range_committed(0, len), "VmBlock MEM_COMMIT failed for {len} bytes");
        // SAFETY: `[0, self.len)` is committed inside `self.base` (or `len == 0`).
        unsafe { std::slice::from_raw_parts(self.base.as_ptr(), len) }
    }

    /// Commits every active byte `[0, self.len)` and returns it as `&mut [u8]`.
    #[cfg(test)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.len;
        assert!(self.ensure_range_committed(0, len), "VmBlock MEM_COMMIT failed for {len} bytes");
        // SAFETY: `[0, self.len)` is committed inside `self.base` (or `len == 0`).
        unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr(), len) }
    }

    /// Re-sizes a pooled block to `new_len <= self.reserve_len`, committing all
    /// pieces in `[0, new_len)` for overwrite and decommitting any committed tail
    /// pieces completely beyond `new_len`.
    pub(crate) fn prepare_for_overwrite(&mut self, new_len: usize) -> bool {
        if new_len > self.reserve_len {
            return false;
        }
        if !self.ensure_range_committed(0, new_len) {
            return false;
        }
        self.len = new_len;
        let needed_pieces = new_len.div_ceil(VM_PIECE);
        let _ = self.decommit_piece_range(needed_pieces, self.committed.len());
        true
    }

    /// Decommits contiguous runs of committed pieces in `[first_piece, last_piece)`
    /// via `VirtualFree(..., MEM_DECOMMIT)` while preserving `self.base`'s
    /// `MEM_RESERVE` virtual address range. Returns bytes reclaimed.
    pub(crate) fn decommit_piece_range(&mut self, first_piece: usize, last_piece: usize) -> usize {
        let limit = last_piece.min(self.committed.len());
        let mut reclaimed = 0usize;
        let mut p = first_piece;
        while p < limit {
            if !self.committed[p] {
                p += 1;
                continue;
            }
            let run_start = p;
            while p < limit && self.committed[p] {
                p += 1;
            }
            let bytes = (p - run_start) * VM_PIECE;
            // SAFETY: `[run_start * VM_PIECE, p * VM_PIECE)` is exclusively owned,
            // non-write-watch `MEM_PRIVATE`, and no slices borrow `&mut self`.
            let ok = unsafe { safe_decommit(self.base.as_ptr().add(run_start * VM_PIECE).cast(), bytes) };
            if ok {
                self.committed[run_start..p].fill(false);
                self.committed_bytes -= bytes;
                COUNTERS.vm_committed_bytes.fetch_sub(bytes as u64, Relaxed);
                COUNTERS.vm_decommitted_bytes.fetch_add(bytes as u64, Relaxed);
                reclaimed += bytes;
            }
        }
        reclaimed
    }

    /// Decommits all wholly contained 64 KiB pieces covering `[byte_start, byte_end)`.
    pub(crate) fn decommit_byte_range(&mut self, byte_start: usize, byte_end: usize) -> usize {
        let clamped_end = byte_end.min(self.len);
        if byte_start >= clamped_end {
            return 0;
        }
        let first_piece = byte_start.div_ceil(VM_PIECE);
        let last_piece = if clamped_end == self.len {
            self.piece_count()
        } else {
            clamped_end / VM_PIECE
        };
        if first_piece >= last_piece {
            return 0;
        }
        self.decommit_piece_range(first_piece, last_piece)
    }

    /// Decommits every committed piece of this block, returning 100% of physical
    /// commit while preserving the `MEM_RESERVE` virtual address range.
    pub(crate) fn decommit_all(&mut self) -> usize {
        let total_pieces = self.committed.len();
        self.decommit_piece_range(0, total_pieces)
    }

    /// Scans committed 64 KiB pieces and decommits contiguous runs of all-zero
    /// pieces in coalesced `MEM_DECOMMIT` calls while keeping logical zeros.
    pub(crate) fn decommit_zero_pieces(&mut self) -> usize {
        let count = self.piece_count();
        let mut reclaimed = 0usize;
        let mut p = 0usize;
        while p < count {
            let is_zero = |idx: usize| -> bool {
                if !self.committed[idx] {
                    return false;
                }
                let start = idx * VM_PIECE;
                if start >= self.len {
                    return false;
                }
                let n = (self.len - start).min(VM_PIECE);
                // SAFETY: `idx` is committed and `[start, start + n)` lies within `self.len`.
                let slice = unsafe { std::slice::from_raw_parts(self.base.as_ptr().add(start), n) };
                is_all_zero(slice)
            };
            if !is_zero(p) {
                p += 1;
                continue;
            }
            let run_start = p;
            p += 1;
            while p < count && is_zero(p) {
                p += 1;
            }
            let bytes = self.decommit_piece_range(run_start, p);
            if bytes > 0 {
                COUNTERS.vm_zero_decommitted_bytes.fetch_add(bytes as u64, Relaxed);
                reclaimed += bytes;
            }
        }
        reclaimed
    }
}

#[cfg(windows)]
impl Drop for VmBlock {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        COUNTERS.vm_reserved_bytes.fetch_sub(self.reserve_len as u64, Relaxed);
        if self.committed_bytes > 0 {
            COUNTERS.vm_committed_bytes.fetch_sub(self.committed_bytes as u64, Relaxed);
        }
        // SAFETY: `self.base` was allocated with `VirtualAlloc(MEM_RESERVE)`.
        unsafe { VirtualFree(self.base.as_ptr().cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(not(windows))]
pub(crate) struct VmBlock {
    bytes: Vec<u8>,
}

#[cfg(not(windows))]
impl VmBlock {
    pub(crate) fn new_zeroed(len: usize) -> Option<Self> {
        Some(Self { bytes: vec![0u8; len] })
    }
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }
    pub(crate) fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
    pub(crate) fn committed_bytes(&self) -> usize {
        self.bytes.capacity()
    }
    pub(crate) fn ensure_range_committed(&mut self, _start: usize, _end: usize) -> bool {
        true
    }
    #[cfg(test)]
    pub(crate) fn as_slice(&mut self) -> &[u8] {
        &self.bytes
    }
    #[cfg(test)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
    pub(crate) fn prepare_for_overwrite(&mut self, new_len: usize) -> bool {
        if new_len > self.bytes.capacity() {
            return false;
        }
        self.bytes.resize(new_len, 0);
        true
    }
    pub(crate) fn decommit_byte_range(&mut self, _byte_start: usize, _byte_end: usize) -> usize {
        0
    }
    pub(crate) fn decommit_all(&mut self) -> usize {
        0
    }
    pub(crate) fn decommit_zero_pieces(&mut self) -> usize {
        0
    }
}

/// Fast word-aligned check that every byte in `bytes` is `0`.
pub(crate) fn is_all_zero(bytes: &[u8]) -> bool {
    // SAFETY: `align_to::<u64>` reads valid aligned `u64` words inside `bytes`.
    let (head, words, tail) = unsafe { bytes.align_to::<u64>() };
    head.iter().all(|&b| b == 0)
        && words.iter().all(|&w| w == 0)
        && tail.iter().all(|&b| b == 0)
}

/// Snapshot backing storage: small heap `Vec<u8>` below [`VM_THRESHOLD`], or
/// [`VmBlock`] above it so large snapshots can `MEM_DECOMMIT` physical pages
/// while preserving `MEM_RESERVE` VA ranges across reuse.
pub(crate) enum SnapshotBuffer {
    Small(Vec<u8>),
    Large(VmBlock),
}

impl SnapshotBuffer {
    pub(crate) fn new_zeroed(len: usize) -> Self {
        if len >= VM_THRESHOLD
            && let Some(block) = VmBlock::new_zeroed(len)
        {
            return Self::Large(block);
        }
        Self::Small(vec![0u8; len])
    }

    pub(crate) fn capacity(&self) -> usize {
        match self {
            Self::Small(v) => v.capacity(),
            Self::Large(b) => b.capacity(),
        }
    }

    pub(crate) fn committed_bytes(&self) -> usize {
        match self {
            Self::Small(v) => v.capacity(),
            Self::Large(b) => b.committed_bytes(),
        }
    }

    #[cfg(test)]
    pub(crate) fn base_addr(&self) -> Option<usize> {
        match self {
            Self::Small(_) => None,
            #[cfg(windows)]
            Self::Large(b) => Some(b.base_addr()),
            #[cfg(not(windows))]
            Self::Large(_) => None,
        }
    }

    pub(crate) fn is_large(&self) -> bool {
        matches!(self, Self::Large(_))
    }

    #[cfg(test)]
    pub(crate) fn as_slice(&mut self) -> &[u8] {
        match self {
            Self::Small(v) => v.as_slice(),
            Self::Large(b) => b.as_slice(),
        }
    }

    #[cfg(test)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Small(v) => v.as_mut_slice(),
            Self::Large(b) => b.as_mut_slice(),
        }
    }

    /// Raw pointer to the start of the reservation (`len` logical bytes).
    ///
    /// # Safety
    /// Any subrange read or written through the returned pointer must have been
    /// committed first via [`ensure_range_committed`].
    pub(crate) unsafe fn raw_ptr(&self) -> *const u8 {
        match self {
            Self::Small(v) => v.as_ptr(),
            #[cfg(windows)]
            Self::Large(b) => b.base.as_ptr(),
            #[cfg(not(windows))]
            Self::Large(b) => b.bytes.as_ptr(),
        }
    }

    /// Mutable raw pointer to the start of the reservation (`len` logical bytes).
    ///
    /// # Safety
    /// Any subrange read or written through the returned pointer must have been
    /// committed first via [`ensure_range_committed`].
    pub(crate) unsafe fn raw_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Small(v) => v.as_mut_ptr(),
            #[cfg(windows)]
            Self::Large(b) => b.base.as_ptr(),
            #[cfg(not(windows))]
            Self::Large(b) => b.bytes.as_mut_ptr(),
        }
    }

    pub(crate) fn ensure_range_committed(&mut self, start: usize, end: usize) -> bool {
        match self {
            Self::Small(_) => true,
            Self::Large(b) => b.ensure_range_committed(start, end),
        }
    }

    pub(crate) fn prepare_for_overwrite(&mut self, new_len: usize) -> bool {
        match self {
            Self::Small(v) => {
                if new_len > v.capacity() {
                    return false;
                }
                v.resize(new_len, 0);
                true
            }
            Self::Large(b) => b.prepare_for_overwrite(new_len),
        }
    }

    pub(crate) fn decommit_all(&mut self) -> usize {
        match self {
            Self::Small(_) => 0,
            Self::Large(b) => b.decommit_all(),
        }
    }

    pub(crate) fn decommit_byte_range(&mut self, start: usize, end: usize) -> usize {
        match self {
            Self::Small(_) => 0,
            Self::Large(b) => b.decommit_byte_range(start, end),
        }
    }

    pub(crate) fn decommit_zero_pieces(&mut self) -> usize {
        match self {
            Self::Small(_) => 0,
            Self::Large(b) => b.decommit_zero_pieces(),
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;

    /// Serializes tests that assert OS-wide `K32GetProcessMemoryInfo` commit drops
    /// so unrelated memory-heavy tests do not perturb process commit concurrently.
    pub(crate) fn commit_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn policy_scales_by_host_tier_and_ram_pressure() {
        let def_32 = TrimPolicy::for_host(false, 32 * GIB, 20 * GIB, 24 * GIB);
        assert!(!def_32.aggressive);
        assert_eq!(def_32.tier, HostTier::Standard32);
        assert_eq!(def_32.decommit_cold_reference_ms, None);
        assert!(def_32.shadow_pool_warm_bytes > 0);

        let agg_16 = TrimPolicy::for_host(true, 16 * GIB, 6 * GIB, 8 * GIB);
        assert!(agg_16.aggressive);
        assert_eq!(agg_16.tier, HostTier::Tight16);
        assert_eq!(agg_16.shadow_pool_warm_bytes, 0);
        assert_eq!(agg_16.texture_spare_warm_bytes, 0);
        assert_eq!(agg_16.scratch_spare_warm_bytes, 0);
        assert!(agg_16.list_idle_warm_count < def_32.list_idle_warm_count);
        assert!(agg_16.decommit_cold_reference_ms.is_some());

        let panic_32 = TrimPolicy::for_host(false, 32 * GIB, GIB, GIB);
        assert!(panic_32.panic_memory);
        assert_eq!(panic_32.shadow_pool_warm_bytes, 0);
        assert_eq!(panic_32.texture_spare_warm_bytes, 0);
    }

    #[test]
    fn safety_guard_rejects_write_watch_and_mapped_views() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, GetWriteWatch, MEM_COMMIT, MEM_RELEASE,
            MEM_RESERVE, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;

        let blocked_before = COUNTERS.unsafe_decommit_blocked.load(Relaxed);

        // 1. Private `MEM_WRITE_WATCH` must NEVER be cleared for decommit, and the
        // guard must not reset any dirty page bits tracked by `GetWriteWatch`.
        // SAFETY: standalone 256 KiB test allocation owned by this test.
        let watched = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * VM_PIECE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!watched.is_null());
        // SAFETY: first page of `watched` is newly committed read-write memory.
        unsafe { std::ptr::write_volatile(watched.cast::<u8>(), 0x5A) };
        assert!(
            !verify_safe_for_decommit(watched.cast_const(), 4 * VM_PIECE),
            "MEM_WRITE_WATCH region must be rejected"
        );
        let mut watch_addrs = [std::ptr::null_mut(); 4];
        let mut watch_count = watch_addrs.len();
        let mut gran = 0u32;
        // SAFETY: `watched` is valid `MEM_WRITE_WATCH` memory and output locals match.
        let ww_rc = unsafe {
            GetWriteWatch(
                0,
                watched,
                4 * VM_PIECE,
                watch_addrs.as_mut_ptr(),
                &mut watch_count,
                &mut gran,
            )
        };
        assert_eq!(ww_rc, 0);
        assert!(
            watch_count >= 1 && watch_addrs.contains(&watched),
            "verify_safe_for_decommit must preserve dirty write-watch state"
        );
        // SAFETY: freeing test allocation.
        unsafe { VirtualFree(watched, 0, MEM_RELEASE) };

        // 2. File-mapping / section view (`MEM_MAPPED`) must NEVER be cleared for decommit.
        // SAFETY: pagefile-backed section mapping owned by this test.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                VM_PIECE as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping view of section created above.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, VM_PIECE) };
        assert!(!view.Value.is_null());
        assert!(
            !verify_safe_for_decommit(view.Value.cast_const(), VM_PIECE),
            "MEM_MAPPED section view must be rejected"
        );
        // SAFETY: unmapping and closing test handles.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(section);
        }

        // 3. Normal private `VirtualAlloc` without `MEM_WRITE_WATCH` is allowed.
        // SAFETY: standalone private committed allocation.
        let plain = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * VM_PIECE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!plain.is_null());
        assert!(verify_safe_for_decommit(plain.cast_const(), 2 * VM_PIECE));
        // SAFETY: freeing plain allocation.
        unsafe { VirtualFree(plain, 0, MEM_RELEASE) };

        assert!(COUNTERS.unsafe_decommit_blocked.load(Relaxed) >= blocked_before + 2);
    }

    /// Walk OS virtual memory regions via `VirtualQuery` and sum committed bytes inside `[base, base + len)`.
    fn os_committed_bytes_in_range(base: usize, len: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };
        let end = base + len;
        let mut cursor = base;
        let mut committed = 0usize;
        while cursor < end {
            // SAFETY: plain zeroed struct.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: querying inside the caller's live `MEM_RESERVE` region.
            let n = unsafe {
                VirtualQuery(cursor as *const std::ffi::c_void, &mut m, size_of_val(&m))
            };
            assert_ne!(n, 0);
            let reg_end = (m.BaseAddress as usize + m.RegionSize).min(end);
            if m.State == MEM_COMMIT {
                committed += reg_end.saturating_sub(cursor);
            }
            cursor = reg_end.max(cursor + 4096);
        }
        committed
    }

    #[test]
    fn vm_block_decommits_preserves_reserve_and_recommits_cleanly() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };

        let _guard = commit_test_lock();
        const LEN: usize = 64 * MIB;
        let mut block = VmBlock::new_zeroed(LEN).expect("VmBlock reserve+commit");
        let base_addr = block.base.as_ptr() as usize;
        assert_eq!(block.len(), LEN);
        assert_eq!(block.committed_bytes(), LEN);

        // Touch pages across the whole range so OS commit reflects all 64 MiB.
        {
            let slice = block.as_mut_slice();
            for (i, chunk) in slice.chunks_exact_mut(VM_PIECE).enumerate() {
                // Leave even-indexed pieces all zeros; fill odd pieces with guaranteed non-zero byte.
                if i % 2 == 1 {
                    let byte = ((i as u8).wrapping_mul(13).wrapping_add(7)) | 1;
                    chunk.fill(byte);
                }
            }
        }
        assert_eq!(os_committed_bytes_in_range(base_addr, LEN), LEN);

        // Aggressive zero-piece decommit reclaims exactly half (32 MiB) while keeping VA.
        let zero_reclaimed = block.decommit_zero_pieces();
        assert_eq!(zero_reclaimed, 32 * MIB);
        assert_eq!(block.committed_bytes(), 32 * MIB);
        assert_eq!(os_committed_bytes_in_range(base_addr, LEN), 32 * MIB);

        let query_state = |offset: usize| -> (u32, usize) {
            // SAFETY: plain zeroed struct.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: querying inside live `base_addr` reservation.
            let n = unsafe {
                VirtualQuery(
                    (base_addr + offset) as *const std::ffi::c_void,
                    &mut m,
                    size_of_val(&m),
                )
            };
            assert_ne!(n, 0);
            (m.State, m.AllocationBase as usize)
        };
        assert_eq!(query_state(0), (MEM_RESERVE, base_addr));
        assert_eq!(query_state(VM_PIECE), (MEM_COMMIT, base_addr));

        // Decommit the rest cleanly; OS kernel commit for the reservation drops to 0 while keeping VA.
        let rest_reclaimed = block.decommit_all();
        assert_eq!(rest_reclaimed, 32 * MIB);
        assert_eq!(block.committed_bytes(), 0);
        assert_eq!(os_committed_bytes_in_range(base_addr, LEN), 0);
        assert_eq!(query_state(VM_PIECE), (MEM_RESERVE, base_addr));

        // Re-materialize via `prepare_for_overwrite` at same VA base without corruption.
        assert!(block.prepare_for_overwrite(16 * MIB));
        assert_eq!(block.base.as_ptr() as usize, base_addr, "MEM_RESERVE VA preserved");
        assert_eq!(block.committed_bytes(), 16 * MIB);
        assert_eq!(os_committed_bytes_in_range(base_addr, LEN), 16 * MIB);
        let slice = block.as_mut_slice();
        assert!(slice.iter().all(|&b| b == 0), "newly re-committed pages are zero-filled");
        slice[0] = 0xAB;
        slice[16 * MIB - 1] = 0xCD;
        assert_eq!((slice[0], slice[16 * MIB - 1]), (0xAB, 0xCD));
    }

    #[test]
    fn stress_decommit_recommit_reuse_no_corruption_and_os_commit_drops() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };

        let _guard = commit_test_lock();
        const TOTAL_BUFFERS: usize = 4;
        const BUF_LEN: usize = 64 * MIB;

        let decomm_before = COUNTERS.vm_decommitted_bytes.load(Relaxed);
        let recomm_before = COUNTERS.vm_recommitted_bytes.load(Relaxed);

        let mut bufs: Vec<SnapshotBuffer> = (0..TOTAL_BUFFERS)
            .map(|_| SnapshotBuffer::new_zeroed(BUF_LEN))
            .collect();
        let va_bases: Vec<usize> = bufs
            .iter()
            .map(|b| match b {
                SnapshotBuffer::Large(vm) => vm.base.as_ptr() as usize,
                SnapshotBuffer::Small(_) => panic!("48 MiB must allocate VmBlock"),
            })
            .collect();

        // Populate every 4 KiB page across 192 MiB so OS commit charges all physical backing.
        for (idx, buf) in bufs.iter_mut().enumerate() {
            let slice = buf.as_mut_slice();
            for page in (0..BUF_LEN).step_by(4096) {
                let tag = ((idx as u8).wrapping_mul(31)) ^ ((page >> 12) as u8);
                slice[page] = tag;
                slice[page + 4095] = tag.wrapping_add(1);
            }
        }

        // Stress 3 cycles of partial-range decommit, full decommit with immediate OS
        // commit verification, and re-commit + verification at unchanged VA bases.
        for round in 0..3usize {
            // Partial decommit of middle `[16 MiB, 32 MiB)` keeps head and tail intact.
            for (idx, buf) in bufs.iter_mut().enumerate() {
                let reclaimed = buf.decommit_byte_range(16 * MIB, 32 * MIB);
                assert_eq!(reclaimed, 16 * MIB);
                let base = va_bases[idx];
                // SAFETY: plain zeroed structs.
                let (mut m_head, mut m_mid): (MEMORY_BASIC_INFORMATION, MEMORY_BASIC_INFORMATION) =
                    unsafe { (std::mem::zeroed(), std::mem::zeroed()) };
                // SAFETY: valid VA queries inside `buf`.
                unsafe {
                    assert_ne!(VirtualQuery(base as *const _, &mut m_head, size_of_val(&m_head)), 0);
                    assert_ne!(
                        VirtualQuery((base + 16 * MIB) as *const _, &mut m_mid, size_of_val(&m_mid)),
                        0
                    );
                }
                assert_eq!(m_head.State, MEM_COMMIT);
                assert_eq!(m_mid.State, MEM_RESERVE);

                // Re-commit middle range cleanly and verify untouched head pattern + zeroed middle.
                assert!(buf.ensure_range_committed(16 * MIB, 32 * MIB));
                let s = buf.as_slice();
                for page in (0..16 * MIB).step_by(4096) {
                    let expected = ((idx as u8).wrapping_mul(31))
                        ^ ((page >> 12) as u8)
                        ^ (round as u8);
                    assert_eq!(s[page], expected);
                }
                assert!(s[16 * MIB..32 * MIB].iter().all(|&b| b == 0));
                // Touch middle pages before full-decommit commit measurement.
                let s_mut = buf.as_mut_slice();
                for page in (16 * MIB..32 * MIB).step_by(4096) {
                    s_mut[page] = 0x55;
                }
            }

            // Full decommit of all 256 MiB must drop OS private commit immediately.
            let commit_before_decommit = host_status().process_private_bytes;
            let mut round_reclaimed = 0usize;
            for buf in &mut bufs {
                round_reclaimed += buf.decommit_all();
                assert_eq!(buf.committed_bytes(), 0);
            }
            assert_eq!(round_reclaimed, TOTAL_BUFFERS * BUF_LEN);

            let commit_decommitted = host_status().process_private_bytes;
            for &base in &va_bases {
                assert_eq!(os_committed_bytes_in_range(base, BUF_LEN), 0);
            }
            eprintln!(
                "[ram_trim stress round {round}] private commit before={:.1} MiB, after={:.1} MiB, delta=-{:.1} MiB",
                commit_before_decommit as f64 / (MIB as f64),
                commit_decommitted as f64 / (MIB as f64),
                commit_before_decommit.saturating_sub(commit_decommitted) as f64 / (MIB as f64),
            );
            assert!(
                commit_decommitted <= commit_before_decommit.saturating_sub(128 * MIB as u64),
                "round {round}: expected OS commit to drop >=128 MiB (before={commit_before_decommit}, after={commit_decommitted})"
            );

            // Re-materialize all buffers at identical VA bases and write next-round pattern.
            for (idx, buf) in bufs.iter_mut().enumerate() {
                assert!(buf.prepare_for_overwrite(BUF_LEN));
                let cur_base = match buf {
                    SnapshotBuffer::Large(vm) => vm.base.as_ptr() as usize,
                    SnapshotBuffer::Small(_) => unreachable!(),
                };
                assert_eq!(cur_base, va_bases[idx], "MEM_RESERVE base changed on reuse");
                let s = buf.as_mut_slice();
                assert!(s.iter().all(|&b| b == 0), "re-committed buffer not zeroed");
                for page in (0..BUF_LEN).step_by(4096) {
                    let tag = ((idx as u8).wrapping_mul(31))
                        ^ ((page >> 12) as u8)
                        ^ ((round + 1) as u8);
                    s[page] = tag;
                    s[page + 4095] = tag.wrapping_add(3);
                }
            }
        }

        assert!(COUNTERS.vm_decommitted_bytes.load(Relaxed) > decomm_before);
        assert!(COUNTERS.vm_recommitted_bytes.load(Relaxed) > recomm_before);
        assert!(diagnostics_line().contains("ram trim"));
    }
}
