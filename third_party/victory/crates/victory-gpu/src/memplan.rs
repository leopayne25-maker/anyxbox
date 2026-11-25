//! (Neptune) Memory caps that follow the machine.
//!
//! The pools and caps below were tuned by hand for one machine: a 16 GB card whose budget
//! the system sets at about 14.4 GB, and a PC with 32 GB of RAM. They are now sized from
//! the card's budget (`Host::video_memory`, the first answer) and the PC's total RAM
//! (`GlobalMemoryStatusEx`), and the numbers are all here, in [`REFERENCE`] and the rules
//! of [`plan`].
//!
//! The rule, per cap: `cap = max(floor, reference * min(have, have_reference) / have_reference)`
//! rounded down to the cap's step, where
//! * the caps that live in the card's memory (windows and title buffers on the card) follow
//!   the budget: `have` is the budget, `have_reference` is 14.4 GiB;
//! * the caps that live in the PC's memory (idle host lists, texture and staging upload
//!   buffers, the window upload pool) follow the RAM: `have` is the total RAM and
//!   `have_reference` is 31 GiB (Windows reports a little under what is installed, so a
//!   "32 GB" PC must count as the reference one).
//!
//! So a machine at least as big as the reference gets exactly the reference values (the ones
//! the launcher used to set), a smaller one gets a share of them by its size, and nothing
//! gets less than its floor (the value the code had before the launcher raised it, or a
//! small one where there was none) nor more than the reference. A switch
//! (`VICTORY_CARD_WINDOWS_MB` and the rest, see [`Plan::apply`]) still wins when it is set:
//! an explicit value is the user's choice.

use std::sync::OnceLock;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// The machine the reference values were tuned for.
const REF_BUDGET: u64 = 14745 * MIB; // 14.4 GiB, rounded down to a whole MiB
const REF_RAM: u64 = 31 * GIB;

/// The caps, in bytes (the list pool's count in lists).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Most bytes of windows whose copy lives on the card (`VICTORY_CARD_WINDOWS_MB`).
    pub card_windows: u64,
    /// Most bytes of title buffers whose copy lives on the card (`VICTORY_CARD_BUFFERS_MB`).
    pub card_buffers: u64,
    /// Most idle host lists kept (`VICTORY_LIST_POOL_MOST`).
    pub list_pool_lists: usize,
    /// Most upload bytes the idle lists keep (`VICTORY_LIST_POOL_MB`).
    pub list_pool_bytes: u64,
    /// Bytes of texture memory staged per frame (`VICTORY_TEXTURE_STAGE_MB`).
    pub texture_stage: u64,
    /// Most bytes the staging buffers' pool keeps (`VICTORY_STAGE_POOL_MB`).
    pub stage_pool: u64,
    /// Most bytes of the window upload pool (`VICTORY_CARD_WINDOW_UPLOAD_MB`).
    pub window_upload: u64,
}

/// What the launcher set for the reference machine (the window upload pool and the stage
/// pool are the code's own old defaults).
pub(crate) const REFERENCE: Plan = Plan {
    card_windows: 4096 * MIB,
    card_buffers: 1536 * MIB,
    list_pool_lists: 2048,
    list_pool_bytes: 2048 * MIB,
    texture_stage: 512 * MIB,
    stage_pool: 256 * MIB,
    window_upload: 512 * MIB,
};

/// `reference` scaled by `have / of` (never above 1), rounded down to `step`, not below
/// `floor`, and never above `reference`.
fn scaled(reference: u64, have: u64, of: u64, step: u64, floor: u64) -> u64 {
    let part = (u128::from(reference) * u128::from(have.min(of)) / u128::from(of)) as u64;
    (part / step * step).max(floor).min(reference)
}

/// The caps for a card whose budget is `budget_bytes` and a PC with `ram_total_bytes` of RAM.
/// Pure: it reads nothing.
pub(crate) fn plan(budget_bytes: u64, ram_total_bytes: u64) -> Plan {
    let v = |reference: u64, floor_mib: u64| scaled(reference, budget_bytes, REF_BUDGET, 64 * MIB, floor_mib * MIB);
    let r = |reference: u64, floor_mib: u64| scaled(reference, ram_total_bytes, REF_RAM, 32 * MIB, floor_mib * MIB);
    Plan {
        // Floors: 256 MB is what even a small card can spare (before, windows had none
        // below a budget of 14 GiB).
        card_windows: v(REFERENCE.card_windows, 256),
        card_buffers: v(REFERENCE.card_buffers, 256),
        // Floors: the code's own old defaults (128 lists, 256 MB, 192 MB a frame), and half
        // the reference for the stage and upload pools.
        list_pool_lists: scaled(REFERENCE.list_pool_lists as u64, ram_total_bytes, REF_RAM, 16, 128) as usize,
        list_pool_bytes: r(REFERENCE.list_pool_bytes, 256),
        texture_stage: r(REFERENCE.texture_stage, 192),
        stage_pool: r(REFERENCE.stage_pool, 128),
        window_upload: r(REFERENCE.window_upload, 128),
    }
}

/// Switch names, in the order of [`Plan::apply`].
const SWITCHES: [&str; 7] = [
    "VICTORY_CARD_WINDOWS_MB",
    "VICTORY_CARD_BUFFERS_MB",
    "VICTORY_LIST_POOL_MOST",
    "VICTORY_LIST_POOL_MB",
    "VICTORY_TEXTURE_STAGE_MB",
    "VICTORY_STAGE_POOL_MB",
    "VICTORY_CARD_WINDOW_UPLOAD_MB",
];

impl Plan {
    /// This plan with every switch that `get` answers with a number (MiB, or lists for
    /// `VICTORY_LIST_POOL_MOST`) put over its cap, and the names of the switches that did so.
    pub(crate) fn apply(mut self, get: &dyn Fn(&str) -> Option<String>) -> (Plan, Vec<&'static str>) {
        let mut set = Vec::new();
        let mut number = |name: &'static str| {
            let v = get(name)?.trim().parse::<u64>().ok()?;
            set.push(name);
            Some(v)
        };
        let mb = |v: Option<u64>, old: u64| v.map_or(old, |v| v << 20);
        self.card_windows = mb(number(SWITCHES[0]), self.card_windows);
        self.card_buffers = mb(number(SWITCHES[1]), self.card_buffers);
        self.list_pool_lists = number(SWITCHES[2]).map_or(self.list_pool_lists, |v| v as usize);
        self.list_pool_bytes = mb(number(SWITCHES[3]), self.list_pool_bytes);
        self.texture_stage = mb(number(SWITCHES[4]), self.texture_stage);
        self.stage_pool = mb(number(SWITCHES[5]), self.stage_pool);
        self.window_upload = mb(number(SWITCHES[6]), self.window_upload);
        (self, set)
    }
}

/// The PC's total RAM in bytes, from `GlobalMemoryStatusEx`.
fn ram_total() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullTotalPhys)
}

/// The caps in force: the plan for this machine with the switches over it. Worked out once,
/// at the first use after the host device has answered for the card's budget, and logged then.
/// With no answer (no device, or no adapter) the caps are those of a card with no budget and
/// nothing is kept, so a later use asks again.
pub(crate) fn current() -> Plan {
    static PLAN: OnceLock<Plan> = OnceLock::new();
    if let Some(p) = PLAN.get() {
        return *p;
    }
    let budget = crate::host::host().and_then(|h| h.video_memory()).map(|[b, ..]| b);
    let ram = ram_total();
    let base = plan(budget.unwrap_or(0), ram.unwrap_or(REF_RAM));
    let (plan, set) = base.apply(&|name| std::env::var(name).ok());
    if budget.is_none() {
        return plan;
    }
    *PLAN.get_or_init(|| {
        let mib = |b: u64| b >> 20;
        tracing::info!(
            "memory caps follow the machine: card budget {} MB, PC RAM {} MB; card windows {} MB, card buffers {} MB, \
             idle lists {} ({} MB), texture stage {} MB a frame, stage pool {} MB, window upload pool {} MB; \
             set by a switch: {}",
            mib(budget.unwrap_or(0)),
            mib(ram.unwrap_or(0)),
            mib(plan.card_windows),
            mib(plan.card_buffers),
            plan.list_pool_lists,
            mib(plan.list_pool_bytes),
            mib(plan.texture_stage),
            mib(plan.stage_pool),
            mib(plan.window_upload),
            if set.is_empty() { "none".to_string() } else { set.join(", ") },
        );
        plan
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A card of `card_gb` GB (the budget is nine tenths of it) and a PC of `ram_gb` GiB.
    fn machine(card_gb: f64, ram_gb: u64) -> Plan {
        plan((card_gb * 0.9 * GIB as f64) as u64, ram_gb * GIB)
    }

    #[test]
    fn reference_machine_gets_the_launchers_values() {
        let p = plan(REF_BUDGET, 32 * GIB);
        assert_eq!(p, REFERENCE);
        assert_eq!(p.card_windows, 4096 << 20);
        assert_eq!(p.card_buffers, 1536 << 20);
        assert_eq!((p.list_pool_lists, p.list_pool_bytes), (2048, 2048 << 20));
        assert_eq!((p.texture_stage, p.stage_pool, p.window_upload), (512 << 20, 256 << 20, 512 << 20));
        // A 16 GB card's budget of 14.4 GiB, a bigger machine, and a PC that reports a
        // little under 32 GiB are the reference too.
        assert_eq!(machine(16.0, 32), REFERENCE);
        assert_eq!(plan(24 * GIB, 128 * GIB), REFERENCE);
        assert_eq!(plan(REF_BUDGET, 31 * GIB + 900 * MIB), REFERENCE);
    }

    #[test]
    fn smaller_cards_and_less_ram_get_less_but_not_below_floors() {
        let (c8, c12) = (machine(8.0, 32), machine(12.0, 32));
        assert!(c8.card_windows < c12.card_windows && c12.card_windows < REFERENCE.card_windows);
        assert!(c8.card_buffers < c12.card_buffers && c12.card_buffers < REFERENCE.card_buffers);
        // 7.2 GiB of 14.4 is half; 10.8 is three quarters.
        assert_eq!((c8.card_windows, c8.card_buffers), (2048 << 20, 768 << 20));
        assert_eq!((c12.card_windows, c12.card_buffers), (3072 << 20, 1152 << 20));
        // RAM caps do not move with the card, and VRAM caps do not move with the RAM.
        assert_eq!(c8.list_pool_bytes, REFERENCE.list_pool_bytes);
        assert_eq!(c8.texture_stage, REFERENCE.texture_stage);
        let r16 = machine(16.0, 16);
        assert_eq!((r16.card_windows, r16.card_buffers), (REFERENCE.card_windows, REFERENCE.card_buffers));
        assert!(r16.list_pool_lists < 2048 && r16.list_pool_bytes < REFERENCE.list_pool_bytes);
        assert!(r16.texture_stage < REFERENCE.texture_stage && r16.stage_pool < REFERENCE.stage_pool);
        assert!(r16.window_upload < REFERENCE.window_upload);
        // 16 GiB of 31 is a little over half; the floors hold for the rest.
        assert_eq!(r16.list_pool_lists, 1056);
        assert_eq!(r16.texture_stage, 256 << 20);
        assert_eq!(r16.stage_pool, 128 << 20);
        let tiny = plan(GIB, GIB);
        assert_eq!((tiny.card_windows, tiny.card_buffers), (256 << 20, 256 << 20));
        assert_eq!(tiny.list_pool_lists, 128);
        assert_eq!((tiny.list_pool_bytes, tiny.texture_stage), (256 << 20, 192 << 20));
        assert_eq!((tiny.stage_pool, tiny.window_upload), (128 << 20, 128 << 20));
    }

    #[test]
    fn nothing_exceeds_the_reference() {
        for card in [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0] {
            for ram in [1, 4, 8, 16, 24, 32, 64, 128] {
                let p = machine(card, ram);
                assert!(p.card_windows <= REFERENCE.card_windows && p.card_buffers <= REFERENCE.card_buffers);
                assert!(p.list_pool_lists <= REFERENCE.list_pool_lists && p.list_pool_bytes <= REFERENCE.list_pool_bytes);
                assert!(p.texture_stage <= REFERENCE.texture_stage && p.stage_pool <= REFERENCE.stage_pool);
                assert!(p.window_upload <= REFERENCE.window_upload);
            }
        }
    }

    #[test]
    fn a_switch_wins_over_the_plan() {
        let base = machine(8.0, 16);
        let get = |name: &str| match name {
            "VICTORY_CARD_WINDOWS_MB" => Some(" 4096 ".to_string()),
            "VICTORY_LIST_POOL_MOST" => Some("3000".to_string()),
            "VICTORY_STAGE_POOL_MB" => Some("0".to_string()),
            "VICTORY_CARD_BUFFERS_MB" => Some("garbage".to_string()),
            _ => None,
        };
        let (p, set) = base.apply(&get);
        assert_eq!(p.card_windows, 4096 << 20);
        assert_eq!(p.list_pool_lists, 3000);
        assert_eq!(p.stage_pool, 0);
        assert_eq!(p.card_buffers, base.card_buffers, "an unreadable value is no choice");
        assert_eq!(p.texture_stage, base.texture_stage);
        assert_eq!(set, ["VICTORY_CARD_WINDOWS_MB", "VICTORY_LIST_POOL_MOST", "VICTORY_STAGE_POOL_MB"]);
        assert!(base.apply(&|_| None).1.is_empty());
    }
}
