//! The environment `anyxboxx run` gives VIctory: the same variables, values and conditions as
//! main's `tools/lancer_victory.ps1` (Neptune base). The golden copy of that script is
//! `tests/fixtures/lancer_victory_neptune.ps1`; `tests/test_anyxboxx_run.py` compares both.

use std::ffi::OsString;
use std::path::Path;

use super::{Profile, RunArgs};

/// Neptune (VIctory Edit.) 2026-11-25c's play.ps1 for the Series X path, with the measured changes
/// of 2026-11-25 on top. Applied unless `--no-tuned-settings`, in this order.
pub const TUNED: &[(&str, &str)] = &[
    ("VICTORY_CARD_WINDOWS_MB", "1536"),
    ("VICTORY_BAKE_TEXTURES_MS", "10000"),
    ("VICTORY_EARLY_TEXTURES_MS", "40"),
    ("VICTORY_WIDE_MERGE", "1"),
    ("VICTORY_RELAY_OWN_QUEUE", "1"),
    ("VICTORY_UNDER_WRITES_WORDS", "1"),
    ("VICTORY_UNDER_WRITES_RUNS", "100000"),
    ("VICTORY_UNDER_WRITES_MASKS", "1"),
    ("VICTORY_RELAY", "2"),
    ("VICTORY_WIDE_REFRESH", "1"),
    ("VICTORY_ASK_ONCE", "1"),
    ("VICTORY_TEXTURE_NO_WAIT", "1"),
    ("VICTORY_CARD_REHASH_RUNS", "1"),
    ("VICTORY_JOIN_COPIES", "1"),
    ("VICTORY_D3D12_BACKGROUND", "0"),
    ("VICTORY_LATE_WAIT_LISTS", "1024"),
    ("VICTORY_WINDOW_TAILS", "1"),
    ("VICTORY_TEXTURE_STAGE_MB", "512"),
    ("VICTORY_CARD_BUFFERS_MIN_KB", "64"),
    ("VICTORY_ASK_QUIET", "8"),
    ("VICTORY_COPY_CHANGED", "1"),
    ("VICTORY_QUIET_BACK", "1"),
    ("VICTORY_LIST_POOL_MOST", "2048"),
    ("VICTORY_LIST_POOL_MB", "512"),
    ("VICTORY_WIDE_THREADS", "12"),
    ("VICTORY_UNDER_WRITES_MASKS_LEAST", "1"),
    ("VICTORY_INDIRECT_RAW", "2"),
    ("VICTORY_TEXTURE_HASH_EVERY", "32"),
    ("VICTORY_LEVELS_ON_DEMAND", "1"),
    ("VICTORY_WINDOW_KB", "1024"),
    ("VICTORY_CARD_BUFFERS_MB", "1536"),
    ("VICTORY_CARD_WINDOW_UPLOAD", "1"),
    ("VICTORY_DROP_DXIL", "1"),
    ("VICTORY_DESC_DATA", "1"),
    ("VICTORY_VRAM_IDLE_MS", "8000,2000,1000"),
    ("VICTORY_PIPELINE_WAIT_MS", "2500"),
    ("VICTORY_TEXTURE_ROOM_PERCENT", "93"),
    ("VICTORY_HEAP_PASS_DIRTY", "1"),
    ("VICTORY_COPY_COALESCE", "256"),
    // PR #358 (measured 2026-11-26, RTX 4070 SUPER, Series X): 13.4 -> 15.2 fps median.
    ("VICTORY_WINDOW_SKIP_SAME", "1"),
    ("VICTORY_MERGE_SUBMISSIONS", "1"),
    ("VICTORY_PAGE_WRITES", "5"),
    // PR #383 (fxceless/victory-gta6 ports), measured 2026-11-26 in one run alternating every 20 s (RTX 4070 SUPER, Series X): 12.7 -> 14.8 fps median, +17 %.
    ("VICTORY_STEP_INDEX", "1"),
    ("VICTORY_CARD_SYNC_BATCH", "1"),
    ("VICTORY_PARALLEL_REFRESH", "1"),
    // The full scan of kept heap slots every 16 passes instead of every pass, measured 2026-11-26 in one run alternating every 20 s (RTX 4070 SUPER, Series X): 14.2 -> 15.2 fps median, frames over 100 ms 4 -> 1.
    ("VICTORY_HEAP_SCAN_ALT", "1"),
    // PR #395: textures picked by index made ready on two threads of their own, measured 2026-11-26 in two 9-minute runs (RTX 5080, Series X): longest texture pass 1,576 -> 213 ms, 10th-percentile fps 1.4 -> 4.1.
    ("VICTORY_TEXTURE_PREPARE_THREADS", "2"),
];

/// Linux (Proton) + AMD measured set (PR #199, RX 9070 XT / RADV / vkd3d-proton).
/// Kept in sync with `crates/anyxboxx-launcher/profiles.toml` (`linux-amd`).
pub const LINUX_AMD: &[(&str, &str)] = &[
    ("VICTORY_SHADER_LOOP_LIMIT", "256"),
    ("VICTORY_LATE_WAIT", "0"),
    ("VICTORY_WIDE_BACK", "0"),
    ("VICTORY_PIPELINE_WAIT_MS", "50"),
    ("VICTORY_TEXTURE_BUDGET_MB", "8192"),
    ("VICTORY_TEXTURE_NO_WAIT", "1"),
    ("VICTORY_CARD_BUFFERS_MIN_MB", "1"),
    ("VICTORY_CARD_BUFFERS_MB", "1024"),
    ("VKD3D_CONFIG", "descriptor_heap"),
];

/// Picture first (`--quality`): applied after the tuned set, before `--override`.
pub const QUALITY: &[(&str, &str)] = &[
    ("VICTORY_CARD_WINDOWS_MB", "0"),
    ("VICTORY_CARD_BUFFERS_MB", "1024"),
    ("VICTORY_LIST_POOL_MOST", "256"),
    ("VICTORY_LIST_POOL_MB", "256"),
    ("VICTORY_TEXTURE_STAGE_MB", "192"),
];

/// Set on every launch, before the tuned set (the script sets them first).
pub const ALWAYS: &[(&str, &str)] = &[("VICTORY_DISCORD", "0"), ("VICTORY_OCEAN_READBACK", "1")];

pub const GAME_DIR: &str = "VICTORY_GAME_DIR";
/// VIctory turns DRED on when the variable exists, whatever its value: only set when asked.
pub const DRED: &str = "VICTORY_D3D12_DRED";
pub const AUTO_A: &str = "VICTORY_AUTO_A";
pub const FRAMETIME_DUMP: &str = "VICTORY_FRAMETIME_DUMP";
pub const CARD_REHASH: &str = "VICTORY_CARD_REHASH_RUNS";
pub const SKIP_PIPELINES: &str = "VICTORY_SKIP_PIPELINES";
pub const XG_DLL: &str = "VICTORY_XG_DLL";
pub const XG_MODE: &str = "VICTORY_XG_MODE";

fn pairs(
    table: &'static [(&'static str, &'static str)],
) -> impl Iterator<Item = (String, OsString)> {
    table
        .iter()
        .map(|(k, v)| ((*k).to_string(), OsString::from(*v)))
}

/// Variables set before the XG library is checked, in the script's order.
pub fn base(args: &RunArgs) -> Vec<(String, OsString)> {
    let mut env = vec![(GAME_DIR.to_string(), args.game_dir.clone().into_os_string())];
    env.extend(pairs(ALWAYS));
    if !args.no_tuned_settings {
        match args.profile {
            Profile::WindowsNvidia => env.extend(pairs(TUNED)),
            Profile::LinuxAmd => env.extend(pairs(LINUX_AMD)),
            Profile::Defaults => {}
        }
    }
    if args.quality {
        env.extend(pairs(QUALITY));
    }
    for (name, value) in &args.overrides {
        env.push((name.clone(), OsString::from(value)));
    }
    if args.auto_a {
        env.push((AUTO_A.to_string(), OsString::from("150")));
    }
    if args.frametime_dump {
        env.push((FRAMETIME_DUMP.to_string(), OsString::from("1")));
    }
    if args.dred {
        env.push((DRED.to_string(), OsString::from("1")));
    }
    if args.batch_card_rehash {
        env.push((CARD_REHASH.to_string(), OsString::from("1")));
    }
    if let Some(skip) = args.skip_pipelines.as_deref().filter(|s| !s.is_empty()) {
        env.push((SKIP_PIPELINES.to_string(), OsString::from(skip)));
    }
    env
}

/// Your own copy of the GDK's offline PC `xg_xs.dll` (never in the repository): Microsoft's
/// library lays out the textures. `absolute` is the resolved path of an existing file.
pub fn xg(absolute: &Path, mode: &str) -> [(String, OsString); 2] {
    [
        (XG_DLL.to_string(), absolute.as_os_str().to_owned()),
        (XG_MODE.to_string(), OsString::from(mode)),
    ]
}
