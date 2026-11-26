//! The environment `anyxboxx run` gives VIctory. The tuned set is the one VIctory's own launcher,
//! `third_party/victory/scripts/play.ps1` (avatarmanz/VIctory-7900XTX), plays with;
//! `tests/test_anyxboxx_run.py` compares both.

use std::ffi::OsString;

use super::{Profile, RunArgs};

/// `third_party/victory/scripts/play.ps1`'s settings without `-Safe`, in its order, less
/// `VICTORY_D3D12_DRED=light` (DRED stays opt-in: `--dred`). Measured by avatarmanz on Windows 11 with
/// an RX 7900 XTX (VIctory's README); not measured on NVIDIA with this base. `VICTORY_RELAY=2` also
/// makes stage 3 of the late waits VIctory's default (`VICTORY_LATE_WAIT` unset). Applied unless
/// `--no-tuned-settings`.
pub const TUNED: &[(&str, &str)] = &[
    ("VICTORY_TEXTURE_NO_WAIT", "1"),
    ("VICTORY_RELAY", "2"),
    ("VICTORY_WIDE_BACK", "1"),
    ("VICTORY_CARD_REHASH_RUNS", "1"),
    ("VICTORY_JOIN_COPIES", "1"),
    ("VICTORY_INDIRECT_WINDOW", "1"),
    ("VICTORY_RELAY_COPY", "1"),
    ("VICTORY_CARD_STEPS", "2"),
    ("VICTORY_INDIRECT_BATCH", "2"),
    ("VICTORY_MAP_THREADS", "1"),
    ("VICTORY_TITLE_PRIORITIES", "0"),
    ("VICTORY_CARD_AWAKE", "1"),
    ("VICTORY_SHOW_AFTER", "1"),
    ("VICTORY_AMBIENT", "0.8"),
    ("VICTORY_THIN", "1"),
    ("VICTORY_JOIN_PARTS", "8"),
    ("VICTORY_UPLOAD_AHEAD", "1"),
];

/// What Linux (Proton) + AMD (RADV through vkd3d-proton) adds to [`TUNED`], measured on an
/// RX 9070 XT (Fedora 44, Mesa 26.2.7 with `tools/linux/mesa`'s patch, GE-Proton11-12, 2026-11-26).
pub const LINUX_AMD_EXTRA: &[(&str, &str)] = &[
    // Torn relay fetches under Wine broke skinned meshes; the check keeps VICTORY_CARD_STEPS=2 safe.
    ("VICTORY_FETCH_CHECK", "1"),
    // The 500 ms default froze a frame once a second while RADV compiles; the programs the title
    // dispatches once (faces) are made and waited for at creation whatever this says.
    ("VICTORY_PIPELINE_WAIT_MS", "0"),
    // Renderer patch 1 (the constants heap read in place). It checks itself at start and stays off
    // where the card does not see the title's memory in place (RADV without tools/linux/anonless).
    ("VICTORY_PORT_CONSTANTS", "1"),
    // The title asks for 1,245,184 views, more than vkd3d-proton's default cap.
    ("VKD3D_CONFIG", "descriptor_heap"),
];

/// Linux (Proton) + AMD: [`TUNED`] then [`LINUX_AMD_EXTRA`] (a test checks it is exactly that).
/// Kept in sync with `crates/anyxboxx-launcher/profiles.toml` (`linux-amd`).
pub const LINUX_AMD: &[(&str, &str)] = &[
    ("VICTORY_TEXTURE_NO_WAIT", "1"),
    ("VICTORY_RELAY", "2"),
    ("VICTORY_WIDE_BACK", "1"),
    ("VICTORY_CARD_REHASH_RUNS", "1"),
    ("VICTORY_JOIN_COPIES", "1"),
    ("VICTORY_INDIRECT_WINDOW", "1"),
    ("VICTORY_RELAY_COPY", "1"),
    ("VICTORY_CARD_STEPS", "2"),
    ("VICTORY_INDIRECT_BATCH", "2"),
    ("VICTORY_MAP_THREADS", "1"),
    ("VICTORY_TITLE_PRIORITIES", "0"),
    ("VICTORY_CARD_AWAKE", "1"),
    ("VICTORY_SHOW_AFTER", "1"),
    ("VICTORY_AMBIENT", "0.8"),
    ("VICTORY_THIN", "1"),
    ("VICTORY_JOIN_PARTS", "8"),
    ("VICTORY_UPLOAD_AHEAD", "1"),
    ("VICTORY_FETCH_CHECK", "1"),
    ("VICTORY_PIPELINE_WAIT_MS", "0"),
    ("VICTORY_PORT_CONSTANTS", "1"),
    ("VKD3D_CONFIG", "descriptor_heap"),
];

/// Picture first (`--quality`): applied after the tuned set, before `--override`. Draw-data windows
/// stay in the PC's memory (read fresh each frame), fewer buffer copies on the card, a smaller
/// texture staging pool. Not measured on this base.
pub const QUALITY: &[(&str, &str)] = &[
    ("VICTORY_CARD_WINDOWS_MB", "0"),
    ("VICTORY_CARD_BUFFERS_MB", "1024"),
    ("VICTORY_TEXTURE_STAGE_MB", "192"),
];

pub const GAME_DIR: &str = "VICTORY_GAME_DIR";
/// VIctory turns DRED on when the variable is set and not `0` (diagnostics builds): only set when asked.
pub const DRED: &str = "VICTORY_D3D12_DRED";
/// VIctory's scripted keys: `<seconds>:<key>,...`; Enter also presses the first pad's A.
pub const KEYS: &str = "VICTORY_KEYS";
pub const CARD_REHASH: &str = "VICTORY_CARD_REHASH_RUNS";
pub const SKIP_PIPELINES: &str = "VICTORY_SKIP_PIPELINES";
/// `--auto-a`: Enter (A) every [`AUTO_A_EVERY_S`] seconds until [`AUTO_A_FOR_S`] seconds.
pub const AUTO_A_EVERY_S: u32 = 4;
pub const AUTO_A_FOR_S: u32 = 150;

/// The `VICTORY_KEYS` script of `--auto-a`: `4:Enter,8:Enter,...,148:Enter`.
pub fn auto_a_keys() -> String {
    (1..)
        .map(|n| n * AUTO_A_EVERY_S)
        .take_while(|&at| at < AUTO_A_FOR_S)
        .map(|at| format!("{at}:Enter"))
        .collect::<Vec<_>>()
        .join(",")
}

fn pairs(
    table: &'static [(&'static str, &'static str)],
) -> impl Iterator<Item = (String, OsString)> {
    table
        .iter()
        .map(|(k, v)| ((*k).to_string(), OsString::from(*v)))
}

/// The game directory, selected profile and dedicated switches, before explicit overrides.
pub fn base(args: &RunArgs) -> Vec<(String, OsString)> {
    let mut env = vec![(GAME_DIR.to_string(), args.game_dir.clone().into_os_string())];
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
    if args.auto_a {
        env.push((KEYS.to_string(), OsString::from(auto_a_keys())));
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
