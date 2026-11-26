use std::ffi::OsString;
use std::path::PathBuf;

use clap::Parser;

use super::*;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    run: RunArgs,
}

fn parse(args: &[&str]) -> RunArgs {
    Cli::parse_from(std::iter::once("anyxboxx").chain(args.iter().copied())).run
}

fn names(env: &[(String, OsString)]) -> Vec<&str> {
    env.iter().map(|(k, _)| k.as_str()).collect()
}

fn value(env: &[(String, OsString)], name: &str) -> Option<OsString> {
    env.iter()
        .rev()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.clone())
}

/// Always set, before the tuned set.
const HEAD: [&str; 1] = ["VICTORY_GAME_DIR"];

#[test]
fn tuned_settings_are_set_by_default_and_can_be_turned_off() {
    let env = settings::base(&parse(&["--game-dir", "G"]));
    assert_eq!(
        env[0],
        ("VICTORY_GAME_DIR".to_string(), OsString::from("G"))
    );
    assert_eq!(env.len(), HEAD.len() + TUNED.len());
    assert!(
        TUNED
            .iter()
            .all(|(k, v)| value(&env, k) == Some(OsString::from(v)))
    );
    let env = settings::base(&parse(&["--game-dir", "G", "--no-tuned-settings"]));
    assert_eq!(names(&env), HEAD);
}

#[test]
fn the_tuned_set_is_victorys_play_script_set() {
    // VIctory's scripts/play.ps1 (checked line by line in tests/test_anyxboxx_run.py).
    assert_eq!(TUNED.len(), 17);
    assert!(TUNED.contains(&("VICTORY_RELAY", "2")));
    assert!(TUNED.contains(&("VICTORY_CARD_STEPS", "2")));
    assert!(TUNED.contains(&("VICTORY_THIN", "1")));
    // Stage 3 of the late waits comes from the relay: the set leaves VICTORY_LATE_WAIT unset.
    assert!(TUNED.iter().all(|(k, _)| *k != "VICTORY_LATE_WAIT"));
}

#[test]
fn linux_amd_is_the_tuned_set_then_the_linux_additions() {
    let joined: Vec<_> = TUNED.iter().chain(LINUX_AMD_EXTRA).copied().collect();
    assert_eq!(LINUX_AMD, joined.as_slice());
    for wanted in [
        ("VICTORY_FETCH_CHECK", "1"),
        ("VICTORY_PIPELINE_WAIT_MS", "0"),
        ("VICTORY_PORT_CONSTANTS", "1"),
        ("VKD3D_CONFIG", "descriptor_heap"),
    ] {
        assert!(LINUX_AMD_EXTRA.contains(&wanted), "{wanted:?}");
    }
}

#[test]
fn quality_is_applied_after_the_tuned_set_and_before_overrides() {
    let env = environment(&parse(&[
        "--game-dir",
        "G",
        "--quality",
        "--override",
        "VICTORY_RELAY=0",
    ]))
    .unwrap();
    assert_eq!(
        value(&env, "VICTORY_CARD_WINDOWS_MB"),
        Some(OsString::from("0"))
    );
    assert_eq!(
        value(&env, "VICTORY_TEXTURE_STAGE_MB"),
        Some(OsString::from("192"))
    );
    assert_eq!(value(&env, "VICTORY_RELAY"), Some(OsString::from("0")));
    let quality_at = env
        .iter()
        .position(|(k, _)| k == "VICTORY_CARD_WINDOWS_MB")
        .unwrap();
    let override_at = env.iter().rposition(|(k, _)| k == "VICTORY_RELAY").unwrap();
    assert!(quality_at < override_at);
}

#[test]
fn overrides_must_be_victory_variables_in_key_value_form() {
    assert!(Cli::try_parse_from(["anyxboxx", "--game-dir", "G", "--override", "PATH=x"]).is_err());
    assert!(
        Cli::try_parse_from(["anyxboxx", "--game-dir", "G", "--override", "VICTORY_RELAY"])
            .is_err()
    );
    let args = parse(&["--game-dir", "G", "--override", "VICTORY_LATE_WAIT=12000"]);
    assert_eq!(
        args.overrides,
        [("VICTORY_LATE_WAIT".to_string(), "12000".to_string())]
    );
}

#[test]
fn the_relay_and_dred_are_never_in_the_tuned_table() {
    // Relay is set by play.ps1's set itself (VICTORY_RELAY=2); DRED stays opt-in (play.ps1's
    // `VICTORY_D3D12_DRED=light` is left out).
    assert!(
        TUNED
            .iter()
            .chain(LINUX_AMD)
            .all(|(k, _)| !k.contains("DRED"))
    );
}

#[test]
fn opt_in_switches_add_their_variable() {
    let env = settings::base(&parse(&[
        "--game-dir",
        "G",
        "--no-tuned-settings",
        "--dred",
        "--batch-card-rehash",
        "--skip-pipelines",
        "3",
        "--auto-a",
    ]));
    let mut expected = HEAD.to_vec();
    expected.extend([
        "VICTORY_KEYS",
        "VICTORY_D3D12_DRED",
        "VICTORY_CARD_REHASH_RUNS",
        "VICTORY_SKIP_PIPELINES",
    ]);
    assert_eq!(names(&env), expected);
    let keys = value(&env, "VICTORY_KEYS").unwrap();
    let keys = keys.to_str().unwrap();
    assert!(
        keys.starts_with("4:Enter,8:Enter,") && keys.ends_with(",148:Enter"),
        "{keys}"
    );
    assert_eq!(keys.split(',').count(), 37);
    assert_eq!(
        value(&env, "VICTORY_SKIP_PIPELINES"),
        Some(OsString::from("3"))
    );
    // An empty value is not passed, like `-SkipPipelines ''` in the script.
    let env = settings::base(&parse(&[
        "--game-dir",
        "G",
        "--no-tuned-settings",
        "--skip-pipelines",
        "",
    ]));
    assert_eq!(names(&env), HEAD);
}

#[test]
fn dred_needs_the_diag_profile() {
    assert!(parse(&["--game-dir", "G", "--dred"]).uses_diag());
    assert!(parse(&["--game-dir", "G", "--diag"]).uses_diag());
    assert!(!parse(&["--game-dir", "G"]).uses_diag());
}

#[test]
fn powershell_names_are_accepted() {
    let args = parse(&[
        "--GameDir",
        "G",
        "--Precompile",
        "--Warm",
        "--Quality",
        "--AutoA",
    ]);
    assert_eq!(args.game_dir, PathBuf::from("G"));
    assert!(args.precompile && args.warm && args.quality && args.auto_a);
}

#[test]
fn the_neptune_only_options_are_gone() {
    // xg_xs.dll, the frame-time CSV: nothing in this VIctory base reads them.
    for gone in ["--xg-dll", "--frametime-dump"] {
        assert!(
            Cli::try_parse_from(["anyxboxx", "--game-dir", "G", gone, "x"]).is_err(),
            "{gone}"
        );
    }
}

#[test]
fn a_repository_without_victory_is_refused_before_building() {
    let empty = std::env::temp_dir().join("anyxboxx-run-no-victory");
    let args = parse(&["--game-dir", "G"]);
    assert!(matches!(run(&args, Some(&empty)), Err(RunError::NoVictory)));
}

#[test]
fn a_prebuilt_victory_has_nothing_to_build() {
    let exe = std::env::current_exe().unwrap();
    let victory = Victory::prebuilt(&exe).unwrap();
    assert_eq!(victory.exe, exe);
    assert!(victory.build_command().is_none());
}

#[test]
fn a_located_victory_is_built_in_release_by_default_and_in_diag_on_request() {
    let repo = std::env::temp_dir().join(format!("anyxboxx-run-located-{}", std::process::id()));
    let victory = repo.join("third_party").join("victory");
    std::fs::create_dir_all(&victory).unwrap();
    std::fs::write(victory.join("Cargo.toml"), b"").unwrap();
    let release = Victory::locate(&repo, false).unwrap();
    let diag = Victory::locate(&repo, true).unwrap();
    std::fs::remove_dir_all(&repo).unwrap();
    assert!(
        release.exe.ends_with("release/victory.exe")
            || release.exe.ends_with(r"release\victory.exe")
    );
    assert!(diag.exe.ends_with(r"diag\victory.exe") || diag.exe.ends_with("diag/victory.exe"));
    let build = release.build_command().unwrap();
    assert_eq!(build.get_program(), "cargo");
    assert!(build.get_args().any(|a| a == "--release"));
    assert!(
        diag.build_command()
            .unwrap()
            .get_args()
            .any(|a| a == "diagnostics")
    );
}

#[test]
fn warm_archives_are_the_game_folder_rpf_files_only_sorted() {
    let dir = std::env::temp_dir().join(format!("anyxboxx-run-warm-{}", std::process::id()));
    let nested = dir.join("sub");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(dir.join("b.rpf"), b"").unwrap();
    std::fs::write(dir.join("A.RPF"), b"").unwrap();
    std::fs::write(dir.join("notes.txt"), b"").unwrap();
    std::fs::write(nested.join("deep.rpf"), b"").unwrap();
    let found = steps::warm_archives(&dir);
    let empty =
        std::env::temp_dir().join(format!("anyxboxx-run-warm-empty-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    let none = steps::warm_archives(&empty);
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&empty).unwrap();
    let names: Vec<String> = found
        .unwrap()
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["A.RPF", "b.rpf"]);
    assert!(matches!(none, Err(RunError::NoArchive(_))));
}

#[test]
fn a_missing_prebuilt_victory_is_refused_before_anything_runs() {
    let args = parse(&[
        "--game-dir",
        "G",
        "--victory-exe",
        "does-not-exist/victory.exe",
    ]);
    assert!(matches!(run(&args, None), Err(RunError::NoVictoryExe(_))));
}

#[test]
fn a_prebuilt_victory_does_not_need_the_repository() {
    // The game folder is checked after the exe: reaching NoGame means no repository was asked for.
    let exe = std::env::current_exe().unwrap();
    let args = parse(&[
        "--game-dir",
        "does-not-exist",
        "--VictoryExe",
        exe.to_str().unwrap(),
    ]);
    assert!(matches!(run(&args, None), Err(RunError::NoGame(_))));
}

#[test]
fn without_a_prebuilt_victory_the_repository_is_required() {
    let args = parse(&["--game-dir", "G"]);
    assert!(matches!(run(&args, None), Err(RunError::NoRepo)));
}

#[test]
fn linux_amd_profile_sets_its_variables_instead_of_windows_tuned() {
    let env = settings::base(&parse(&["--game-dir", "G", "--profile", "linux-amd"]));
    assert_eq!(env.len(), HEAD.len() + LINUX_AMD.len());
    assert!(
        LINUX_AMD
            .iter()
            .all(|(k, v)| value(&env, k) == Some(OsString::from(v)))
    );
    assert_eq!(
        value(&env, "VKD3D_CONFIG"),
        Some(OsString::from("descriptor_heap"))
    );
    assert_eq!(
        value(&env, "VICTORY_PIPELINE_WAIT_MS"),
        Some(OsString::from("0"))
    );
    let windows = settings::base(&parse(&["--game-dir", "G"]));
    assert_eq!(value(&windows, "VICTORY_PIPELINE_WAIT_MS"), None);
    assert_eq!(value(&windows, "VKD3D_CONFIG"), None);
}

#[test]
fn defaults_profile_and_no_tuned_settings_leave_only_base_variables() {
    let defaults = settings::base(&parse(&["--game-dir", "G", "--Profile", "defaults"]));
    assert_eq!(names(&defaults), HEAD);
    let disabled = settings::base(&parse(&[
        "--game-dir",
        "G",
        "--profile",
        "linux-amd",
        "--no-tuned-settings",
    ]));
    assert_eq!(names(&disabled), HEAD);
}

#[test]
fn the_shader_warm_steps_use_victorys_shaders_command() {
    // This base's `victory warm` takes no files (it remakes pipelines from earlier runs);
    // `victory shaders <files>` translates every shader of them into the cache.
    let victory = Victory {
        manifest: None,
        exe: PathBuf::from("victory.exe"),
        diag: false,
    };
    let archives = [PathBuf::from("a.rpf")];
    let warm = victory.warm_command(&archives, &Vec::new());
    let args: Vec<_> = warm.get_args().collect();
    assert_eq!(args, ["shaders", "a.rpf"]);
    let game = std::env::temp_dir().join(format!("anyxboxx-run-precompile-{}", std::process::id()));
    std::fs::create_dir_all(&game).unwrap();
    for name in [SHADERS, GAME_EXE].iter().chain(PRECOMPILED_DLLS.iter()) {
        std::fs::write(game.join(name), b"").unwrap();
    }
    let precompile = victory.precompile_command(&game, &Vec::new()).unwrap();
    assert_eq!(precompile.get_args().next().unwrap(), "shaders");
    std::fs::remove_dir_all(&game).unwrap();
}
