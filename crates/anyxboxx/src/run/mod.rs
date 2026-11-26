//! `anyxboxx run`: builds VIctory (cargo does nothing when it is current) and launches it on the
//! user's own game folder. It replaced main's `tools/lancer_victory.ps1` (same checks, build, warm
//! and precompile steps, exit code); its tuned set is VIctory's own `scripts/play.ps1`.
//! Warm and precompile share one invocation when both are requested.

mod settings;
mod steps;

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Args, ValueEnum};

pub use settings::{LINUX_AMD, LINUX_AMD_EXTRA, QUALITY, TUNED, auto_a_keys};
pub use steps::{
    Env, GAME_EXE, PRECOMPILED_DLLS, SHADERS, Victory, environment, game_exe, status, warm_archives,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Profile {
    /// Windows: VIctory's play.ps1 set (`TUNED`; measured on an RX 7900 XTX, not on NVIDIA).
    #[default]
    WindowsNvidia,
    /// Linux (Proton) + AMD measured set (`LINUX_AMD`: `TUNED` plus the Linux additions, including
    /// `VKD3D_CONFIG=descriptor_heap`).
    LinuxAmd,
    /// VIctory's own defaults (no profile variables).
    Defaults,
}

/// A `KEY=VALUE` pair of `--override`; the key must be a VICTORY_* name.
fn parse_override(text: &str) -> Result<(String, String), String> {
    let (key, value) = text
        .split_once('=')
        .ok_or_else(|| format!("expected KEY=VALUE, got {text:?}"))?;
    if !key.starts_with("VICTORY_") {
        return Err(format!("--override takes VICTORY_* names only: {key:?}"));
    }
    Ok((key.to_string(), value.to_string()))
}

/// Options of `anyxboxx run`; the PowerShell names (`--GameDir`...) are accepted as aliases.
#[derive(Debug, Clone, Args)]
pub struct RunArgs {
    /// Folder of the game package (kept outside this repository); must hold game_scarlett_master.exe.
    #[arg(long, alias = "GameDir")]
    pub game_dir: PathBuf,
    /// Passed unchanged to VICTORY_SKIP_PIPELINES.
    #[arg(long, alias = "SkipPipelines")]
    pub skip_pipelines: Option<String>,
    /// Do not set the tuned variables (VICTORY_GAME_DIR stays set).
    #[arg(long, alias = "NoTunedSettings")]
    pub no_tuned_settings: bool,
    /// Launch profile whose variables are applied unless `--no-tuned-settings` is set.
    #[arg(long, alias = "Profile", value_enum, default_value_t = Profile::WindowsNvidia)]
    pub profile: Profile,
    /// Picture first: windows and buffers read fresh each frame (not measured on this VIctory base).
    #[arg(long, alias = "Quality")]
    pub quality: bool,
    /// KEY=VALUE VICTORY_* variables set after the tuned set and --quality, for A/B runs. Repeatable.
    #[arg(long = "override", alias = "Override", value_parser = parse_override)]
    pub overrides: Vec<(String, String)>,
    /// Taps Enter (also the first pad's A) every 4 s through the start screens for 150 s (VICTORY_KEYS).
    #[arg(long, alias = "AutoA")]
    pub auto_a: bool,
    /// Build and run the diag profile with the diagnostics feature instead of the release build.
    #[arg(long, alias = "Diag")]
    pub diag: bool,
    /// Turn on device-removed data (VICTORY_D3D12_DRED); needs the diagnostics build, so implies --diag.
    #[arg(long, alias = "Dred")]
    pub dred: bool,
    /// Run `victory shaders` on the game folder's own .rpf archives before the game (minutes, once per translator version).
    #[arg(long, alias = "Warm")]
    pub warm: bool,
    /// Warm shaders.rpf, the game executable and its two DLLs before the game.
    #[arg(long, alias = "Precompile")]
    pub precompile: bool,
    /// Set VICTORY_CARD_REHASH_RUNS=1 (hash adjacent GPU memory ranges in batches).
    #[arg(long, alias = "BatchCardRehash")]
    pub batch_card_rehash: bool,
    /// Use this victory.exe, built beforehand, and skip the cargo build (for Proton, where there
    /// is no cargo: cross-build VIctory on Linux with `--target x86_64-pc-windows-msvc`).
    /// The repository is then not needed.
    #[arg(long, alias = "VictoryExe")]
    pub victory_exe: Option<PathBuf>,
    /// Repository root (default: ANYXBOXX_REPO, then the folders above the binary and the
    /// current folder that hold third_party/victory/Cargo.toml).
    #[arg(long)]
    pub repo: Option<PathBuf>,
}

impl RunArgs {
    /// The diag profile serves `--diag`, and `--dred` (which needs its diagnostics).
    pub fn uses_diag(&self) -> bool {
        self.diag || self.dred
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("third_party/victory/Cargo.toml is missing: this copy of AnyXboxX is incomplete.")]
    NoVictory,
    #[error("victory.exe not found: {0:?}.")]
    NoVictoryExe(String),
    #[error("repository not found (pass --repo or set ANYXBOXX_REPO).")]
    NoRepo,
    #[error("game_scarlett_master.exe not found in {0:?}.")]
    NoGame(String),
    #[error("cargo build failed (exit code {0}).")]
    BuildFailed(i32),
    #[error("--precompile needs {0:?}.")]
    PrecompileInput(String),
    #[error("--warm found no .rpf archive in {0:?}.")]
    NoArchive(String),
    #[error("victory shaders failed (exit code {0}).")]
    WarmFailed(i32),
    #[error("cannot start {0}: {1}")]
    Spawn(String, std::io::Error),
}

/// Runs the launch like the script, step for step; returns VIctory's exit code. With
/// `--victory-exe` the build is skipped and the repository is not needed.
pub fn run(args: &RunArgs, repo: Option<&Path>) -> Result<i32, RunError> {
    let victory = match (&args.victory_exe, repo) {
        (Some(exe), _) => Victory::prebuilt(exe)?,
        (None, Some(repo)) => Victory::locate(repo, args.uses_diag())?,
        (None, None) => return Err(RunError::NoRepo),
    };
    game_exe(&args.game_dir)?;
    let warm = steps::warm_inputs(args)?;
    if let Some(mut build) = victory.build_command() {
        let profile = if args.uses_diag() {
            "diag profile, diagnostics feature"
        } else {
            "release"
        };
        println!("Building VIctory ({profile})...");
        let code = status(&mut build, "cargo")?;
        if code != 0 {
            return Err(RunError::BuildFailed(code));
        }
    }
    let env = environment(args)?;
    if args.warm {
        println!(
            "Warming the shader cache from {} game files (several minutes the first time)...",
            warm.len()
        );
    } else if args.precompile {
        println!("Precompiling the shaders of shaders.rpf, the game and its DLLs into the cache...");
    }
    if !warm.is_empty() {
        warm_or_fail(&mut victory.warm_command(&warm, &env))?;
    }
    status(&mut victory.game_command(&args.game_dir, &env), "victory")
}

fn warm_or_fail(command: &mut Command) -> Result<(), RunError> {
    match status(command, "victory")? {
        0 => Ok(()),
        code => Err(RunError::WarmFailed(code)),
    }
}

/// Repository root: `--repo`, `ANYXBOXX_REPO`, then the first folder above the binary or the
/// current folder that holds third_party/victory/Cargo.toml.
pub fn find_repo(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(repo) = explicit {
        return Some(repo.to_path_buf());
    }
    if let Some(repo) = std::env::var_os("ANYXBOXX_REPO") {
        return Some(PathBuf::from(repo));
    }
    let starts = [
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf)),
        std::env::current_dir().ok(),
    ];
    starts.into_iter().flatten().find_map(|start| {
        start
            .ancestors()
            .find(|dir| {
                dir.join("third_party")
                    .join("victory")
                    .join("Cargo.toml")
                    .is_file()
            })
            .map(Path::to_path_buf)
    })
}

#[cfg(test)]
mod tests;
