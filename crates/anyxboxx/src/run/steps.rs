//! The steps of a launch as reusable pieces: the CLI (`anyxboxx run`) chains them like main's
//! script did, the graphical launcher (`crates/anyxboxx-launcher`) drives them one by one.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{RunArgs, RunError, settings};

pub const GAME_EXE: &str = "game_scarlett_master.exe";
/// The game's two DLLs warmed with its shaders by `--precompile`.
pub const PRECOMPILED_DLLS: [&str; 2] = ["XboxBvhBuilder.dll", "amd_fidelityfx_upscaler_xbox.dll"];
pub const SHADERS: &str = "shaders.rpf";
/// VIctory's command that translates every shader of the files it is given into its cache.
pub const SHADERS_COMMAND: &str = "shaders";

/// Environment variables given to VIctory, in the order they are set.
pub type Env = Vec<(String, OsString)>;

/// Where VIctory's sources and its built `victory.exe` are. `manifest` is `None` for a
/// `victory.exe` built beforehand (`--victory-exe`): there is nothing to build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Victory {
    pub manifest: Option<PathBuf>,
    pub exe: PathBuf,
    /// Built with the diag profile and the diagnostics feature (`--diag`, `--dred`).
    pub diag: bool,
}

impl Victory {
    /// VIctory in `repo`, release build by default; its target folder is
    /// `third_party\victory\target` unless `CARGO_TARGET_DIR` is set (VIctory is its own workspace).
    pub fn locate(repo: &Path, diag: bool) -> Result<Self, RunError> {
        let victory = repo.join("third_party").join("victory");
        let manifest = victory.join("Cargo.toml");
        if !manifest.is_file() {
            return Err(RunError::NoVictory);
        }
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map_or_else(|| victory.join("target"), PathBuf::from);
        let profile = if diag { "diag" } else { "release" };
        Ok(Victory {
            manifest: Some(manifest),
            exe: target.join(profile).join("victory.exe"),
            diag,
        })
    }

    /// A `victory.exe` built beforehand, used as is: under Proton there is no cargo, so VIctory
    /// is cross-built on the Linux side (`--target x86_64-pc-windows-msvc`) and only run here.
    pub fn prebuilt(exe: &Path) -> Result<Self, RunError> {
        if !exe.is_file() {
            return Err(RunError::NoVictoryExe(exe.display().to_string()));
        }
        Ok(Victory {
            manifest: None,
            exe: exe.to_path_buf(),
            diag: false,
        })
    }

    /// `cargo build` of the release profile (or the diag profile with diagnostics): does nothing
    /// when victory.exe is current. `None` for a prebuilt `victory.exe`.
    pub fn build_command(&self) -> Option<Command> {
        let manifest = self.manifest.as_ref()?;
        let mut command = Command::new("cargo");
        if self.diag {
            command.args(["build", "--profile", "diag", "-p", "victory"]);
            command.args(["--features", "diagnostics"]);
        } else {
            command.args(["build", "--release", "-p", "victory"]);
        }
        command.arg("--manifest-path").arg(manifest);
        Some(command)
    }

    /// `victory shaders <archives>` with the game's environment (the cache key depends on it): every
    /// shader found translated into the shader cache. (This VIctory base's `warm` makes the host
    /// pipelines of earlier runs instead and takes no files.)
    pub fn warm_command(&self, archives: &[PathBuf], env: &Env) -> Command {
        let mut command = Command::new(&self.exe);
        command.arg(SHADERS_COMMAND).args(archives);
        command.envs(pairs(env));
        command
    }

    /// `victory shaders` on shaders.rpf, the game and its two DLLs (`--precompile`), with the game's
    /// environment. Only these files of the game folder are read, never the folder around them.
    pub fn precompile_command(&self, game_dir: &Path, env: &Env) -> Result<Command, RunError> {
        let (shaders, dlls) = precompile_files(game_dir)?;
        let mut command = Command::new(&self.exe);
        command
            .arg(SHADERS_COMMAND)
            .arg(&shaders)
            .arg(game_dir.join(GAME_EXE))
            .args(&dlls);
        command.envs(pairs(env));
        Ok(command)
    }

    /// `victory run <game> --execute` with the game's environment.
    pub fn game_command(&self, game_dir: &Path, env: &Env) -> Command {
        let mut command = Command::new(&self.exe);
        command
            .arg("run")
            .arg(game_dir.join(GAME_EXE))
            .arg("--execute");
        command.envs(pairs(env));
        command
    }
}

fn precompile_files(game_dir: &Path) -> Result<(PathBuf, [PathBuf; 2]), RunError> {
    let dlls = PRECOMPILED_DLLS.map(|name| game_dir.join(name));
    let shaders = game_dir.join(SHADERS);
    for needed in std::iter::once(&shaders).chain(&dlls) {
        if !needed.is_file() {
            return Err(RunError::PrecompileInput(needed.display().to_string()));
        }
    }
    Ok((shaders, dlls))
}

/// Validate the requested inputs and preserve archive order. On Windows the shader archive
/// can have different casing in the directory listing and still be the same precompile input.
pub(super) fn warm_inputs(args: &RunArgs) -> Result<Vec<PathBuf>, RunError> {
    let mut inputs = if args.warm {
        warm_archives(&args.game_dir)?
    } else {
        Vec::new()
    };
    if args.precompile {
        let (shaders, dlls) = precompile_files(&args.game_dir)?;
        for input in [shaders, args.game_dir.join(GAME_EXE)].into_iter().chain(dlls) {
            let already_present = inputs.iter().any(|path| {
                path == &input
                    || cfg!(windows) && path.as_os_str().eq_ignore_ascii_case(input.as_os_str())
            });
            if !already_present {
                inputs.push(input);
            }
        }
    }
    Ok(inputs)
}

/// The `.rpf` archives directly in `game_dir`, sorted by name (never the folder's other trees).
pub fn warm_archives(game_dir: &Path) -> Result<Vec<PathBuf>, RunError> {
    let entries = std::fs::read_dir(game_dir)
        .map_err(|e| RunError::Spawn(game_dir.display().to_string(), e))?;
    let mut archives: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("rpf"))
        })
        .collect();
    if archives.is_empty() {
        return Err(RunError::NoArchive(game_dir.display().to_string()));
    }
    archives.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(archives)
}

/// The game executable of `game_dir`, which must exist.
pub fn game_exe(game_dir: &Path) -> Result<PathBuf, RunError> {
    let exe = game_dir.join(GAME_EXE);
    if exe.exists() {
        Ok(exe)
    } else {
        Err(RunError::NoGame(game_dir.display().to_string()))
    }
}

/// The game's environment (`settings::base`). A `Result` for the launch steps that share it.
pub fn environment(args: &RunArgs) -> Result<Env, RunError> {
    Ok(settings::base(args))
}

fn pairs(env: &Env) -> impl Iterator<Item = (&OsStr, &OsStr)> {
    env.iter()
        .map(|(k, v)| (OsStr::new(k.as_str()), v.as_os_str()))
}

/// Waits for the command; a process killed without exit code counts as 1, like PowerShell.
pub fn status(command: &mut Command, name: &str) -> Result<i32, RunError> {
    let status = command
        .status()
        .map_err(|e| RunError::Spawn(name.to_string(), e))?;
    Ok(status.code().unwrap_or(1))
}
