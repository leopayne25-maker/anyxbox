//! The steps of a launch as reusable pieces: the CLI (`anyxboxx run`) chains them like main's
//! script did, the graphical launcher (`crates/anyxboxx-launcher`) drives them one by one.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
        let shaders = game_dir.join(SHADERS);
        let dlls: Vec<PathBuf> = PRECOMPILED_DLLS.iter().map(|d| game_dir.join(d)).collect();
        for needed in std::iter::once(&shaders).chain(&dlls) {
            if !needed.is_file() {
                return Err(RunError::PrecompileInput(needed.display().to_string()));
            }
        }
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

/// Ask Cargo for the executable path so custom targets and target directories work.
pub(super) fn build_executable(command: &mut Command) -> Result<PathBuf, RunError> {
    let output = command
        .arg("--message-format=json-render-diagnostics")
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| RunError::Spawn("cargo".into(), e))?;
    if !output.status.success() {
        return Err(RunError::BuildFailed(output.status.code().unwrap_or(1)));
    }
    let executables: Vec<_> = output
        .stdout
        .split(|&byte| byte == b'\n')
        .filter_map(|line| serde_json::from_slice::<serde_json::Value>(line).ok())
        .filter_map(|message| {
            let target = &message["target"];
            let is_binary = target["kind"]
                .as_array()
                .is_some_and(|kind| kind.iter().any(|k| k == "bin"));
            if message["reason"] != "compiler-artifact" || target["name"] != "victory" || !is_binary {
                return None;
            }
            message["executable"]
                .as_str()
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
        })
        .collect();
    let [executable] = executables.as_slice() else {
        return Err(RunError::BuildArtifactCount(executables.len()));
    };
    if !executable.is_file() {
        return Err(RunError::NoVictoryExe(executable.display().to_string()));
    }
    Ok(executable.clone())
}

/// Waits for the command; a process killed without exit code counts as 1, like PowerShell.
pub fn status(command: &mut Command, name: &str) -> Result<i32, RunError> {
    let status = command
        .status()
        .map_err(|e| RunError::Spawn(name.to_string(), e))?;
    Ok(status.code().unwrap_or(1))
}
