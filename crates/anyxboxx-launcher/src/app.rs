//! The window's state and the actions behind its buttons. Long work (folder checks, VIctory's
//! translate and game) runs on threads; this module drains their results once per frame.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Receiver};
use std::time::Instant;

use crate::app_check::ManifestState;
use crate::app_launch::Banner;
use crate::cache;
use crate::host::{self, Adapter, GpuVendor, Host};
use crate::job::{self, Event, Step};
use crate::launch;
use crate::launch_progress::{self, Stage, Tracker};
use crate::manifest::Outcome;
use crate::picker::{self, Pick};
use crate::processes::Tasklist;
use crate::profile::{Profile, Profiles};
use crate::progress::{Progress, Summary};
use crate::session::{self, Settings};
use crate::setup::{self, Item};
use crate::splash::Splash;
use crate::store::{self, Stored};
use crate::toast::Toasts;
use crate::verify::Report;

/// Lines kept in the log view.
const LOG_CAPACITY: usize = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Home,
    Setup,
    Files,
    Shaders,
    Play,
}

/// The game folder check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verify {
    NotRun,
    Running,
    Done(Report),
    Failed(String),
}

/// A job running on its thread, whether it is the shader step, its stages and its cancel switch.
pub struct Running {
    pub events: Receiver<Event>,
    pub shaders_only: bool,
    pub stages: Vec<Stage>,
    pub cancel: Arc<AtomicBool>,
}

pub struct App {
    pub repo: Option<PathBuf>,
    /// Prebuilt `victory.exe` passed via `--victory-exe`: skips the cargo build and needs no repo.
    pub victory_exe: Option<PathBuf>,
    pub profiles: Profiles,
    pub host: Host,
    pub gpu: GpuVendor,
    pub adapter_name: Option<String>,
    adapters: Option<Vec<Adapter>>,
    pub setup: Vec<Item>,
    pub settings: Settings,
    pub tab: Tab,
    pub verify: Verify,
    pub(crate) verify_rx: Option<Receiver<Result<Report, String>>>,
    pub(crate) verify_started: Option<std::time::Instant>,
    /// The optional SHA-256 manifest job; it never gates PLAY.
    pub manifest: ManifestState,
    pub(crate) manifest_rx: Option<Receiver<Outcome>>,
    pub running: Option<Running>,
    pub log: VecDeque<String>,
    pub summary: Summary,
    pub notice: Option<String>,
    /// Set when the game is up and the launcher may close (see `keep_open_after_launch`).
    pub close_requested: bool,
    pub confirm_clean: bool,
    pub cache_dir: Option<PathBuf>,
    pub splash: Splash,
    pub toasts: Toasts,
    /// The step the running job is on (`Building VIctory`, `Running the game`...).
    pub step: Option<String>,
    pub drawer_open: bool,
    /// Whether the shader cache folder holds something (looked at every couple of seconds).
    pub cache_ready: bool,
    pub cache_checked_at: f64,
    /// When the page on screen was switched to, in egui seconds (for the page transition).
    pub tab_changed_at: f64,
    pub shown_tab: Tab,
    /// Start the game as soon as the folder check passes (PLAY pressed before it ended).
    pub launch_when_ready: bool,
    /// The launch screen's model: stage, progress, pace.
    pub tracker: Tracker,
    /// The outcome of the last job, shown as a card until dismissed or replaced.
    pub banner: Option<Banner>,
    /// Shows the launch screen without running anything (`--preview-launch`, debug builds only).
    pub preview_launch: bool,
    pub(crate) clock: Instant,
    pick_rx: Option<Receiver<(Pick, Option<PathBuf>)>>,
    settings_path: Option<PathBuf>,
    last_saved: Stored,
}

/// The checklist as the machine is now; `adapters` is what DXGI listed (None when it failed).
fn read_setup(adapters: Option<&[crate::host::Adapter]>) -> Vec<Item> {
    let system_root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
    let program_files = std::env::var_os("ProgramFiles(x86)")
        .map_or_else(|| PathBuf::from("C:\\Program Files (x86)"), PathBuf::from);
    let kits = program_files.join("Windows Kits").join("10").join("bin");
    let facts = setup::probe(
        &PathBuf::from(system_root),
        &kits,
        adapters.map(<[_]>::to_vec),
    );
    setup::checklist(&facts)
}

impl App {
    /// The window's first state: the setup checklist, the GPU and the host are read now.
    pub fn new() -> Result<Self, String> {
        let profiles = Profiles::embedded().map_err(|e| e.to_string())?;
        let settings_path = store::current_path();
        let settings = settings_path
            .as_deref()
            .and_then(store::load)
            .map(Stored::into_settings)
            .unwrap_or_default();
        let settings_copy = settings.clone();
        let host = host::detect_host();
        let adapters = host::enumerate_adapters().ok();
        let gpu = adapters
            .as_deref()
            .map_or(GpuVendor::Unknown, host::vendor_of);
        let adapter_name = adapters
            .as_deref()
            .and_then(host::main_adapter)
            .map(|a| a.name.clone());
        let mut app = App {
            repo: anyxboxx::run::find_repo(None),
            victory_exe: None,
            profiles,
            host,
            gpu,
            adapter_name,
            setup: read_setup(adapters.as_deref()),
            adapters,
            settings,
            tab: Tab::Home,
            verify: Verify::NotRun,
            verify_rx: None,
            verify_started: None,
            manifest: ManifestState::NotRun,
            manifest_rx: None,
            running: None,
            log: VecDeque::new(),
            summary: Summary::default(),
            notice: None,
            close_requested: false,
            confirm_clean: false,
            cache_dir: cache::current_cache_dir(),
            splash: Splash::new(),
            toasts: Toasts::default(),
            step: None,
            drawer_open: false,
            cache_ready: false,
            cache_checked_at: 0.0,
            tab_changed_at: 0.0,
            shown_tab: Tab::Home,
            last_saved: Stored::from_settings(&settings_copy),
            launch_when_ready: false,
            tracker: Tracker::default(),
            banner: None,
            preview_launch: false,
            clock: Instant::now(),
            pick_rx: None,
            settings_path,
        };
        app.find_game_folder();
        if app.settings.game_path().is_some() {
            app.start_verify();
        }
        Ok(app)
    }

    /// Keeps the saved game folder when it still holds the game, else looks for one next to the
    /// launcher and in its parents.
    fn find_game_folder(&mut self) {
        let last = self.settings.game_path();
        let launcher = picker::launcher_dir();
        let found = picker::detect(launcher.as_deref(), last.as_deref());
        if let Some(dir) = found {
            self.settings.game_dir = dir.display().to_string();
        }
    }

    /// Opens the folder dialog on its own thread.
    pub fn pick_game_folder(&mut self) {
        self.open_dialog(Pick::GameFolder);
    }

    fn open_dialog(&mut self, kind: Pick) {
        if self.pick_rx.is_some() {
            return;
        }
        let start = match kind {
            Pick::GameFolder => self.settings.game_path(),
        };
        self.pick_rx = Some(picker::open(kind, start));
    }

    /// Uses `dir` as the game folder when it is one (error toast otherwise), then checks it.
    pub fn set_game_folder(&mut self, dir: &std::path::Path) {
        match picker::validate_game_folder(dir) {
            Ok(()) => {
                self.settings.game_dir = dir.display().to_string();
                self.manifest = ManifestState::NotRun;
                self.notice = Some(format!("Game folder saved: {}", dir.display()));
                self.start_verify();
            }
            Err(reason) => self.notice = Some(format!("Cannot use this folder: {reason}")),
        }
    }

    /// Takes the answer of a dialog, if one came.
    fn poll_dialog(&mut self) {
        let Some(rx) = &self.pick_rx else {
            return;
        };
        let Ok((kind, chosen)) = rx.try_recv() else {
            return;
        };
        self.pick_rx = None;
        let Some(path) = chosen else {
            return;
        };
        match kind {
            Pick::GameFolder => self.set_game_folder(&path),
        }
    }

    /// A folder (or a file of it) dropped on the window becomes the game folder.
    pub fn handle_drops(&mut self, ctx: &eframe::egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()));
        if let Some(path) = dropped {
            self.set_game_folder(&picker::folder_of_drop(&path));
        }
    }

    /// The profile for this host and GPU.
    pub fn profile(&self) -> &Profile {
        self.profiles.select(self.host, self.gpu)
    }

    /// Reads the checklist again (after the user installed something).
    pub fn refresh_setup(&mut self) {
        self.setup = read_setup(self.adapters.as_deref());
    }

    /// The repository folder, or an empty path when none was found (the plan is then only shown).
    pub fn repo_or_empty(&self) -> PathBuf {
        self.repo.clone().unwrap_or_default()
    }

    /// Where VIctory keeps saves and data: `VICTORY_DATA_DIR`, else `%LOCALAPPDATA%\victory\data`.
    pub fn data_dir_label(&self) -> String {
        let data = std::env::var_os("VICTORY_DATA_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("LOCALAPPDATA")
                    .map(|l| PathBuf::from(l).join("victory").join("data"))
            });
        data.map_or("unknown".to_string(), |d| d.display().to_string())
    }

    pub fn is_busy(&self) -> bool {
        self.running.is_some() || matches!(self.verify, Verify::Running)
    }

    /// The last progress report of the translate, for the bar.
    pub fn progress(&self) -> Option<&Progress> {
        self.summary.last_progress.as_ref()
    }

    /// Starts the play steps, or the shader preloading alone; refused while a game runs.
    pub fn start(&mut self, precompile_only: bool) {
        if self.is_busy() {
            return;
        }
        if let Err(error) = session::guard(&self.settings, &Tasklist) {
            self.notice = Some(error.to_string());
            return;
        }
        let profile = self.profile().clone();
        let repo = match (&self.repo, &self.victory_exe) {
            (Some(repo), _) => repo.clone(),
            (None, Some(_)) => PathBuf::new(),
            (None, None) => {
                self.notice = Some(
                    "AnyXboxX folder not found: run the launcher from the repository (target\\release) or pass --victory-exe <path>."
                        .to_string(),
                );
                return;
            }
        };
        let plan = match session::make_plan(
            &self.settings,
            &repo,
            self.victory_exe.as_deref(),
            &profile,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                self.notice = Some(error.to_string());
                return;
            }
        };
        if !precompile_only {
            // NVIDIA Smooth Motion lives in the driver's profile, not in VIctory's environment:
            // written now, so the game starts with the switch as the Play page shows it. A failure
            // is reported and the game starts all the same.
            let on = self.settings.options.smooth_motion;
            match crate::smooth_motion::sync(on) {
                Ok(Some(done)) => self.log_line(format!("== {done} ==")),
                Ok(None) => {}
                Err(error) if on => {
                    self.notice = Some(format!("Smooth Motion could not be turned on: {error}"));
                }
                Err(_) => {}
            }
        }
        let steps: Result<Vec<Step>, anyxboxx::run::RunError> = if precompile_only {
            launch::precompile_steps(&plan, self.settings.options.warm_first)
        } else {
            launch::play_steps(&plan, self.settings.options.warm_first)
        };
        match steps {
            Ok(steps) => self.spawn_steps(steps, precompile_only),
            Err(error) => self.notice = Some(error.to_string()),
        }
    }

    fn spawn_steps(&mut self, steps: Vec<Step>, shaders_only: bool) {
        let labels: Vec<String> = steps.iter().map(|s| s.label.clone()).collect();
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        job::spawn_cancellable(steps, tx, Arc::clone(&cancel));
        self.running = Some(Running {
            events: rx,
            shaders_only,
            stages: launch_progress::stages_of(&labels),
            cancel,
        });
        self.step = None;
        self.summary = Summary::default();
        self.tracker = Tracker::start(self.now_secs());
        self.banner = None;
        self.notice = None;
    }

    /// Cleans the shader cache after the user confirmed.
    pub fn clean_cache(&mut self) {
        self.confirm_clean = false;
        let Some(dir) = self.cache_dir.clone() else {
            self.notice = Some("No cache folder: LOCALAPPDATA is not set.".to_string());
            return;
        };
        let cleaned = cache::clean_cache(&dir);
        // With the cache gone, the warm marker would make the next launch skip the warm.
        let forgotten =
            match crate::warm_state::marker_path(std::env::var_os("LOCALAPPDATA").as_deref()) {
                Some(marker) if cleaned.is_ok() => crate::warm_state::forget_marker(&marker)
                    .map_err(|e| format!(" (could not delete {}: {e})", marker.display())),
                _ => Ok(()),
            };
        let mut notice = match cleaned {
            Ok(cache::CleanOutcome::Removed) => format!("Shader cache removed: {}", dir.display()),
            Ok(cache::CleanOutcome::NothingToClean) => "No shader cache to remove.".to_string(),
            Err(error) => error.to_string(),
        };
        if let Err(warning) = forgotten {
            notice.push_str(&warning);
        }
        self.notice = Some(notice);
    }

    /// Drains the results of the threads started earlier.
    pub fn poll(&mut self) {
        self.poll_dialog();
        self.poll_verify();
        self.poll_manifest();
        self.poll_job();
    }

    /// Writes the settings file when something changed since the last write.
    pub fn save_settings_if_changed(&mut self) {
        let now = Stored::from_settings(&self.settings);
        if now == self.last_saved {
            return;
        }
        if let Some(path) = &self.settings_path
            && let Err(error) = store::save(path, &now)
        {
            self.notice = Some(format!("Could not save the settings: {error}"));
        }
        self.last_saved = now;
    }

    pub(crate) fn log_line(&mut self, line: String) {
        self.log.push_back(line);
        while self.log.len() > LOG_CAPACITY {
            self.log.pop_front();
        }
    }
}
