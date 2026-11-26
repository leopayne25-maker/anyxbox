//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! The title creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// Guest shader container bytes: raw while hot, zlib-compressed once cold and
/// translated metadata is captured.
#[cfg(windows)]
enum ContainerBlob {
    Raw(Arc<[u8]>),
    Compressed { packed: Vec<u8>, orig_len: usize },
}

#[cfg(not(windows))]
enum ContainerBlob {
    Raw(Arc<[u8]>),
}

impl ContainerBlob {
    fn bytes_arc(&self) -> Arc<[u8]> {
        match self {
            Self::Raw(bytes) => Arc::clone(bytes),
            #[cfg(windows)]
            Self::Compressed { packed, orig_len } => {
                Arc::from(decompress_container(packed, *orig_len))
            }
        }
    }

    #[cfg(test)]
    fn to_vec(&self) -> Vec<u8> {
        self.bytes_arc().to_vec()
    }

    #[cfg(windows)]
    fn compress(&mut self) -> u64 {
        let Self::Raw(raw) = self else {
            return 0;
        };
        let orig_len = raw.len();
        if orig_len < 1024 {
            return 0;
        }
        let mut buf = vec![0u8; zlib_rs::compress_bound(orig_len)];
        let (packed, rc) =
            zlib_rs::compress_slice(&mut buf, raw, zlib_rs::DeflateConfig::new(1));
        if rc != zlib_rs::ReturnCode::Ok {
            return 0;
        }
        let packed_len = packed.len();
        if packed_len + 256 >= orig_len {
            return 0;
        }
        let saved = (orig_len - packed_len) as u64;
        let compressed = packed.to_vec();
        crate::ram_trim::COUNTERS
            .shader_container_retained_bytes
            .fetch_sub(saved, std::sync::atomic::Ordering::Relaxed);
        *self = Self::Compressed {
            packed: compressed,
            orig_len,
        };
        saved
    }
}

#[cfg(windows)]
fn decompress_container(packed: &[u8], orig_len: usize) -> Vec<u8> {
    if orig_len == 0 {
        return Vec::new();
    }
    let mut out = vec![0u8; orig_len];
    let mut state = zlib_rs::Inflate::new(true, 15);
    match state.decompress(packed, &mut out, zlib_rs::InflateFlush::Finish) {
        Ok(zlib_rs::Status::StreamEnd) if state.total_out() as usize == orig_len => out,
        _ => Vec::new(),
    }
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    #[cfg_attr(not(windows), allow(dead_code))]
    container_len: usize,
    container: Mutex<ContainerBlob>,
    /// Whether it is the program that resolves the indirect diffuse light
    /// (`crate::ambient`).
    #[cfg(windows)]
    pub(crate) resolves_light: bool,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
    /// Cached [`of_builder`] flag so raw containers can be compressed safely.
    #[cfg(windows)]
    builder_spilled: OnceLock<bool>,
    /// Number of active PSO builds currently borrowing this shader's DXIL.
    #[cfg(windows)]
    building_dxil: std::sync::atomic::AtomicU32,
    /// Timestamp (`host::now_ms`) when `dxil` was last needed.
    #[cfg(windows)]
    last_dxil_ms: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// For the memory lines: containers kept and their bytes, translations
/// kept and the bytes of their host code. None of them is ever let go.
static KEPT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the shader table and the pipeline library keep in memory:
/// containers and their bytes, translations and their bytes, the bytes of
/// the library's file (kept whole while the library lives) and what its
/// last save wrote.
#[cfg(windows)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn kept() -> [u64; 7] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        KEPT[0].load(Relaxed),
        KEPT[1].load(Relaxed),
        KEPT[2].load(Relaxed),
        KEPT[3].load(Relaxed),
        library::BLOB.load(Relaxed),
        library::SAVED.load(Relaxed),
        library::TRIMMED_BYTES.load(Relaxed),
    ]
}

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            KEPT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            KEPT[1].fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            crate::ram_trim::COUNTERS
                .shader_container_orig_bytes
                .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            crate::ram_trim::COUNTERS
                .shader_container_retained_bytes
                .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            Arc::new(Shader {
                stage,
                hash: full.to_hex()[..16].to_owned(),
                container_len: bytes.len(),
                container: Mutex::new(ContainerBlob::Raw(Arc::from(bytes))),
                #[cfg(windows)]
                resolves_light: bytes.windows(crate::ambient::PROGRAM.len()).any(|w| w == crate::ambient::PROGRAM),
                #[cfg(windows)]
                translated: OnceLock::new(),
                #[cfg(windows)]
                builder_spilled: OnceLock::new(),
                #[cfg(windows)]
                building_dxil: std::sync::atomic::AtomicU32::new(0),
                #[cfg(windows)]
                last_dxil_ms: std::sync::atomic::AtomicU64::new(0),
            })
        })
        .clone()
}

/// Puts the translation of a container of the title's into the cache on
/// disk, and with `compute` its host pipeline into the driver's, before any
/// run wants them. Whether it translated, and whether it has a pipeline
/// (`None`: not asked for, or left out).
#[cfg(windows)]
pub fn warm(stage: &'static str, container: &[u8], compute: bool) -> (bool, Option<bool>) {
    let full = blake3::hash(container);
    // Not through the table, which lets nothing go.
    let shader = Arc::new(Shader {
        stage,
        hash: full.to_hex()[..16].to_owned(),
        container_len: container.len(),
        container: Mutex::new(ContainerBlob::Raw(Arc::from(container))),
        resolves_light: container.windows(crate::ambient::PROGRAM.len()).any(|w| w == crate::ambient::PROGRAM),
        translated: OnceLock::new(),
        builder_spilled: OnceLock::new(),
        building_dxil: std::sync::atomic::AtomicU32::new(0),
        last_dxil_ms: std::sync::atomic::AtomicU64::new(0),
    });
    let translated = shader.translated().is_some();
    let made = (translated && compute).then(|| make::warm(vec![shader])).flatten();
    (translated, made)
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    pub(crate) used: Vec<victory_shader::UsedResource>,
    /// Whether the shader's code can store to buffer memory ([`stores`]);
    /// true when its code is not known.
    pub(crate) stores: bool,
    /// The host DXIL, moved out of `host`: read only to make a pipeline. Dropped
    /// once cold or with `VICTORY_DROP_DXIL=1` and reloaded transparently from
    /// the shader cache / guest container when another pipeline needs it ([`Shader::dxil`]).
    dxil: Mutex<Option<Arc<[u8]>>>,
}

/// `VICTORY_DROP_DXIL=1` (off by default): shaders' host DXIL is let go once their pipeline is
/// made; otherwise `TrimPolicy` releases cold post-PSO DXIL on idle sweeps.
#[cfg(windows)]
pub(crate) fn drop_dxil() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DROP_DXIL").is_some_and(|v| v == "1"))
}

/// DXIL bytes let go, times and bytes brought back, for the ram line.
#[cfg(windows)]
pub(crate) static DXIL_DROPS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(windows)]
pub(crate) struct DxilPin<'a>(&'a Shader);

#[cfg(windows)]
impl Drop for DxilPin<'_> {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::AcqRel;
        if self.0.building_dxil.fetch_sub(1, AcqRel) == 1 {
            let policy = crate::ram_trim::TrimPolicy::cached();
            if drop_dxil() || policy.shader_dxil_idle_ms == 0 {
                self.0.release_dxil();
            }
            if policy.shader_container_idle_ms == 0 {
                self.0.compress_container_if_ready();
            }
        }
    }
}

#[cfg(windows)]
impl Shader {
    /// Pins DXIL bytecode in RAM while a PSO creation call is using it.
    pub(crate) fn pin_dxil(&self) -> DxilPin<'_> {
        use std::sync::atomic::Ordering::{AcqRel, Relaxed};
        self.building_dxil.fetch_add(1, AcqRel);
        self.last_dxil_ms.store(crate::host::now_ms(), Relaxed);
        DxilPin(self)
    }

    pub(super) fn with_container<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        let bytes = self
            .container
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bytes_arc();
        f(&bytes)
    }

    /// The host DXIL of a translated shader: kept, or reloaded transparently
    /// from the shader cache / guest container after [`Shader::release_dxil`].
    pub(crate) fn dxil(&self) -> Option<Arc<[u8]>> {
        use std::sync::atomic::Ordering::Relaxed;
        let t = self.translated()?;
        self.last_dxil_ms.store(crate::host::now_ms(), Relaxed);
        {
            let held = t.dxil.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(bytes) = held.as_ref() {
                return Some(bytes.clone());
            }
        }
        let again = self.with_container(|bytes| crate::shader_cache::host_shader(bytes).ok())?.dxil;
        let arc: Arc<[u8]> = Arc::from(again);
        let bytes_len = arc.len() as u64;
        let mut held = t.dxil.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = held.as_ref() {
            return Some(existing.clone());
        }
        *held = Some(arc.clone());
        DXIL_DROPS[1].fetch_add(1, Relaxed);
        DXIL_DROPS[2].fetch_add(bytes_len, Relaxed);
        KEPT[3].fetch_add(bytes_len, Relaxed);
        crate::ram_trim::COUNTERS
            .shader_dxil_reloaded_bytes
            .fetch_add(bytes_len, Relaxed);
        Some(arc)
    }

    /// Drops retained post-PSO DXIL bytecode if no build currently pins it.
    pub(crate) fn release_dxil(&self) -> u64 {
        use std::sync::atomic::Ordering::{Acquire, Relaxed};
        if self.building_dxil.load(Acquire) > 0 {
            return 0;
        }
        let Some(Some(t)) = self.translated.get() else {
            return 0;
        };
        let mut guard = t.dxil.lock().unwrap_or_else(|e| e.into_inner());
        if self.building_dxil.load(Acquire) > 0 {
            return 0;
        }
        let Some(dropped) = guard.take() else {
            return 0;
        };
        let len = dropped.len() as u64;
        drop(dropped);
        KEPT[3].fetch_sub(len, Relaxed);
        DXIL_DROPS[0].fetch_add(len, Relaxed);
        crate::ram_trim::COUNTERS
            .shader_dxil_dropped_bytes
            .fetch_add(len, Relaxed);
        len
    }

    fn compress_container_if_ready(&self) -> u64 {
        use std::sync::atomic::Ordering::Acquire;
        if self.translated.get().is_none() || self.building_dxil.load(Acquire) > 0 {
            return 0;
        }
        let Ok(mut guard) = self.container.try_lock() else {
            return 0;
        };
        guard.compress()
    }

    /// The translation, made on first use; a shader that does not
    /// translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        use std::sync::atomic::Ordering::Relaxed;
        let out = self
            .translated
            .get_or_init(|| {
                let done = self.with_container(|bytes| {
                    let done = translate(bytes);
                    // `VICTORY_SHADER_SAVE=<dir>`: each shader as the title gave it (`.cdxil`) and as
                    // translated (`.dxil`), by the hash crash reports name it with
                    if let Some(dir) = std::env::var_os("VICTORY_SHADER_SAVE") {
                        let dir = std::path::PathBuf::from(dir);
                        let _ = std::fs::create_dir_all(&dir);
                        let stem = format!("{}.{}", self.hash, self.stage);
                        let _ = std::fs::write(dir.join(format!("{stem}.cdxil")), bytes);
                        if let Ok(t) = &done
                            && let Some(dxil) = t.dxil.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
                        {
                            let _ = std::fs::write(dir.join(format!("{stem}.dxil")), &**dxil);
                        }
                    }
                    done
                });
                if let Ok(t) = &done {
                    let dxil_len = t
                        .dxil
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .as_ref()
                        .map_or(0, |d| d.len() as u64);
                    KEPT[2].fetch_add(1, Relaxed);
                    KEPT[3].fetch_add(dxil_len, Relaxed);
                }
                match &done {
                    Ok(t) => tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        samplers = t.host.samplers.len(),
                        "shader translated"
                    ),
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref();
        if out.is_some() {
            self.last_dxil_ms.store(crate::host::now_ms(), Relaxed);
        }
        out
    }
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    // The translation depends on it (a view index past a capped heap).
    crate::descriptor::host::probe_view_limit();
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    // (`ACCC`: native code another compiler made, with `VICTORY_WAVE32_NVIDIA=1`)
    let programs = match c
        .part(FourCC::DXIL)
        .or_else(|| c.part(FourCC::ACCC).filter(|_| victory_shader::emit::nvidia::enabled()))
    {
        Some(part) => victory_shader::native_programs(part.data).map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    let mut host = crate::shader_cache::host_shader(container)?;
    let dxil = Mutex::new(Some(Arc::from(std::mem::take(&mut host.dxil))));
    let stores = programs.is_empty() || programs.iter().any(stores);
    let used = programs
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
            )
        })
        .map(|p| p.used)
        .unwrap_or_default();
    Ok(Translated { host, used, stores, dxil })
}

/// The compute pipelines of the console's ray-tracing structure builder
/// (`XboxBvhBuilder.dll`; its programs are the ones with a spilled
/// user-data table) run **all or none**. Its stages feed each other
/// through memory: a stage that runs on what a dropped stage never wrote
/// computes its store offsets from garbage. The structures lie in plain
/// title memory, between the title's own objects, and a root view has no
/// size, so those stores land on the neighbours (run 98, with 14 of 35
/// stages translated: the title died within a second of the first builder
/// submission, on pointers that had become bounding-box bit patterns).
///
/// So a builder dispatch runs only while every builder pipeline the title
/// has created has a host pipeline; the first builder dispatch makes them
/// all on the spot instead of over the next frames (a stage "still being
/// made" would be a dropped stage). One that cannot be made turns the
/// builder off for the rest of the run.
///
/// shortcut: and only with `VICTORY_BVH_BUILDER=1`, until a run has shown
/// the whole builder and the host side of title memory by address
/// (`memory.rs`) working on a card; nothing reads the structures before
/// the programs that walk them translate.
#[cfg(windows)]
struct Builder {
    pipelines: Vec<Arc<make::Pending>>,
    /// Every one of `pipelines` has a host pipeline.
    whole: bool,
    /// One could not be made.
    off: bool,
}

#[cfg(windows)]
static BUILDER: Mutex<Builder> = Mutex::new(Builder {
    pipelines: Vec::new(),
    whole: false,
    off: false,
});

/// Whether `shaders` are a pipeline of the structure builder.
#[cfg(windows)]
fn of_builder(shaders: &[Arc<Shader>]) -> bool {
    let [shader] = shaders else { return false };
    if shader.stage != "compute" {
        return false;
    }
    *shader.builder_spilled.get_or_init(|| {
        shader.with_container(|bytes| {
            use victory_shader::{Container, FourCC};
            let Ok(c) = Container::parse(bytes) else {
                return false;
            };
            c.part(FourCC::DXIL)
                .or_else(|| c.part(FourCC::ACCC).filter(|_| victory_shader::emit::nvidia::enabled()))
                .and_then(|part| victory_shader::native_programs(part.data).ok())
                .is_some_and(|programs| programs.iter().any(victory_shader::spill::is_spilled))
        })
    })
}

/// `VICTORY_BVH_QUARANTINE=1`: what the structure builder's stages store
/// stays in the host's copies and is not copied back into title memory. A
/// check for a first run of the whole builder: a stage that stores where
/// it should not can then not reach the title's own objects (run 98), and
/// the programs that walk the structures read the host's copies anyway.
/// shortcut: not a way to run: the title also reads results back (sizes,
/// counters) and gets none, and a copy's pages are overwritten from title
/// memory when the title writes them.
#[cfg(windows)]
pub(crate) fn builder_quarantined(this: This) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    is_builder(this) && *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_QUARANTINE").is_some_and(|v| v == "1"))
}

/// Whether the title's pipeline `this` is one of the structure builder's.
#[cfg(windows)]
pub(crate) fn is_builder(this: This) -> bool {
    !this.is_null() && pipeline(this).builder
}

/// A stage of the builder was dropped after all: none runs from here on.
#[cfg(windows)]
pub(crate) fn builder_off(why: &'static str) {
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    if !b.off {
        b.off = true;
        tracing::warn!(why, "the ray-tracing structure builder is turned off for the rest of the run: its stages run all or none");
    }
}

/// At a dispatch of a builder pipeline: whether the builder runs (see
/// [`Builder`]).
#[cfg(windows)]
pub(crate) fn builder_runs() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    if !*ALLOWED.get_or_init(|| std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1")) {
        return false;
    }
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    if b.off {
        return false;
    }
    if b.whole {
        return true;
    }
    let missing = b.pipelines.iter().filter(|p| !make::now(p)).count();
    if missing > 0 {
        b.off = true;
        tracing::warn!(
            missing,
            of = b.pipelines.len(),
            "the ray-tracing structure builder stays off: not all of its programs have a host pipeline, and its stages run all or none"
        );
        return false;
    }
    tracing::info!(
        pipelines = b.pipelines.len(),
        "the ray-tracing structure builder runs: all of its programs have host pipelines"
    );
    b.whole = true;
    true
}

/// Whether a console program can store to buffer memory: it has a buffer,
/// typed-buffer, scalar-memory, flat or global store or atomic, or code
/// that is not all known (an instruction of those kinds outside our table,
/// an indirect jump, code that does not decode). Stores to textures
/// (image instructions) and to the program's own scratch memory do not
/// count: neither reaches a buffer's host copy. After upstream #413,
/// which leaves scalar-memory stores out and counts image stores.
#[cfg(windows)]
fn stores(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        let name = i.name();
        match i.encoding {
            Encoding::Mubuf
            | Encoding::Mtbuf
            | Encoding::Smem
            | Encoding::Flat
            | Encoding::Global => name.is_none_or(|n| {
                (n.contains("store") || n.contains("atomic")) && !n.contains("scratch")
            }),
            Encoding::Sop1 => name.is_none_or(|n| matches!(n, "s_setpc_b64" | "s_swappc_b64")),
            _ => false,
        }
    })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// One of the ray-tracing structure builder's ([`Builder`]).
    #[cfg(windows)]
    builder: bool,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// The stages and hashes of title pipeline `this`, as `[mesh 0123..,
/// pixel 4567..]` (diagnostics); "no pipeline" for none.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn shaders_of(this: This) -> String {
    if this.is_null() {
        return "no pipeline".to_owned();
    }
    let names: Vec<String> = pipeline(this).shaders.iter().map(|s| format!("{} {}", s.stage, s.hash)).collect();
    format!("[{}]", names.join(", "))
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container_len, hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (Neptune: in every build, so that a release run says which pipelines its dropped
/// draws were for).
#[cfg(windows)]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(windows)]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        "no pipeline set".to_owned()
    } else {
        let p = pipeline(this);
        #[cfg(feature = "diagnostics")]
        let skipped = skipped_shader(p).map(|hash| format!("skipped shader {hash}"));
        #[cfg(not(feature = "diagnostics"))]
        let skipped: Option<String> = None;
        skipped.or_else(|| make::missing(&p.host)).unwrap_or_else(|| {
            // (Neptune) Its host pipeline is there: the draw was left out for another
            // reason (its views or buffers could not be made ready, or a filter of ours).
            let shaders: Vec<String> = p.shaders.iter().map(|s| format!("{} {}", s.stage, s.hash)).collect();
            format!("pipeline made, draw not run [{}]", shaders.join(", "))
        })
    };
    note_why(why);
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(windows)]
pub(crate) fn note_why(why: String) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(0, || format!("DROPPED {why}"));
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(windows)]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders.clone(), graphics));
    #[cfg(windows)]
    make::early(&host);
    #[cfg(windows)]
    let builder = of_builder(&shaders);
    #[cfg(windows)]
    if builder {
        // Another stage: the group is looked at again at its next dispatch.
        let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
        b.pipelines.push(host.clone());
        b.whole = false;
    }
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            shaders.extend(copy(stage, code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;

/// Lets the pipeline library's mapped file go from memory when no pipeline
/// has loaded for a while (`library::trim`); called by the host's sweep.
#[cfg(windows)]
pub(crate) fn trim_library() {
    library::trim();
}

/// Releases cold post-PSO shader DXIL blobs and zlib-compresses idle guest
/// containers (`SHADERS`), keeping full hash/container reloadability on demand.
#[cfg(windows)]
pub(crate) fn trim_shaders_with(policy: &crate::ram_trim::TrimPolicy, now: u64) {
    use std::sync::atomic::Ordering::{Acquire, Relaxed};
    let list: Vec<Arc<Shader>> = {
        let guard = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = guard.as_ref() else {
            return;
        };
        map.values().cloned().collect()
    };
    for shader in list {
        if shader.translated.get().is_none() || shader.building_dxil.load(Acquire) > 0 {
            continue;
        }
        let idle = now.saturating_sub(shader.last_dxil_ms.load(Relaxed));
        if policy.panic_memory || idle >= policy.shader_dxil_idle_ms {
            shader.release_dxil();
        }
        if policy.panic_memory || idle >= policy.shader_container_idle_ms {
            shader.compress_container_if_ready();
        }
    }
}

/// The pipeline library's part of the report when the host device is
/// removed (`library::removal_note`).
#[cfg(windows)]
pub(crate) fn library_removal_note() -> String {
    library::removal_note()
}

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| {
                (
                    s.stage,
                    s.container
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// The stream layout the title builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }
}

#[cfg(all(test, windows))]
mod ram_trim_tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;

    fn minimal_stage_container(slot_offset: usize, extra_padding: usize) -> Vec<u8> {
        // Minimal console `DXIL` part carrying one valid `s_endpgm` native program at `slot_offset`.
        const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
        const SLOT_CS: usize = 0x60;
        const RECORD_SIZE: usize = 0x54;
        const STAGE_DATA: usize = 0x14c + 8 * (4 + 10 * 10 * 4);
        let code = 0xbf81_0000u32.to_le_bytes(); // s_endpgm
        let mut payload = 6u64.to_le_bytes().to_vec();
        payload.resize(SLOT_CS + 8, 0);
        for at in [0x08usize, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, SLOT_CS] {
            payload[at..at + 8].copy_from_slice(&ABSENT.to_le_bytes());
        }
        let record = payload.len() as u64;
        payload[slot_offset..slot_offset + 8].copy_from_slice(&record.to_le_bytes());
        let code_at = record + RECORD_SIZE as u64 + STAGE_DATA as u64;
        payload.extend_from_slice(&code_at.to_le_bytes());
        payload.extend_from_slice(&(code.len() as u64).to_le_bytes());
        payload.resize(record as usize + 0x50, 0);
        payload.extend_from_slice(&(1u32).to_le_bytes()); // 1 VGPR
        let mut stage_data = vec![0u8; STAGE_DATA];
        if slot_offset == SLOT_CS {
            // CS thread group (1, 1, 1) at `STAGE_DATA + 0x28`.
            for (i, v) in [1u32, 1, 1].into_iter().enumerate() {
                stage_data[0x28 + i * 4..0x28 + (i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        payload.extend_from_slice(&stage_data);
        payload.extend_from_slice(&code);

        let mut fssm = b"FSSM".to_vec();
        fssm.extend_from_slice(&0x101u32.to_le_bytes());
        fssm.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        fssm.extend_from_slice(&payload);

        let mut buf = vec![0u8; zlib_rs::compress_bound(fssm.len())];
        let (stream, rc) =
            zlib_rs::compress_slice(&mut buf, &fssm, zlib_rs::DeflateConfig::new(6));
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        let mut xbch = b"XBCH".to_vec();
        xbch.extend_from_slice(&3u32.to_le_bytes());
        xbch.extend_from_slice(&(stream.len() as u64).to_le_bytes());
        xbch.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
        xbch.extend_from_slice(stream);

        let pad = vec![0x42u8; extra_padding];
        let blobs: Vec<(u32, &[u8])> = if extra_padding > 0 {
            vec![(0x0a, b"xbop"), (0x0d, &xbch), (0x12, &pad)]
        } else {
            vec![(0x0a, b"xbop"), (0x0d, &xbch)]
        };
        let mut dxil_part = vec![0u8; 0x18];
        dxil_part.extend_from_slice(b"XDAT");
        dxil_part.extend_from_slice(&(blobs.len() as u32).to_le_bytes());
        dxil_part.extend_from_slice(&0x24u32.to_le_bytes());
        let mut off = dxil_part.len() + 12 * blobs.len();
        for &(kind, blob) in &blobs {
            for w in [kind, off as u32, blob.len() as u32] {
                dxil_part.extend_from_slice(&w.to_le_bytes());
            }
            off += blob.len();
        }
        for (_, blob) in blobs {
            dxil_part.extend_from_slice(blob);
        }
        let dxbc_parts = [(victory_shader::FourCC::DXIL, dxil_part.as_slice())];
        let table = 32 + 4 * dxbc_parts.len();
        let total = table + dxbc_parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"DXBC");
        out.extend_from_slice(&[0; 16]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(dxbc_parts.len() as u32).to_le_bytes());
        let mut offset = table;
        for (_, d) in &dxbc_parts {
            out.extend_from_slice(&(offset as u32).to_le_bytes());
            offset += 8 + d.len();
        }
        for (fourcc, d) in dxbc_parts {
            out.extend_from_slice(&fourcc.0);
            out.extend_from_slice(&(d.len() as u32).to_le_bytes());
            out.extend_from_slice(d);
        }
        out
    }

    #[test]
    fn real_shader_drops_dxil_post_pso_compresses_and_reloads_on_demand() {
        let orig = minimal_stage_container(0x60, 64 << 10);
        assert!(orig.len() > (64 << 10));
        let shader = super::shader("compute", &orig);

        let t = shader.translated().expect("minimal CS translates");
        let _ = t;
        assert!(!super::of_builder(std::slice::from_ref(&shader)));

        let dxil_first = shader.dxil().expect("DXIL available");
        let dxil_len = dxil_first.len() as u64;
        assert!(dxil_len > 0);
        drop(dxil_first);

        // Pinning DXIL during PSO creation prevents release.
        let pin = shader.pin_dxil();
        assert_eq!(shader.release_dxil(), 0, "pinned DXIL must stay in RAM");
        drop(pin);

        let kept_before = KEPT[3].load(Relaxed);
        let dropped_before = crate::ram_trim::COUNTERS
            .shader_dxil_dropped_bytes
            .load(Relaxed);
        assert_eq!(shader.release_dxil(), dxil_len);
        assert!(KEPT[3].load(Relaxed) <= kept_before.saturating_sub(dxil_len));
        assert!(
            crate::ram_trim::COUNTERS
                .shader_dxil_dropped_bytes
                .load(Relaxed)
                >= dropped_before + dxil_len
        );

        // Cold container compression shrinks RAM significantly and roundtrips losslessly.
        let saved = shader.compress_container_if_ready();
        assert!(saved > (60 << 10), "expected >60 KiB saved, got {saved}");
        shader.with_container(|decoded| assert_eq!(decoded, orig.as_slice()));

        // A second PSO later requesting DXIL transparently reloads identical bytes.
        let reloaded_before = crate::ram_trim::COUNTERS
            .shader_dxil_reloaded_bytes
            .load(Relaxed);
        let dxil_again = shader.dxil().expect("DXIL reloads after drop + compression");
        assert_eq!(dxil_again.len() as u64, dxil_len);
        assert!(
            crate::ram_trim::COUNTERS
                .shader_dxil_reloaded_bytes
                .load(Relaxed)
                >= reloaded_before + dxil_len
        );
        drop(dxil_again);
        let _ = shader.release_dxil();
    }

    #[test]
    fn graphics_shader_compresses_and_reloads_dxil_without_builder() {
        let orig = minimal_stage_container(0x30, 64 << 10);
        let shader = super::shader("pixel", &orig);

        // Translate only — never call `of_builder()`, matching default runs.
        assert!(shader.translated().is_some(), "minimal PS translates");
        assert!(shader.builder_spilled.get().is_none());

        let dxil_len = shader.dxil().expect("PS DXIL available").len() as u64;
        assert!(dxil_len > 0);
        assert_eq!(shader.release_dxil(), dxil_len);

        let saved = shader.compress_container_if_ready();
        assert!(saved > (60 << 10), "graphics shader container must compress (saved={saved})");
        let dxil_reloaded = shader.dxil().expect("PS DXIL reloads from compressed container");
        assert_eq!(dxil_reloaded.len() as u64, dxil_len);
        drop(dxil_reloaded);
        let _ = shader.release_dxil();
    }
}
