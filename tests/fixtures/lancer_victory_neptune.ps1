<#
.SYNOPSIS
  Builds (cargo skips the work when current) and launches VIctory (third_party/victory, Neptune base) on the user's own
  game folder.

.PARAMETER GameDir
  Folder of the game package (kept outside this repository). It must contain game_scarlett_master.exe.

.PARAMETER Warm
  Optional. Before the game, runs 'victory.exe warm' on the .rpf archives of the game folder only (never the folder
  itself: a game folder can hold other trees, which warm would walk too). It translates their shaders into the cache
  and makes their compute pipelines, with the same environment variables as the game (the cache key depends on them).
  About 7 minutes the first time; later launches reuse the cache.

.PARAMETER Precompile
  Optional. Before the game, runs 'victory.exe warm' on shaders.rpf, the game executable and its two DLLs
  (XboxBvhBuilder.dll, amd_fidelityfx_upscaler_xbox.dll), with the same environment variables as the game.
  Combine with -Warm to cover every archive as well.

.PARAMETER AutoA
  Optional. Taps A through the start screens for 150 s (VICTORY_AUTO_A=150), as Neptune's play.ps1 does by default.

.PARAMETER FrametimeDump
  Optional. Writes frametime.csv into the cache folder, one row a second (VICTORY_FRAMETIME_DUMP=1).

.PARAMETER Diag
  Optional. Builds and runs the diag profile with the diagnostics feature instead of the release build the frame rates
  were measured with. -Dred implies it.

.PARAMETER SkipPipelines
  Optional. Passed unchanged to VICTORY_SKIP_PIPELINES when given.

.PARAMETER NoTunedSettings
  Optional. Does not set Neptune's switch set below (only VICTORY_GAME_DIR, VICTORY_DISCORD and
  VICTORY_OCEAN_READBACK are set).

.PARAMETER Dred
  Optional. Turns on device-removed data (VICTORY_D3D12_DRED) to diagnose a device loss; needs the diagnostics build,
  so it implies -Diag. Off by default: it notes every command list operation and slows each frame.

.PARAMETER XgDll
  Optional. Path to your own copy of xg_xs.dll, the offline PC build of the GDK's texture layout library (never in this
  repository). VIctory then lays out the title's textures with it (VICTORY_XG_DLL).

.PARAMETER XgMode
  Optional, with -XgDll. 'real' (default) serves the title from xg_xs.dll; 'compare' keeps VIctory's layouts and logs
  every difference with the library (VICTORY_XG_MODE).

.PARAMETER BatchCardRehash
  Optional. Sets VICTORY_CARD_REHASH_RUNS=1 even with -NoTunedSettings (Neptune's switch set already sets it).

.PARAMETER Override
  Optional. A hashtable of VICTORY_* variables set after the tuned set (and after -NoTunedSettings), for A/B runs:
  -Override @{ VICTORY_RELAY = '0'; VICTORY_PIPELINE_WAIT_MS = '12000' }. Applied last, so it wins over the set.

.PARAMETER Quality
  Optional. Applies the picture-first set on top of the tuned set (after -NoTunedSettings too, before -Override):
  windows and buffers are read fresh each frame instead of kept as card copies, with more VRAM headroom. It costs
  about half the frame rate: measured about 7 to 9 fps against 14 to 17 for the default set, with 51 to 93 texture
  refusals against 296 (run auto-ram-2326). Picture first, frame rate second.

.NOTES
  The switch set is Neptune (VIctory Edit.) 2026-11-25c's play.ps1 for the Series X path (no PC FidelityFX DLL, no
  VICTORY_AUTO_A unless -AutoA), with the measured changes of 2026-11-25 on top (heap pass on dirty slots, copy
  coalescing, 1536 MB windows, texture room 93 %, idle times 8 s / 2 s / 1 s, 512 MB list pool). Measured by the
  maintainer on 2026-11-25 (RTX 4070 SUPER, i7-13700KF, Series X 4K, release build, after warm): median about 13.8 fps
  in the world (15.6 to 17.0 over the last minute), 72.5 ms a frame. VICTORY_CARD_WINDOWS_MB was sized for a 16 GB card:
  4096 spilled VRAM into system RAM on a 12 GB card; on a smaller card, lower VICTORY_CARD_BUFFERS_MB first.
  Discord Rich Presence stays off (VICTORY_DISCORD=0).
  The ocean's height maps are copied back for the title's CPU (VICTORY_OCEAN_READBACK=1), or boats sit on the sea bed.

.EXAMPLE
  .\tools\lancer_victory.ps1 -GameDir D:\Games\MyPackage -Warm

.EXAMPLE
  .\tools\lancer_victory.ps1 -GameDir D:\Games\MyPackage -XgDll D:\MyGdk\xg_xs.dll
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$GameDir,
    [switch]$Warm,
    [switch]$Precompile,
    [switch]$AutoA,
    [switch]$FrametimeDump,
    [switch]$Diag,
    [string]$SkipPipelines,
    [switch]$NoTunedSettings,
    [switch]$Dred,
    [string]$XgDll,
    [ValidateSet('real', 'compare')][string]$XgMode = 'real',
    [switch]$BatchCardRehash,
    [switch]$Quality,
    [hashtable]$Override = @{}
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$victory = Join-Path $repo 'third_party\victory'
$manifest = Join-Path $victory 'Cargo.toml'

if (-not (Test-Path -LiteralPath $manifest)) {
    throw "third_party/victory/Cargo.toml is missing: this copy of AnyXboxX is incomplete."
}

$gameExe = Join-Path $GameDir 'game_scarlett_master.exe'
if (-not (Test-Path -LiteralPath $gameExe)) {
    throw "game_scarlett_master.exe not found in '$GameDir'."
}

# VIctory is its own Cargo workspace: its target folder is third_party\victory\target
# unless CARGO_TARGET_DIR is set.
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $victory 'target' }
$useDiag = $Diag -or $Dred
if ($useDiag) {
    $profileArgs = @('--profile', 'diag', '-p', 'victory', '--features', 'diagnostics')
    $exe = Join-Path $target 'diag\victory.exe'
} else {
    $profileArgs = @('--release', '-p', 'victory')
    $exe = Join-Path $target 'release\victory.exe'
}

# Always ask cargo: it does nothing when victory.exe is current, and rebuilds it after a pull that changes
# third_party/victory (an old victory.exe would ignore the settings the new tree reads).
Write-Host "Building VIctory ($(if ($useDiag) { 'diag profile, diagnostics feature' } else { 'release' }))..."
# Windows PowerShell 5.1 wraps redirected native stderr (`2>`) in NativeCommandError, which
# $ErrorActionPreference = 'Stop' turns into a terminating error on benign warnings; rely on $LASTEXITCODE.
$prevErrorAction = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & cargo build @profileArgs --manifest-path $manifest
} finally {
    $ErrorActionPreference = $prevErrorAction
}
if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit code $LASTEXITCODE)." }

# No victory.toml is needed: VIctory reads the game folder from VICTORY_GAME_DIR.
$env:VICTORY_GAME_DIR = $GameDir
# Neptune's Discord Rich Presence stays off (our copy also needs VICTORY_DISCORD=1 to turn it on).
$env:VICTORY_DISCORD = '0'
# The title's CPU reads the ocean height maps its wave passes write on the GPU: boats float on them. Off by default in
# our copy (texback.rs); without it they read memory nothing wrote and boats sit on the sea bed. Linear layouts only
# (VICTORY_TEXTURE_BACK_MODES left unset): the height maps are linear, XOR modes turned the water to foam.
$env:VICTORY_OCEAN_READBACK = '1'
if (-not $NoTunedSettings) {
    # Neptune (VIctory Edit.) 2026-11-25c, play.ps1, Series X path. Its VICTORY_WAVE_OWN_LDS_GROUPS is left out: no code
    # of this tree reads it (Neptune's shader translator, from soldsawn/victory-nvidia-8gb, dropped that wave rule).
    $tuned = [ordered]@{
        VICTORY_CARD_WINDOWS_MB          = '1536'      # draw data on the card; 4096 spilled VRAM on a 12 GB card
        VICTORY_BAKE_TEXTURES_MS         = '10000'     # textures uploaded before the character bake
        VICTORY_EARLY_TEXTURES_MS        = '40'
        VICTORY_WIDE_MERGE               = '1'         # shader results merged back on several threads
        VICTORY_RELAY_OWN_QUEUE          = '1'         # copy-back fetches on a GPU queue of their own
        VICTORY_UNDER_WRITES_WORDS       = '1'         # small changes under shader writes do not force a wait
        VICTORY_UNDER_WRITES_RUNS        = '100000'
        VICTORY_UNDER_WRITES_MASKS       = '1'
        VICTORY_RELAY                    = '2'
        VICTORY_WIDE_REFRESH             = '1'
        VICTORY_ASK_ONCE                 = '1'
        VICTORY_TEXTURE_NO_WAIT          = '1'
        VICTORY_CARD_REHASH_RUNS         = '1'
        VICTORY_JOIN_COPIES              = '1'
        VICTORY_D3D12_BACKGROUND         = '0'
        VICTORY_LATE_WAIT_LISTS          = '1024'
        VICTORY_WINDOW_TAILS             = '1'
        VICTORY_TEXTURE_STAGE_MB         = '512'
        VICTORY_CARD_BUFFERS_MIN_KB      = '64'
        VICTORY_ASK_QUIET                = '8'         # quiet memory asked about once a frame
        VICTORY_COPY_CHANGED             = '1'         # only bytes that differ go up to the card
        VICTORY_QUIET_BACK               = '1'         # copied-back shader results are not read again
        VICTORY_LIST_POOL_MOST           = '2048'      # command lists kept, not made anew each frame
        VICTORY_LIST_POOL_MB             = '512'
        VICTORY_WIDE_THREADS             = '12'
        VICTORY_UNDER_WRITES_MASKS_LEAST = '1'         # small changes go up in one pass per buffer
        VICTORY_INDIRECT_RAW             = '2'         # indirect draws use the game's own arguments
        VICTORY_TEXTURE_HASH_EVERY       = '32'
        VICTORY_LEVELS_ON_DEMAND         = '1'         # card memory only for the mip levels the game holds
        VICTORY_WINDOW_KB                = '1024'
        VICTORY_CARD_BUFFERS_MB          = '1536'
        VICTORY_CARD_WINDOW_UPLOAD       = '1'
        VICTORY_DROP_DXIL                = '1'
        VICTORY_DESC_DATA                = '1'         # shaders that read a descriptor as data translate
        VICTORY_VRAM_IDLE_MS             = '8000,2000,1000'    # an unused texture is kept 8 s while the card has room
        VICTORY_PIPELINE_WAIT_MS         = '2500'      # a frame waits at most 2.5 s for a shader being compiled
        VICTORY_TEXTURE_ROOM_PERCENT     = '93'        # textures use up to 93 % of the card's room
        VICTORY_HEAP_PASS_DIRTY          = '1'         # the heap pass looks only at slots marked since its last pass
        VICTORY_COPY_COALESCE            = '256'       # changed runs up to 256 bytes apart go up as one range
        VICTORY_WINDOW_SKIP_SAME         = '1'         # windows leave out pages that hold what they last sent (#358)
        VICTORY_MERGE_SUBMISSIONS        = '1'         # consecutive title submissions as one host submission (#358)
        VICTORY_PAGE_WRITES              = '5'         # host copies under finished reads (#358)
        VICTORY_STEP_INDEX               = '1'         # step copies found by an index (#383)
        VICTORY_CARD_SYNC_BATCH          = '1'         # card copies recorded in three steps (#383)
        VICTORY_PARALLEL_REFRESH         = '1'         # texture checks on the helper pool (#383)
        VICTORY_HEAP_SCAN_ALT            = '1'         # full heap scan every 16 passes, not every pass
        VICTORY_TEXTURE_PREPARE_THREADS  = '2'         # textures picked by index made ready off the recording thread (#395)
    }
    foreach ($name in $tuned.Keys) {
        Set-Item -Path "Env:$name" -Value $tuned[$name]
    }
}
if ($Quality) {
    # Picture first: windows and buffers are read fresh each frame instead of kept as card copies (0 = no windows),
    # with more VRAM headroom for the textures. About half the frame rate of the default set.
    $env:VICTORY_CARD_WINDOWS_MB = '0'
    $env:VICTORY_CARD_BUFFERS_MB = '1024'
    $env:VICTORY_LIST_POOL_MOST = '256'
    $env:VICTORY_LIST_POOL_MB = '256'
    $env:VICTORY_TEXTURE_STAGE_MB = '192'
}
foreach ($name in $Override.Keys) {
    if ($name -notlike 'VICTORY_*') { throw "-Override takes VICTORY_* names only: '$name'." }
    Set-Item -Path "Env:$name" -Value ([string]$Override[$name])
}
if ($AutoA) {
    $env:VICTORY_AUTO_A = '150'
}
if ($FrametimeDump) {
    $env:VICTORY_FRAMETIME_DUMP = '1'
}
# VIctory turns DRED on when the variable exists, whatever its value: only set it when asked.
if ($Dred) {
    $env:VICTORY_D3D12_DRED = '1'
}
if ($BatchCardRehash) {
    $env:VICTORY_CARD_REHASH_RUNS = '1'
}
if ($PSBoundParameters.ContainsKey('SkipPipelines') -and $SkipPipelines) {
    $env:VICTORY_SKIP_PIPELINES = $SkipPipelines
}
# Your own copy of the GDK's offline PC xg_xs.dll (never in this repository): Microsoft's library lays out the textures.
if ($PSBoundParameters.ContainsKey('XgDll') -and $XgDll) {
    if (-not (Test-Path -LiteralPath $XgDll -PathType Leaf)) {
        throw "xg_xs.dll not found: '$XgDll'."
    }
    $env:VICTORY_XG_DLL = (Resolve-Path -LiteralPath $XgDll).Path
    $env:VICTORY_XG_MODE = $XgMode
}

if ($Warm) {
    # Same environment as the game below: the cache key depends on it. Only the game folder's own .rpf files.
    $archives = @(Get-ChildItem -LiteralPath $GameDir -Filter '*.rpf' -File | Sort-Object Name | ForEach-Object { $_.FullName })
    if ($archives.Count -eq 0) {
        throw "-Warm found no .rpf archive in '$GameDir'."
    }
    Write-Host "Warming the shader cache from $($archives.Count) archives (about 7 minutes the first time)..."
    $prevErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $exe warm @archives
    } finally {
        $ErrorActionPreference = $prevErrorAction
    }
    if ($LASTEXITCODE -ne 0) { throw "victory warm failed (exit code $LASTEXITCODE)." }
}

if ($Precompile) {
    $shaders = Join-Path $GameDir 'shaders.rpf'
    $dlls = @('XboxBvhBuilder.dll', 'amd_fidelityfx_upscaler_xbox.dll') | ForEach-Object { Join-Path $GameDir $_ }
    foreach ($needed in @($shaders) + $dlls) {
        if (-not (Test-Path -LiteralPath $needed -PathType Leaf)) {
            throw "-Precompile needs '$needed'."
        }
    }
    Write-Host 'Precompiling the shaders of shaders.rpf, the game and its DLLs into the cache...'
    $prevErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $exe warm $shaders $gameExe @dlls
    } finally {
        $ErrorActionPreference = $prevErrorAction
    }
    if ($LASTEXITCODE -ne 0) { throw "victory warm failed (exit code $LASTEXITCODE)." }
}

$prevErrorAction = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $exe run $gameExe --execute
} finally {
    $ErrorActionPreference = $prevErrorAction
}
exit $LASTEXITCODE
