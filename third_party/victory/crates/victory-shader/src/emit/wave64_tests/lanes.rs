//! Cross-lane instructions of console wave64 (and wave32) programs as the console runs them,
//! in every form the host may take: 64-lane host waves (AMD) and 32-lane host waves, with or
//! without the exchange between the two halves (NVIDIA). Ported from AnyXboxX `local/perf-linux`
//! (VIctory PRs #284, #297, #300 and #334), without its paired form, which this tree lacks.

use super::*;

// Operands as the hardware numbers them: SGPR n is n, VCC 106, EXEC 126, an inline integer
// n 128 + n, a literal (the next dword) 255, VGPR n 256 + n.
const VCC: u32 = 106;
const EXEC: u32 = 126;
const LIT: u32 = 255;

const fn v(n: u32) -> u32 {
    256 + n
}

const fn k(n: u32) -> u32 {
    128 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}

const fn vop3(op: u32, dst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | dst, src0 | src1 << 9]
}

const fn ds(op: u32, vdst: u32, addr: u32, data: u32, offset: u32) -> [u32; 2] {
    [0xD800_0000 | op << 18 | offset, vdst << 24 | data << 8 | addr]
}

/// `buffer_store_dword v<data>, v<addr>, s[8:11], 0 offen offset:<offset>`
const fn store(data: u32, addr: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8002_0000 | data << 8 | addr]
}

/// `v<regs>` stored as each thread's dwords of the UAV, the thread's index (v0) times 16 on.
fn stores(regs: [u32; 4]) -> Vec<u32> {
    let mut w = vec![vop2(0x1A, 15, k(4), 0)]; // v_lshlrev_b32 v15, 4, v0
    for (n, r) in regs.into_iter().enumerate() {
        w.extend(store(r, 15, 4 * n as u32));
    }
    w
}

/// `native` as [`super::emit`] gives it to a host whose waves have at most `host_wave` lanes,
/// validated by `dxil.dll` when present.
fn emitted(native: &NativeProgram, host_wave: u32) -> HostShader {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = super::super::Container::parse(&bytes).expect("container");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let shader = super::super::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let shader = shader.unwrap_or_else(|e| panic!("emit for {host_wave}-lane host waves: {e}"));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// `shader` on the hardware device: one group, root constants `s0..s3`, a raw UAV of `dwords`
/// zeroed dwords, which come back; None (and a line) without a device that takes it.
fn run(shader: &HostShader, constants: [u32; 4], dwords: u32) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program: {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let mut out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out.swap_remove(0))
}

/// Panics at the first dword of `got` that is not the console's (`want`), naming the thread
/// and which of its four dwords.
fn same_as_console(label: &str, got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len());
    if let Some(n) = (0..want.len()).find(|&n| got[n] != want[n]) {
        panic!("{label}: thread {}, dword {}: {:#x}, the console's {:#x}", n / 4, n % 4, got[n], want[n]);
    }
}

/// `body` as a one-wave compute program of `threads` threads.
fn program(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    native_of(body, wave64, threads)
}

/// RDNA 2 `ds_bpermute_b32` in wave64 reads within the caller's own 32-lane half (ISA 10.4.4):
/// lane 32 + n reading "lane 1" gets lane 33's dword, not lane 1's. Before, the 64-lane host
/// form read across the whole wave, and the 32-lane exchange form refused the program.
#[test]
fn bpermute_wave64_stays_in_each_half() {
    let mut w = vec![vop1(0x01, 1, 0)]; // v_mov_b32 v1, s0 (byte address)
    w.extend(ds(0xB3, 2, 1, 0, 4)); // ds_bpermute_b32 v2, v1, v0 offset:4
    w.extend(stores([2, 0, 0, 0]));
    let native = program(&w, true, 64);
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for host_wave in [64, 32] {
        let shader = emitted(&native, host_wave);
        for address in [0u32, 120, 124] {
            let Some(got) = run(&shader, [address, 0, 0, 0], 256) else { return };
            let index = ((address + 4) >> 2) & 31;
            let want: Vec<u32> = (0..64).flat_map(|lane| [(lane & 32) | index, lane, lane, lane]).collect();
            same_as_console(&format!("{host_wave}-lane host, address {address}"), &got, &want);
        }
    }
    eprintln!("ran on the hardware device: ds_bpermute stayed in each half");
}

/// A DPP `row_mask` (0x6: rows 1 and 2, console lanes 16..47) of a console wave64 program run
/// as two 32-lane host waves, with and without the halves' exchange: the upper host wave holds
/// console rows 2 and 3, so the mask tests the console lane's row, not the host lane's.
#[test]
fn wave64_dpp_row_mask_uses_console_rows_in_every_form() {
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for with_exchange in [false, true] {
        let mut w = vop3(0x169, 1, v(0), k(3)).to_vec(); // v_mul_lo_u32 v1, v0, 3
        w.extend([
            vop2(0x25, 1, k(1), 1), // v_add_nc_u32 v1, 1, v1
            vop1(0x01, 4, LIT),     // v_mov_b32 v4, 1000
            1000,
            vop1(0x01, 4, 0xFA), // v_mov_b32 v4, v1 row_shr:1 row_mask:0x6 bank_mask:0xf
            0x6 << 28 | 0xF << 24 | 0x111 << 8 | 1,
        ]);
        if with_exchange {
            w.extend([
                vopc(0xC1, LIT, 4), // v_cmp_lt_u32 vcc, 500, v4
                500,
                sop1(0x10, 16, VCC), // s_bcnt1_i32_b64 s16, vcc
                vop1(0x01, 5, 16),   // v_mov_b32 v5, s16
            ]);
        } else {
            w.push(vop1(0x01, 5, k(0))); // v_mov_b32 v5, 0
        }
        w.extend(stores([4, 5, 1, 0]));
        let native = program(&w, true, 64);
        let x = |n: u32| 3 * n + 1;
        let shifted = |n: u32| if matches!(n >> 4, 1 | 2) && n & 15 != 0 { x(n - 1) } else { 1000 };
        let count = if with_exchange { (0..64u32).filter(|&n| shifted(n) > 500).count() as u32 } else { 0 };
        let want: Vec<u32> = (0..64u32).flat_map(|n| [shifted(n), count, x(n), n]).collect();
        for host_wave in [64, 32] {
            let shader = emitted(&native, host_wave);
            let Some(got) = run(&shader, [0; 4], 256) else { return };
            same_as_console(&format!("{host_wave}-lane host, exchange {with_exchange}"), &got, &want);
        }
    }
    eprintln!("ran on the hardware device: DPP row_mask matched console rows 0..3");
}

/// RDNA 2 `v_writelane_b32` ignores EXEC (SGPR spills into VGPR lanes inside divergent code)
/// and both `v_writelane_b32` and `v_readlane_b32` take the lane index's low 5 (wave32) or 6
/// (wave64) bits.
#[test]
fn writelane_ignores_exec_and_lane_indices_are_masked() {
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());

    // Wave32: EXEC narrowed to the even lanes (no branch), a write to odd lane 5 through index
    // s1 = 37, EXEC restored, read back at index 37.
    let mut w32 = vec![
        vop1(0x01, 6, v(0)),    // v_mov_b32 v6, v0
        vop2(0x1B, 3, k(1), 0), // v_and_b32 v3, 1, v0
        sop1(0x03, 12, EXEC),   // s_mov_b32 s12, exec_lo
        vopc(0xD2, k(0), 3),    // v_cmpx_eq_u32 exec_lo, 0, v3 (even lanes)
    ];
    w32.extend(vop3(0x361, 6, 0, 1)); // v_writelane_b32 v6, s0, s1
    w32.push(sop1(0x03, EXEC, 12)); // s_mov_b32 exec_lo, s12
    w32.extend(vop3(0x360, 14, v(6), 1)); // v_readlane_b32 s14, v6, s1
    w32.push(vop1(0x01, 7, 14)); // v_mov_b32 v7, s14
    w32.extend(stores([6, 7, 0, 0]));
    let val32 = 0xDEAD_BEEF;
    let want32: Vec<u32> = (0..32u32).flat_map(|t| [if t == 5 { val32 } else { t }, val32, t, t]).collect();
    let shader32 = emitted(&program(&w32, false, 32), 32);
    let Some(got32) = run(&shader32, [val32, 37, 0, 0], 128) else { return };
    same_as_console("wave32", &got32, &want32);

    // Wave64: EXEC narrowed to the lanes below 32 (no branch), a write to inactive lane 45
    // through index s1 = 109, EXEC restored, read back at index 109. (shortcut: inside a lane
    // branch the host lanes outside it do not run the write at all; the AnyXboxX test of
    // that case runs on its paired form, which this tree lacks.)
    let mut w64 = vec![
        vop1(0x01, 6, v(0)),  // v_mov_b32 v6, v0
        sop1(0x04, 12, EXEC), // s_mov_b64 s[12:13], exec
        vopc(0xD4, k(32), 0), // v_cmpx_gt_u32 exec, 32, v0 (lanes 0..31)
    ];
    w64.extend(vop3(0x361, 6, 0, 1)); // v_writelane_b32 v6, s0, s1
    w64.push(sop1(0x04, EXEC, 12)); // s_mov_b64 exec, s[12:13]
    w64.extend(vop3(0x360, 14, v(6), 1)); // v_readlane_b32 s14, v6, s1
    w64.push(vop1(0x01, 7, 14)); // v_mov_b32 v7, s14
    w64.extend(stores([6, 7, 0, 0]));
    let val64 = 0xCAFE_BABE;
    let want64: Vec<u32> = (0..64u32).flat_map(|t| [if t == 45 { val64 } else { t }, val64, t, t]).collect();
    let native64 = program(&w64, true, 64);
    for host_wave in [64, 32] {
        let shader = emitted(&native64, host_wave);
        let Some(got) = run(&shader, [val64, 109, 0, 0], 256) else { return };
        same_as_console(&format!("wave64 guarded, {host_wave}-lane host waves"), &got, &want64);
    }

    // Wave64, every lane running: reads and a write at unmasked indices s0 = 104 (lane 40)
    // and s1 = 69 (lane 5), on 64-lane host waves and as two exchanging 32-lane ones.
    let mut wex = vop3(0x169, 6, v(0), k(3)).to_vec(); // v_mul_lo_u32 v6, v0, 3
    wex.push(vop2(0x25, 6, k(1), 6)); // v_add_nc_u32 v6, 1, v6
    wex.extend(vop3(0x360, 14, v(6), 0)); // v_readlane_b32 s14, v6, s0 (lane 40)
    wex.extend(vop3(0x360, 15, v(6), 1)); // v_readlane_b32 s15, v6, s1 (lane 5)
    wex.extend(vop3(0x361, 6, 2, 0)); // v_writelane_b32 v6, s2, s0 (lane 40)
    wex.push(vop1(0x01, 7, 14)); // v_mov_b32 v7, s14
    wex.push(vop1(0x01, 8, 15)); // v_mov_b32 v8, s15
    wex.extend(stores([6, 7, 8, 0]));
    let native_ex = program(&wex, true, 64);
    let val_ex = 0x1234_5678;
    let want_ex: Vec<u32> = (0..64u32)
        .flat_map(|t| [if t == 40 { val_ex } else { 3 * t + 1 }, 3 * 40 + 1, 3 * 5 + 1, t])
        .collect();
    for host_wave in [64, 32] {
        let shader = emitted(&native_ex, host_wave);
        let Some(got) = run(&shader, [104, 69, val_ex, 0], 256) else { return };
        same_as_console(&format!("wave64 on {host_wave}-lane host waves"), &got, &want_ex);
    }
    eprintln!("ran on the hardware device: writelane ignored EXEC, lane indices masked");
}

/// `v_readlane_b32` reads the lane's VGPR whether that lane runs there or not, here from inside
/// a lane branch (odd lanes) inside a loop, of lanes the branch leaves out too: the per-wave
/// stack the title's ray-traced shadow programs keep in a VGPR's lanes. The read is made of the
/// VGPR's lanes kept in LDS (`LaneMirror`), with no `WaveReadLaneAt`, on every host: dxil-spirv
/// (vkd3d-proton) recursed without end moving such a read out of its branches, so the shadow
/// programs' pipelines were never made. Before, the read was a `WaveReadLaneAt` of a lane the
/// host does not run there.
#[test]
fn lane_reads_of_lanes_outside_the_branch_need_no_shuffle() {
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    const SOPP: u32 = 0xBF80_0000;
    let sopp = |op: u32, simm: i16| SOPP | op << 16 | u32::from(simm as u16);
    let mut w = vop3(0x169, 6, v(0), k(3)).to_vec(); // v_mul_lo_u32 v6, v0, 3
    w.push(vop2(0x25, 6, k(1), 6)); // v_add_nc_u32 v6, 1, v6
    w.push(vop1(0x01, 7, k(0))); // v_mov_b32 v7, 0
    w.push(sop1(0x03, 20, k(0))); // s_mov_b32 s20, 0
    w.push(sop1(0x03, 21, EXEC)); // s_mov_b32 s21, exec_lo
    let top = w.len();
    w.push(vop2(0x1B, 3, k(1), 0)); // v_and_b32 v3, 1, v0
    w.push(vopc(0xD2, k(1), 3)); // v_cmpx_eq_u32 exec_lo, 1, v3 (odd lanes)
    let skip = w.len();
    w.push(sopp(8, 0)); // s_cbranch_execz past the body (set below)
    w.extend(vop3(0x360, 14, v(6), 20)); // v_readlane_b32 s14, v6, s20 (even lanes too)
    w.push(vop2(0x25, 7, 14, 7)); // v_add_nc_u32 v7, s14, v7
    w[skip] = sopp(8, (w.len() - skip - 1) as i16);
    w.push(sop1(0x03, EXEC, 21)); // s_mov_b32 exec_lo, s21
    w.push(0x8000_0000 | 20 << 16 | k(1) << 8 | 20); // s_add_u32 s20, s20, 1
    w.push(0xBF00_0000 | 10 << 16 | 20); // s_cmp_lt_u32 s20, s0
    let back = w.len();
    w.push(sopp(5, -((back + 1 - top) as i16))); // s_cbranch_scc1 back to the top
    w.extend(stores([6, 7, 0, 0]));
    let native = program(&w, false, 32);
    let n = 5u32;
    let sum: u32 = (0..n).map(|i| 3 * i + 1).sum();
    let want: Vec<u32> = (0..32u32).flat_map(|t| [3 * t + 1, if t % 2 == 1 { sum } else { 0 }, t, t]).collect();
    for spirv in [false, true] {
        let was = lower::TEST_DXIL_SPIRV.replace(Some(spirv));
        lower::LANE_SHUFFLES.set(0);
        let shader = emitted(&native, 32);
        lower::TEST_DXIL_SPIRV.set(was);
        assert_eq!(lower::LANE_SHUFFLES.get(), 0, "a WaveReadLaneAt (dxil-spirv {spirv})");
        let Some(got) = run(&shader, [n, 0, 0, 0], 128) else { return };
        same_as_console(&format!("v_readlane in a lane branch in a loop (dxil-spirv {spirv})"), &got, &want);
    }
    eprintln!("ran on the hardware device: lane reads of lanes outside the branch matched the console");
}

/// A DPP quad permutation where the host takes DXIL through dxil-spirv is made of the host's
/// quad operations (no `WaveReadLaneAt`, see `lane_reads_of_lanes_outside_the_branch_need_no_shuffle`),
/// lanes 4n..4n+3 a quad as on the console; elsewhere a lane read, as before.
#[test]
fn dpp_quad_permutations_are_quad_operations_through_dxil_spirv() {
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut w = vop3(0x169, 1, v(0), k(3)).to_vec(); // v_mul_lo_u32 v1, v0, 3
    w.push(vop2(0x25, 1, k(1), 1)); // v_add_nc_u32 v1, 1, v1
    for (dst, perm) in [(4u32, 0xB1u32), (5, 0x4E), (6, 0x1B)] {
        // v_mov_b32 v<dst>, v1 quad_perm:<perm> row_mask:0xf bank_mask:0xf bound_ctrl:1
        w.extend([vop1(0x01, dst, 0xFA), 0xF << 28 | 0xF << 24 | 1 << 19 | perm << 8 | 1]);
    }
    w.extend(stores([4, 5, 6, 1]));
    let native = program(&w, false, 32);
    let x = |n: u32| 3 * n + 1;
    let want: Vec<u32> = (0..32u32).flat_map(|t| [x(t ^ 1), x(t ^ 2), x(t ^ 3), x(t)]).collect();
    for spirv in [false, true] {
        let was = lower::TEST_DXIL_SPIRV.replace(Some(spirv));
        lower::LANE_SHUFFLES.set(0);
        let shader = emitted(&native, 32);
        lower::TEST_DXIL_SPIRV.set(was);
        let shuffles = lower::LANE_SHUFFLES.get();
        assert_eq!(shuffles == 0, spirv, "{shuffles} WaveReadLaneAt (dxil-spirv {spirv})");
        let Some(got) = run(&shader, [0; 4], 128) else { return };
        same_as_console(&format!("quad permutations (dxil-spirv {spirv})"), &got, &want);
    }
    eprintln!("ran on the hardware device: DPP quad permutations matched the console");
}

/// A lane outside EXEC gives 0 to the lanes whose `ds_swizzle_b32` reads it.
#[test]
fn guarded_swizzles_read_zero_from_inactive_lanes() {
    let mut w = vec![
        vop2(0x25, 1, k(1), 0), // v1 = lane + 1, never 0
        vop1(0x01, 2, k(42)),
        vop1(0x01, 3, k(42)),
        vop2(0x1B, 4, k(1), 0),
        vopc(0xC2, k(0), 4),
        sop1(0x24, 12, VCC), // the even lanes
    ];
    w.extend(ds(0x35, 2, 1, 0, 0x801B)); // reverse each quad: an odd lane read
    w.extend(ds(0x35, 3, 1, 0, 0x401F)); // swap 16: another even lane read
    w.push(sop1(0x04, EXEC, 12));
    w.extend(stores([1, 2, 3, 4]));
    let native = program(&w, true, 64);
    let want: Vec<u32> = (0..64)
        .flat_map(|lane| {
            let (quad, swap) = if lane & 1 == 0 { (0, (lane ^ 16) + 1) } else { (42, 42) };
            [lane + 1, quad, swap, lane & 1]
        })
        .collect();
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for host_wave in [64, 32] {
        let shader = emitted(&native, host_wave);
        let Some(got) = run(&shader, [0; 4], 256) else { return };
        same_as_console(&format!("{host_wave}-lane host"), &got, &want);
    }
    eprintln!("ran on the hardware device: guarded swizzles in both halves");
}

/// One wave, two consecutive swizzles, and a distinct EXEC mask for each (VIctory PR #284's
/// sweep), on 64-lane and 32-lane host forms. Run with
/// `generated_swizzle_sweep -- --ignored --nocapture`; `SWIZZLE_CASE=n` (0..255) replays one.
#[test]
#[ignore = "explicit GPU sweep"]
fn generated_swizzle_sweep() {
    // Each odd multiplier permutes the six bits of lane numbers modulo 64; the cutoff selects
    // exactly that many lanes, including empty and full masks.
    fn set_mask(words: &mut Vec<u32>, multiplier: u32, key: u32, cutoff: u32) {
        words.extend(vop3(0x169, 4, v(0), k(multiplier))); // v_mul_lo_u32
        words.push(vop2(0x1B, 4, k(63), 4)); // v_and_b32
        words.push(vop2(0x1D, 4, k(key), 4)); // v_xor_b32
        words.push(vopc(0xC4, k(cutoff), 4)); // v_cmp_gt_u32
        words.push(sop1(0x24, 12, VCC)); // save EXEC and select the active lanes
    }
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let cases = match std::env::var("SWIZZLE_CASE") {
        Ok(raw) => {
            let case = raw.parse::<u32>().expect("SWIZZLE_CASE must be an integer");
            assert!(case < 256, "SWIZZLE_CASE must be in 0..256");
            case..=case
        }
        Err(_) => 0..=255,
    };
    for case in cases {
        let quad = case; // the four selectors of a quad permute
        let bits = (case * 15401 + 731) & 0x7fff;
        let m = 1 + 2 * (case % 32);
        let key = (case * 13) % 64;
        let count = case % 65;
        let second_count = (case * 17 + 23) % 65;
        let mut words = vec![vop2(0x25, 1, k(1), 0), vop1(0x01, 2, k(42))];
        set_mask(&mut words, m, key, count);
        words.extend(ds(0x35, 1, 1, 0, 0x8000 | quad)); // in place
        words.push(sop1(0x04, EXEC, 12));
        set_mask(&mut words, 64 - m, key ^ 63, second_count);
        words.extend(ds(0x35, 2, 1, 0, bits));
        words.push(sop1(0x04, EXEC, 12));
        words.push(vop2(0x1A, 15, k(3), 0)); // two output dwords per lane
        words.extend(store(1, 15, 0));
        words.extend(store(2, 15, 4));
        let native = native(&words);
        let active = |lane: u32| ((lane * m % 64) ^ key) < count;
        let second = |lane: u32| ((lane * (64 - m) % 64) ^ (key ^ 63)) < second_count;
        let after_quad = |lane: u32| {
            let src = (lane & !3) + ((quad >> (2 * (lane % 4))) & 3);
            if !active(lane) {
                lane + 1
            } else if active(src) {
                src + 1
            } else {
                0
            }
        };
        let want: Vec<u32> = (0..64u32)
            .flat_map(|lane| {
                let within = (((lane & 31) & bits) | ((bits >> 5) & 31)) ^ (bits >> 10);
                let src = (lane & 32) + within;
                let value = if !second(lane) {
                    42
                } else if second(src) {
                    after_quad(src)
                } else {
                    0
                };
                [after_quad(lane), value]
            })
            .collect();
        for host_wave in [64, 32] {
            let shader = emitted(&native, host_wave);
            let Some(got) = run(&shader, [0; 4], 128) else { return };
            if let Some(n) = (0..want.len()).find(|&n| got[n] != want[n]) {
                panic!(
                    "case {case}, {host_wave}-lane host, lane {}, dword {}: {:#x}, the console's {:#x} \
                     (quad {quad:#04x}, bitmask {bits:#06x})",
                    n / 2,
                    n % 2,
                    got[n],
                    want[n]
                );
            }
        }
    }
    eprintln!("ran on the hardware device: the swizzle sweep");
}

/// RDNA 2 DPP16 and DPP8 inactive source-lane semantics (`EXEC`, `bound_ctrl`, `fi`):
/// - With `fi:0` and `bound_ctrl:0`, reading an inactive source lane leaves `vdst` untouched.
/// - With `fi:0` and `bound_ctrl:1` (DPP16), reading an inactive source lane supplies `0`.
/// - With `fi:1` (fetch-inactive), reading an inactive source lane reads its VGPR value.
/// - Reading an active source lane transfers its live value.
/// Tested in guarded wave32, divergent-branched wave32, and split wave64-on-wave32.
#[test]
fn dpp16_and_dpp8_honor_inactive_source_lanes_and_fi() {
    // dpp8:[1,0,3,2,5,4,7,6] (each lane swaps with its odd/even neighbour)
    const DPP8_SWAP1: u32 = 1 | 0 << 3 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21;
    let dpp16 = |ctrl: u32, bound_ctrl: bool, fi: bool| -> u32 {
        0xFF00_0000 | u32::from(bound_ctrl) << 19 | u32::from(fi) << 18 | ctrl << 8 | 1
    };
    let sopp = |op: u32, simm16: i16| -> u32 { 0xBF80_0000 | op << 16 | u32::from(simm16 as u16) };

    let build_guarded = |wave64: bool| -> Vec<u32> {
        let mut w = vop3(0x169, 1, v(0), k(10)).to_vec(); // v_mul_lo_u32 v1, v0, 10
        w.extend([
            vop2(0x25, 1, k(1), 1), // v_add_nc_u32 v1, 1, v1 (10*lane + 1)
            vop1(0x01, 4, LIT),
            0xAAAA,
            vop1(0x01, 5, LIT),
            0xBBBB,
            vop1(0x01, 6, LIT),
            0xCCCC,
            vop1(0x01, 7, LIT),
            0xDDDD,
            vop2(0x1B, 2, k(1), 0), // v_and_b32 v2, 1, v0
            vopc(0xC2, k(0), 2),    // v_cmp_eq_u32 vcc, 0, v2 (even lanes active)
            sop1(if wave64 { 0x24 } else { 0x3C }, 12, VCC), // s_and_saveexec_b64/b32 s12, vcc
            // 1. DPP16 quad_perm:[1,0,3,2] (0xB1) bound_ctrl:0 fi:0 -> inactive odd source: keep v4
            vop1(0x01, 4, 0xFA),
            dpp16(0xB1, false, false),
            // 2. DPP16 quad_perm:[1,0,3,2] (0xB1) bound_ctrl:1 fi:0 -> inactive odd source: write 0
            vop1(0x01, 5, 0xFA),
            dpp16(0xB1, true, false),
            // 3. DPP16 quad_perm:[1,0,3,2] (0xB1) bound_ctrl:0 fi:1 -> fetch inactive odd source
            vop1(0x01, 6, 0xFA),
            dpp16(0xB1, false, true),
            // 4. DPP8 swap-1 fi:0 (0xE9) -> inactive odd source: keep v7
            vop1(0x01, 7, 0xE9),
            DPP8_SWAP1 << 8 | 1,
            // 5. DPP16 quad_perm:[2,3,0,1] (0x4E) bound_ctrl:0 fi:0 -> active even source added to v5 (0)
            vop2(0x25, 5, 0xFA, 5),
            dpp16(0x4E, false, false),
            sop1(if wave64 { 0x04 } else { 0x03 }, EXEC, 12),
        ]);
        w.extend(stores([4, 5, 6, 7]));
        w
    };

    let want_for = |lanes: u32| -> Vec<u32> {
        let val = |n: u32| 10 * n + 1;
        (0..lanes)
            .flat_map(|n| {
                if n % 2 == 0 {
                    [0xAAAA, val(n ^ 2), val(n ^ 1), 0xDDDD]
                } else {
                    [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD]
                }
            })
            .collect()
    };

    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());

    // 1. Guarded wave32 compute
    let shader32 = emitted(&program(&build_guarded(false), false, 32), 32);
    if let Some(got32) = run(&shader32, [0; 4], 128) {
        same_as_console("guarded wave32", &got32, &want_for(32));
    }

    // 2. Divergent wave32 branch (guard == None; odd host lanes branched away)
    let mut w32_br = vop3(0x169, 1, v(0), k(10)).to_vec();
    w32_br.extend([
        vop2(0x25, 1, k(1), 1),
        vop1(0x01, 4, LIT),
        0xAAAA,
        vop1(0x01, 5, LIT),
        0xBBBB,
        vop1(0x01, 6, LIT),
        0xCCCC,
        vop1(0x01, 7, LIT),
        0xDDDD,
        vop2(0x1B, 2, k(1), 0),
        vopc(0xC2, k(0), 2),
        sop1(0x3C, 12, VCC), // s_and_saveexec_b32 s12, vcc
        sopp(0x08, 8),       // s_cbranch_execz over 4 x 2-dword DPP instructions
        vop1(0x01, 4, 0xFA),
        dpp16(0xB1, false, false),
        vop1(0x01, 5, 0xFA),
        dpp16(0xB1, true, false),
        vop1(0x01, 7, 0xE9),
        DPP8_SWAP1 << 8 | 1,
        vop2(0x25, 5, 0xFA, 5),
        dpp16(0x4E, false, false),
        sop1(0x03, EXEC, 12),
    ]);
    w32_br.extend(stores([4, 5, 6, 7]));
    let shader32_br = emitted(&program(&w32_br, false, 32), 32);
    let want32_br: Vec<u32> = (0..32u32)
        .flat_map(|n| {
            if n % 2 == 0 {
                [0xAAAA, 10 * (n ^ 2) + 1, 0xCCCC, 0xDDDD]
            } else {
                [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD]
            }
        })
        .collect();
    if let Some(got32_br) = run(&shader32_br, [0; 4], 128) {
        same_as_console("branched wave32", &got32_br, &want32_br);
    }

    // 3. Guarded wave64 compute on both 64-lane and 32-lane hosts
    let native64 = program(&build_guarded(true), true, 64);
    for host_wave in [64, 32] {
        let shader64 = emitted(&native64, host_wave);
        if let Some(got64) = run(&shader64, [0; 4], 256) {
            same_as_console(&format!("guarded wave64 on {host_wave}-lane host"), &got64, &want_for(64));
        }
    }
    eprintln!("ran on the hardware device: DPP16 and DPP8 honored EXEC, bound_ctrl, and fi");
}
