//! Lane masks a join takes the same bits of on every way in ([`super::Lower::joined_dword`]):
//! console wave64 programs of 64 threads (one wave), lowered for a host whose waves are 64
//! lanes wide and validated by `dxil.dll` when present. Translation only: nothing here makes a
//! device.
//!
//! In each program the lanes below s0 make the mask M (`v_cmp_gt_u32 vcc, s0, v0`, kept in
//! s[20:21]) and count it (`s_bcnt1_i32_b64`: a mask read as a number, so the lowering declares
//! the console's wave); the even lanes run an arm that moves s[20:21] away and back; after the
//! join every lane stores s20 and s21 as numbers. On the console both moves copy the register
//! whatever EXEC holds, so the join holds M's bits whichever arm ran last.

use std::collections::HashMap;

use super::super::{Lower, MASK_ELSEWHERE, Parts, TEST_HOST_WAVE, lower};
#[cfg(windows)]
use crate::emit::validator::Validator;
use crate::ir::{self, Reg, Value};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, resource, structure};

// Operands as the hardware numbers them: SGPR n is n, VCC 106, EXEC 126, an inline integer n
// 128 + n, VGPR n 256 + n.
const VCC: u32 = 106;
const EXEC: u32 = 126;

const fn k(n: u32) -> u32 {
    128 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

/// A branch by `imm` dwords from the next instruction.
const fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u16 as u32
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

const S_MOV_B32: u32 = 0x03;
const S_MOV_B64: u32 = 0x04;
const S_BCNT1_I32_B64: u32 = 0x10;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
const S_CBRANCH_SCC1: u32 = 0x05;
const S_CBRANCH_EXECZ: u32 = 0x08;

/// The lanes below s0 in vcc, counted (s7) and copied to s[20:21] (M), and the even lanes in
/// vcc.
const MASK_AND_EVEN_LANES: [u32; 5] = [
    vopc(0xC4, 0, 0), // v_cmp_gt_u32 vcc, s0, v0
    sop1(S_BCNT1_I32_B64, 7, VCC),
    sop1(S_MOV_B64, 20, VCC),
    vop2(0x1B, 1, k(1), 0), // v_and_b32 v1, 1, v0
    vopc(0xC2, k(0), 1),    // v_cmp_eq_u32 vcc, 0, v1
];

/// s[20:21] moved to s[24:25], the pair used for something else, and moved back.
const AWAY_AND_BACK: [u32; 3] = [
    sop1(S_MOV_B64, 24, 20),
    sop1(S_MOV_B64, 20, k(5)),
    sop1(S_MOV_B64, 20, 24),
];

/// s20 and s21 as numbers, stored as the thread's two dwords of the UAV (7 dwords).
const STORE_S20_S21: [u32; 7] = [
    vop1(0x01, 2, 20),       // v_mov_b32 v2, s20
    vop1(0x01, 3, 21),       // v_mov_b32 v3, s21
    vop2(0x1A, 15, k(3), 0), // v_lshlrev_b32 v15, 3, v0
    0xE070_1000,
    0x8002_0000 | 2 << 8 | 15, // buffer_store_dword v2, v15, s[8:11], 0 offen
    0xE070_1004,
    0x8002_0000 | 3 << 8 | 15, // buffer_store_dword v3, v15, s[8:11], 0 offen offset:4
];

/// A wave64 compute program of 64 threads (the thread index in v0) with 4 root constants in
/// s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    let mut words = vec![
        sop1(S_MOV_B32, 5, k(0)), // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(0xBF81_0000); // s_endpgm
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `body` lowered for a host of 64-lane waves (the console's wave is one host wave), with what
/// the lowering finds the joins of s20 and s21 after the arm take on every way in
/// ([`Lower::same_bits`]) and the dwords of M (vcc as the first compare writes it).
struct Outcome {
    result: Result<Option<u32>, ShaderError>,
    joins: Vec<Option<Value>>,
    m: [Value; 2],
}

fn lowered(body: &[u32]) -> Outcome {
    let native = native(body);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let parts = Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    };
    let first = p.blocks[0]
        .insts
        .iter()
        .find(|i| i.name() == "v_cmp_gt_u32")
        .expect("the compare");
    let m = [Reg::S(VCC as u8), Reg::S(VCC as u8 + 1)].map(|r| first.write(r).expect("vcc"));
    let was = TEST_HOST_WAVE.replace(Some(64));
    let bad = HashMap::new();
    let l = Lower::new(&parts, &bad);
    // the values of s20 and s21 the stores read: the joins after the arm
    let mut joins = Vec::new();
    for reg in [Reg::S(20), Reg::S(21)] {
        let read = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| i.name() == "v_mov_b32")
            .find_map(|i| i.read(reg))
            .expect("the read");
        joins.push(l.same_bits(read));
    }
    let result = lower(&parts).map(|lowered| {
        #[cfg(windows)]
        if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
            let mut copy = lowered.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        lowered.wave_size
    });
    TEST_HOST_WAVE.set(was);
    Outcome { result, joins, m }
}

/// The arm is a lane branch (`s_cbranch_execz` past it): the even lanes come to the join
/// through it, the odd ones past it. Whichever came last on the console, s[20:21] holds M, so
/// the joins are M's dwords and every lane stores M's ballot.
#[test]
fn a_mask_moved_away_and_back_in_a_lane_branch_joins_as_its_own_number() {
    let mut body = MASK_AND_EVEN_LANES.to_vec();
    body.extend([sop1(S_AND_SAVEEXEC_B64, 12, VCC), sopp(S_CBRANCH_EXECZ, 3)]);
    body.extend(AWAY_AND_BACK);
    body.push(sop1(S_MOV_B64, EXEC, 12)); // the join: s_mov_b64 exec, s[12:13]
    body.extend(STORE_S20_S21);
    let l = lowered(&body);
    assert_eq!(l.joins, vec![Some(l.m[0]), Some(l.m[1])]);
    assert_eq!(l.result, Ok(Some(64)));
}

/// The first program without the count: nothing else reads a mask as a number, so the lowering
/// does not declare the console's wave size and the host may run the program in waves of 32,
/// whose ballots hold half of M. The moved copy stays refused there.
#[test]
fn a_moved_mask_stays_refused_where_the_consoles_wave_is_not_declared() {
    let mut body: Vec<u32> = MASK_AND_EVEN_LANES
        .iter()
        .copied()
        .filter(|&w| w != sop1(S_BCNT1_I32_B64, 7, VCC))
        .collect();
    body.extend([sop1(S_AND_SAVEEXEC_B64, 12, VCC), sopp(S_CBRANCH_EXECZ, 3)]);
    body.extend(AWAY_AND_BACK);
    body.push(sop1(S_MOV_B64, EXEC, 12));
    body.extend(STORE_S20_S21);
    let l = lowered(&body);
    // (the web is the same: only the wave size is missing)
    assert_eq!(l.joins, vec![Some(l.m[0]), Some(l.m[1])]);
    assert!(
        matches!(&l.result, Err(ShaderError::Unsupported { what, .. }) if *what == MASK_ELSEWHERE),
        "{:?}",
        l.result
    );
}

/// The arm is skipped by the wave's own test (`s_cbranch_scc1` on root constant s1 == 0) inside
/// the even lanes' code: a join every lane comes to the same way, where the moved-back copy was
/// still written where the odd lanes of M do not run.
#[test]
fn a_mask_moved_away_and_back_behind_a_uniform_branch_joins_as_its_own_number() {
    let mut body = MASK_AND_EVEN_LANES.to_vec();
    body.extend([
        sop1(S_AND_SAVEEXEC_B64, 12, VCC),
        sopp(S_CBRANCH_EXECZ, 12), // past the even lanes' code
        0xBF06_8001,               // s_cmp_eq_u32 s1, 0
        sopp(S_CBRANCH_SCC1, 3),   // past the moves
    ]);
    body.extend(AWAY_AND_BACK);
    body.extend(STORE_S20_S21);
    body.push(sop1(S_MOV_B64, EXEC, 12));
    let l = lowered(&body);
    assert_eq!(l.joins, vec![Some(l.m[0]), Some(l.m[1])]);
    assert_eq!(l.result, Ok(Some(64)));
}

/// The title's shape: M kept across a loop (four iterations, the wave's count in s6) whose
/// body moves it away and back in the even lanes' arm. The joins in the loop and at its head
/// form one web with M the only value from outside.
#[test]
fn a_mask_kept_across_a_loop_and_moved_in_a_lane_branch_joins_as_its_own_number() {
    let mut body = MASK_AND_EVEN_LANES.to_vec();
    body.push(sop1(S_MOV_B32, 6, k(0))); // s_mov_b32 s6, 0
    // the loop: 9 dwords
    body.extend([sop1(S_AND_SAVEEXEC_B64, 12, VCC), sopp(S_CBRANCH_EXECZ, 3)]);
    body.extend(AWAY_AND_BACK);
    body.extend([
        sop1(S_MOV_B64, EXEC, 12),
        sop2(0x00, 6, 6, k(1)), // s_add_u32 s6, s6, 1
        0xBF0A_8406,            // s_cmp_lt_u32 s6, 4
        sopp(S_CBRANCH_SCC1, -9),
    ]);
    body.extend(STORE_S20_S21);
    let l = lowered(&body);
    assert_eq!(l.joins, vec![Some(l.m[0]), Some(l.m[1])]);
    assert_eq!(l.result, Ok(Some(64)));
}

/// The even lanes' arm puts another mask in s[20:21] (the lanes below s1). On the console the
/// join then holds that mask if some even lane ran the arm, M if none did: which arm ran last
/// decides, for every lane. A join per lane would give the odd lanes M while the console has
/// the other mask, so this stays refused.
#[test]
fn a_join_of_two_different_masks_in_a_lane_branch_read_as_a_number_stays_refused() {
    let mut body = MASK_AND_EVEN_LANES.to_vec();
    // v_cmp_gt_u32 s[22:23], s1, v0 (VOP3), where every lane runs
    body.splice(2..2, [0xD4C4_0016, 1 | 256 << 9]);
    body.extend([
        sop1(S_AND_SAVEEXEC_B64, 12, VCC),
        sopp(S_CBRANCH_EXECZ, 1),
        sop1(S_MOV_B64, 20, 22), // s_mov_b64 s[20:21], s[22:23]
        sop1(S_MOV_B64, EXEC, 12),
    ]);
    body.extend(STORE_S20_S21);
    let l = lowered(&body);
    assert_eq!(l.joins, vec![None, None]);
    assert!(
        matches!(&l.result, Err(ShaderError::Unsupported { what, .. }) if *what == MASK_ELSEWHERE),
        "{:?}",
        l.result
    );
}
