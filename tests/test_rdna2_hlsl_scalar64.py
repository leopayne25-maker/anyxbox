"""64-bit scalar shifts and bit operations in the HLSL recompiler (emu/rdna2_hlsl_scalar64.py).

Every 64-bit scalar instruction of emu/rdna2_exec_ops.py and emu/rdna2_exec_bits.py is checked
byte for byte against the reference interpreter on GPU (or WARP) in both wave32 and wave64,
including shift/width edge cases (0, 31, 32, 63, 64), SCC updates, register aliasing, and
ISA section 6.2.1 (p. 43) 32-bit literal expansion to 64 bits.
"""
import pytest

import test_rdna2_hlsl_wave64 as wave64
from emu import rdna2_exec_ops, rdna2_gpu, rdna2_hlsl, rdna2_hlsl_graphics
from test_rdna2_exec import const, program, sop1, sop2, sopc, vop1, vopc
from test_rdna2_hlsl import compare, store_lane, with_lane_offset

LITERAL, NULL, VCC_LO, MINUS_ONE = 255, 125, 106, 193
SLOTS_PER_BASE = 16                             # 15 * 128 bytes fits the signed 12-bit global offset
SLOTS_PER_BASE_64 = 8                           # 7 * 256 bytes
gpu = pytest.mark.skipif(not rdna2_gpu.available(), reason="dxc or anyxboxx_gpu_run.exe missing")


def mov32(register, value):
    return sop1("s_mov_b32", register, LITERAL), value & 0xFFFF_FFFF


def mov64(register, value):
    return (*mov32(register, value), *mov32(register + 1, value >> 32))


def scc_into(register):
    return sop2("s_cselect_b32", register, const(1), const(0))


def emit_stores(registers, wave_size=32):
    """Copy each SGPR into v1 and store it to OUTPUT; bumps s2 before the 12-bit offset overflows."""
    limit = SLOTS_PER_BASE_64 if wave_size == 64 else SLOTS_PER_BASE
    store = wave64.store_lane if wave_size == 64 else store_lane
    stride = limit * (wave64.SLOT if wave_size == 64 else 128)
    words = []
    for index, register in enumerate(registers):
        if index and index % limit == 0:
            words += [sop2("s_add_u32", 2, 2, LITERAL), stride]
        words += [vop1("v_mov_b32", 1, register), *store(1, index % limit)]
    return words


def test_all_exec_scalar_ops_are_covered_by_hlsl():
    missing = set(rdna2_exec_ops.SCALAR) - rdna2_hlsl.coverage_names()
    assert missing == set()


def test_64_bit_scalar_refusals_and_null_bitset():
    with pytest.raises(rdna2_hlsl.Untranslatable, match="s127"):
        rdna2_hlsl.translate(program(sop2("s_lshl_b64", 127, 4, const(1))))
    with pytest.raises(rdna2_hlsl.Untranslatable, match="s127"):
        rdna2_hlsl.translate(program(sop1("s_bitset1_b64", 127, const(5))))
    with pytest.raises(rdna2_hlsl.Untranslatable):
        rdna2_hlsl.translate(program(sop2("s_lshl_b64", 4, 242, const(1))))     # inline float on 64-bit S0
    with pytest.raises(rdna2_hlsl.Untranslatable):
        rdna2_hlsl.translate(program(sop1("s_flbit_i32_b64", 4, 242)))
    hlsl = rdna2_hlsl.translate(program(sop1("s_bitset0_b64", NULL, const(5)),
                                        sop1("s_bitset1_b64", NULL, const(40))))
    assert "s_null &=" not in hlsl and "s_null |=" not in hlsl


@gpu
def test_shifts_and_scc_match_interpreter_on_gpu():
    words = [
        *mov64(4, 0x8123_4567_89AB_CDEF),
        *mov64(6, 0x7123_4567_89AB_CDEF),
        # s_lshl_b64 across 0, 1, 31, 32, 33, 63 and 64 (wraps & 63) + zero SCC
        sop2("s_lshl_b64", 8, 4, const(0)), scc_into(10),
        sop2("s_lshl_b64", 12, 4, const(1)), scc_into(14),
        sop2("s_lshl_b64", 16, 4, const(31)), scc_into(18),
        sop2("s_lshl_b64", 20, 4, const(32)), scc_into(22),
        sop2("s_lshl_b64", 24, 4, const(63)), scc_into(26),
        sop2("s_lshl_b64", 28, 4, const(64)),
        sop2("s_lshl_b64", 30, 4, LITERAL), 0x0000_0021,                        # literal S1 = 33
        *mov64(32, 0x0000_0001_0000_0000),
        sop2("s_lshl_b64", 32, 32, const(32)), scc_into(34),                    # alias + result 0 → SCC = 0
        # s_lshr_b64 across 0, 1, 31, 32, 40, 63 + zero SCC
        sop2("s_lshr_b64", 36, 4, const(0)),
        sop2("s_lshr_b64", 38, 4, const(1)),
        sop2("s_lshr_b64", 40, 4, const(31)),
        sop2("s_lshr_b64", 42, 4, const(32)),
        sop2("s_lshr_b64", 44, 4, const(40)), scc_into(46),
        sop2("s_lshr_b64", 48, 4, const(63)), scc_into(50),
        *mov64(52, 0x0000_0000_0000_0001),
        sop2("s_lshr_b64", 52, 52, const(1)), scc_into(54),                     # result 0 → SCC = 0
        # s_ashr_i64 on negative and positive values across 0, 1, 31, 32, 33, 63
        sop2("s_ashr_i64", 56, 4, const(0)),
        sop2("s_ashr_i64", 58, 4, const(1)),
        sop2("s_ashr_i64", 60, 4, const(31)),
        sop2("s_ashr_i64", 62, 4, const(32)),
        sop2("s_ashr_i64", 64, 4, const(33)),
        sop2("s_ashr_i64", 66, 4, const(63)), scc_into(68),
        sop2("s_ashr_i64", 70, 6, const(32)),
        sop2("s_ashr_i64", 72, 6, const(63)), scc_into(74),                     # positive >> 63 = 0 → SCC = 0
    ]
    regs = [8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74]
    expected, actual = compare(with_lane_offset(*words, *emit_stores(regs)))
    assert actual == expected


@gpu
def test_bfe_and_bfm_64_match_interpreter_on_gpu():
    words = [
        *mov64(4, 0xFEDC_BA98_7654_3210),
        # s_bfe_u64: width 0, 1, 16, 32, 40, 63, 64, 127 across low/high boundary
        sop2("s_bfe_u64", 8, 4, const(4)), scc_into(10),                        # width 0 → 0, SCC = 0
        sop2("s_bfe_u64", 12, 4, LITERAL), 16 << 16 | 36, scc_into(14),
        sop2("s_bfe_u64", 16, 4, LITERAL), 32 << 16 | 16,
        sop2("s_bfe_u64", 18, 4, LITERAL), 40 << 16 | 12,
        sop2("s_bfe_u64", 20, 4, LITERAL), 64 << 16 | 4,
        sop2("s_bfe_u64", 22, 4, LITERAL), 127 << 16 | 60,
        # s_bfe_i64: width 0, 4 (neg), 8 (pos), 32 (neg), 32 (pos), 40 (neg), 63, 64, 127
        sop2("s_bfe_i64", 24, 4, const(4)), scc_into(26),                       # width 0 → 0, SCC = 0
        sop2("s_bfe_i64", 28, 4, LITERAL), 4 << 16 | 60, scc_into(30),          # 0xF → -1
        sop2("s_bfe_i64", 32, 4, LITERAL), 8 << 16 | 24,                        # 0x76 → +118
        sop2("s_bfe_i64", 34, 4, LITERAL), 32 << 16 | 16,                       # bit 31 = 1 → sign-extended
        sop2("s_bfe_i64", 36, 4, LITERAL), 32 << 16 | 0,                        # 0x76543210 → positive
        sop2("s_bfe_i64", 38, 4, LITERAL), 40 << 16 | 20,                       # width > 32, negative
        sop2("s_bfe_i64", 40, 4, LITERAL), 63 << 16 | 0,
        sop2("s_bfe_i64", 42, 4, LITERAL), 64 << 16 | 8,
        sop2("s_bfe_i64", 44, 4, LITERAL), 127 << 16 | 60,
        # s_bfm_b64: width 0, 1, 31, 32, 40, 63 with various offsets + SCC preserved
        sopc("s_cmp_eq_u32", const(1), const(1)),                               # SCC = 1
        sop2("s_bfm_b64", 46, const(0), const(5)), scc_into(48),                # width 0, SCC still 1
        sop2("s_bfm_b64", 50, const(1), const(63)),
        sop2("s_bfm_b64", 52, const(32), const(0)),
        sop2("s_bfm_b64", 54, const(32), const(32)),
        sop2("s_bfm_b64", 56, const(40), const(8)),
        sop2("s_bfm_b64", 58, const(63), const(1)),
        sop2("s_bfm_b64", 60, LITERAL, const(20)), 0x0000_0025,                 # literal width = 37
    ]
    regs = [8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30,
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
    expected, actual = compare(with_lane_offset(*words, *emit_stores(regs)))
    assert actual == expected


@gpu
def test_bitcmp_bitset_brev_flbit_wqm_quadmask_replicate_match_on_gpu():
    words = [
        *mov64(4, 0x8000_0100_0000_000F),
        # s_bitcmp0_b64 / s_bitcmp1_b64
        sopc("s_bitcmp1_b64", 4, const(0)), scc_into(8),
        sopc("s_bitcmp1_b64", 4, const(40)), scc_into(9),
        sopc("s_bitcmp1_b64", 4, const(63)), scc_into(10),
        sopc("s_bitcmp1_b64", 4, const(8)), scc_into(11),
        sopc("s_bitcmp0_b64", 4, const(8)), scc_into(12),
        sopc("s_bitcmp0_b64", 4, const(64)), scc_into(13),                      # 64 & 63 == 0 → bit 0 is 1
        # s_bitset0_b64 / s_bitset1_b64 (low word, high word, alias, NULL, SCC preserved)
        *mov64(14, 0),
        sopc("s_cmp_eq_u32", const(1), const(1)),                               # SCC = 1
        sop1("s_bitset1_b64", 14, const(3)),
        sop1("s_bitset1_b64", 14, const(40)),
        sop1("s_bitset1_b64", 14, const(63)),
        sop1("s_bitset0_b64", 14, const(3)),
        sop1("s_bitset0_b64", NULL, const(10)),
        scc_into(16),                                                           # SCC must remain 1
        *mov64(18, 0x0000_0000_0000_0025),
        sop1("s_bitset1_b64", 18, 18),                                          # alias: index = s18 = 37
        # s_brev_b64 (including self-alias)
        sop1("s_brev_b64", 20, 4),
        *mov64(22, 0x0123_4567_89AB_CDEF),
        sop1("s_brev_b64", 22, 22),
        # s_flbit_i32_b64
        *mov64(24, 0),
        sop1("s_flbit_i32_b64", 26, 24),                                        # 0 → -1
        sop1("s_flbit_i32_b64", 27, 4),                                         # bit 63 set → 0
        *mov64(28, 0x0000_0001_0000_0000),
        sop1("s_flbit_i32_b64", 30, 28),                                        # bit 32 → 31
        sop1("s_flbit_i32_b64", 31, const(1)),                                  # bit 0 → 63
        # s_flbit_i32_i64
        sop1("s_flbit_i32_i64", 32, 24),                                        # 0 → -1
        sop1("s_flbit_i32_i64", 33, MINUS_ONE),                                 # -1 → -1
        *mov64(34, 0xFFFF_FFFF_7FFF_FFFF),
        sop1("s_flbit_i32_i64", 36, 34),                                        # → 32
        *mov64(38, 0x7FFF_FFFF_FFFF_FFFF),
        sop1("s_flbit_i32_i64", 40, 38),                                        # → 1
        sop1("s_flbit_i32_i64", 41, const(1)),                                  # → 63
        # s_bitreplicate_b64_b32 (including self-alias)
        *mov32(42, 0x8000_5A01),
        sop1("s_bitreplicate_b64_b32", 44, 42),
        *mov32(46, 0xAAAA_5555),
        sop1("s_bitreplicate_b64_b32", 46, 46),
        # s_wqm_b64 / s_quadmask_b64 (non-zero & zero + SCC)
        *mov64(48, 0x0001_0000_0000_0008),
        sop1("s_wqm_b64", 50, 48), scc_into(52),
        sop1("s_quadmask_b64", 54, 48), scc_into(56),
        sop1("s_wqm_b64", 58, 24), scc_into(60),
        sop1("s_quadmask_b64", 62, 24), scc_into(64),
    ]
    regs = [8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 26, 27, 30, 31,
            32, 33, 36, 40, 41, 44, 45, 46, 47, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64]
    expected, actual = compare(with_lane_offset(*words, *emit_stores(regs)))
    assert actual == expected


@gpu
def test_32_bit_literal_expansion_in_64_bit_scalar_ops_matches_on_gpu():
    """ISA section 6.2.1 (p. 43): signed 64-bit S0 sign-extends a 32-bit literal; unsigned/bitwise zero-extends."""
    words = [
        sop2("s_ashr_i64", 40, LITERAL, const(1)), 0x8000_0000, scc_into(50),
        sop2("s_ashr_i64", 42, LITERAL, const(4)), 0x7FFF_FFFF,
        *mov32(60, 40 << 16 | 4),
        sop2("s_bfe_i64", 44, LITERAL, 60), 0x8000_0000,
        sop2("s_bfe_u64", 46, LITERAL, 60), 0x8000_0000,
        sop1("s_flbit_i32_i64", 51, LITERAL), 0xFFFF_FFF0,
        sop1("s_flbit_i32_b64", 52, LITERAL), 0xFFFF_FFF0,
        sop2("s_lshl_b64", 54, LITERAL, const(4)), 0x8000_0001,
        sop1("s_mov_b64", 56, LITERAL), 0x8000_0000,
    ]
    regs = [40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 54, 55, 56, 57]
    expected, actual = compare(with_lane_offset(*words, *emit_stores(regs)))
    assert actual == expected


@gpu
def test_64_bit_scalar_ops_match_interpreter_in_wave64():
    words = [
        vopc("v_cmp_lt_u32", const(25), 0),                                     # vcc: lanes 26..63
        sop2("s_lshl_b64", 8, VCC_LO, const(5)),
        sop2("s_lshr_b64", 10, VCC_LO, const(20)),
        sop2("s_ashr_i64", 12, VCC_LO, const(35)),
        sop2("s_bfe_u64", 14, VCC_LO, LITERAL), 24 << 16 | 24,
        sop2("s_bfe_i64", 16, VCC_LO, LITERAL), 24 << 16 | 36,
        sop2("s_bfm_b64", 18, const(36), const(12)),
        sopc("s_bitcmp1_b64", VCC_LO, const(40)), scc_into(24),
        sop1("s_mov_b64", 26, VCC_LO),
        sop1("s_bitset0_b64", 26, const(63)),
        sop1("s_bitset1_b64", 26, const(2)),
        sop1("s_brev_b64", 28, VCC_LO),
        sop1("s_flbit_i32_b64", 30, 10),
        sop1("s_flbit_i32_i64", 31, VCC_LO),
        sop1("s_bitreplicate_b64_b32", 32, VCC_LO),
        sop1("s_wqm_b64", 34, 10),
        sop1("s_quadmask_b64", 36, VCC_LO),
    ]
    regs = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
    expected, actual = wave64.compare(wave64.with_lane_offset(*words, *emit_stores(regs, wave_size=64)))
    assert actual == expected


@gpu
def test_64_bit_scalar_helpers_compile_in_vertex_shader():
    code = program(
        *mov64(4, 0x1234_5678_9ABC_DEF0),
        sop2("s_lshl_b64", 6, 4, const(5)),
        sop2("s_bfe_u64", 8, 6, LITERAL), 16 << 16 | 28,
        sop1("s_flbit_i32_b64", 10, 8),
        sop1("s_bitreplicate_b64_b32", 12, 10),
    )
    hlsl = rdna2_hlsl_graphics.translate_vertex(code)
    dxil = rdna2_gpu.compile_dxil(hlsl, model="vs_6_0")
    assert dxil[:4] == b"DXBC"
