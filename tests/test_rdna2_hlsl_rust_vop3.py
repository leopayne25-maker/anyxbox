"""Native VOP3 integer translation and interpreter parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, program, sop1, vop1, vop3
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_add3_u32", "v_lshl_add_u32", "v_add_lshl_u32", "v_lshl_or_b32", "v_and_or_b32", "v_or3_b32", "v_xor3_b32", "v_xad_u32",
       "v_bfe_u32", "v_bfe_i32", "v_bfi_b32", "v_bfm_b32", "v_alignbit_b32", "v_alignbyte_b32", "v_perm_b32", "v_min3_u32", "v_max3_u32", "v_med3_u32",
       "v_min3_i32", "v_max3_i32", "v_med3_i32", "v_mul_lo_u32", "v_mul_hi_u32", "v_mul_hi_i32",
       "v_mad_u32_u24", "v_mad_i32_i24", "v_bcnt_u32_b32")


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("mask", [0xffffffff, 0x55555555])
def test_vop3_text_and_warp(name, mask, monkeypatch):
    data = program(vop1("v_mov_b32", 1, LITERAL), 0x87654321,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *vop3(name, 1, V + 1, 255, V, abs_=7, neg=7, omod=3), 0x80000021)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[1])


def test_vop3_clamp_rejected_by_name():
    result = rust(program(*vop3("v_add3_u32", 1, V, V, V, clamp=1)))
    assert result.returncode != 0 and b"v_add3_u32" in result.stderr


@pytest.mark.parametrize("name", ["v_fma_f32", "v_fma_legacy_f32", "v_mad_f32", "v_min3_f32", "v_max3_f32", "v_med3_f32"])
@pytest.mark.parametrize("abs_,neg,omod,clamp", [(0, 0, 0, 0), (5, 3, 2, 1)])
def test_vop3_float_modifiers(name, abs_, neg, omod, clamp, monkeypatch):
    data = program(vop1("v_mov_b32", 1, LITERAL), 0x3f400000,
                   *vop3(name, 1, V + 1, 242, 242, abs_=abs_, neg=neg, omod=omod, clamp=clamp))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[1])
