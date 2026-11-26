"""VOP2 HLSL text and WARP lane parity against the interpreter."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, program, sop1, vop1, vop2
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_cndmask_b32", "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32", "v_and_b32", "v_or_b32",
       "v_xor_b32", "v_xnor_b32", "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32", "v_min_u32",
       "v_max_u32", "v_min_i32", "v_max_i32", "v_mul_u32_u24", "v_mul_hi_u32_u24", "v_mul_i32_i24",
       "v_mul_hi_i32_i24")


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("a,b,mask,vcc", [(0, 0, 0xffffffff, 0),
                                           (0xffffffff, 0x80000000, 0x55555555, 0xaaaaaaaa),
                                           (0x12345678, 0xfedcba98, 0xaaaaaaaa, 0x12345678)])
def test_vop2_text_and_warp(name, a, b, mask, vcc, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop1("v_mov_b32", 1, LITERAL), a, vop1("v_mov_b32", 2, LITERAL), b,
                   vop1("v_mov_b32", 3, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 106, LITERAL), vcc, sop1("s_mov_b32", 126, LITERAL), mask,
                   vop2(name, 3, V + 1, 2))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[3]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[3])


@pytest.mark.parametrize("name", OPS)
def test_vop2_literal_and_lane_source(name, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop2(name, 1, LITERAL, 0), 0x80000002)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[1])


def test_untranslated_vop2_is_named():
    result = rust(program(vop2("v_add_f32", 1, V, 0)))
    assert result.returncode != 0 and b"v_add_f32" in result.stderr


@pytest.mark.parametrize("src", [0xe9, 0xea, 0xf9, 0xfa])
def test_vop2_extensions_are_named(src):
    result = rust(program(vop2("v_add_nc_u32", 1, src, 0), 0))
    assert result.returncode != 0 and b"v_add_nc_u32" in result.stderr


def test_vop2_destination_can_alias_source(monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop1("v_mov_b32", 1, LITERAL), 0xffffffff, vop2("v_add_nc_u32", 1, V + 1, 0))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[1])
