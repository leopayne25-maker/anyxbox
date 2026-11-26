"""VOP3P dot-product HLSL text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, program, sop1, vop1
from test_rdna2_exec_dot import vop3p
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_dot2_i32_i16", "v_dot2_u32_u16", "v_dot4_i32_i8", "v_dot4_u32_u8", "v_dot8_i32_i4", "v_dot8_u32_u4")


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("a,b,c,mask", [(0, 0, 0, 0xffffffff), (0xffffffff, 0xffffffff, 0xffffffff, 0x55555555),
                                         (0x80007fff, 0x7fff0002, 0x12345678, 0xaaaaaaaa),
                                         (0x87654321, 0xf0f0f0f0, 3, 0xffffffff)])
def test_dot_text_and_warp(name, a, b, c, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop1("v_mov_b32", 1, LITERAL), a, vop1("v_mov_b32", 2, LITERAL), b,
                   vop1("v_mov_b32", 3, LITERAL), c, vop1("v_mov_b32", 4, LITERAL), 0xabcdef12,
                   sop1("s_mov_b32", 126, LITERAL), mask, *vop3p(name, 4, V + 1, V + 2, V + 3))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[4]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[4])


@pytest.mark.parametrize("mods", [{"op_sel": 1}, {"op_sel_hi": 6}, {"neg_lo": 1}, {"neg_hi": 1}, {"clamp": 1}])
def test_dot_unsupported_modifiers_are_named(mods):
    result = rust(program(*vop3p("v_dot4_i32_i8", 1, V, V, **mods)))
    assert result.returncode != 0 and b"v_dot4_i32_i8" in result.stderr


@pytest.mark.parametrize("name", OPS)
def test_dot_literal_and_lane_source(name, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(*vop3p(name, 4, LITERAL, V, V), 0x9f3ca5e1)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[4]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[4])
