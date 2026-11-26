"""Carry/borrow VOP2 text and WARP lane and VCC parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, program, sop1, vop1, vop2
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32")


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("a,b,vcc,mask", [(0, 0, 0, 0xffffffff), (0, 0, 0xffffffff, 0xffffffff),
                                          (0xffffffff, 0, 0x55555555, 0xffffffff),
                                          (0xffffffff, 1, 0xaaaaaaaa, 0x55555555),
                                          (1, 1, 0xffffffff, 0xaaaaaaaa),
                                          (0x12345678, 0xfedcba98, 0x12345678, 0xf0f0f0f0)])
def test_vop2_carry_text_and_warp(name, a, b, vcc, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop1("v_mov_b32", 1, LITERAL), a, vop1("v_mov_b32", 2, LITERAL), b,
                   vop1("v_mov_b32", 3, LITERAL), 0x12345678, sop1("s_mov_b32", 106, LITERAL), vcc,
                   sop1("s_mov_b32", 126, LITERAL), mask, vop2(name, 3, V + 1, 2))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[3]); "
                        "if (lane == 0) g_memory.Store(128, s[106]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(132)) == struct.pack("<33I", *wave.v[3], wave.s[106])


@pytest.mark.parametrize("name", OPS)
def test_vop2_carry_literal_and_lane_source(name, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 106, LITERAL), 0x55555555, vop2(name, 1, LITERAL, 0), 0xfffffff0)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]); "
                        "if (lane == 0) g_memory.Store(128, s[106]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(132)) == struct.pack("<33I", *wave.v[1], wave.s[106])
