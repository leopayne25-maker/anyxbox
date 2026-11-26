"""DS lane permutations: text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import ds
from test_rdna2_hlsl_rust_parity import rust


@pytest.mark.parametrize("name,offset", [("ds_permute_b32", 0), ("ds_bpermute_b32", 12),
                                         ("ds_swizzle_b32", 0), ("ds_swizzle_b32", 0x801b),
                                         ("ds_swizzle_b32", 0x041f), ("ds_swizzle_b32", 0x7fff)])
@pytest.mark.parametrize("mask,collisions", [(0xffffffff, False), (0x55555555, False),
                                               (0x80000001, True), (0, True)])
def test_ds_lanes_text_and_warp(name, offset, mask, collisions, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    addr = vop1("v_mov_b32", 1, const(0)) if collisions else vop2("v_lshlrev_b32", 1, const(2), 0)
    data = program(addr, vop1("v_mov_b32", 3, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *ds(name, 1, data0=0, vdst=3, offset0=offset & 255, offset1=offset >> 8))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[3]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[3])


@pytest.mark.parametrize("offset", [0x8100, 0xbfff, 0xc000, 0xe000])
def test_unsupported_swizzle_modes_are_named(offset):
    result = rust(program(*ds("ds_swizzle_b32", 1, data0=0, vdst=3,
                              offset0=offset & 255, offset1=offset >> 8)))
    assert result.returncode != 0 and b"ds_swizzle_b32" in result.stderr


@pytest.mark.parametrize("name", ["ds_permute_b32", "ds_bpermute_b32", "ds_swizzle_b32"])
def test_ds_lane_destination_aliases_data(name, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop1("v_mov_b32", 3, LITERAL), 0x12345678,
                   vop2("v_lshlrev_b32", 1, const(2), 0),
                   *ds(name, 1, data0=3, vdst=3, offset0=0x1b, offset1=0x80 if name == "ds_swizzle_b32" else 0))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[3]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[3])
