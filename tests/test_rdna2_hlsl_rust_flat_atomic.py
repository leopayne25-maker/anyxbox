"""GLOBAL/FLAT atomic word accesses: text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import global_
from test_rdna2_hlsl_rust_flat import flat
from test_rdna2_hlsl_rust_parity import rust

OPS = ("add", "swap", "and", "or", "xor", "umin", "umax")


@pytest.mark.parametrize("kind", OPS)
@pytest.mark.parametrize("family", ["global", "flat"])
@pytest.mark.parametrize("glc,mask", [(False, 0xffffffff), (True, 0x55555555)])
def test_atomic_text_and_warp(kind, family, glc, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    encoder = flat if family == "flat" else global_
    kwargs = {} if family == "flat" else {"saddr": 4}
    low, high = encoder(f"{family}_atomic_{kind}", 1, data=4, vdst=10, offset=4, **kwargs)
    absolute = [vop2("v_add_nc_u32", 1, LITERAL, 1), 256] if family == "flat" else []
    data = program(sop1("s_mov_b32", 4, LITERAL), 256, sop1("s_mov_b32", 5, LITERAL), 0,
                   vop2("v_lshlrev_b32", 1, const(2), 0), *absolute, vop1("v_mov_b32", 2, const(0)),
                   vop1("v_mov_b32", 4, LITERAL), 0x100, vop1("v_mov_b32", 10, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 126, LITERAL), mask, low | (int(glc) << 16), high)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytearray(1024)
    struct.pack_into("<32I", initial, 260, *range(32))
    wave = rdna2_exec.Wave()
    wave.memory.write(0, bytes(initial))
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(512u + lane * 4u, v[10]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<32I", expected, 512, *wave.v[10])
    assert rdna2_gpu.run(hlsl, bytes(initial)) == bytes(expected)


@pytest.mark.parametrize("name", ["global_atomic_sub", "global_atomic_smin", "flat_atomic_cmpswap", "flat_atomic_add_x2"])
def test_unimplemented_atomic_is_named(name):
    encoder = flat if name.startswith("flat_") else global_
    result = rust(program(*encoder(name, 1, data=4, vdst=10)))
    assert result.returncode != 0 and name.encode() in result.stderr


def test_global_atomic_negative_offset_and_alias(monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    low, high = global_("global_atomic_add", 1, data=4, vdst=4, offset=-4)
    data = program(vop2("v_lshlrev_b32", 1, const(2), 0), vop2("v_add_nc_u32", 1, LITERAL, 1), 256,
                   vop1("v_mov_b32", 2, const(0)), vop1("v_mov_b32", 4, LITERAL), 7,
                   low | (1 << 16), high)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytes(1024)
    wave = rdna2_exec.Wave()
    wave.memory.write(0, initial)
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(512u + lane * 4u, v[4]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<32I", expected, 512, *wave.v[4])
    assert rdna2_gpu.run(hlsl, initial) == bytes(expected)
