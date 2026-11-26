"""GLOBAL/FLAT byte loads: HLSL and WARP/interpreter parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import OFF, global_
from test_rdna2_hlsl_rust_flat import flat
from test_rdna2_hlsl_rust_parity import rust


@pytest.mark.parametrize("family", ["global", "global_off", "flat"])
@pytest.mark.parametrize("width", ["ubyte", "sbyte"])
@pytest.mark.parametrize("byte_offset", [0, 1, 2, 3])
@pytest.mark.parametrize("mask", [0xffffffff, 0x55555555])
def test_byte_load_text_and_warp(family, width, byte_offset, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    is_flat = family == "flat"
    name = "flat" if is_flat else "global"
    encoder = flat if is_flat else global_
    kwargs = {} if is_flat else {"saddr": 4 if family == "global" else OFF}
    absolute = [] if family == "global" else [vop2("v_add_nc_u32", 1, LITERAL, 1), 256]
    data = program(sop1("s_mov_b32", 4, LITERAL), 256, sop1("s_mov_b32", 5, LITERAL), 0,
                   vop2("v_lshlrev_b32", 1, const(2), 0), *absolute,
                   vop1("v_mov_b32", 2, const(0)), vop1("v_mov_b32", 10, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *encoder(f"{name}_load_{width}", 1, vdst=10, offset=byte_offset, **kwargs))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytearray(1024)
    initial[256:384] = bytes([0, 127, 128, 255]) * 32
    wave = rdna2_exec.Wave()
    wave.memory.write(0, bytes(initial))
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(512u + lane * 4u, v[10]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<32I", expected, 512, *wave.v[10])
    assert rdna2_gpu.run(hlsl, bytes(initial)) == bytes(expected)


@pytest.mark.parametrize("name", ["flat_store_byte", "global_store_short", "global_load_ushort"])
def test_unported_subword_forms_are_named(name):
    encoder = flat if name.startswith("flat_") else global_
    result = rust(program(*encoder(name, 1, data=4, vdst=10)))
    assert result.returncode != 0 and name.encode() in result.stderr
