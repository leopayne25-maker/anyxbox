"""Untyped MUBUF byte loads: HLSL text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import mubuf
from test_rdna2_hlsl_rust_parity import rust


@pytest.mark.parametrize("width", ["ubyte", "sbyte"])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@pytest.mark.parametrize("records,stride,mask", [(19, 0, 0xffffffff), (0, 0, 0x55555555),
                                                 (2, 16, 0xaaaaaaaa)])
def test_buffer_byte_load(width, offset, records, stride, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 20, LITERAL), 256, sop1("s_mov_b32", 21, LITERAL), stride << 16,
                   sop1("s_mov_b32", 22, LITERAL), records, sop1("s_mov_b32", 24, LITERAL), 8,
                   vop2("v_lshlrev_b32", 1, const(2), 0), vop1("v_mov_b32", 10, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *mubuf(f"buffer_load_{width}", 1, 10, 20, offen=1, offset=offset, soffset=24))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytearray(1024)
    initial[264:392] = bytes([0, 127, 128, 255]) * 32
    wave = rdna2_exec.Wave()
    wave.memory.write(0, bytes(initial))
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(512u + lane * 4u, v[10]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<32I", expected, 512, *wave.v[10])
    assert rdna2_gpu.run(hlsl, bytes(initial)) == bytes(expected)


@pytest.mark.parametrize("name", ["buffer_store_byte", "buffer_load_ushort"])
def test_unsupported_subword_form_is_named(name):
    result = rust(program(*mubuf(name, 1, 10, 20, offen=1)))
    assert result.returncode != 0 and name.encode() in result.stderr


def test_buffer_byte_offset_overflow_and_negative_soffset(monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 20, LITERAL), 256, sop1("s_mov_b32", 22, LITERAL), 32,
                   vop1("v_mov_b32", 1, LITERAL), 0xfffffff0,
                   *mubuf("buffer_load_sbyte", 1, 10, 20, offen=1, offset=32))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[10]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(512))[:128] == struct.pack("<32I", *wave.v[10])
    refused = rust(program(*mubuf("buffer_load_sbyte", 1, 10, 20, soffset=193)))
    assert refused.returncode != 0 and b"buffer_load_sbyte" in refused.stderr
