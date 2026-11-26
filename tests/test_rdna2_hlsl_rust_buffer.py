"""Untyped MUBUF word accesses: HLSL text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import mubuf
from test_rdna2_hlsl_rust_parity import rust


@pytest.mark.parametrize("width", ["dword", "dwordx2", "dwordx3", "dwordx4"])
@pytest.mark.parametrize("records,stride,mask", [(48, 0, 0xffffffff), (0, 0, 0x55555555),
                                                 (2, 16, 0x55555555), (0, 16, 0xffffffff)])
def test_buffer_word_access(width, records, stride, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    initial_regs = [part for i in range(4) for part in (vop1("v_mov_b32", 4 + i, LITERAL), 0x12340000 + i)]
    data = program(sop1("s_mov_b32", 20, LITERAL), 256, sop1("s_mov_b32", 21, LITERAL), stride << 16,
                   sop1("s_mov_b32", 22, LITERAL), records, sop1("s_mov_b32", 24, LITERAL), 8,
                   vop2("v_lshlrev_b32", 1, const(4), 0), *initial_regs,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *mubuf(f"buffer_store_{width}", 1, 4, 20, offen=1, offset=4, soffset=24),
                   *mubuf(f"buffer_load_{width}", 1, 10, 20, offen=1, offset=4, soffset=24))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytes(2048)
    wave = rdna2_exec.Wave()
    wave.memory.write(0, initial)
    rdna2_exec.run(data, wave)
    stores = " ".join(f"g_memory.Store({1024 + i * 128}u + lane * 4u, v[{10 + i}]);" for i in range(4))
    hlsl = hlsl.replace("  return;\n", f"  {stores}\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    for i in range(4):
        struct.pack_into("<32I", expected, 1024 + i * 128, *wave.v[10 + i])
    assert rdna2_gpu.run(hlsl, initial) == bytes(expected)


def test_raw_buffer_offset_overflow_is_out_of_bounds(monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 20, LITERAL), 256, sop1("s_mov_b32", 22, LITERAL), 32,
                   vop1("v_mov_b32", 1, LITERAL), 0xfffffff0,
                   *mubuf("buffer_load_dword", 1, 10, 20, offen=1, offset=32))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[10]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(512))[:128] == struct.pack("<32I", *wave.v[10])


def test_buffer_idxen_and_overlap_are_named():
    low, high = mubuf("buffer_load_dwordx2", 1, 10, 20, offen=1)
    for data in (program(low | (1 << 13), high), program(*mubuf("buffer_load_dwordx2", 1, 1, 20, offen=1))):
        result = rust(data)
        assert result.returncode != 0 and b"buffer_load_dwordx2" in result.stderr


@pytest.mark.parametrize("soffset", [125, 128, 24])
def test_buffer_without_vector_offset(soffset, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 20, LITERAL), 256, sop1("s_mov_b32", 22, LITERAL), 32,
                   sop1("s_mov_b32", 24, LITERAL), 8,
                   *mubuf("buffer_load_dword", 1, 10, 20, offset=4, soffset=soffset))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytearray(512)
    struct.pack_into("<16I", initial, 240, *range(16))
    wave = rdna2_exec.Wave()
    wave.memory.write(0, bytes(initial))
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[10]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<32I", expected, 0, *wave.v[10])
    assert rdna2_gpu.run(hlsl, bytes(initial)) == bytes(expected)


def test_negative_inline_buffer_offset_is_named():
    result = rust(program(*mubuf("buffer_load_dword", 1, 10, 20, offset=4, soffset=193)))
    assert result.returncode != 0 and b"buffer_load_dword" in result.stderr
