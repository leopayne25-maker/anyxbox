"""GLOBAL/FLAT word access text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import OFF, OPCODES, global_
from test_rdna2_hlsl_rust_parity import rust


def flat(name, addr, data=0, vdst=0, offset=0):
    return (0x37 << 26 | OPCODES["FLAT", name] << 18 | (offset & 0xfff),
            vdst << 24 | OFF << 16 | data << 8 | addr)


@pytest.mark.parametrize("family", ["global", "global_off", "flat"])
@pytest.mark.parametrize("width", ["dword", "dwordx2", "dwordx3", "dwordx4"])
@pytest.mark.parametrize("mask", [0xffffffff, 0x55555555])
def test_word_access_text_and_warp(family, width, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    is_flat = family == "flat"
    name = "flat" if is_flat else "global"
    encoder = flat if is_flat else global_
    kwargs = {} if is_flat else {"saddr": 4 if family == "global" else OFF}
    offset = 4 if is_flat else -4
    initialize = [part for i in range(4) for part in (vop1("v_mov_b32", 4 + i, LITERAL), 0x12340000 + i)]
    absolute = [] if family == "global" else [vop2("v_add_nc_u32", 1, LITERAL, 1), 256]
    data = program(sop1("s_mov_b32", 4, LITERAL), 256, sop1("s_mov_b32", 5, LITERAL), 0,
                   vop2("v_lshlrev_b32", 1, const(4), 0), *absolute,
                   vop1("v_mov_b32", 2, const(0)), *initialize,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *encoder(f"{name}_store_{width}", 1, data=4, offset=offset, **kwargs),
                   *encoder(f"{name}_load_{width}", 1, vdst=10, offset=offset, **kwargs))
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


def test_unsafe_flat_forms_are_named():
    for data, name in [(program(*flat("flat_load_dword", 1, vdst=10, offset=0x800)), b"flat_load_dword"),
                       (program(*flat("flat_load_dwordx2", 1, vdst=1)), b"flat_load_dwordx2"),
                       (program(*global_("global_atomic_sub", 1, data=4)), b"global_atomic_sub")]:
        result = rust(data)
        assert result.returncode != 0 and name in result.stderr


@pytest.mark.parametrize("mask", [0xffffffff, 0x55555555])
def test_global_addtid_text_and_warp(mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 4, LITERAL), 256, sop1("s_mov_b32", 5, LITERAL), 0,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *global_("global_store_dword_addtid", 0, data=0, saddr=4, offset=-4),
                   *global_("global_load_dword_addtid", 0, vdst=10, saddr=4, offset=-4))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytes(512)
    wave = rdna2_exec.Wave()
    wave.memory.write(0, initial)
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[10]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<32I", expected, 0, *wave.v[10])
    assert rdna2_gpu.run(hlsl, initial) == bytes(expected)


def test_global_addtid_requires_scalar_base():
    result = rust(program(*global_("global_load_dword_addtid", 0, vdst=10, saddr=OFF)))
    assert result.returncode != 0 and b"global_load_dword_addtid" in result.stderr
