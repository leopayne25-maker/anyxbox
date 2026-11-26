"""DS LDS word accesses: text and WARP/interpreter parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import ds
from test_rdna2_hlsl_rust_parity import rust

FORMS = (("", "b32"), ("", "b64"), ("", "b96"), ("", "b128"),
         ("2", "b32"), ("2", "b64"), ("2st64", "b32"), ("2st64", "b64"))


@pytest.mark.parametrize("mode,width", FORMS)
@pytest.mark.parametrize("mask", [0xffffffff, 0x55555555])
def test_lds_write_then_read(mode, width, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    write = f"ds_write{mode}_{width}"
    read = f"ds_read{mode}_{width}"
    initial = [part for n in range(2, 8) for part in (vop1("v_mov_b32", n, LITERAL), 0xa0b00000 + n)]
    offset1 = 4 if mode == "2st64" else 128 if mode == "2" else 0
    data = program(vop2("v_lshlrev_b32", 1, const(4), 0), *initial,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *ds(write, 1, data0=2, data1=6, offset0=0, offset1=offset1),
                   *ds(read, 1, vdst=10, offset0=0, offset1=offset1))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    stores = " ".join(f"g_memory.Store({i * 128}u + lane * 4u, v[{10 + i}]);" for i in range(4))
    hlsl = hlsl.replace("  return;\n", f"  {stores}\n  return;\n")
    expected = b"".join(struct.pack("<32I", *wave.v[10 + i]) for i in range(4))
    assert rdna2_gpu.run(hlsl, bytes(512)) == expected


@pytest.mark.parametrize("name", ["ds_write_b8", "ds_read_u16", "ds_sub_u32"])
def test_untranslated_ds_is_named(name):
    result = rust(program(*ds(name, 1, data0=2, vdst=10)))
    assert result.returncode != 0 and name.encode() in result.stderr


def test_lds_full_16_bit_offset(monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop2("v_lshlrev_b32", 1, const(4), 0), vop1("v_mov_b32", 2, LITERAL), 0x12345678,
                   *ds("ds_write_b32", 1, data0=2, offset0=8, offset1=1),
                   *ds("ds_read_b32", 1, vdst=10, offset0=8, offset1=1))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[10]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[10])
