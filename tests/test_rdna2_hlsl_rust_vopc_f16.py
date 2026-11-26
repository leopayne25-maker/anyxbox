"""Half-float VOPC text and WARP/interpreter parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, OPCODES, program, sop1, vop1, vopc
from test_rdna2_hlsl_rust_parity import rust

OPS = tuple(name for (family, name) in OPCODES if family == "VOPC" and name.endswith("_f16"))


def check(data, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) { g_memory.Store(0, s[106]); "
                        "g_memory.Store(4, s[126]); }\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(8)) == struct.pack("<II", wave.s[106], wave.s[126])


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("a,b,mask", [(0x3c00, 0x4000, 0xffffffff), (0x7e01, 0x3c00, 0x55555555),
                                       (0xbc00, 0x7c01, 0xaaaaaaaa), (0x8000, 0, 0xffffffff),
                                       (0x0001, 0x8001, 0xffffffff)])
def test_float16_compare(name, a, b, mask, monkeypatch):
    data = program(vop1("v_mov_b32", 1, LITERAL), b | 0xdead0000,
                   sop1("s_mov_b32", 126, LITERAL), mask, sop1("s_mov_b32", 106, LITERAL), 0x12345678,
                   vopc(name, LITERAL, 1), a | 0xabcd0000)
    check(data, monkeypatch)


@pytest.mark.parametrize("code", [*range(240, 249), 125, 128, 192, 193, 208, 256])
def test_float16_source_encodings(code, monkeypatch):
    data = program(vop1("v_mov_b32", 1, LITERAL), 0x3c00, vopc("v_cmp_eq_f16", code, 1))
    check(data, monkeypatch)


@pytest.mark.parametrize("src", [0xe9, 0xea, 0xf9, 0xfa])
def test_float16_extensions_refused(src):
    result = rust(program(vopc("v_cmp_eq_f16", src, 0), 0))
    assert result.returncode != 0 and b"v_cmp_eq_f16" in result.stderr


@pytest.mark.parametrize("value,class_bit", [(0x7c01, 0), (0x7e00, 1), (0xfc00, 2), (0xbc00, 3),
                                             (0x8001, 4), (0x8000, 5), (0, 6), (1, 7),
                                             (0x3c00, 8), (0x7c00, 9)])
def test_float16_class_mask(value, class_bit, monkeypatch):
    data = program(vop1("v_mov_b32", 1, LITERAL), 1 << class_bit,
                   vopc("v_cmp_class_f16", LITERAL, 1), value)
    check(data, monkeypatch)
