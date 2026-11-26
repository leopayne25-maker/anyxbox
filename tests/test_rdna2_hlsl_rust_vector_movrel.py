"""VOP1 indexed vector moves and swaps: text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, const, program, sop1, vop1
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_movrels_b32", "v_movreld_b32", "v_movrelsd_b32", "v_movrelsd_2_b32", "v_swap_b32", "v_swaprel_b32")


def check(data, regs, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    stores = " ".join(f"g_memory.Store({i * 128}u + lane * 4u, v[{register}]);" for i, register in enumerate(regs))
    hlsl = hlsl.replace("  return;\n", f"  {stores}\n  return;\n")
    expected = b"".join(struct.pack("<32I", *wave.v[register]) for register in regs)
    assert rdna2_gpu.run(hlsl, bytes(len(expected))) == expected


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("mask", [0xffffffff, 0x55555555, 0])
def test_indexed_vector_move(name, mask, monkeypatch):
    values = (10, 11, 12, 13, 20, 21, 22, 23)
    initial = [word for n in values for word in (vop1("v_mov_b32", n, LITERAL), 0x1000 + n)]
    m0 = 1 | 2 << 16 if name in ("v_movrelsd_2_b32", "v_swaprel_b32") else 2
    data = program(*initial, sop1("s_mov_b32", 124, LITERAL), m0,
                   sop1("s_mov_b32", 126, LITERAL), mask, vop1(name, 20, V + 10))
    check(data, values, monkeypatch)


@pytest.mark.parametrize("code,extra", [(const(7), ()), (LITERAL, (0xabcdef12,)), (V, ())])
def test_indexed_destination_source_forms(code, extra, monkeypatch):
    data = program(sop1("s_mov_b32", 124, LITERAL), 2, vop1("v_movreld_b32", 20, code), *extra)
    check(data, (20, 22), monkeypatch)


@pytest.mark.parametrize("name", [name for name in OPS if name != "v_movreld_b32"])
def test_indexed_vector_source_must_be_vgpr(name):
    result = rust(program(vop1(name, 20, const(7))))
    assert result.returncode != 0 and name.encode() in result.stderr


@pytest.mark.parametrize("name", ["v_movrels_b32", "v_movreld_b32"])
def test_indexed_move_at_last_register(name, monkeypatch):
    if name == "v_movrels_b32":
        op = vop1(name, 20, V + 250)
    else:
        op = vop1(name, 250, V + 20)
    data = program(vop1("v_mov_b32", 255, LITERAL), 0x12345678,
                   vop1("v_mov_b32", 20, LITERAL), 0x87654321,
                   sop1("s_mov_b32", 124, LITERAL), 5, op)
    check(data, (20, 255), monkeypatch)


def test_swap_same_register(monkeypatch):
    data = program(vop1("v_mov_b32", 10, LITERAL), 0xabcdef12, vop1("v_swap_b32", 10, V + 10))
    check(data, (10,), monkeypatch)
