"""Instructions scalaires 64 bits RDNA 2 → HLSL (masques de lignes du wave64, paires de SGPR).

Une valeur 64 bits occupe deux SGPR (`s[n]` mot bas, `s[n + 1]` mot haut). Les modèles reçoivent
{a0} {a1} (S0 64 bits) ou {a} (S0 32 bits), {b0} {b1} (S1 64 bits) ou {b} (S1 32 bits), et
{d0} {d1} (destination) ; chaque résultat est calculé dans des temporaires avant d'être écrit, la
destination pouvant être une source. La sémantique (SCC compris) suit l'interpréteur de référence
(emu/rdna2_exec_ops.py et emu/rdna2_exec_bits.py, d'après AMD « RDNA 2 Instruction Set Architecture »,
novembre 2020, chapitre 5 et section 6.2.1 p. 43 pour l'extension des littéraux 32 bits sur 64 bits),
contre lequel la traduction est vérifiée sur GPU et WARP. Ces instructions servent aux deux tailles
de wave.
"""
from emu.rdna2_hlsl_ops import NULL, Untranslatable, hexa, scalar_target, source, source_high

LOGIC = {"and": "{x} & {y}", "or": "{x} | {y}", "xor": "{x} ^ {y}", "andn2": "{x} & ~{y}",
         "orn2": "{x} | ~{y}", "nand": "~({x} & {y})", "nor": "~({x} | {y})", "xnor": "~({x} ^ {y})"}
# EXEC = combinaison de S0 (x) et de l'ancien EXEC (y) ; D = ancien EXEC (saveexec) ou nouveau (wrexec).
EXEC_COMBINE = {**LOGIC, "andn1": "~{x} & {y}", "orn1": "~{x} | {y}"}
LITERAL, SCALAR_LAST = 255, 127
FIRST_ONE = "({0} != 0u ? firstbitlow({0}) : ({1} != 0u ? 32u + firstbitlow({1}) : 0xFFFFFFFFu))"

# Largeur des opérandes lorsqu'ils restent sur 32 bits dans une instruction 64 bits (rdna2_shapes.SHAPES).
S0_32 = frozenset({"s_bfm_b64", "s_bitset0_b64", "s_bitset1_b64", "s_bitreplicate_b64_b32"})
S1_32 = frozenset({"s_lshl_b64", "s_lshr_b64", "s_ashr_i64", "s_bfe_u64", "s_bfe_i64", "s_bfm_b64",
                   "s_bitcmp0_b64", "s_bitcmp1_b64"})
DST_32 = frozenset({"s_bcnt0_i32_b64", "s_bcnt1_i32_b64", "s_ff0_i32_b64", "s_ff1_i32_b64",
                    "s_flbit_i32_b64", "s_flbit_i32_i64", "s_cmp_eq_u64", "s_cmp_lg_u64",
                    "s_bitcmp0_b64", "s_bitcmp1_b64"})
# ISA RDNA 2 §6.2.1 (p. 43) : un littéral 32 bits sur S0 signé 64 bits est étendu en signe.
SIGNED_LITERAL_S0 = frozenset({"s_ashr_i64", "s_bfe_i64", "s_flbit_i32_i64"})

HELPERS = """uint2 Lshl64(uint lo, uint hi, uint n) {
  n &= 63u;
  if (n == 0u) return uint2(lo, hi);
  if (n >= 32u) return uint2(0u, lo << (n - 32u));
  return uint2(lo << n, (hi << n) | (lo >> (32u - n)));
}
uint2 Lshr64(uint lo, uint hi, uint n) {
  n &= 63u;
  if (n == 0u) return uint2(lo, hi);
  if (n >= 32u) return uint2(hi >> (n - 32u), 0u);
  return uint2((lo >> n) | (hi << (32u - n)), hi >> n);
}
uint2 Ashr64(uint lo, uint hi, uint n) {
  n &= 63u;
  uint sign = asuint(asint(hi) >> 31);
  if (n == 0u) return uint2(lo, hi);
  if (n >= 32u) return uint2(asuint(asint(hi) >> (n - 32u)), sign);
  return uint2((lo >> n) | (hi << (32u - n)), asuint(asint(hi) >> n));
}
uint2 SBfeU64(uint lo, uint hi, uint c) {
  uint w = (c >> 16) & 0x7Fu;
  if (w == 0u) return uint2(0u, 0u);
  uint2 s = Lshr64(lo, hi, c);
  if (w >= 64u) return s;
  if (w >= 32u) return uint2(s.x, w == 32u ? 0u : (s.y & ((1u << (w - 32u)) - 1u)));
  return uint2(s.x & ((1u << w) - 1u), 0u);
}
uint2 SBfeI64(uint lo, uint hi, uint c) {
  uint w = (c >> 16) & 0x7Fu;
  if (w == 0u) return uint2(0u, 0u);
  uint2 s = Ashr64(lo, hi, c);
  if (w >= 64u) return s;
  if (w == 32u) return uint2(s.x, asuint(asint(s.x) >> 31));
  if (w > 32u) {
    uint sh = 64u - w;
    return uint2(s.x, asuint((asint(s.y) << sh) >> sh));
  }
  uint sh = 32u - w;
  uint r = asuint((asint(s.x) << sh) >> sh);
  return uint2(r, asuint(asint(r) >> 31));
}
uint2 SBfmB64(uint a, uint b) {
  uint w = a & 63u;
  uint lo = w == 0u ? 0u : (w >= 32u ? 0xFFFFFFFFu : ((1u << w) - 1u));
  uint hi = w <= 32u ? 0u : ((1u << (w - 32u)) - 1u);
  return Lshl64(lo, hi, b);
}
uint BitReplicate32(uint v) {
  uint r = 0u;
  [unroll] for (uint i = 0u; i < 16u; ++i) { if (((v >> i) & 1u) != 0u) r |= 3u << (i * 2u); }
  return r;
}
uint FlbitU64(uint lo, uint hi) {
  if (hi != 0u) return 31u - firstbithigh(hi);
  return lo != 0u ? 63u - firstbithigh(lo) : 0xFFFFFFFFu;
}
uint FlbitI64(uint lo, uint hi) {
  bool neg = (hi & 0x80000000u) != 0u;
  return FlbitU64(neg ? ~lo : lo, neg ? ~hi : hi);
}
"""


def _pair(code, name, literal=None, signed_literal=False):
    if code == LITERAL:
        if literal is None:
            raise Untranslatable(f"non traduite en HLSL : {name} avec littéral manquant (64 bits)")
        high = "0xFFFFFFFFu" if signed_literal and (literal & 0x8000_0000) else "0u"
        return hexa(literal), high
    if code == SCALAR_LAST:
        return "s[127]", "0u"
    try:
        return source(code), source_high(code)
    except Untranslatable as error:
        raise Untranslatable(f"non traduite en HLSL : {name} ({error})") from error


def _target(code, name, wide=True):
    if code == NULL:
        return "s_null", "s_null"
    if wide and code >= SCALAR_LAST:
        raise Untranslatable(f"non traduite en HLSL : {name} écrit 64 bits au-delà de s127 (s{code})")
    return scalar_target(code), scalar_target(code + 1) if wide else "s_null"


def _logic(operation):
    expression = LOGIC[operation]
    low, high = expression.format(x="{a0}", y="{b0}"), expression.format(x="{a1}", y="{b1}")
    return f"{{{{ uint r0 = {low}, r1 = {high}; scc = (r0 | r1) != 0u; {{d0}} = r0; {{d1}} = r1; }}}}"


def _exec(operation, keep_new):
    expression = EXEC_COMBINE[operation]
    low, high = expression.format(x="{a0}", y="e0"), expression.format(x="{a1}", y="e1")
    written = "s[126], s[127]" if keep_new else "e0, e1"
    first, second = written.split(", ")
    return (f"{{{{ uint e0 = s[126], e1 = s[127]; uint r0 = {low}, r1 = {high}; s[126] = r0; s[127] = r1; "
            f"scc = (r0 | r1) != 0u; {{d0}} = {first}; {{d1}} = {second}; }}}}")


SCALAR64 = {
    "s_mov_b64": "{{ uint r0 = {a0}, r1 = {a1}; {d0} = r0; {d1} = r1; }}",
    "s_cmov_b64": "if (scc) {{ uint r0 = {a0}, r1 = {a1}; {d0} = r0; {d1} = r1; }}",
    "s_cselect_b64": "{{ uint r0 = scc ? {a0} : {b0}, r1 = scc ? {a1} : {b1}; {d0} = r0; {d1} = r1; }}",
    "s_not_b64": "{{ uint r0 = ~{a0}, r1 = ~{a1}; scc = (r0 | r1) != 0u; {d0} = r0; {d1} = r1; }}",
    "s_bcnt1_i32_b64": "{{ uint r = countbits({a0}) + countbits({a1}); scc = r != 0u; {d0} = r; }}",
    "s_bcnt0_i32_b64": "{{ uint r = 64u - countbits({a0}) - countbits({a1}); scc = r != 0u; {d0} = r; }}",
    "s_ff1_i32_b64": "{{ uint x0 = {a0}, x1 = {a1}; {d0} = " + FIRST_ONE.format("x0", "x1") + "; }}",
    "s_ff0_i32_b64": "{{ uint x0 = ~{a0}, x1 = ~{a1}; {d0} = " + FIRST_ONE.format("x0", "x1") + "; }}",
    "s_cmp_eq_u64": "scc = {a0} == {b0} && {a1} == {b1};",
    "s_cmp_lg_u64": "scc = {a0} != {b0} || {a1} != {b1};",
    "s_lshl_b64": "{{ uint2 r = Lshl64({a0}, {a1}, {b}); scc = (r.x | r.y) != 0u; {d0} = r.x; {d1} = r.y; }}",
    "s_lshr_b64": "{{ uint2 r = Lshr64({a0}, {a1}, {b}); scc = (r.x | r.y) != 0u; {d0} = r.x; {d1} = r.y; }}",
    "s_ashr_i64": "{{ uint2 r = Ashr64({a0}, {a1}, {b}); scc = (r.x | r.y) != 0u; {d0} = r.x; {d1} = r.y; }}",
    "s_bfe_u64": "{{ uint2 r = SBfeU64({a0}, {a1}, {b}); scc = (r.x | r.y) != 0u; {d0} = r.x; {d1} = r.y; }}",
    "s_bfe_i64": "{{ uint2 r = SBfeI64({a0}, {a1}, {b}); scc = (r.x | r.y) != 0u; {d0} = r.x; {d1} = r.y; }}",
    "s_bfm_b64": "{{ uint2 r = SBfmB64({a}, {b}); {d0} = r.x; {d1} = r.y; }}",
    "s_bitcmp0_b64": "{{ uint i = {b} & 63u; scc = (((i < 32u ? {a0} : {a1}) >> (i & 31u)) & 1u) == 0u; }}",
    "s_bitcmp1_b64": "{{ uint i = {b} & 63u; scc = (((i < 32u ? {a0} : {a1}) >> (i & 31u)) & 1u) != 0u; }}",
    "s_bitset0_b64": "{{ uint i = {a} & 63u; if (i < 32u) {d0} &= ~(1u << i); else {d1} &= ~(1u << (i - 32u)); }}",
    "s_bitset1_b64": "{{ uint i = {a} & 63u; if (i < 32u) {d0} |= (1u << i); else {d1} |= (1u << (i - 32u)); }}",
    "s_brev_b64": "{{ uint r0 = reversebits({a1}), r1 = reversebits({a0}); {d0} = r0; {d1} = r1; }}",
    "s_flbit_i32_b64": "{d0} = FlbitU64({a0}, {a1});",
    "s_flbit_i32_i64": "{d0} = FlbitI64({a0}, {a1});",
    "s_bitreplicate_b64_b32": "{{ uint x = {a}; {d0} = BitReplicate32(x & 0xFFFFu); {d1} = BitReplicate32(x >> 16); }}",
    "s_wqm_b64": "{{ uint r0 = WqmB32({a0}), r1 = WqmB32({a1}); scc = (r0 | r1) != 0u; {d0} = r0; {d1} = r1; }}",
    "s_quadmask_b64": "{{ uint r0 = QuadMaskB32({a0}) | (QuadMaskB32({a1}) << 8u); scc = r0 != 0u; {d0} = r0; {d1} = 0u; }}",
}
SCALAR64.update({f"s_{operation}_b64": _logic(operation) for operation in LOGIC})
SCALAR64.update({f"s_{operation}_saveexec_b64": _exec(operation, False) for operation in EXEC_COMBINE})
SCALAR64.update({f"s_{operation}_wrexec_b64": _exec(operation, True) for operation in ("andn1", "andn2")})


def statement(name, fields, literal=None):
    """Ligne HLSL d'une instruction scalaire 64 bits de SCALAR64 (champs SOP1, SOP2 ou SOPC)."""
    sdst = fields.get("sdst", NULL)
    if name in ("s_bitset0_b64", "s_bitset1_b64") and sdst == NULL:
        return ""
    values = {}
    if "ssrc0" in fields:
        code = fields["ssrc0"]
        if name in S0_32:
            values["a"] = source(code, literal)
        else:
            values["a0"], values["a1"] = _pair(code, name, literal, signed_literal=name in SIGNED_LITERAL_S0)
    if "ssrc1" in fields:
        code = fields["ssrc1"]
        if name in S1_32:
            values["b"] = source(code, literal)
        else:
            values["b0"], values["b1"] = _pair(code, name, literal)
    values["d0"], values["d1"] = _target(sdst, name, wide=name not in DST_32)
    return SCALAR64[name].format(**values)
