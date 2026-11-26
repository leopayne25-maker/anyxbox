
// float32 -> float16, subnormals kept (half_from_float): round to nearest even with overflow to infinity,
// or toward zero with finite values past the range giving 65504.
uint BufHalfFromFloat(uint f, bool rtz) {
  uint sign = (f >> 16) & 0x8000u, e = (f >> 23) & 0xFFu, m = f & 0x7FFFFFu;
  if (e == 0xFFu) {
    uint payload = m >> 13;
    return sign | 0x7C00u | (m != 0u && payload == 0u ? 1u : payload);
  }
  if (e < 101u) return sign;
  uint full = m | 0x800000u;
  bool normal = e > 112u;
  uint drop = normal ? 13u : 126u - e;
  uint q = full >> drop, r = full & ((1u << drop) - 1u), half = 1u << (drop - 1u);
  if (!rtz && (r > half || (r == half && (q & 1u) != 0u))) q += 1u;
  uint bits = normal ? ((e - 113u) << 10) + q : q;
  return sign | min(bits, rtz ? 31743u : 0x7C00u);
}

uint BufNarrow(uint info, uint value) {
  uint kind = info & 7u;
  if (kind == 5u || kind == 6u) return value & 0xFFFFu;
  return BufHalfFromFloat(value, kind == 7u);
}

uint BufWiden(uint info, uint c, uint h) {
  uint kind = info & 7u;
  if (kind == 6u) return BufSigned(h, 16u);
  if (kind == 7u && BufWidth(info, c) != 16u) return BufHalf(h);
  return h;
}

void BufWrite(uint address, uint size, uint4 e) {
  uint base = address & ~3u, a = address & 3u, shift = a * 8u, need = (a + size + 3u) >> 2;
  uint data[5] = { e.x, e.y, e.z, e.w, 0u };
  uint mask[5] = { 0u, 0u, 0u, 0u, 0u };
  for (uint j = 0u; j < 4u; j++)
    mask[j] = size >= 4u * j + 4u ? 0xFFFFFFFFu : (size > 4u * j ? (1u << (8u * (size - 4u * j))) - 1u : 0u);
  for (uint k = 0u; k < need; k++) {
    uint value = data[k] << shift, bits = mask[k] << shift;
    if (shift != 0u && k > 0u) {
      value |= data[k - 1u] >> (32u - shift);
      bits |= mask[k - 1u] >> (32u - shift);
    }
    if (bits == 0xFFFFFFFFu) g_memory.Store(base + 4u * k, value);
    else if (bits != 0u) {
      g_memory.InterlockedAnd(base + 4u * k, ~bits);
      g_memory.InterlockedOr(base + 4u * k, value & bits);
    }
  }
}

uint DotPacked(uint a, uint b, uint c, uint width, bool sgn) {
  // v_dot2/4/8 : c + somme des produits des éléments de `width` bits, modulo 2^32 (calcul non signé).
  uint r = c;
  [unroll] for (uint i = 0u; i < 32u; i += width) {
    uint x = (a >> i) & ((1u << width) - 1u), y = (b >> i) & ((1u << width) - 1u);
    if (sgn) {
      x = asuint(asint(a << (32u - width - i)) >> (32u - width));
      y = asuint(asint(b << (32u - width - i)) >> (32u - width));
    }
    r += x * y;
  }
  return r;
}

uint MsadU8(uint a, uint b) {
  // v_msad_u8 : octets où la référence b vaut 0 exclus de la somme.
  uint r = 0u;
  // [loop] et non [unroll] : déroulée, cette boucle est mal compilée par le pilote AMD RDNA 2 (#151).
  [loop] for (uint i = 0u; i < 32u; i += 8u) {
    uint x = (a >> i) & 0xFFu, y = (b >> i) & 0xFFu;
    if (y != 0u) r += x > y ? x - y : y - x;
  }
  return r;
}
uint SadWindow(uint lo, uint hi, uint k, uint b, bool masked) {
  uint w = AlignBit(hi, lo, 8u * k);              // S0[8k+31:8k]
  return masked ? MsadU8(w, b) : SadU8(w, b);
}
uint2 QsadPk(uint lo, uint hi, uint b, uint2 c, bool masked) {
  uint r0 = (SadWindow(lo, hi, 0u, b, masked) + (c.x & 0xFFFFu)) & 0xFFFFu;
  uint r1 = (SadWindow(lo, hi, 1u, b, masked) + (c.x >> 16)) & 0xFFFFu;
  uint r2 = (SadWindow(lo, hi, 2u, b, masked) + (c.y & 0xFFFFu)) & 0xFFFFu;
  uint r3 = (SadWindow(lo, hi, 3u, b, masked) + (c.y >> 16)) & 0xFFFFu;
  return uint2(r0 | (r1 << 16), r2 | (r3 << 16));
}
uint4 MqsadU32(uint lo, uint hi, uint b, uint4 c) {
  return uint4(SadWindow(lo, hi, 0u, b, true) + c.x, SadWindow(lo, hi, 1u, b, true) + c.y,
               SadWindow(lo, hi, 2u, b, true) + c.z, SadWindow(lo, hi, 3u, b, true) + c.w);
}

uint F16ToF32(uint h) {
  uint s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
  if (e == 31u) return m != 0u ? 0x7FC00000u : s | 0x7F800000u;
  if (e != 0u) return s | ((e + 112u) << 23) | (m << 13);
  if (m == 0u) return s;
  uint p = firstbithigh(m);                       // dénormal f16 : normal en f32
  return s | ((p + 103u) << 23) | ((m << (23u - p)) & 0x7FFFFFu);
}
uint F32ToF16(uint f) {
  uint s = (f >> 16) & 0x8000u, a = f & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return 0x7E00u;
  if (a >= 0x477FF000u) return s | 0x7C00u;
  if (a >= 0x38800000u) return s | ((a - 0x38000000u + 0xFFFu + ((a >> 13) & 1u)) >> 13);
  if (a < 0x33000000u) return s;
  uint shift = 126u - (a >> 23), m = (a & 0x7FFFFFu) | 0x800000u, q = m >> shift;
  uint rest = m & ((1u << shift) - 1u), tie = 1u << (shift - 1u);
  return s | (q + ((rest > tie || (rest == tie && (q & 1u) != 0u)) ? 1u : 0u));
}
uint F16Add(uint a, uint b) {
  precise float r = asfloat(F16ToF32(a)) + asfloat(F16ToF32(b));
  uint h = F32ToF16(asuint(r));
  return (h & 0x7FFFu) == 0u ? a & b & 0x8000u : h;    // somme exactement nulle : -0 seulement pour -0 + -0
}
uint F16Sub(uint a, uint b) { return F16Add(a, b ^ 0x8000u); }
uint F16Mul(uint a, uint b) {
  precise float r = asfloat(F16ToF32(a)) * asfloat(F16ToF32(b));
  return F32ToF16(asuint(r));
}
uint F16Fma(uint a, uint b, uint c) {
  precise float x = asfloat(F16ToF32(a)), y = asfloat(F16ToF32(b)), z = asfloat(F16ToF32(c));
  precise float p = x * y;                        // exact : deux mantisses de 11 bits
  precise float s = p + z;
  if ((asuint(s) & 0x7FFFFFFFu) == 0u) return (a ^ b) & c & 0x8000u;    // résultat exactement nul
  precise float t = s - p;
  precise float e = (p - (s - t)) + (z - t);      // erreur exacte de s (TwoSum)
  uint r = asuint(s);
  if ((r & 0x7F800000u) != 0x7F800000u && e != 0.0f && (r & 1u) == 0u)
    r += ((e > 0.0f) == (s > 0.0f)) ? 1u : 0xFFFFFFFFu;    // arrondi à l'impair
  return F32ToF16(r);
}
uint F16Ldexp(uint a, uint n) {
  int k = clamp(asint(n << 16) >> 16, -64, 64);   // au-delà de ±64, le f16 vaut déjà 0 ou l'infini
  precise float r = asfloat(F16ToF32(a)) * asfloat(uint(k + 127) << 23);
  return F32ToF16(asuint(r));
}
bool F16IsNan(uint h) { return (h & 0x7FFFu) > 0x7C00u; }
uint F16Key(uint h) { return (h & 0x8000u) != 0u ? ~h & 0xFFFFu : h | 0x8000u; }   // -0 juste sous +0
uint F16Min(uint a, uint b) {
  if (F16IsNan(a)) return F16IsNan(b) ? 0x7E00u : b;
  if (F16IsNan(b)) return a;
  return F16Key(a) <= F16Key(b) ? a : b;
}
uint F16Max(uint a, uint b) {
  if (F16IsNan(a)) return F16IsNan(b) ? 0x7E00u : b;
  if (F16IsNan(b)) return a;
  return F16Key(a) >= F16Key(b) ? a : b;
}
uint F16Clamp(uint h) { return F16IsNan(h) || (h & 0x8000u) != 0u ? 0u : min(h, 0x3C00u); }
uint F16Rndne(uint h) {                           // au plus proche pair, en entiers : pas de Round_ne (#227)
  uint s = h & 0x8000u, a = h & 0x7FFFu, e = a >> 10;
  if (a > 0x7C00u) return 0x7E00u;
  if (e >= 25u) return h;                         // |x| ≥ 1024 et ±infini : déjà entier
  if (e < 14u) return s;                          // |x| < 0.5
  if (e == 14u) return a == 0x3800u ? s : s | 0x3C00u;    // 0.5 ≤ |x| < 1 : 0.5 va à 0
  uint f = 25u - e, mask = (1u << f) - 1u, half = 1u << (f - 1u);    // bits sous l'unité
  uint t = a & ~mask, r = a & mask;
  bool up = r > half || (r == half && ((t >> f) & 1u) != 0u);
  return s | (t + (up ? 1u << f : 0u));           // la retenue passe dans l'exposant
}

uint ClassF16(uint h) {
  uint e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, n = (h >> 15) & 1u;
  if (e == 0x1Fu) return m != 0u ? ((m >> 9) & 1u) : (n != 0u ? 2u : 9u);
  if (e == 0u) return m == 0u ? (n != 0u ? 5u : 6u) : (n != 0u ? 4u : 7u);
  return n != 0u ? 3u : 8u;
}
int Sext16(uint x) {
  // Low 16 bits of x, sign-extended (bits 31:16 are ignored).
  return asint(x << 16) >> 16;
}
uint SatI16(int x) { return asuint(clamp(x, -32768, 32767)); }
uint AddSatI16(uint a, uint b, bool subtract) {
  // Signed 16-bit a + b (or a - b) saturated, from the sign bits in uint. Written as
  // SatI16(Sext16(a) + Sext16(b)) then masked to 16 bits, the sign-extended high half reached the
  // destination on an NVIDIA RTX 4060 Laptop GPU (WARP and Intel UHD were right).
  uint r = (subtract ? a - b : a + b) & 0xFFFFu;
  uint o = subtract ? (a ^ b) & (a ^ r) : (a ^ r) & (b ^ r);
  return (o & 0x8000u) != 0u ? ((a & 0x8000u) != 0u ? 0x8000u : 0x7FFFu) : r;
}
uint AddSatI32(uint a, uint b, bool sat) {
  // a + b, or with `sat` the signed sum saturated to [0x80000000, 0x7FFFFFFF].
  uint r = a + b;
  bool overflow = ((a ^ r) & (b ^ r) & 0x80000000u) != 0u;
  return sat && overflow ? ((a & 0x80000000u) != 0u ? 0x80000000u : 0x7FFFFFFFu) : r;
}
uint SubSatI32(uint a, uint b, bool sat) {
  uint r = a - b;
  bool overflow = ((a ^ b) & (a ^ r) & 0x80000000u) != 0u;
  return sat && overflow ? ((a & 0x80000000u) != 0u ? 0x80000000u : 0x7FFFFFFFu) : r;
}

uint F32Clamp01(uint f) {
  if ((f & 0x7FFFFFFFu) > 0x7F800000u || (f & 0x80000000u) != 0u) return 0u;
  return min(f, 0x3F800000u);
}

[numthreads(32, 1, 1)]
void main(uint lane : SV_GroupIndex) {
  uint s[128];
  uint v[256];
  uint s_null = 0u;
  bool scc = false;
  // Dérouler l'initialisation scalaire évite un résultat incorrect sur WARP (#142).
  [unroll] for (uint i = 0; i < 128; ++i) s[i] = 0u;
  for (uint j = 0; j < 256; ++j) v[j] = 0u;
  v[0] = lane;
  s[126] = 0xFFFFFFFFu;
  // La mémoire groupshared n'est pas remise à zéro par D3D12 : on la vide, comme l'interpréteur.
  for (uint w = lane; w < 8128u; w += 32u) g_lds[w] = 0u;
  GroupMemoryBarrierWithGroupSync();
