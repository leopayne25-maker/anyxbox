  // Arrondi au plus près pair d'un double vers f32, dénormaux et dépassement compris.
  uint lo, hi; asuint(x, lo, hi);
  uint s = hi & 0x80000000u, ef = (hi >> 20) & 0x7FFu;
  if (ef == 0u) return s;
  int e = int(ef) - 1023;
  uint sig = 0x80000000u | ((hi & 0xFFFFFu) << 11) | (lo >> 21);
  bool sticky = (lo & 0x1FFFFFu) != 0u;
  uint shift = e < -126 ? uint(8 + (-126 - e)) : 8u;
  if (shift > 32u) return s;
  uint kept = shift == 32u ? 0u : sig >> shift;
  uint rest = shift == 32u ? sig : sig & ((1u << shift) - 1u);
  uint half = 1u << (shift - 1u);
  if (rest > half || (rest == half && (sticky || (kept & 1u) != 0u))) kept += 1u;
  if (e < -126) return s | kept;
  if (kept == 0x1000000u) { kept = 0x800000u; e += 1; }
  if (e > 127) return s | 0x7F800000u;
  return s | (uint(e + 127) << 23) | (kept & 0x7FFFFFu);
}
bool EqF32(uint a, uint b) {
  return !IsNanF32(a) && !IsNanF32(b) && (a == b || ((a | b) & 0x7FFFFFFFu) == 0u);
}
uint ScaleF32(uint a) {
  // ldexp(a, 64) ; infini et NaN inchangés.
  return ((a >> 23) & 0xFFu) == 0xFFu ? a : F64ToF32(F32ToF64(a) * asdouble(0u, 0x43F00000u));
}
bool QuotientIsDenormalF32(uint n, uint d) {
  // |n / d| arrondi est un dénormal non nul : 2^-150 < |n / d| < (2^24 - 1) × 2^-150 (produits exacts).
  precise double x = F32ToF64(n & 0x7FFFFFFFu), y = F32ToF64(d & 0x7FFFFFFFu);
  precise double low = y * asdouble(0u, 0x36900000u), high = y * ((double)0xFFFFFFu * asdouble(0u, 0x36900000u));
  return x > low && x < high;
}
uint DivScaleF32(uint s0, uint s1, uint s2, out bool vcc) {
  vcc = false;
  uint a1 = s1 & 0x7FFFFFFFu, a2 = s2 & 0x7FFFFFFFu, e1 = a1 >> 23, e2 = a2 >> 23;
  if (a1 == 0u || a2 == 0u) return 0x7FC00000u;
  if (int(e2) - int(e1) >= 96) { vcc = true; return EqF32(s0, s1) ? ScaleF32(s0) : s0; }
  if (e1 == 0u) return ScaleF32(s0);
  if (e1 != 0xFFu && e2 != 0xFFu && QuotientIsDenormalF32(s2, s1)) { vcc = true; return EqF32(s0, s2) ? ScaleF32(s0) : s0; }
  return e2 <= 23u ? ScaleF32(s0) : s0;
}
uint DivFmasF32(uint a, uint b, uint c, bool vcc) {
  bool ia = (a & 0x7FFFFFFFu) == 0x7F800000u, ib = (b & 0x7FFFFFFFu) == 0x7F800000u;
  bool ic = (c & 0x7FFFFFFFu) == 0x7F800000u;
  if (IsNanF32(a) || IsNanF32(b) || IsNanF32(c) || ia || ib || ic) {
    uint ps = (a ^ b) & 0x80000000u;
    bool za = (a & 0x7FFFFFFFu) == 0u, zb = (b & 0x7FFFFFFFu) == 0u;
    if (IsNanF32(a) || IsNanF32(b) || IsNanF32(c) || (ia && zb) || (ib && za)
        || ((ia || ib) && ic && ps != (c & 0x80000000u))) return 0x7FC00000u;
    return ia || ib ? ps | 0x7F800000u : c;
  }
  precise double p = F32ToF64(a) * F32ToF64(b), z = F32ToF64(c);
  precise double r = p + z;
  precise double t = r - p;
  precise double err = (p - (r - t)) + (z - t);
  uint lo, hi; asuint(r, lo, hi);
  if (err != 0.0 && (lo & 1u) == 0u) {              // arrondi à l'impair : le bit 0 garde l'inexactitude
    if ((err > 0.0) == (r > 0.0)) lo |= 1u; else { if (lo == 0u) hi -= 1u; lo -= 1u; }
    r = asdouble(lo, hi);
  }
  if (vcc) r *= (c & 0x7FFFFFFFu) >= 0x40000000u ? asdouble(0u, 0x43F00000u) : asdouble(0u, 0x3BF00000u);
  return F64ToF32(r);
}
uint DivFixupF32(uint q, uint d, uint n) {
  uint sign = (d ^ n) & 0x80000000u, a1 = d & 0x7FFFFFFFu, a2 = n & 0x7FFFFFFFu;
  if (a2 > 0x7F800000u) return n | 0x00400000u;
  if (a1 > 0x7F800000u) return d | 0x00400000u;
  if ((a1 == 0u && a2 == 0u) || (a1 == 0x7F800000u && a2 == 0x7F800000u)) return 0xFFC00000u;
  if (a1 == 0u || a2 == 0x7F800000u) return sign | 0x7F800000u;
  if (a1 == 0x7F800000u || a2 == 0u) return sign;
  if (int(a2 >> 23) - int(a1 >> 23) < -150) return sign;
  return IsNanF32(q) ? sign | 0x7F800000u : sign | (q & 0x7FFFFFFFu);
}
uint2 DivFixupF64(uint2 q, uint2 d, uint2 n) {
  uint sign = (d.y ^ n.y) & 0x80000000u, dh = d.y & 0x7FFFFFFFu, nh = n.y & 0x7FFFFFFFu;
  bool dzero = dh == 0u && d.x == 0u, nzero = nh == 0u && n.x == 0u;
  bool dinf = dh == 0x7FF00000u && d.x == 0u, ninf = nh == 0x7FF00000u && n.x == 0u;
  if (IsNanF64(n.x, n.y)) return uint2(n.x, n.y | 0x00080000u);
  if (IsNanF64(d.x, d.y)) return uint2(d.x, d.y | 0x00080000u);
  if ((dzero && nzero) || (dinf && ninf)) return uint2(0u, 0xFFF80000u);
  if (dzero || ninf) return uint2(0u, sign | 0x7FF00000u);
  if (dinf || nzero) return uint2(0u, sign);
  if (int(nh >> 20) - int(dh >> 20) < -1075) return uint2(0u, sign);
  return IsNanF64(q.x, q.y) ? uint2(0u, sign | 0x7FF00000u) : uint2(q.x, sign | (q.y & 0x7FFFFFFFu));
}


static const uint UNORM8[256] = { 0x00000000u, 0x3B808081u, 0x3C008081u, 0x3C40C0C1u, 0x3C808081u, 0x3CA0A0A1u, 0x3CC0C0C1u, 0x3CE0E0E1u, 0x3D008081u, 0x3D109091u, 0x3D20A0A1u, 0x3D30B0B1u, 0x3D40C0C1u, 0x3D50D0D1u, 0x3D60E0E1u, 0x3D70F0F1u, 0x3D808081u, 0x3D888889u, 0x3D909091u, 0x3D989899u, 0x3DA0A0A1u, 0x3DA8A8A9u, 0x3DB0B0B1u, 0x3DB8B8B9u, 0x3DC0C0C1u, 0x3DC8C8C9u, 0x3DD0D0D1u, 0x3DD8D8D9u, 0x3DE0E0E1u, 0x3DE8E8E9u, 0x3DF0F0F1u, 0x3DF8F8F9u, 0x3E008081u, 0x3E048485u, 0x3E088889u, 0x3E0C8C8Du, 0x3E109091u, 0x3E149495u, 0x3E189899u, 0x3E1C9C9Du, 0x3E20A0A1u, 0x3E24A4A5u, 0x3E28A8A9u, 0x3E2CACADu, 0x3E30B0B1u, 0x3E34B4B5u, 0x3E38B8B9u, 0x3E3CBCBDu, 0x3E40C0C1u, 0x3E44C4C5u, 0x3E48C8C9u, 0x3E4CCCCDu, 0x3E50D0D1u, 0x3E54D4D5u, 0x3E58D8D9u, 0x3E5CDCDDu, 0x3E60E0E1u, 0x3E64E4E5u, 0x3E68E8E9u, 0x3E6CECEDu, 0x3E70F0F1u, 0x3E74F4F5u, 0x3E78F8F9u, 0x3E7CFCFDu, 0x3E808081u, 0x3E828283u, 0x3E848485u, 0x3E868687u, 0x3E888889u, 0x3E8A8A8Bu, 0x3E8C8C8Du, 0x3E8E8E8Fu, 0x3E909091u, 0x3E929293u, 0x3E949495u, 0x3E969697u, 0x3E989899u, 0x3E9A9A9Bu, 0x3E9C9C9Du, 0x3E9E9E9Fu, 0x3EA0A0A1u, 0x3EA2A2A3u, 0x3EA4A4A5u, 0x3EA6A6A7u, 0x3EA8A8A9u, 0x3EAAAAABu, 0x3EACACADu, 0x3EAEAEAFu, 0x3EB0B0B1u, 0x3EB2B2B3u, 0x3EB4B4B5u, 0x3EB6B6B7u, 0x3EB8B8B9u, 0x3EBABABBu, 0x3EBCBCBDu, 0x3EBEBEBFu, 0x3EC0C0C1u, 0x3EC2C2C3u, 0x3EC4C4C5u, 0x3EC6C6C7u, 0x3EC8C8C9u, 0x3ECACACBu, 0x3ECCCCCDu, 0x3ECECECFu, 0x3ED0D0D1u, 0x3ED2D2D3u, 0x3ED4D4D5u, 0x3ED6D6D7u, 0x3ED8D8D9u, 0x3EDADADBu, 0x3EDCDCDDu, 0x3EDEDEDFu, 0x3EE0E0E1u, 0x3EE2E2E3u, 0x3EE4E4E5u, 0x3EE6E6E7u, 0x3EE8E8E9u, 0x3EEAEAEBu, 0x3EECECEDu, 0x3EEEEEEFu, 0x3EF0F0F1u, 0x3EF2F2F3u, 0x3EF4F4F5u, 0x3EF6F6F7u, 0x3EF8F8F9u, 0x3EFAFAFBu, 0x3EFCFCFDu, 0x3EFEFEFFu, 0x3F008081u, 0x3F018182u, 0x3F028283u, 0x3F038384u, 0x3F048485u, 0x3F058586u, 0x3F068687u, 0x3F078788u, 0x3F088889u, 0x3F09898Au, 0x3F0A8A8Bu, 0x3F0B8B8Cu, 0x3F0C8C8Du, 0x3F0D8D8Eu, 0x3F0E8E8Fu, 0x3F0F8F90u, 0x3F109091u, 0x3F119192u, 0x3F129293u, 0x3F139394u, 0x3F149495u, 0x3F159596u, 0x3F169697u, 0x3F179798u, 0x3F189899u, 0x3F19999Au, 0x3F1A9A9Bu, 0x3F1B9B9Cu, 0x3F1C9C9Du, 0x3F1D9D9Eu, 0x3F1E9E9Fu, 0x3F1F9FA0u, 0x3F20A0A1u, 0x3F21A1A2u, 0x3F22A2A3u, 0x3F23A3A4u, 0x3F24A4A5u, 0x3F25A5A6u, 0x3F26A6A7u, 0x3F27A7A8u, 0x3F28A8A9u, 0x3F29A9AAu, 0x3F2AAAABu, 0x3F2BABACu, 0x3F2CACADu, 0x3F2DADAEu, 0x3F2EAEAFu, 0x3F2FAFB0u, 0x3F30B0B1u, 0x3F31B1B2u, 0x3F32B2B3u, 0x3F33B3B4u, 0x3F34B4B5u, 0x3F35B5B6u, 0x3F36B6B7u, 0x3F37B7B8u, 0x3F38B8B9u, 0x3F39B9BAu, 0x3F3ABABBu, 0x3F3BBBBCu, 0x3F3CBCBDu, 0x3F3DBDBEu, 0x3F3EBEBFu, 0x3F3FBFC0u, 0x3F40C0C1u, 0x3F41C1C2u, 0x3F42C2C3u, 0x3F43C3C4u, 0x3F44C4C5u, 0x3F45C5C6u, 0x3F46C6C7u, 0x3F47C7C8u, 0x3F48C8C9u, 0x3F49C9CAu, 0x3F4ACACBu, 0x3F4BCBCCu, 0x3F4CCCCDu, 0x3F4DCDCEu, 0x3F4ECECFu, 0x3F4FCFD0u, 0x3F50D0D1u, 0x3F51D1D2u, 0x3F52D2D3u, 0x3F53D3D4u, 0x3F54D4D5u, 0x3F55D5D6u, 0x3F56D6D7u, 0x3F57D7D8u, 0x3F58D8D9u, 0x3F59D9DAu, 0x3F5ADADBu, 0x3F5BDBDCu, 0x3F5CDCDDu, 0x3F5DDDDEu, 0x3F5EDEDFu, 0x3F5FDFE0u, 0x3F60E0E1u, 0x3F61E1E2u, 0x3F62E2E3u, 0x3F63E3E4u, 0x3F64E4E5u, 0x3F65E5E6u, 0x3F66E6E7u, 0x3F67E7E8u, 0x3F68E8E9u, 0x3F69E9EAu, 0x3F6AEAEBu, 0x3F6BEBECu, 0x3F6CECEDu, 0x3F6DEDEEu, 0x3F6EEEEFu, 0x3F6FEFF0u, 0x3F70F0F1u, 0x3F71F1F2u, 0x3F72F2F3u, 0x3F73F3F4u, 0x3F74F4F5u, 0x3F75F5F6u, 0x3F76F6F7u, 0x3F77F7F8u, 0x3F78F8F9u, 0x3F79F9FAu, 0x3F7AFAFBu, 0x3F7BFBFCu, 0x3F7CFCFDu, 0x3F7DFDFEu, 0x3F7EFEFFu, 0x3F800000u };

uint ImageWidth(uint4 t) { return (((t.y >> 30) & 3u) | ((t.z & 0xFFFu) << 2)) + 1u; }
uint ImageHeight(uint4 t) { return ((t.z >> 14) & 0x3FFFu) + 1u; }

int2 ImageOffset(uint off_word) {
  int ox = (int(off_word & 0x3Fu) << 26) >> 26;
  int oy = (int((off_word >> 8) & 0x3Fu) << 26) >> 26;
  return int2(ox, oy);
}

bool ImageUseLinear(uint4 q, uint has_lod, uint lod_bits) {
  uint filt = (has_lod != 0u && asfloat(lod_bits) > 0.0f) ? ((q.z >> 22) & 3u) : ((q.z >> 20) & 3u);
  return filt == 1u;
}

int ImageClamp(int index, uint size, uint mode) {
  int s = int(size);
  if (mode == 0u) { int r = index % s; return r < 0 ? r + s : r; }
  if (mode == 1u) {
    int p = 2 * s;
    int r = index % p;
    if (r < 0) r += p;
    return r < s ? r : p - 1 - r;
  }
  if (mode == 2u) return clamp(index, 0, s - 1);
  if (mode == 3u) {
    if (index < 0) index = -1 - index;
    return min(index, s - 1);
  }
  if (mode == 7u) {
    if (index < 0) index = -1 - index;
    return index < s ? index : -1;
  }
  if (mode == 6u) return (index >= 0 && index < s) ? index : -1;
  return clamp(index, 0, s - 1);
}

int ImageIndex(uint bits, uint size, uint unnormalized, uint mode, int off) {
  precise float value = asfloat(bits);
  if (unnormalized == 0u) value = value * float(size);
  uint raw = asuint(value);
  if ((raw & 0x7F800000u) == 0x7F800000u && (raw & 0x007FFFFFu) != 0u) return ImageClamp(off, size, mode);
  int index = int(floor(clamp(value, -2147483520.0f, 2147483520.0f))) + off;
  return ImageClamp(index, size, mode);
}

int2 ImageBilinearAxis(uint bits, uint size, uint unnormalized, int off, out float frac) {
  precise float value = asfloat(bits);
  if (unnormalized == 0u) value = value * float(size);
  uint raw = asuint(value);
  if ((raw & 0x7F800000u) == 0x7F800000u && (raw & 0x007FFFFFu) != 0u) {
    frac = 0.0f;
    return int2(off, off);
  }
  precise float shifted = value - 0.5f;
  precise float clamped = clamp(shifted, -2147483520.0f, 2147483520.0f);
  precise float fl = floor(clamped);
  precise float f = clamped - fl;
  frac = f;
  int i0 = int(fl) + off;
  return int2(i0, i0 + 1);
}

uint4 ImageBorderColor(uint border) {
  if (border == 1u) return uint4(0u, 0u, 0u, 0x3f800000u);
  if (border == 2u) return uint4(0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u);
  return uint4(0u, 0u, 0u, 0u);
}

uint4 ImageTexel(uint4 t, int x, int y) {
  uint format = (t.y >> 20) & 0x1FFu;
  uint pixel = uint(y) * ImageWidth(t) + uint(x);
  uint base = t.x << 8;
  if (format == 77u) return g_memory.Load4(base + pixel * 16u);
  if (format == 64u) {
    uint2 v2 = g_memory.Load2(base + pixel * 8u);
    return uint4(v2.x, v2.y, 0u, 0x3f800000u);
  }
  if (format == 62u || format == 63u) {
    uint2 v2 = g_memory.Load2(base + pixel * 8u);
    return uint4(v2.x, v2.y, 0u, 1u);
  }
  if (format == 22u) {
    return uint4(g_memory.Load(base + pixel * 4u), 0u, 0u, 0x3f800000u);
  }
  if (format == 20u || format == 21u) {
    return uint4(g_memory.Load(base + pixel * 4u), 0u, 0u, 1u);
  }
  if (format == 56u) {
    uint w = g_memory.Load(base + pixel * 4u);
    return uint4(UNORM8[w & 255u], UNORM8[(w >> 8) & 255u], UNORM8[(w >> 16) & 255u], UNORM8[w >> 24]);
  }
  if (format == 14u) {
    uint byte_addr = base + pixel * 2u;
    uint w = g_memory.Load(byte_addr & ~3u);
    uint shift = (byte_addr & 2u) * 8u;
    return uint4(UNORM8[(w >> shift) & 255u], UNORM8[(w >> (shift + 8u)) & 255u], 0u, 0x3f800000u);
  }
  if (format == 1u) {
    uint byte_addr = base + pixel;
    uint w = g_memory.Load(byte_addr & ~3u);
    uint shift = (byte_addr & 3u) * 8u;
    return uint4(UNORM8[(w >> shift) & 255u], 0u, 0u, 0x3f800000u);
  }
  return uint4(0u, 0u, 0u, 0u);
}

uint4 ImageFetch(uint4 t, int x, int y, uint border) {
  if (x < 0 || y < 0) return ImageBorderColor(border);
  return ImageTexel(t, x, y);
}

uint ImageSelect(uint4 c, uint sel) {
  if (sel >= 4u) return c[sel - 4u];
  return sel == 1u ? 0x3f800000u : 0u;
}

uint4 ImageSwizzle(uint4 c, uint swz) {
  return uint4(ImageSelect(c, swz & 7u), ImageSelect(c, (swz >> 3) & 7u),
               ImageSelect(c, (swz >> 6) & 7u), ImageSelect(c, (swz >> 9) & 7u));
}

uint ImageLerp(uint a_bits, uint b_bits, float f_in) {
  precise float a = asfloat(a_bits);
  precise float b = asfloat(b_bits);
  precise float f = f_in;
  precise float diff = b - a;
  precise float prod = diff * f;
  precise float res = a + prod;
  return asuint(res);
}

uint ImageBilerp(uint c00, uint c10, uint c01, uint c11, float fx, float fy) {
  return ImageLerp(ImageLerp(c00, c10, fx), ImageLerp(c01, c11, fx), fy);
}

uint ImageCompare(uint z_bits, uint t_bits, uint func) {
  float z = asfloat(z_bits);
  float t = asfloat(t_bits);
  bool p = false;
  if (func == 1u) p = (z < t);
  else if (func == 2u) p = (z == t);
  else if (func == 3u) p = (z <= t);
  else if (func == 4u) p = (z > t);
  else if (func == 5u) p = (z != t);
  else if (func == 6u) p = (z >= t);
  else if (func == 7u) p = true;
  return p ? 0x3f800000u : 0u;
}

uint4 ImageSampleColor(uint4 t, uint4 q, uint s_bits, uint t_bits, uint unorm_bit, int2 off, uint has_lod, uint lod_bits) {
  uint un = unorm_bit | ((q.x >> 15) & 1u);
  uint cx = q.x & 7u;
  uint cy = (q.x >> 3) & 7u;
  uint border = (q.w >> 30) & 3u;
  uint w = ImageWidth(t);
  uint h = ImageHeight(t);
  uint4 raw;
  if (ImageUseLinear(q, has_lod, lod_bits)) {
    float fx = 0.0f;
    float fy = 0.0f;
    int2 xb = ImageBilinearAxis(s_bits, w, un, off.x, fx);
    int2 yb = ImageBilinearAxis(t_bits, h, un, off.y, fy);
    int cx0 = ImageClamp(xb.x, w, cx);
    int cx1 = ImageClamp(xb.y, w, cx);
    int cy0 = ImageClamp(yb.x, h, cy);
    int cy1 = ImageClamp(yb.y, h, cy);
    uint4 c00 = ImageFetch(t, cx0, cy0, border);
    uint4 c10 = ImageFetch(t, cx1, cy0, border);
    uint4 c01 = ImageFetch(t, cx0, cy1, border);
    uint4 c11 = ImageFetch(t, cx1, cy1, border);
    raw = uint4(ImageBilerp(c00.x, c10.x, c01.x, c11.x, fx, fy),
                ImageBilerp(c00.y, c10.y, c01.y, c11.y, fx, fy),
                ImageBilerp(c00.z, c10.z, c01.z, c11.z, fx, fy),
                ImageBilerp(c00.w, c10.w, c01.w, c11.w, fx, fy));
  } else {
    int x = ImageIndex(s_bits, w, un, cx, off.x);
    int y = ImageIndex(t_bits, h, un, cy, off.y);
    raw = ImageFetch(t, x, y, border);
  }
  return ImageSwizzle(raw, t.w);
}

uint4 ImageSampleCompare(uint4 t, uint4 q, uint z_bits, uint s_bits, uint t_bits, uint unorm_bit, int2 off, uint has_lod, uint lod_bits) {
  uint un = unorm_bit | ((q.x >> 15) & 1u);
  uint cx = q.x & 7u;
  uint cy = (q.x >> 3) & 7u;
  uint func = (q.x >> 12) & 7u;
  uint border = (q.w >> 30) & 3u;
  uint w = ImageWidth(t);
  uint h = ImageHeight(t);
  uint pcf;
  if (ImageUseLinear(q, has_lod, lod_bits)) {
    float fx = 0.0f;
    float fy = 0.0f;
    int2 xb = ImageBilinearAxis(s_bits, w, un, off.x, fx);
    int2 yb = ImageBilinearAxis(t_bits, h, un, off.y, fy);
    int cx0 = ImageClamp(xb.x, w, cx);
    int cx1 = ImageClamp(xb.y, w, cx);
    int cy0 = ImageClamp(yb.x, h, cy);
    int cy1 = ImageClamp(yb.y, h, cy);
    uint r00 = ImageCompare(z_bits, ImageFetch(t, cx0, cy0, border).x, func);
    uint r10 = ImageCompare(z_bits, ImageFetch(t, cx1, cy0, border).x, func);
    uint r01 = ImageCompare(z_bits, ImageFetch(t, cx0, cy1, border).x, func);
    uint r11 = ImageCompare(z_bits, ImageFetch(t, cx1, cy1, border).x, func);
    pcf = ImageBilerp(r00, r10, r01, r11, fx, fy);
  } else {
    int x = ImageIndex(s_bits, w, un, cx, off.x);
    int y = ImageIndex(t_bits, h, un, cy, off.y);
    pcf = ImageCompare(z_bits, ImageFetch(t, x, y, border).x, func);
  }
  return ImageSwizzle(uint4(pcf, pcf, pcf, 0x3f800000u), t.w);
}
