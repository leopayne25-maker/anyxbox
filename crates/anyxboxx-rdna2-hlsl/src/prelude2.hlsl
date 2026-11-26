
uint4 ImageGather4(uint4 t, uint4 q, uint s_bits, uint t_bits, uint unorm_bit, uint channel, int2 off) {
  uint un = unorm_bit | ((q.x >> 15) & 1u);
  uint cx = q.x & 7u;
  uint cy = (q.x >> 3) & 7u;
  uint border = (q.w >> 30) & 3u;
  uint w = ImageWidth(t);
  uint h = ImageHeight(t);
  float fx = 0.0f;
  float fy = 0.0f;
  int2 xb = ImageBilinearAxis(s_bits, w, un, off.x, fx);
  int2 yb = ImageBilinearAxis(t_bits, h, un, off.y, fy);
  int cx0 = ImageClamp(xb.x, w, cx);
  int cx1 = ImageClamp(xb.y, w, cx);
  int cy0 = ImageClamp(yb.x, h, cy);
  int cy1 = ImageClamp(yb.y, h, cy);
  uint sel = (t.w >> (3u * channel)) & 7u;
  return uint4(ImageSelect(ImageFetch(t, cx0, cy1, border), sel),
               ImageSelect(ImageFetch(t, cx1, cy1, border), sel),
               ImageSelect(ImageFetch(t, cx1, cy0, border), sel),
               ImageSelect(ImageFetch(t, cx0, cy0, border), sel));
}

uint4 ImageLoad(uint4 t, uint x, uint y, uint mip) {
  if (mip != 0u || x >= ImageWidth(t) || y >= ImageHeight(t)) return uint4(0u, 0u, 0u, 0u);
  return ImageSwizzle(ImageTexel(t, int(x), int(y)), t.w);
}

uint ImageFloatToUnorm8(uint bits) {
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) return 0u;
  precise float f = asfloat(bits);
  if (f <= 0.0f) return 0u;
  if (f >= 1.0f) return 255u;
  precise float scaled = f * 255.0f;
  precise float rounded = floor(scaled + 0.5f);
  return uint(rounded);
}

void ImageStore(uint4 t, uint x, uint y, uint mip, uint dmask, uint4 vals) {
  uint w = ImageWidth(t);
  uint h = ImageHeight(t);
  if (mip != 0u || x >= w || y >= h || dmask == 0u) return;
  uint format = (t.y >> 20) & 0x1FFu;
  uint pixel = y * w + x;
  uint base = t.x << 8;
  if (format == 77u) {
    uint addr = base + pixel * 16u;
    if ((dmask & 1u) != 0u) g_memory.Store(addr, vals.x);
    if ((dmask & 2u) != 0u) g_memory.Store(addr + 4u, vals.y);
    if ((dmask & 4u) != 0u) g_memory.Store(addr + 8u, vals.z);
    if ((dmask & 8u) != 0u) g_memory.Store(addr + 12u, vals.w);
  } else if (format == 64u || format == 62u || format == 63u) {
    uint addr = base + pixel * 8u;
    if ((dmask & 1u) != 0u) g_memory.Store(addr, vals.x);
    if ((dmask & 2u) != 0u) g_memory.Store(addr + 4u, vals.y);
  } else if (format == 22u || format == 20u || format == 21u) {
    if ((dmask & 1u) != 0u) g_memory.Store(base + pixel * 4u, vals.x);
  } else if (format == 56u) {
    uint addr = base + pixel * 4u;
    uint word = (dmask == 0xFu) ? 0u : g_memory.Load(addr);
    if ((dmask & 1u) != 0u) word = (word & 0xFFFFFF00u) | ImageFloatToUnorm8(vals.x);
    if ((dmask & 2u) != 0u) word = (word & 0xFFFF00FFu) | (ImageFloatToUnorm8(vals.y) << 8);
    if ((dmask & 4u) != 0u) word = (word & 0xFF00FFFFu) | (ImageFloatToUnorm8(vals.z) << 16);
    if ((dmask & 8u) != 0u) word = (word & 0x00FFFFFFu) | (ImageFloatToUnorm8(vals.w) << 24);
    g_memory.Store(addr, word);
  } else if (format == 14u) {
    uint byte_addr = base + pixel * 2u;
    uint aligned = byte_addr & ~3u;
    uint shift = (byte_addr & 2u) * 8u;
    uint clear_mask = 0u;
    uint bits = 0u;
    if ((dmask & 1u) != 0u) { clear_mask |= 0xFFu << shift; bits |= ImageFloatToUnorm8(vals.x) << shift; }
    if ((dmask & 2u) != 0u) { clear_mask |= 0xFFu << (shift + 8u); bits |= ImageFloatToUnorm8(vals.y) << (shift + 8u); }
    if (clear_mask != 0u) {
      g_memory.InterlockedAnd(aligned, ~clear_mask);
      g_memory.InterlockedOr(aligned, bits);
    }
  } else if (format == 1u) {
    if ((dmask & 1u) != 0u) {
      uint byte_addr = base + pixel;
      uint aligned = byte_addr & ~3u;
      uint shift = (byte_addr & 3u) * 8u;
      g_memory.InterlockedAnd(aligned, ~(0xFFu << shift));
      g_memory.InterlockedOr(aligned, ImageFloatToUnorm8(vals.x) << shift);
    }
  }
}

uint4 ImageResInfo(uint4 t, uint mip) {
  return mip == 0u ? uint4(ImageWidth(t), ImageHeight(t), 1u, 1u) : uint4(0u, 0u, 0u, 1u);
}

uint ImageAtomic32(uint4 t, uint x, uint y, uint op, uint data0, uint data1) {
  uint w = ImageWidth(t);
  uint h = ImageHeight(t);
  if (x >= w || y >= h) return 0u;
  uint format = (t.y >> 20) & 0x1FFu;
  if (format != 20u && format != 21u && format != 22u) return 0u;
  uint addr = (t.x << 8) + (y * w + x) * 4u;
  uint old = 0u;
  if (op == 0u) { g_memory.InterlockedExchange(addr, data0, old); return old; }
  if (op == 1u) { g_memory.InterlockedCompareExchange(addr, data1, data0, old); return old; }
  if (op == 2u) { g_memory.InterlockedAdd(addr, data0, old); return old; }
  if (op == 3u) { g_memory.InterlockedAdd(addr, 0u - data0, old); return old; }
  if (op == 5u) { g_memory.InterlockedMin(addr, data0, old); return old; }
  if (op == 7u) { g_memory.InterlockedMax(addr, data0, old); return old; }
  if (op == 8u) { g_memory.InterlockedAnd(addr, data0, old); return old; }
  if (op == 9u) { g_memory.InterlockedOr(addr, data0, old); return old; }
  if (op == 10u) { g_memory.InterlockedXor(addr, data0, old); return old; }
  uint cur = g_memory.Load(addr);
  for (uint iter = 0u; iter < 256u; ++iter) {
    uint next = cur;
    if (op == 4u) {
      next = (asint(data0) < asint(cur)) ? data0 : cur;
    } else if (op == 6u) {
      next = (asint(data0) > asint(cur)) ? data0 : cur;
    } else if (op == 11u) {
      next = (cur >= data0) ? 0u : (cur + 1u);
    } else if (op == 12u) {
      next = (cur == 0u || cur > data0) ? data0 : (cur - 1u);
    } else if (op == 13u) {
      bool cur_nan = ((cur & 0x7F800000u) == 0x7F800000u) && ((cur & 0x007FFFFFu) != 0u);
      bool cmp_nan = ((data1 & 0x7F800000u) == 0x7F800000u) && ((data1 & 0x007FFFFFu) != 0u);
      next = (!cur_nan && !cmp_nan && (asfloat(cur) == asfloat(data1))) ? data0 : cur;
    } else if (op == 14u) {
      bool cur_nan = ((cur & 0x7F800000u) == 0x7F800000u) && ((cur & 0x007FFFFFu) != 0u);
      bool dat_nan = ((data0 & 0x7F800000u) == 0x7F800000u) && ((data0 & 0x007FFFFFu) != 0u);
      if (cur_nan || dat_nan) next = (cur_nan && !dat_nan) ? data0 : cur;
      else next = (asfloat(data0) < asfloat(cur)) ? data0 : cur;
    } else if (op == 15u) {
      bool cur_nan = ((cur & 0x7F800000u) == 0x7F800000u) && ((cur & 0x007FFFFFu) != 0u);
      bool dat_nan = ((data0 & 0x7F800000u) == 0x7F800000u) && ((data0 & 0x007FFFFFu) != 0u);
      if (cur_nan || dat_nan) next = (cur_nan && !dat_nan) ? data0 : cur;
      else next = (asfloat(data0) > asfloat(cur)) ? data0 : cur;
    }
    uint prev;
    g_memory.InterlockedCompareExchange(addr, cur, next, prev);
    if (prev == cur) return prev;
    cur = prev;
  }
  return cur;
}


static const uint BUF_FORMAT_INFO[128] = { 0x00000000u, 0x00007049u, 0x0000704Au, 0x0000704Bu, 0x0000704Cu, 0x0000784Du, 0x0000784Eu, 0x0000F089u, 0x0000F08Au, 0x0000F08Bu, 0x0000F08Cu, 0x0000F88Du, 0x0000F88Eu, 0x0000F08Fu, 0x000E7091u, 0x000E7092u, 0x000E7093u, 0x000E7094u, 0x000E7895u, 0x000E7896u, 0x0001F90Du, 0x0001F90Eu, 0x0001F90Fu, 0x001EF111u, 0x001EF112u, 0x001EF113u, 0x001EF114u, 0x001EF915u, 0x001EF916u, 0x001EF117u, 0x0254A119u, 0x0254A11Au, 0x0254A11Bu, 0x0254A11Cu, 0x0254A91Du, 0x0254A91Eu, 0x00000000u, 0x02949119u, 0x0294911Au, 0x0294911Bu, 0x0294911Cu, 0x0294991Du, 0x0294991Eu, 0x00000000u, 0x4A521121u, 0x4A521122u, 0x4A521123u, 0x4A521124u, 0x4A521925u, 0x4A521926u, 0x0A529121u, 0x0A529122u, 0x0A529123u, 0x0A529124u, 0x0A529925u, 0x0A529926u, 0x39CE7121u, 0x39CE7122u, 0x39CE7123u, 0x39CE7124u, 0x39CE7925u, 0x39CE7926u, 0x003FFA15u, 0x003FFA16u, 0x003FFA17u, 0x7BDEF221u, 0x7BDEF222u, 0x7BDEF223u, 0x7BDEF224u, 0x7BDEFA25u, 0x7BDEFA26u, 0x7BDEF227u, 0x07FFFB1Du, 0x07FFFB1Eu, 0x07FFFB1Fu, 0xFFFFFC25u, 0xFFFFFC26u, 0xFFFFFC27u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u };

uint BufFormatInfo(uint code, uint4 d) {
  if (((d.y >> 31) & 1u) != 0u || ((d.w >> 23) & 1u) != 0u) return 0u;
  return BUF_FORMAT_INFO[code & 0x7Fu];
}
uint BufWidth(uint info, uint c) { return ((info >> (12u + 5u * c)) & 31u) + 1u; }
uint BufCount(uint info) { return (info >> 3) & 7u; }
uint BufSize(uint info) { return (info >> 6) & 31u; }

// m / (2^n - 1) arrondi au plus proche, 1 <= n <= 16 : l'écriture binaire du quotient est m répété ;
// T (hi:lo, 64 bits) en garde les 64 premiers bits, la mantisse et le bit d'arrondi en sont tirés
// (jamais d'égalité : la suite répétée n'est jamais nulle). Méthode de norm_bits (Leonida), en mots de 32 bits.
uint BufUnorm(uint m, uint n) {
  if (m == 0u) return 0u;
  if (m == (1u << n) - 1u) return 0x3F800000u;
  uint hi = 0u, lo = 0u;
  for (int top = 64 - int(n); top > -int(n); top -= int(n)) {
    if (top >= 32) hi |= m << uint(top - 32);
    else if (top >= 0) {
      lo |= m << uint(top);
      if (uint(top) + n > 32u) hi |= m >> uint(32 - top);
    } else lo |= m >> uint(-top);
  }
  uint p = firstbithigh(hi);
  uint q = p >= 23u ? hi >> (p - 23u) : (hi << (23u - p)) | (lo >> (p + 9u));
  uint r = p >= 24u ? (hi >> (p - 24u)) & 1u : (lo >> (p + 8u)) & 1u;
  return ((p + 95u) << 23) + (q - 0x800000u) + r;
}

uint BufSigned(uint field, uint n) { return asuint(asint(field << (32u - n)) >> (32u - n)); }

uint BufSnorm(uint field, uint n) {
  int x = asint(BufSigned(field, n));
  int limit = int((1u << (n - 1u)) - 1u);
  if (x <= -limit) return 0xBF800000u;
  uint magnitude = BufUnorm(uint(abs(x)), n - 1u);
  return x < 0 ? magnitude | 0x80000000u : magnitude;
}

uint BufHalf(uint h) {
  uint sign = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 0x3FFu;
  if (e == 31u) return sign | 0x7F800000u | (m << 13);
  if (e != 0u) return sign | ((e + 112u) << 23) | (m << 13);
  if (m == 0u) return sign;
  uint p = firstbithigh(m);
  return sign | ((p + 103u) << 23) | ((m << (23u - p)) & 0x7FFFFFu);
}

uint BufComponent(uint kind, uint field, uint n) {
  if (kind == 1u) return BufUnorm(field, n);
  if (kind == 2u) return BufSnorm(field, n);
  if (kind == 3u) return asuint(float(field));
  if (kind == 4u) return asuint(float(asint(BufSigned(field, n))));
  if (kind == 6u) return BufSigned(field, n);
  if (kind == 7u && n == 16u) return BufHalf(field);
  return field;
}

// d16: p / q (p <= q < 2^16) rounded once to float16, nearest even, subnormals kept (half_from_ratio).
uint BufHalfRatio(uint p, uint q) {
  if (p == 0u) return 0u;
  uint shift = firstbithigh(q) - firstbithigh(p);   // p << shift has q's top bit
  if ((p << shift) < q) shift++;
  uint e = min(shift, 14u), steps = e + 10u;
  // floor(p * 2^steps / q) by long division: the remainder stays below q, so 2r fits easily.
  uint s = p >= q ? 1u : 0u, r = p >= q ? p - q : p;
  for (uint i = 0u; i < steps; i++) {
    r <<= 1;
    s <<= 1;
    if (r >= q) { s |= 1u; r -= q; }
  }
  if (2u * r > q || (2u * r == q && (s & 1u) != 0u)) s += 1u;
  return ((14u - e) << 10) + s;
}

// d16 UNORM/SNORM: the float32 of the exact quotient's float16 (normalized_half), which BufNarrow gives back.
uint BufNormHalf(uint kind, uint field, uint n) {
  if (kind == 1u) return BufHalf(BufHalfRatio(field, (1u << n) - 1u));
  int x = asint(BufSigned(field, n));
  int limit = int((1u << (n - 1u)) - 1u);
  if (x <= -limit) return 0xBF800000u;
  uint magnitude = BufHalf(BufHalfRatio(uint(abs(x)), uint(limit)));
  return x < 0 ? magnitude | 0x80000000u : magnitude;
}

uint4 BufDecode(uint info, uint4 w, bool d16) {
  uint4 c = uint4(0u, 0u, 0u, 0u);
  uint position = 0u;
  uint kind = info & 7u;
  bool exact16 = d16 && (kind == 1u || kind == 2u);
  for (uint i = 0u; i < BufCount(info); i++) {
    uint n = BufWidth(info, i);
    uint word = w[position >> 5];
    // Champ par décalage à gauche puis à droite : avec (word >> p) & masque, le pilote AMD RDNA 2 perd le >> 16 des moitiés hautes.
    uint field = n >= 32u ? word : (word << (32u - n - (position & 31u))) >> (32u - n);
    c[i] = exact16 ? BufNormHalf(kind, field, n) : BufComponent(kind, field, n);
    position += n;
  }
  return c;
}

uint4 BufRead(uint address, uint size) {
  uint base = address & ~3u, shift = (address & 3u) * 8u, need = ((address & 3u) + size + 3u) >> 2;
  uint raw[5] = { 0u, 0u, 0u, 0u, 0u };
  for (uint k = 0u; k < need; k++) raw[k] = g_memory.Load(base + 4u * k);
  uint4 w;
  for (uint j = 0u; j < 4u; j++) w[j] = shift == 0u ? raw[j] : (raw[j] >> shift) | (raw[j + 1u] << (32u - shift));
  return w;
}

uint BufSelect(uint4 c, uint info, uint sel) {
  if (info == 0u) return 0u;
  uint kind = info & 7u;
  if (sel == 1u) return kind == 5u || kind == 6u ? 1u : 0x3F800000u;
  if (sel >= 4u && sel - 4u < BufCount(info)) return c[sel - 4u];
  return 0u;
}

uint BufField(uint kind, uint value, uint n) {
  if (n >= 32u) return value;
  if (kind == 7u) return value & ((1u << n) - 1u);
  if (kind == 5u) return min(value, (1u << n) - 1u);
  int high = int((1u << (n - 1u)) - 1u);
  return uint(clamp(asint(value), -high - 1, high)) & ((1u << n) - 1u);
}

uint4 BufEncode(uint info, uint4 v) {
  uint4 w = uint4(0u, 0u, 0u, 0u);
  uint position = 0u;
  for (uint i = 0u; i < BufCount(info); i++) {
    uint n = BufWidth(info, i);
    w[position >> 5] |= BufField(info & 7u, v[i], n) << (position & 31u);
    position += n;
  }
  return w;
}

bool BufStorable(uint info, bool d16) {
  uint kind = info & 7u;
  if (d16) return kind == 5u || kind == 6u || kind == 7u;
  return ((info >> 11) & 1u) != 0u;
}

bool BufStoreSelectsOk(uint info, uint word3, uint count, bool d16) {
  uint n = BufCount(info);
  if (!BufStorable(info, d16) || n != count) return false;
  for (uint i = 0u; i < n; i++) if (((word3 >> (3u * i)) & 7u) != 4u + i) return false;
  return n != 2u || ((word3 >> 6) & 7u) <= 1u;
}

// d16 (emu/rdna2_buffer_d16.py) : 32-bit integer formats are not loaded as 16 bits.
uint BufD16Info(uint info) {
  uint kind = info & 7u;
  return (kind == 5u || kind == 6u) && BufWidth(info, 0u) == 32u ? 0u : info;
}
