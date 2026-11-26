//! DPP16 and DPP8 lane selection, inactive-lane EXEC handling, and write masks.

use super::{Attr, Bin, Cast, Fields, Inst, Lower, Operand, Reg, V, Why, host_dxil_spirv, pred};
use crate::gfx10::Dpp;

impl Lower<'_> {
    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    pub(super) fn dpp(&mut self, inst: &Inst, dpp: Dpp, guard: Option<V>) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, whether it lies in the row, write masks, bound_ctrl, and fetch-inactive.
        let (src, inside, masks, zero, fi) = match dpp {
            Dpp::Lanes { sel, fi } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false, fi)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                fi,
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl, fi)
            }
        };
        // Avoid passing out-of-bounds lane indices to waveReadLaneAt on row shifts.
        let safe_src = match inside {
            Some(ok) => self.s.m.select(c, ok, src, lane),
            None => src,
        };
        let quad_op = if let Dpp::Row { ctrl, .. } = dpp
            && (self.helpers_read(quad) || (ctrl <= 0xFF && host_dxil_spirv()))
        {
            // A quad permutation by the host's quad operations: also where
            // dxil-spirv would take a `WaveReadLaneAt` out of its branches.
            Some([0, 1, 2, 3].map(|i| (ctrl >> (2 * i) & 3) as u8))
        } else {
            None
        };
        let mut r = if mesh_quad {
            self.quad_pick(x, safe_src)
        } else if let Some(lanes) = quad_op {
            self.quad_read(x, lanes)
        } else {
            self.read_lane(x, safe_src)
        };
        // Unless fetch-inactive (`fi:1`) is set, reading an inactive source lane
        // (`EXEC == 0`) yields 0 when `bound_ctrl:1` or leaves `vdst` untouched otherwise.
        let src_active = if fi {
            None
        } else if mesh_quad {
            guard.map(|g| {
                let g32 = self.s.m.cast(c, Cast::ZExt, g, self.s.t.i32);
                let picked = self.quad_pick(g32, safe_src);
                let z = k(self, 0);
                self.s.m.cmp(c, pred::NE, picked, z)
            })
        } else if let Some(lanes) = quad_op {
            guard.map(|g| {
                let g32 = self.s.m.cast(c, Cast::ZExt, g, self.s.t.i32);
                let picked = self.quad_read(g32, lanes);
                let z = k(self, 0);
                self.s.m.cmp(c, pred::NE, picked, z)
            })
        } else {
            let mine = guard.unwrap_or_else(|| self.s.m.bool(true));
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let ballot = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[mine]);
            let lo = self.s.m.extract(c, ballot, 0);
            let word = if self.s.wave_size == Some(32) {
                lo
            } else {
                let hi = self.s.m.extract(c, ballot, 1);
                let k32 = k(self, 32);
                let upper = self.s.m.cmp(c, pred::IUGE, safe_src, k32);
                self.s.m.select(c, upper, hi, lo)
            };
            let (k31, one, z) = (k(self, 31), k(self, 1), k(self, 0));
            let bit = self.s.m.bin(c, Bin::And, safe_src, k31);
            let shifted = self.s.m.bin(c, Bin::LShr, word, bit);
            let on = self.s.m.bin(c, Bin::And, shifted, one);
            Some(self.s.m.cmp(c, pred::NE, on, z))
        };
        let valid = match (inside, src_active) {
            (Some(a), Some(b)) => Some(self.s.m.bin(c, Bin::And, a, b)),
            (a, b) => a.or(b),
        };
        let mut enable = None;
        if let Some(valid) = valid {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, valid, r, z);
            } else {
                enable = Some(valid);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            // (in wave64 on 32-lane host waves rows 2 and 3 are lanes 32..63)
            let row_lane = if self.exchange.is_some() || self.halves.is_some() {
                self.console_lane()
            } else {
                lane
            };
            let ri = self.s.m.bin(c, Bin::LShr, row_lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }
}
