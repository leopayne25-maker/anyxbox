//! Cross-lane row permutations (`v_permlane16_b32`, `v_permlanex16_b32`), a
//! child module of [`super`] so it shares the emitter's state.

use super::{Attr, Bin, Fields, Inst, Lower, V, Why, pred};

impl Lower<'_> {
    /// Lowers `v_permlane16_b32` (`cross == false`) and `v_permlanex16_b32`
    /// (`cross == true`).
    ///
    /// Each lane reads `src0` of a lane in its own 32-lane half picked by
    /// nibble `lane & 15` of `{src2, src1}`: within its own row of 16
    /// (`v_permlane16`) or the other row of its half (`v_permlanex16`).
    /// `op_sel[0]` is `fi` (fetch inactive) and `op_sel[1]` is `bound_ctrl`
    /// (supply `0` when reading an inactive source lane; otherwise `vdst`
    /// stays untouched).
    pub(super) fn permlane16(
        &mut self,
        inst: &Inst,
        cross: bool,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        self.cross_lane(false)?;
        let c = self.cur;
        let x = self.src(inst, 0)?;
        let (s1, s2) = (self.src(inst, 1)?, self.src(inst, 2)?);
        let i32t = self.s.t.i32;
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let [k15, k8, k7, k2, k16, not31, not15] =
            [15, 8, 7, 2, 16, !31, !15].map(|n| self.s.m.i32(n));
        let low = self.s.m.bin(c, Bin::And, lane, k15);
        // Nibble `low` of the 64-bit selector: src1 holds lanes 0-7.
        let hi_word = self.s.m.cmp(c, pred::IUGE, low, k8);
        let word = self.s.m.select(c, hi_word, s2, s1);
        let n = self.s.m.bin(c, Bin::And, low, k7);
        let sh = self.s.m.bin(c, Bin::Shl, n, k2);
        let pick = self.s.m.bin(c, Bin::LShr, word, sh);
        let pick = self.s.m.bin(c, Bin::And, pick, k15);
        let row = if cross {
            // The other row of this 32-lane half.
            let half = self.s.m.bin(c, Bin::And, lane, not31);
            let other = self.s.m.bin(c, Bin::And, lane, k16);
            let other = self.s.m.bin(c, Bin::Xor, other, k16);
            self.s.m.bin(c, Bin::Or, half, other)
        } else {
            self.s.m.bin(c, Bin::And, lane, not15)
        };
        let from = self.s.m.bin(c, Bin::Or, row, pick);
        let mut r = self.read_lane(x, from);
        let fi = v.opsel & 1 != 0;
        let bound_ctrl = v.opsel & 2 != 0;
        let mut write_guard = guard;
        if !fi {
            let mine = guard.unwrap_or_else(|| self.s.m.bool(true));
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let ballot = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[mine]);
            let lo = self.s.m.extract(c, ballot, 0);
            let word = if self.s.wave_size == Some(32) {
                lo
            } else {
                let hi = self.s.m.extract(c, ballot, 1);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.cmp(c, pred::IUGE, from, k32);
                self.s.m.select(c, upper, hi, lo)
            };
            let [k31, k1, k0] = [31, 1, 0].map(|n| self.s.m.i32(n));
            let bit = self.s.m.bin(c, Bin::And, from, k31);
            let shifted = self.s.m.bin(c, Bin::LShr, word, bit);
            let on = self.s.m.bin(c, Bin::And, shifted, k1);
            let src_active = self.s.m.cmp(c, pred::NE, on, k0);
            if bound_ctrl {
                r = self.s.m.select(c, src_active, r, k0);
            } else {
                write_guard = Some(match write_guard {
                    Some(g) => self.s.m.bin(c, Bin::And, g, src_active),
                    None => src_active,
                });
            }
        }
        self.write_i(inst, r, write_guard)
    }
}
