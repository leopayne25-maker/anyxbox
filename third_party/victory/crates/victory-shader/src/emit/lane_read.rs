//! First-lane and indexed lane reads used by the DXIL emitter.

#[cfg(test)]
use super::LANE_SHUFFLES;
use super::{Attr, Bin, Lower, V, flags, pred};

impl Lower<'_> {
    /// `x` in the first lane of this host wave where `guard` holds, and whether it has such a
    /// lane: the console reads the first lane of EXEC, and a lane that got here with EXEC clear
    /// (a guard, not a branch) is not one. With none, `x` of the wave's first lane here.
    ///
    /// A helper lane of a pixel shader takes its own value: the console's whole-quad mode has
    /// helper lanes in EXEC, so a waterfall loop serves them too, and the host's wave operations
    /// leave them out. Once only helper lanes are left in such a loop none of them ever matched:
    /// they ran to the iteration limit, and left with no sample for their quad's derivatives,
    /// along every triangle's edges (upstream #908, the fix of #1733 for this tree's form).
    pub(super) fn first_lane(&mut self, x: V, guard: Option<V>) -> (V, V) {
        let (first, any) = self.first_live_lane(x, guard);
        if !self.s.pixel || !super::super::helper_own() {
            return (first, any);
        }
        let helper = self.helper_lane();
        let c = self.cur;
        let yes = self.s.m.bool(true);
        (
            self.s.m.select(c, helper, x, first),
            self.s.m.select(c, helper, yes, any),
        )
    }

    fn first_live_lane(&mut self, x: V, guard: Option<V>) -> (V, V) {
        let c = self.cur;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let first = self
            .s
            .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
        let Some(g) = guard else {
            return (first, self.s.m.bool(true));
        };
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[g]);
        let (lo, hi) = (self.s.m.extract(c, r, 0), self.s.m.extract(c, r, 1));
        let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        // FirstbitLo (dx.op 32): the lowest set bit
        let flo = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let fhi = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
        let in_lo = self.s.m.cmp(c, pred::NE, lo, z);
        let in_hi = self.s.m.cmp(c, pred::NE, hi, z);
        let any = self.s.m.bin(c, Bin::Or, in_lo, in_hi);
        let lane = self.s.m.select(c, in_lo, flo, fhi);
        // no lane in EXEC: read this lane (one that is here); the read is not used
        let me = self
            .s
            .op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let lane = self.s.m.select(c, any, lane, me);
        let at = self.read_active_lane(x, lane);
        (self.s.m.select(c, any, at, first), any)
    }

    pub(super) fn read_lane(&mut self, x: V, lane: V) -> V {
        #[cfg(test)]
        LANE_SHUFFLES.set(LANE_SHUFFLES.get() + 1);
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }
}
