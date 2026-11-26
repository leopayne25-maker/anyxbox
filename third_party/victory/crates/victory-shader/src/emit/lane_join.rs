//! Lane masks a join takes the same bits of on every way in ([`Lower::joined_dword`]).
//!
//! The console compiler moves a lane mask out of its register pair where it needs those
//! registers for something else (another pair, `s_mov_b64`; a lane of a VGPR, which the IR
//! gives as `s_mov_b32` to and from a parked scalar, `ir::unspill`) and moves it back before
//! the code that reads it. A join after such code then brings the mask on one way and the
//! moved-back copy on another (the title's water volumetrics: a mask kept across a loop of
//! theirs, read as a number where it is parked). The copy is written where not every lane of
//! the mask runs, so the ballot there is not its number ([`super::MASK_ELSEWHERE`]).
//!
//! On the console a scalar move copies the register's bits whatever EXEC holds, and a scalar
//! register at a block entry holds what the last block to run before it left there: one of the
//! values its join names (in the IR's SSA, built from the machine code's registers), whichever
//! way the wave came and whichever arm of a lane branch ran last. Where every value a join may
//! take is that same mask, moved or joined again (a web of moves and joins that takes in one
//! value from outside it, as in Braun et al.'s removal of redundant joins), the register holds
//! that value's bits on every way in, so its number is that value's number: the ballot where
//! that value is written (or its number, if it is one), under the rules of every other mask
//! read as data. Nothing is chosen per lane: whichever way a lane takes, it brings that number.
//! A web that takes in two different values is not this case and stays refused: there the
//! console's register depends on which arm ran last, which a per-lane join does not know.
//!
//! A ballot is the console's mask only where the host's lanes are the console's and its wave
//! the console's (or the two host waves of one console wave exchange their halves, or the
//! paired form asks both lanes of each thread): the lowering declares the console's wave size
//! for code that reads masks as numbers (`console_lanes`), but does not see a mask read through
//! a join of numbers. Without that the copy stays refused.

use std::collections::HashSet;

use super::{Fields, Lower, NULL, Operand, Reg, V, Val, Value, Why};

/// Webs of moves and joins larger than this are not followed (the title's have a few joins).
const MAX_WEB: usize = 1 << 12;

impl Lower<'_> {
    /// The number a mask brings into the data join `join` (its dword `arg`), as
    /// [`Self::mask_dword`] gives it; where that has no ballot for it
    /// ([`Self::elsewhere`]) and every way into `join` brings the bits of one other value
    /// ([`Self::same_bits`]), that value's number, where a ballot is of the console's wave
    /// ([`Self::console_ballots`]).
    pub(super) fn joined_dword(&mut self, join: Value, arg: Value) -> Result<V, Why> {
        let lo = self.x.st.highs.get(&arg).copied().unwrap_or(arg);
        if self.x.st.masks.contains_key(&lo)
            && self.elsewhere(lo)
            && self.console_ballots()
            && let Some(x) = self.same_bits(join)
            && x != arg
        {
            return self.number_of(x);
        }
        self.mask_dword(arg)
    }

    /// Whether a ballot gives a mask of the console's whole wave: the host's lanes are the
    /// console's (`console_lanes`: the console's wave size is declared), and a host wave is the
    /// console's wave, or the two halves exchange theirs, or the paired form asks both lanes.
    fn console_ballots(&self) -> bool {
        self.console_lanes
            && (self.halves.is_none() || self.exchange.is_some() || self.twin.is_some())
    }

    /// `x` as a number: its value, or for a mask, its ballot ([`Self::mask_dword`]).
    fn number_of(&mut self, x: Value) -> Result<V, Why> {
        let st = self.x.st;
        let mask = st.masks.contains_key(&x) || st.highs.contains_key(&x);
        match self.vals.get(&x).copied() {
            Some(Val::I32(n)) => Ok(n),
            Some(Val::Missing(_)) | None if mask => self.mask_dword(x),
            Some(Val::Missing(why)) => Err(why),
            _ => self.i32_of(x),
        }
    }

    /// The dword a scalar move writes `v` from: `s_mov_b32`, or the same dword of the pair
    /// `s_mov_b64` copies (both copy the register whatever EXEC holds).
    fn moved_from(&self, v: Value) -> Option<Value> {
        let &b = self.def_block.get(&v)?;
        let inst = self.x.p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))?;
        let dwords = match inst.name() {
            "s_mov_b32" => 1,
            "s_mov_b64" => 2,
            _ => return None,
        };
        let Fields::Sop {
            dst: Some(Operand::Scalar { code: dst, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        let Some(&Operand::Scalar { code: src, .. }) = src.first() else {
            return None;
        };
        let Reg::S(to) = inst.writes.iter().find(|w| w.1 == v)?.0 else {
            return None;
        };
        let h = to.checked_sub(*dst).filter(|&h| h < dwords)?;
        if src == NULL {
            return None;
        }
        inst.read(Reg::S(src.checked_add(h)?))
    }

    /// The value whose bits `v` holds wherever it is read, where `v` is a join: followed back
    /// through scalar moves, the joins and moves its values come from make a web, and the
    /// one value from outside the web that comes into it is the answer. None if two different
    /// values come in (or none, or the web is too large).
    ///
    /// Exact by induction along any run: a move holds its source's bits, a join one of its
    /// values', so each member holds what came in last. The value from outside is written
    /// before every member on each way to it (each member's value is written earlier on that
    /// way, back to the value from outside), and is written again before a member is read only
    /// where that member is written again too: a member that held an earlier write of it past a
    /// later one would be read where its own write does not dominate.
    pub(super) fn same_bits(&self, v: Value) -> Option<Value> {
        let back = |mut v: Value| {
            for _ in 0..MAX_WEB {
                match self.moved_from(v) {
                    Some(s) => v = s,
                    None => break,
                }
            }
            v
        };
        let start = back(v);
        self.phi_at.get(&start)?;
        let (mut web, mut work, mut from) = (HashSet::from([start]), vec![start], None);
        while let Some(w) = work.pop() {
            let (b, i) = self.phi_at[&w];
            for &a in &self.x.p.blocks[b].phis[i].args {
                let a = back(a);
                if web.contains(&a) {
                    continue;
                }
                if self.phi_at.contains_key(&a) {
                    if web.len() >= MAX_WEB {
                        return None;
                    }
                    web.insert(a);
                    work.push(a);
                } else if *from.get_or_insert(a) != a {
                    return None;
                }
            }
        }
        from
    }
}

#[cfg(test)]
#[path = "lane_join_tests.rs"]
mod tests;
