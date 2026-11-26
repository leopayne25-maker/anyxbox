//! DS lane permutation operations.
const ACTIVE: &str = "((s[126] >> lane) & 1u) != 0";

pub fn statement(name: &str, word: u64) -> Result<String, String> {
    let addr = (word >> 32) & 0xff;
    let data = (word >> 40) & 0xff;
    let dest = (word >> 56) & 0xff;
    let offset = word & 0xffff;
    let target = format!("(v[{addr}] + {offset}u) >> 2");
    let value = match name {
        "ds_permute_b32" => format!("ForwardLane(lane, {ACTIVE}, v[{data}], {target})"),
        "ds_bpermute_b32" => format!("ReadLane(lane, (({ACTIVE}) ? v[{data}] : 0u), {target})"),
        "ds_swizzle_b32" => {
            let source = if offset & 0xff00 == 0x8000 {
                format!(
                    "((lane & ~3u) + ((0x{:X}u >> ((lane & 3u) * 2u)) & 3u))",
                    offset & 0xff
                )
            } else if offset & 0x8000 == 0 {
                let (and, or, xor) = (offset & 31, (offset >> 5) & 31, (offset >> 10) & 31);
                format!("(((lane & {and}u) | {or}u) ^ {xor}u)")
            } else {
                return Err(format!(
                    "untranslated instruction: {name}: unsupported offset 0x{offset:04X}"
                ));
            };
            format!("ReadLane(lane, (({ACTIVE}) ? v[{data}] : 0u), {source})")
        }
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    Ok(format!(
        "{{ uint value = {value}; if ({ACTIVE}) v[{dest}] = value; }}"
    ))
}
