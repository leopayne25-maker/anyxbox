//! SOP2 32-bit arithmetic and bit operations, matching the Python HLSL emitter.
use crate::scalar::source;
use anyxboxx_rdna2_ref::scalar::fields;

pub fn statement(name: &str, word: u32, literal: Option<u32>) -> Result<String, String> {
    let f = fields(word);
    let a = source(f.ssrc0, literal).map_err(|e| format!("{name}: {e}"))?;
    let b = source(f.ssrc1, literal).map_err(|e| format!("{name}: {e}"))?;
    let d = if f.sdst == 125 {
        "s_null".into()
    } else {
        format!("s[{}]", f.sdst)
    };
    let line = match name {
        "s_cselect_b32" => format!("{d} = scc ? {a} : {b};"),
        "s_add_u32" => {
            format!("{{ uint x = {a}, y = {b}; uint r = x + y; scc = r < x; {d} = r; }}")
        }
        "s_sub_u32" => format!("{{ uint x = {a}, y = {b}; scc = x < y; {d} = x - y; }}"),
        "s_addc_u32" => format!(
            "{{ uint x = {a}, y = {b}, k = scc ? 1u : 0u; uint r = x + y + k; scc = (r < x) || (k != 0u && r == x); {d} = r; }}"
        ),
        "s_subb_u32" => format!(
            "{{ uint x = {a}, y = {b}, k = scc ? 1u : 0u; scc = (x < y) || (k != 0u && x == y); {d} = x - y - k; }}"
        ),
        "s_add_i32" => format!(
            "{{ uint x = {a}, y = {b}; uint r = x + y; scc = ((~(x ^ y) & (x ^ r)) >> 31) != 0; {d} = r; }}"
        ),
        "s_sub_i32" => format!(
            "{{ uint x = {a}, y = {b}; uint r = x - y; scc = (((x ^ y) & (x ^ r)) >> 31) != 0; {d} = r; }}"
        ),
        "s_mul_i32" => format!("{d} = {a} * {b};"),
        "s_pack_ll_b32_b16" => format!("{d} = ({b} << 16) | ({a} & 0xFFFFu);"),
        "s_pack_lh_b32_b16" => format!("{d} = ({b} & 0xFFFF0000u) | ({a} & 0xFFFFu);"),
        "s_pack_hh_b32_b16" => format!("{d} = ({b} & 0xFFFF0000u) | ({a} >> 16);"),
        "s_and_b32" => logic(&d, &format!("{a} & {b}")),
        "s_or_b32" => logic(&d, &format!("{a} | {b}")),
        "s_xor_b32" => logic(&d, &format!("{a} ^ {b}")),
        "s_andn2_b32" => logic(&d, &format!("{a} & ~{b}")),
        "s_orn2_b32" => logic(&d, &format!("{a} | ~{b}")),
        "s_nand_b32" => logic(&d, &format!("~({a} & {b})")),
        "s_nor_b32" => logic(&d, &format!("~({a} | {b})")),
        "s_xnor_b32" => logic(&d, &format!("~({a} ^ {b})")),
        "s_lshl_b32" => logic(&d, &format!("{a} << ({b} & 31u)")),
        "s_lshr_b32" => logic(&d, &format!("{a} >> ({b} & 31u)")),
        "s_ashr_i32" => logic(&d, &format!("asuint(asint({a}) >> ({b} & 31u))")),
        "s_bfm_b32" => format!("{d} = ((1u << ({a} & 31u)) - 1u) << ({b} & 31u);"),
        "s_min_u32" => minmax(&d, &a, &b, "min", "<", false),
        "s_max_u32" => minmax(&d, &a, &b, "max", ">", false),
        "s_min_i32" => minmax(&d, &a, &b, "min", "<", true),
        "s_max_i32" => minmax(&d, &a, &b, "max", ">", true),
        "s_bfe_u32" => {
            logic(&d, &format!("SBfeU32({a}, {b})")).replace("scc = r != 0;", "scc = r != 0u;")
        }
        "s_bfe_i32" => {
            logic(&d, &format!("SBfeI32({a}, {b})")).replace("scc = r != 0;", "scc = r != 0u;")
        }
        "s_absdiff_i32" => format!(
            "{{ uint x = {a} - {b}; uint r = asint(x) < 0 ? 0u - x : x; scc = r != 0u; {d} = r; }}"
        ),
        "s_mul_hi_u32" => format!("{d} = MulHiU32({a}, {b});"),
        "s_mul_hi_i32" => format!("{d} = MulHiI32({a}, {b});"),
        "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
            let n = name.as_bytes()[6] - b'0';
            format!(
                "{{ uint x = {a}, t = x << {n}u; uint r = t + {b}; scc = (x >> {}u) != 0u || r < t; {d} = r; }}",
                32 - n
            )
        }
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    Ok(line)
}

fn logic(d: &str, expr: &str) -> String {
    format!("{{ uint r = {expr}; scc = r != 0; {d} = r; }}")
}

fn minmax(d: &str, a: &str, b: &str, op: &str, cmp: &str, signed: bool) -> String {
    if signed {
        format!(
            "{{ int x = asint({a}), y = asint({b}); scc = x {cmp} y; {d} = asuint({op}(x, y)); }}"
        )
    } else {
        format!("{{ uint x = {a}, y = {b}; scc = x {cmp} y; {d} = {op}(x, y); }}")
    }
}
