//! Native VOP3 integer operations.
use crate::scalar::source;
use anyxboxx_rdna2_ref::vop3::Fields;

pub fn statement(name: &str, word: u64, literal: Option<u32>) -> Result<String, String> {
    let f = Fields::new(word);
    if matches!(
        name,
        "v_readlane_b32" | "v_readfirstlane_b32" | "v_writelane_b32"
    ) {
        if (name == "v_writelane_b32" && f.sources[0] >= 256)
            || (name != "v_readfirstlane_b32" && f.sources[1] >= 256)
        {
            return Err(format!(
                "untranslated instruction: {name}: scalar operand required"
            ));
        }
        let operand = |code| {
            if code >= 256 {
                Ok(format!("v[{}]", code - 256))
            } else {
                source(code, literal).map_err(|e| format!("{name}: {e}"))
            }
        };
        let a = operand(f.sources[0])?;
        let b = operand(f.sources[1])?;
        let _ = operand(f.sources[2])?;
        let target = if f.vdst == 125 {
            "s_null".into()
        } else {
            format!("s[{}]", f.vdst)
        };
        return Ok(match name {
            "v_readlane_b32" => format!("{target} = ReadLane(lane, {a}, {b});"),
            "v_readfirstlane_b32" => {
                format!("{target} = ReadLane(lane, {a}, firstbitlow(s[126] == 0u ? 1u : s[126]));")
            }
            _ => format!("if (lane == ({b} & 31u)) v[{}] = {a};", f.vdst),
        });
    }
    let is_float = matches!(
        name,
        "v_fma_f32" | "v_mad_f32" | "v_fma_legacy_f32" | "v_min3_f32" | "v_max3_f32" | "v_med3_f32"
    );
    if !is_float
        && (!matches!(
            name,
            "v_add3_u32"
                | "v_lshl_add_u32"
                | "v_add_lshl_u32"
                | "v_lshl_or_b32"
                | "v_and_or_b32"
                | "v_or3_b32"
                | "v_xor3_b32"
                | "v_xad_u32"
                | "v_bfe_u32"
                | "v_bfe_i32"
                | "v_bfi_b32"
                | "v_bfm_b32"
                | "v_alignbit_b32"
                | "v_alignbyte_b32"
                | "v_perm_b32"
                | "v_min3_u32"
                | "v_max3_u32"
                | "v_med3_u32"
                | "v_min3_i32"
                | "v_max3_i32"
                | "v_med3_i32"
                | "v_mul_lo_u32"
                | "v_mul_hi_u32"
                | "v_mul_hi_i32"
                | "v_mad_u32_u24"
                | "v_mad_i32_i24"
                | "v_bcnt_u32_b32"
        ) || f.clamp)
    {
        return Err(format!("untranslated instruction: {name}"));
    }
    let operand = |code| {
        if code >= 256 {
            Ok(format!("v[{}]", code - 256))
        } else {
            source(code, literal).map_err(|e| format!("{name}: {e}"))
        }
    };
    let [a, b, c] = f.sources.map(operand);
    let modify = |mut value: String, index: u32| {
        if is_float && f.abs >> index & 1u32 != 0u32 {
            value = format!("({value} & 0x7FFFFFFFu)");
        }
        if is_float && f.neg >> index & 1u32 != 0u32 {
            value = format!("({value} ^ 0x80000000u)");
        }
        value
    };
    let (a, b, c) = (modify(a?, 0), modify(b?, 1), modify(c?, 2));
    let mut expression = match name {
        "v_min3_f32" => format!("asuint(min(min(asfloat({a}), asfloat({b})), asfloat({c})))"),
        "v_max3_f32" => format!("asuint(max(max(asfloat({a}), asfloat({b})), asfloat({c})))"),
        "v_med3_f32" => format!(
            "asuint(max(min(asfloat({a}), asfloat({b})), min(max(asfloat({a}), asfloat({b})), asfloat({c}))))"
        ),
        "v_fma_f32" => format!("FmaF32({a}, {b}, {c})"),
        "v_fma_legacy_f32" => format!("FmaLegacyF32({a}, {b}, {c})"),
        "v_mad_f32" => format!("asuint(mad(asfloat({a}), asfloat({b}), asfloat({c})))"),
        "v_add3_u32" => format!("({a} + {b} + {c})"),
        "v_lshl_add_u32" => format!("(({a} << ({b} & 31u)) + {c})"),
        "v_add_lshl_u32" => format!("(({a} + {b}) << ({c} & 31u))"),
        "v_lshl_or_b32" => format!("(({a} << ({b} & 31u)) | {c})"),
        "v_and_or_b32" => format!("(({a} & {b}) | {c})"),
        "v_or3_b32" => format!("({a} | {b} | {c})"),
        "v_xor3_b32" => format!("({a} ^ {b} ^ {c})"),
        "v_xad_u32" => format!("(({a} ^ {b}) + {c})"),
        "v_bfe_u32" => format!("(({a} >> ({b} & 31u)) & ((1u << ({c} & 31u)) - 1u))"),
        "v_bfe_i32" => format!("BfeI32({a}, {b}, {c})"),
        "v_bfi_b32" => format!("(({a} & {b}) | (~{a} & {c}))"),
        "v_bfm_b32" => format!("(((1u << ({a} & 31u)) - 1u) << ({b} & 31u))"),
        "v_alignbit_b32" => format!("AlignBit({a}, {b}, {c})"),
        "v_alignbyte_b32" => format!("AlignBit({a}, {b}, ({c} & 3u) * 8u)"),
        "v_perm_b32" => format!("PermB32({a}, {b}, {c})"),
        "v_min3_u32" => format!("min(min({a}, {b}), {c})"),
        "v_max3_u32" => format!("max(max({a}, {b}), {c})"),
        "v_med3_u32" => format!("max(min({a}, {b}), min(max({a}, {b}), {c}))"),
        "v_min3_i32" => format!("asuint(min(min(asint({a}), asint({b})), asint({c})))"),
        "v_max3_i32" => format!("asuint(max(max(asint({a}), asint({b})), asint({c})))"),
        "v_med3_i32" => format!(
            "asuint(max(min(asint({a}), asint({b})), min(max(asint({a}), asint({b})), asint({c}))))"
        ),
        "v_mul_lo_u32" => format!("({a} * {b})"),
        "v_mul_hi_u32" => format!("MulHiU32({a}, {b})"),
        "v_mul_hi_i32" => format!("MulHiI32({a}, {b})"),
        "v_mad_u32_u24" => format!("(({a} & 0xFFFFFFu) * ({b} & 0xFFFFFFu) + {c})"),
        "v_mad_i32_i24" => {
            format!("asuint(((asint({a}) << 8) >> 8) * ((asint({b}) << 8) >> 8) + asint({c}))")
        }
        "v_bcnt_u32_b32" => format!("(countbits({a}) + {b})"),
        _ => unreachable!(),
    };
    if is_float && (f.clamp || f.omod != 0) {
        let scale = match f.omod {
            1 => "2.0f",
            2 => "4.0f",
            3 => "0.5f",
            _ => "1.0f",
        };
        let scaled = format!("asfloat({expression}) * {scale}");
        expression = if f.clamp {
            format!("asuint(saturate({scaled}))")
        } else {
            format!("asuint({scaled})")
        };
    }
    Ok(format!(
        "if (((s[126] >> lane) & 1u) != 0) v[{}] = {expression};",
        f.vdst
    ))
}
