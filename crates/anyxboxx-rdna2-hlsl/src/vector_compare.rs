//! VOPC integer comparisons for wave32.
use crate::scalar::source;

pub fn statement(name: &str, word: u32, literal: Option<u32>) -> Result<String, String> {
    let code = word & 0x1ff;
    if matches!(code, 0xe9 | 0xea | 0xf9 | 0xfa) {
        return Err(format!("untranslated instruction: {name} with SDWA or DPP"));
    }
    let a = if code >= 256 {
        format!("v[{}]", code - 256)
    } else {
        source(code, literal).map_err(|e| format!("{name}: {e}"))?
    };
    let b = format!("v[{}]", (word >> 9) & 0xff);
    let (prefix, suffix) = name
        .rsplit_once('_')
        .ok_or_else(|| format!("untranslated instruction: {name}"))?;
    let (op, target) = if let Some(op) = prefix.strip_prefix("v_cmpx_") {
        (op, "s[126]")
    } else if let Some(op) = prefix.strip_prefix("v_cmp_") {
        (op, "s[106]")
    } else {
        return Err(format!("untranslated instruction: {name}"));
    };
    if suffix == "f16" {
        let ah = half_operand(code, &a);
        let bh = format!("({b} & 0xFFFFu)");
        let condition = if op == "class" {
            format!("((({bh}) >> ClassF16({ah})) & 1u) != 0u")
        } else {
            float_condition(op, &format!("F16ToF32({ah})"), &format!("F16ToF32({bh})"))
                .ok_or_else(|| format!("untranslated instruction: {name}"))?
        };
        return Ok(format!(
            "{{ bool t = (((s[126] >> lane) & 1u) != 0) && ({condition}); {target} = Ballot(lane, t); }}"
        ));
    }
    if suffix == "f32" {
        let condition = float_condition(op, &a, &b)
            .ok_or_else(|| format!("untranslated instruction: {name}"))?;
        return Ok(format!(
            "{{ bool t = (((s[126] >> lane) & 1u) != 0) && ({condition}); {target} = Ballot(lane, t); }}"
        ));
    }
    if !matches!(suffix, "u32" | "i32" | "u16" | "i16") {
        return Err(format!("untranslated instruction: {name}"));
    }
    let condition = match op {
        "f" => "false".into(),
        "t" | "tru" => "true".into(),
        _ => {
            let symbol = match op {
                "lt" => "<",
                "eq" => "==",
                "le" => "<=",
                "gt" => ">",
                "lg" | "ne" | "neq" => "!=",
                "ge" => ">=",
                _ => return Err(format!("untranslated instruction: {name}")),
            };
            match suffix {
                "u16" => format!("(({a}) & 0xFFFFu) {symbol} (({b}) & 0xFFFFu)"),
                "i16" => {
                    format!("(asint(({a}) << 16u) >> 16) {symbol} (asint(({b}) << 16u) >> 16)")
                }
                "i32" => format!("asint({a}) {symbol} asint({b})"),
                _ => format!("({a}) {symbol} ({b})"),
            }
        }
    };
    Ok(format!(
        "{{ bool t = (((s[126] >> lane) & 1u) != 0) && ({condition}); {target} = Ballot(lane, t); }}"
    ))
}

fn float_condition(op: &str, a: &str, b: &str) -> Option<String> {
    let nan = format!("IsNanF32({a}) || IsNanF32({b})");
    let ordered = format!("!IsNanF32({a}) && !IsNanF32({b})");
    let x = format!("asfloat({a})");
    let y = format!("asfloat({b})");
    Some(match op {
        "f" => "false".into(),
        "t" | "tru" => "true".into(),
        "o" => format!("({ordered})"),
        "u" => format!("({nan})"),
        "nge" => format!("({nan} || !({x} >= {y}))"),
        "nlg" => format!("({nan} || !({x} < {y} || {x} > {y}))"),
        "ngt" => format!("({nan} || !({x} > {y}))"),
        "nle" => format!("({nan} || !({x} <= {y}))"),
        "nlt" => format!("({nan} || !({x} < {y}))"),
        "class" => format!("((({b}) >> ClassF32({a})) & 1u) != 0u"),
        "lg" => format!("({ordered} && ({x} < {y} || {x} > {y}))"),
        "lt" | "le" | "gt" | "ge" => {
            let symbol = match op {
                "lt" => "<",
                "le" => "<=",
                "gt" => ">",
                _ => ">=",
            };
            format!("({ordered} && {x} {symbol} {y})")
        }
        "eq" => format!("{x} == {y}"),
        "ne" | "neq" => format!("{x} != {y}"),
        _ => return None,
    })
}

pub(crate) fn half_operand(code: u32, expression: &str) -> String {
    const INLINE: [u32; 9] = [
        0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4400, 0xc400, 0x3118,
    ];
    if (240..=248).contains(&code) {
        format!("0x{:X}u", INLINE[(code - 240) as usize])
    } else {
        format!("({expression} & 0xFFFFu)")
    }
}
