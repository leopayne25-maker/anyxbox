//! JSON rendering of an [`Image`], the format `tests/test_rust_pe_parity.py` compares with the
//! Python loader. Hand-written to keep the crate free of extra dependencies.

use std::fmt::Write;

use crate::{Image, Tls};

/// One JSON object; `pages` is added when page protections were asked for.
pub fn render(image: &Image, pages: Option<&[u8]>) -> String {
    let mut fields = vec![
        ("base", image.base.to_string()),
        ("size", image.size.to_string()),
        ("entry", image.entry.to_string()),
        ("relocated", image.relocated.to_string()),
        ("is_dll", image.is_dll.to_string()),
        (
            "imports",
            array(image.imports.iter().map(|import| {
                object(&[
                    ("dll", string(&import.dll)),
                    ("name", string(&import.name)),
                    ("slot", import.slot.to_string()),
                ])
            })),
        ),
        (
            "exports",
            map(image
                .exports
                .iter()
                .map(|(k, v)| (k.clone(), v.to_string()))),
        ),
        (
            "ordinals",
            map(image
                .ordinals
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))),
        ),
        (
            "forwarders",
            map(image.forwarders.iter().map(|(k, v)| (k.clone(), string(v)))),
        ),
        (
            "tls",
            image.tls.as_ref().map_or_else(|| "null".to_string(), tls),
        ),
    ];
    if let Some(pages) = pages {
        fields.push(("pages", array(pages.iter().map(u8::to_string))));
    }
    object(&fields)
}

fn tls(tls: &Tls) -> String {
    object(&[
        ("template_start", tls.template_start.to_string()),
        ("template_end", tls.template_end.to_string()),
        ("zero_fill", tls.zero_fill.to_string()),
        ("index_address", tls.index_address.to_string()),
        ("callbacks", array(tls.callbacks.iter().map(u64::to_string))),
    ])
}

fn object(fields: &[(&str, String)]) -> String {
    map(fields.iter().map(|(k, v)| ((*k).to_string(), v.clone())))
}

fn map(entries: impl Iterator<Item = (String, String)>) -> String {
    let body: Vec<String> = entries
        .map(|(k, v)| format!("{}:{v}", string(&k)))
        .collect();
    format!("{{{}}}", body.join(","))
}

fn array(items: impl Iterator<Item = String>) -> String {
    format!("[{}]", items.collect::<Vec<_>>().join(","))
}

/// A JSON string literal (RFC 8259 escaping).
pub fn string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                write!(out, "\\u{:04x}", u32::from(c)).expect("writing to a String");
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_escaped() {
        assert_eq!(string("a\"b\\c\n\u{1}é"), "\"a\\\"b\\\\c\\n\\u0001é\"");
    }
}
