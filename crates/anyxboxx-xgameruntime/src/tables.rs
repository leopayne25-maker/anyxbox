//! The two tables the C++ runtime reads, parsed from the same files, so both sides cannot drift:
//! - `served_interfaces.inc`: `(object, provider, interface)` pairs that AnyXboxX serves itself;
//! - `gdk_api_table.inc`: every GDK function with its provider, interface and vtable slot
//!   (generated from the SDK's `xgameruntime.lib` by `tools/gdk_interfaces.py`).

use crate::guid::Guid;

const SERVED_TEXT: &str = include_str!("../../../runtime/xgameruntime/served_interfaces.inc");
const GDK_TEXT: &str = include_str!("../../../runtime/xgameruntime/gdk_api_table.inc");

/// One `ANYXBOXX_SERVED_INTERFACE(object, provider, interface)` line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServedEntry {
    pub object: String,
    pub provider: Guid,
    pub iid: Guid,
}

/// One `ANYXBOXX_GDK_API(function, provider, interface, slot)` line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdkApi {
    pub function: String,
    pub provider: Guid,
    pub iid: Guid,
    pub slot: usize,
}

/// The arguments of `MACRO(a, "b", "c", 3)` on one line, quotes removed.
fn arguments(line: &str, macro_name: &str) -> Option<Vec<String>> {
    let body = line.trim().strip_prefix(macro_name)?.strip_prefix('(')?;
    let body = body.strip_suffix(')')?;
    Some(
        body.split(',')
            .map(|part| part.trim().trim_matches('"').to_string())
            .collect(),
    )
}

/// Parses `served_interfaces.inc`; malformed lines are a build error of the table, so they panic.
pub fn parse_served(text: &str) -> Vec<ServedEntry> {
    text.lines()
        .filter_map(|line| arguments(line, "ANYXBOXX_SERVED_INTERFACE"))
        .map(|args| {
            let [object, provider, iid] = args.as_slice() else {
                panic!("served_interfaces.inc: {args:?}");
            };
            ServedEntry {
                object: object.clone(),
                provider: Guid::parse(provider).expect("served provider GUID"),
                iid: Guid::parse(iid).expect("served interface GUID"),
            }
        })
        .collect()
}

/// Parses `gdk_api_table.inc`.
pub fn parse_gdk(text: &str) -> Vec<GdkApi> {
    text.lines()
        .filter_map(|line| arguments(line, "ANYXBOXX_GDK_API"))
        .map(|args| {
            let [function, provider, iid, slot] = args.as_slice() else {
                panic!("gdk_api_table.inc: {args:?}");
            };
            GdkApi {
                function: function.clone(),
                provider: Guid::parse(provider).expect("GDK provider GUID"),
                iid: Guid::parse(iid).expect("GDK interface GUID"),
                slot: slot.parse().expect("GDK slot"),
            }
        })
        .collect()
}

/// The served table of the repository.
pub fn served() -> Vec<ServedEntry> {
    parse_served(SERVED_TEXT)
}

/// The GDK function table of the repository.
pub fn gdk() -> Vec<GdkApi> {
    parse_gdk(GDK_TEXT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_table_matches_the_cpp_list() {
        let served = served();
        assert_eq!(served.len(), 30);
        let task_queue = served
            .iter()
            .find(|e| e.object == "TaskQueueInterface")
            .unwrap();
        assert_eq!(task_queue.iid, task_queue.provider);
        assert_eq!(
            served
                .iter()
                .filter(|e| e.object == "StoreInterface")
                .count(),
            4
        );
    }

    #[test]
    fn gdk_table_gives_slots_in_vtable_order() {
        let gdk = gdk();
        let first = gdk
            .iter()
            .find(|a| a.function == "XAppBroadcastGetStatus")
            .unwrap();
        assert_eq!(first.slot, 5);
        assert_eq!(gdk.len(), 396);
    }

    #[test]
    fn lines_of_another_macro_are_ignored() {
        assert_eq!(
            arguments("  ANYXBOXX_GDK_API(F, \"a\", \"b\", 3)", "ANYXBOXX_GDK_API").unwrap(),
            ["F", "a", "b", "3"]
        );
        assert!(arguments("ANYXBOXX_SERVED_INTERFACE(F, \"a\")", "ANYXBOXX_GDK_API").is_none());
    }
}
