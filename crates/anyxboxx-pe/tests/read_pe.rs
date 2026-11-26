//! `read_pe` on images built from source (tests/common), the cases of tests/test_modules.py,
//! test_native_tls*.py and test_loader-style checks of the Python loader.

mod common;

use anyxboxx_pe::{
    DEFAULT_BASE, LoadError, PROT_EXEC, PROT_READ, PROT_WRITE, THUNKS, default_allocate,
    page_protections, read_pe,
};
use common::{DLL_BASE, EXE_BASE, Pe, Symbol, Target, rva};

const MOVED: u64 = 0x2_0000_0000;
const TLS_AT: usize = 0x100;
const CALLBACKS_AT: usize = 0x140;

fn load(file: &[u8]) -> Result<anyxboxx_pe::Image, LoadError> {
    read_pe(file, default_allocate)
}

fn forced(base: u64) -> impl FnOnce(Option<u64>, u64, bool) -> Result<u64, LoadError> {
    move |_, _, _| Ok(base)
}

#[test]
fn imports_name_every_iat_slot_by_name_or_ordinal() {
    let pe = Pe {
        imports: vec![
            ("kernel32.dll", Symbol::Name("ExitProcess")),
            ("xmem.dll", Symbol::Ordinal(12)),
            ("kernel32.dll", Symbol::Name("Sleep")),
        ],
        ..Pe::default()
    };
    let (file, iat) = pe.build();
    let image = load(&file).unwrap();
    let found: Vec<_> = image
        .imports
        .iter()
        .map(|i| (i.dll.as_str(), i.name.as_str(), i.slot))
        .collect();
    let slot =
        |dll: &str, name: &str| DLL_BASE + u64::from(iat[&(dll.to_string(), name.to_string())]);
    assert_eq!(
        found,
        [
            (
                "kernel32.dll",
                "ExitProcess",
                slot("kernel32.dll", "ExitProcess")
            ),
            ("kernel32.dll", "Sleep", slot("kernel32.dll", "Sleep")),
            ("xmem.dll", "#12", slot("xmem.dll", "#12")),
        ]
    );
    assert!(image.is_dll && !image.relocated);
    assert_eq!(image.entry, DLL_BASE + u64::from(rva(0)));
}

#[test]
fn exports_by_name_ordinal_and_forwarder() {
    let pe = Pe {
        exports: vec![
            ("Double", Target::Code(0x10)),
            ("Twice", Target::Forward("libB.Double")),
            ("Add1", Target::Code(0x20)),
        ],
        ..Pe::default()
    };
    let image = load(&pe.build().0).unwrap();
    let at = |offset| DLL_BASE + u64::from(rva(offset));
    assert_eq!(image.exports.get("Double"), Some(&at(0x10)));
    assert_eq!(image.exports.get("Add1"), Some(&at(0x20)));
    assert_eq!(image.ordinals.get(&1), Some(&at(0x10)));
    assert_eq!(image.ordinals.get(&3), Some(&at(0x20)));
    assert_eq!(
        image.forwarders.get("Twice").map(String::as_str),
        Some("libB.Double")
    );
    assert_eq!(
        image.forwarders.get("#2").map(String::as_str),
        Some("libB.Double")
    );
    assert!(!image.exports.contains_key("Twice"));
}

#[test]
fn a_forced_base_relocates_pointers_entry_and_exports() {
    let pe = Pe {
        pointers: vec![(0x80, 0x10)],
        exports: vec![("Double", Target::Code(0x10))],
        ..Pe::default()
    };
    let image = read_pe(&pe.build().0, forced(MOVED)).unwrap();
    assert!(image.relocated);
    assert_eq!(image.base, MOVED);
    assert_eq!(image.entry, MOVED + u64::from(rva(0)));
    assert_eq!(image.exports["Double"], MOVED + u64::from(rva(0x10)));
}

#[test]
fn an_unusable_preferred_base_falls_back_only_when_relocatable() {
    let relocatable = Pe {
        base: Some(THUNKS),
        pointers: vec![(0x80, 0)],
        ..Pe::default()
    };
    let image = load(&relocatable.build().0).unwrap();
    assert_eq!(image.base, DEFAULT_BASE);
    assert!(image.relocated);
    let fixed = Pe {
        base: Some(THUNKS),
        ..Pe::default()
    };
    assert!(matches!(
        load(&fixed.build().0),
        Err(LoadError::UnusableBase)
    ));
}

fn tls_image() -> Vec<u8> {
    let mut code = vec![0u8; CALLBACKS_AT + 24];
    // IMAGE_TLS_DIRECTORY64: template [0x180, 0x190), index at 0x1A0, callbacks at 0x140.
    code[TLS_AT + 32..TLS_AT + 36].copy_from_slice(&0x30u32.to_le_bytes());
    let pe = Pe {
        dll: false,
        code,
        tls: Some(TLS_AT),
        pointers: vec![
            (TLS_AT, 0x180),
            (TLS_AT + 8, 0x190),
            (TLS_AT + 16, 0x1A0),
            (TLS_AT + 24, CALLBACKS_AT),
            (CALLBACKS_AT, 0x40),
            (CALLBACKS_AT + 8, 0x60),
        ],
        ..Pe::default()
    };
    pe.build().0
}

#[test]
fn tls_directory_and_callbacks_follow_the_chosen_base() {
    let file = tls_image();
    let at = |base: u64, offset| base + u64::from(rva(offset));
    for base in [EXE_BASE, MOVED] {
        let tls = read_pe(&file, forced(base)).unwrap().tls.unwrap();
        assert_eq!(
            (tls.template_start, tls.template_end),
            (at(base, 0x180), at(base, 0x190))
        );
        assert_eq!(tls.zero_fill, 0x30);
        assert_eq!(tls.index_address, at(base, 0x1A0));
        assert_eq!(tls.callbacks, [at(base, 0x40), at(base, 0x60)]);
    }
}

#[test]
fn foreign_or_unsupported_images_are_refused() {
    let (file, _) = Pe {
        machine: 0x14C,
        ..Pe::default()
    }
    .build();
    assert!(matches!(load(&file), Err(LoadError::NotPe32Plus)));
    let (file, _) = Pe {
        dll: false,
        entry: None,
        ..Pe::default()
    }
    .build();
    assert!(matches!(load(&file), Err(LoadError::BadHeaders)));
    let (mut file, _) = Pe::default().build();
    Pe::set_directory(&mut file, 13, 0x1000, 0x40);
    assert!(matches!(load(&file), Err(LoadError::Unsupported(_))));
    let (mut file, _) = Pe::default().build();
    Pe::set_directory(&mut file, 14, 0x1000, 0x48);
    assert!(matches!(load(&file), Err(LoadError::Unsupported(_))));
    assert!(load(b"MZ not a PE").is_err());
    assert!(load(&[]).is_err());
}

#[test]
fn a_dll_without_entry_point_is_accepted() {
    let image = load(
        &Pe {
            entry: None,
            ..Pe::default()
        }
        .build()
        .0,
    )
    .unwrap();
    assert_eq!(image.entry, 0);
}

#[test]
fn pages_take_the_section_rights() {
    let (file, _) = Pe::default().build();
    let pages = page_protections(&file).unwrap();
    assert_eq!(pages[0], PROT_READ);
    assert!(
        pages[1..]
            .iter()
            .all(|&p| p == PROT_READ | PROT_WRITE | PROT_EXEC)
    );
}

#[test]
fn a_malformed_relocation_block_is_refused_not_skipped() {
    const DIRECTORIES: usize = 0x80 + 4 + 20 + 112;
    const DIR_RELOC: usize = 5;
    const HEADERS_SIZE: usize = 0x400;
    let pe = Pe {
        pointers: vec![(0x80, 0x10)],
        ..Pe::default()
    };
    let (mut file, _) = pe.build();
    let at = DIRECTORIES + 8 * DIR_RELOC;
    let table_rva = u32::from_le_bytes(file[at..at + 4].try_into().unwrap()) as usize;
    let block = table_rva - common::SECTION_RVA as usize + HEADERS_SIZE;
    // SizeOfBlock smaller than the block header: pefile stopped there and left the pointer as is.
    file[block + 4..block + 8].copy_from_slice(&6u32.to_le_bytes());
    assert!(matches!(
        read_pe(&file, forced(MOVED)),
        Err(LoadError::Relocation(_))
    ));
}
