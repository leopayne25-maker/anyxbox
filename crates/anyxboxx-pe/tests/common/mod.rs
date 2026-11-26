//! Small PE32+ x86-64 images built from source for the tests, as `tests/pe_builder.py` does:
//! one RWX section at RVA 0x1000 (code, then import, export and relocation tables). No game,
//! console or SDK file is ever used.

use std::collections::HashMap;

pub const SECTION_RVA: u32 = 0x1000;
pub const CODE_SIZE: usize = 0x400;
pub const DLL_BASE: u64 = 0x1_8000_0000;
pub const EXE_BASE: u64 = 0x1_4000_0000;
const FILE_ALIGN: usize = 0x200;
const HEADERS_SIZE: usize = 0x400;
const PE_OFFSET: usize = 0x80;
const ORDINAL_FLAG: u64 = 1 << 63;
const REL_BASED_DIR64: u16 = 10;
const SECTION_RWX: u32 = 0xE000_0020;

pub fn rva(offset: usize) -> u32 {
    SECTION_RVA + offset as u32
}

#[derive(Clone)]
pub enum Symbol {
    Name(&'static str),
    Ordinal(u16),
}

#[derive(Clone)]
pub enum Target {
    Code(usize),
    Forward(&'static str),
}

/// Description of the image to build; `Default` is a DLL with no tables.
#[derive(Clone)]
pub struct Pe {
    pub code: Vec<u8>,
    pub imports: Vec<(&'static str, Symbol)>,
    pub exports: Vec<(&'static str, Target)>,
    /// `(offset of a qword in the code, offset it points to)`, fixed up by relocation.
    pub pointers: Vec<(usize, usize)>,
    pub dll: bool,
    pub base: Option<u64>,
    pub entry: Option<usize>,
    pub tls: Option<usize>,
    pub machine: u16,
}

impl Default for Pe {
    fn default() -> Self {
        Pe {
            code: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            pointers: Vec::new(),
            dll: true,
            base: None,
            entry: Some(0),
            tls: None,
            machine: 0x8664,
        }
    }
}

struct Body(Vec<u8>);

impl Body {
    fn put(&mut self, data: &[u8], alignment: usize) -> usize {
        let start = self.0.len().div_ceil(alignment) * alignment;
        self.0.resize(start, 0);
        self.0.extend_from_slice(data);
        start
    }
    fn u16(&mut self, at: usize, v: u16) {
        self.0[at..at + 2].copy_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, at: usize, v: u32) {
        self.0[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, at: usize, v: u64) {
        self.0[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
}

/// IAT slot RVA of each import, keyed by `(dll, name or #ordinal)`.
pub type Iat = HashMap<(String, String), u32>;

impl Pe {
    pub fn base(&self) -> u64 {
        self.base
            .unwrap_or(if self.dll { DLL_BASE } else { EXE_BASE })
    }

    pub fn build(&self) -> (Vec<u8>, Iat) {
        let mut body = Body(vec![0; CODE_SIZE]);
        let (import_dir, iat) = imports(&mut body, &self.imports);
        let export_dir = exports(&mut body, &self.exports);
        let reloc_dir = relocations(&mut body, &self.pointers);
        assert!(self.code.len() <= CODE_SIZE, "test code too long");
        body.0[..self.code.len()].copy_from_slice(&self.code);
        for &(offset, target) in &self.pointers {
            body.u64(offset, self.base() + u64::from(rva(target)));
        }
        let raw_size = body.0.len().div_ceil(FILE_ALIGN) * FILE_ALIGN;
        body.0.resize(raw_size, 0);
        let mut directories = [(0u32, 0u32); 16];
        directories[0] = export_dir;
        directories[1] = import_dir;
        directories[5] = reloc_dir;
        if let Some(tls) = self.tls {
            directories[9] = (rva(tls), 40);
        }
        let mut file = self.headers(raw_size, &directories);
        file.extend_from_slice(&body.0);
        (file, iat)
    }

    /// Sets data directory `index` of a built image (to declare unsupported features).
    pub fn set_directory(file: &mut [u8], index: usize, rva: u32, size: u32) {
        let at = PE_OFFSET + 4 + 20 + 112 + 8 * index;
        file[at..at + 4].copy_from_slice(&rva.to_le_bytes());
        file[at + 4..at + 8].copy_from_slice(&size.to_le_bytes());
    }

    fn headers(&self, raw_size: usize, directories: &[(u32, u32); 16]) -> Vec<u8> {
        let size_of_image = SECTION_RVA as usize + raw_size.div_ceil(0x1000) * 0x1000;
        let mut h = Body(vec![0; HEADERS_SIZE]);
        h.0[0..2].copy_from_slice(b"MZ");
        h.u32(0x3C, PE_OFFSET as u32);
        h.0[PE_OFFSET..PE_OFFSET + 4].copy_from_slice(b"PE\0\0");
        let coff = PE_OFFSET + 4;
        h.u16(coff, self.machine);
        h.u16(coff + 2, 1);
        h.u16(coff + 16, 240);
        h.u16(coff + 18, if self.dll { 0x2022 } else { 0x0022 });
        let opt = coff + 20;
        h.u16(opt, 0x20B);
        h.u32(opt + 16, self.entry.map_or(0, rva));
        h.u64(opt + 24, self.base());
        h.u32(opt + 32, 0x1000);
        h.u32(opt + 36, FILE_ALIGN as u32);
        h.u32(opt + 56, size_of_image as u32);
        h.u32(opt + 60, HEADERS_SIZE as u32);
        h.u16(opt + 68, 3);
        h.u16(
            opt + 70,
            if self.pointers.is_empty() {
                0x0100
            } else {
                0x0160
            },
        );
        h.u32(opt + 108, 16);
        for (index, &(dir_rva, size)) in directories.iter().enumerate() {
            h.u32(opt + 112 + 8 * index, dir_rva);
            h.u32(opt + 116 + 8 * index, size);
        }
        let section = opt + 240;
        h.0[section..section + 5].copy_from_slice(b".text");
        h.u32(section + 8, raw_size as u32);
        h.u32(section + 12, SECTION_RVA);
        h.u32(section + 16, raw_size as u32);
        h.u32(section + 20, HEADERS_SIZE as u32);
        h.u32(section + 36, SECTION_RWX);
        h.0
    }
}

fn imports(body: &mut Body, imports: &[(&'static str, Symbol)]) -> ((u32, u32), Iat) {
    let mut groups: Vec<(&str, Vec<Symbol>)> = Vec::new();
    for (dll, symbol) in imports {
        match groups.iter_mut().find(|(name, _)| name == dll) {
            Some((_, list)) => list.push(symbol.clone()),
            None => groups.push((dll, vec![symbol.clone()])),
        }
    }
    let mut iat = Iat::new();
    if groups.is_empty() {
        return ((0, 0), iat);
    }
    let table = body.put(&vec![0; 20 * (groups.len() + 1)], 4);
    for (index, (dll, symbols)) in groups.iter().enumerate() {
        let lookup = body.put(&vec![0; 8 * (symbols.len() + 1)], 8);
        let address = body.put(&vec![0; 8 * (symbols.len() + 1)], 8);
        for (slot, symbol) in symbols.iter().enumerate() {
            let (entry, key) = match symbol {
                Symbol::Ordinal(o) => (ORDINAL_FLAG | u64::from(*o), format!("#{o}")),
                Symbol::Name(n) => {
                    let hint_name = [&[0u8, 0][..], n.as_bytes(), &[0]].concat();
                    (u64::from(rva(body.put(&hint_name, 2))), (*n).to_string())
                }
            };
            body.u64(lookup + 8 * slot, entry);
            body.u64(address + 8 * slot, entry);
            iat.insert(((*dll).to_string(), key), rva(address + 8 * slot));
        }
        let name = body.put(&[dll.as_bytes(), &[0]].concat(), 2);
        let descriptor = table + 20 * index;
        body.u32(descriptor, rva(lookup));
        body.u32(descriptor + 12, rva(name));
        body.u32(descriptor + 16, rva(address));
    }
    ((rva(table), 20 * (groups.len() as u32 + 1)), iat)
}

fn exports(body: &mut Body, exports: &[(&'static str, Target)]) -> (u32, u32) {
    if exports.is_empty() {
        return (0, 0);
    }
    let count = exports.len();
    let start = body.put(&[0; 40], 4);
    let functions = body.put(&vec![0; 4 * count], 4);
    let names = body.put(&vec![0; 4 * count], 4);
    let ordinals = body.put(&vec![0; 2 * count], 2);
    let dll_name = body.put(b"test.dll\0", 1);
    let mut sorted: Vec<(&str, usize)> =
        exports.iter().enumerate().map(|(i, e)| (e.0, i)).collect();
    sorted.sort();
    for (slot, (name, index)) in sorted.into_iter().enumerate() {
        let name_rva = rva(body.put(&[name.as_bytes(), &[0]].concat(), 1));
        body.u32(names + 4 * slot, name_rva);
        body.u16(ordinals + 2 * slot, index as u16);
    }
    for (index, (_, target)) in exports.iter().enumerate() {
        let value = match target {
            Target::Forward(to) => rva(body.put(&[to.as_bytes(), &[0]].concat(), 1)),
            Target::Code(offset) => rva(*offset),
        };
        body.u32(functions + 4 * index, value);
    }
    let end = body.0.len();
    body.u32(start + 12, rva(dll_name));
    body.u32(start + 16, 1);
    body.u32(start + 20, count as u32);
    body.u32(start + 24, count as u32);
    body.u32(start + 28, rva(functions));
    body.u32(start + 32, rva(names));
    body.u32(start + 36, rva(ordinals));
    (rva(start), (end - start) as u32)
}

fn relocations(body: &mut Body, pointers: &[(usize, usize)]) -> (u32, u32) {
    if pointers.is_empty() {
        return (0, 0);
    }
    let mut entries: Vec<u16> = pointers
        .iter()
        .map(|&(offset, _)| (REL_BASED_DIR64 << 12) | offset as u16)
        .collect();
    entries.sort_unstable();
    if entries.len() % 2 == 1 {
        entries.push(0);
    }
    let mut block = Vec::new();
    block.extend_from_slice(&SECTION_RVA.to_le_bytes());
    block.extend_from_slice(&(8 + 2 * entries.len() as u32).to_le_bytes());
    for entry in entries {
        block.extend_from_slice(&entry.to_le_bytes());
    }
    (rva(body.put(&block, 4)), block.len() as u32)
}
