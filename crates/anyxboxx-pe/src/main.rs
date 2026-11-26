//! `anyxboxx-pe <image> [--base 0x...] [--pages]`: prints the loaded image as one JSON line.
//!
//! `--base` forces the final base (as a custom `allocate` does in `loader.read_pe`); `--pages`
//! adds the protection of each page. A refused image prints its error on stderr, exit code 1.

use std::process::ExitCode;

use anyxboxx_pe::{LoadError, default_allocate, json, page_protections, read_pe};

const USAGE: &str = "usage: anyxboxx-pe <image> [--base 0x...] [--pages]";

struct Options {
    path: String,
    base: Option<u64>,
    pages: bool,
}

fn parse(arguments: &[String]) -> Result<Options, String> {
    let mut options = Options {
        path: String::new(),
        base: None,
        pages: false,
    };
    let mut rest = arguments.iter();
    while let Some(argument) = rest.next() {
        match argument.as_str() {
            "--pages" => options.pages = true,
            "--base" => {
                let value = rest.next().ok_or("--base needs a value")?;
                let digits = value.trim_start_matches("0x");
                let base = u64::from_str_radix(digits, 16).map_err(|e| format!("--base: {e}"))?;
                options.base = Some(base);
            }
            path if options.path.is_empty() && !path.starts_with("--") => {
                options.path = path.to_string();
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    if options.path.is_empty() {
        return Err(USAGE.to_string());
    }
    Ok(options)
}

fn run(options: &Options) -> Result<String, String> {
    let data = std::fs::read(&options.path).map_err(|e| format!("{}: {e}", options.path))?;
    let forced = options.base;
    let allocate =
        |preferred: Option<u64>, size: u64, relocatable: bool| -> Result<u64, LoadError> {
            match forced {
                Some(base) => Ok(base),
                None => default_allocate(preferred, size, relocatable),
            }
        };
    let image = read_pe(&data, allocate).map_err(|e| e.to_string())?;
    let pages = if options.pages {
        Some(page_protections(&data).map_err(|e| e.to_string())?)
    } else {
        None
    };
    Ok(json::render(&image, pages.as_deref()))
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match parse(&arguments).and_then(|options| run(&options)) {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("anyxboxx-pe: {message}");
            ExitCode::FAILURE
        }
    }
}
