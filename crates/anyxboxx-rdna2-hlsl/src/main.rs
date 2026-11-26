use std::io::{self, Read};

fn main() {
    let mut data = Vec::new();
    let result = io::stdin()
        .read_to_end(&mut data)
        .map_err(|error| error.to_string())
        .and_then(|_| anyxboxx_rdna2_hlsl::translate(&data));
    match result {
        Ok(hlsl) => print!("{hlsl}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
