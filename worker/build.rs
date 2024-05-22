use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR").unwrap();
    let protoc_path = protoc_fetcher::protoc("26.1", Path::new(&out_dir)).unwrap();

    env::set_var("PROTOC", protoc_path);
    tonic_build::compile_protos("../api/proto/candlefl/v1/candlefl.proto")?;
    Ok(())
}
