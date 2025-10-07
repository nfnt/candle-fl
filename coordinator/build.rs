fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("../api/proto/candlefl/v1/candlefl.proto")?;
    Ok(())
}
