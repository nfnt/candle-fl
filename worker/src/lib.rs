pub mod candlefl {
    tonic::include_proto!("candlefl.v1");
}
pub mod handler;
pub mod ml;

/// Pick the best available device: CUDA, then Metal, then CPU.
///
/// # Errors
///
/// Returns an error if a CUDA or Metal device is reported available but
/// fails to initialize.
pub fn select_device() -> candle_core::Result<candle_core::Device> {
    if candle_core::utils::cuda_is_available() {
        candle_core::Device::new_cuda(0)
    } else if candle_core::utils::metal_is_available() {
        candle_core::Device::new_metal(0)
    } else {
        Ok(candle_core::Device::Cpu)
    }
}
