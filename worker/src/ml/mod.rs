use candle_core::{D, DType, Device, Error, safetensors::Load};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss, ops};
use safetensors::SafeTensors;
use tracing::info;

use crate::ml::model::LeNet;

pub use crate::ml::dataloader::Dataloader;

mod dataloader;
mod model;

/// Load the full `FashionMNIST` training set.
///
/// # Errors
///
/// Returns an error if the dataset fails to download or load.
fn prepare_data(dev: &Device) -> Result<Dataloader, Error> {
    let dataset = candle_datasets::vision::fashion_mnist::load()?;

    let inputs = dataset.train_images.to_device(dev)?;
    let targets = dataset.train_labels.to_dtype(DType::U32)?.to_device(dev)?;

    Ok(Dataloader::new(inputs, targets, 64))
}

/// Build a freshly initialized `LeNet` model.
///
/// # Errors
///
/// Returns an error if the model's parameters fail to initialize.
fn prepare_model(dev: &Device) -> Result<(VarMap, LeNet), Error> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);

    // Creating the model builds 'varmap' parameters
    let model = LeNet::try_new(&vs)?;

    Ok((varmap, model))
}

/// Train a `LeNet` model on `data`, starting from `weights`.
///
/// # Errors
///
/// Returns an error if `weights` is missing a parameter the model needs,
/// or a tensor operation during the forward/backward pass fails.
///
/// # Panics
///
/// Panics if the model's parameter lock is poisoned.
fn train(weights: &SafeTensors, data: &Dataloader, dev: &Device) -> Result<VarMap, Error> {
    info!("starting training");

    let (varmap, model) = prepare_model(dev)?;

    // Load weights
    {
        let mut tensor_data = varmap.data().lock().unwrap();
        for (name, var) in tensor_data.iter_mut() {
            let data = weights.tensor(name)?;
            var.set(&data.load(dev)?)?;
        }
    }

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 0.001,
            ..Default::default()
        },
    )?;

    let mut sum_loss = 0f32;
    let mut total = 0;

    for batch in data {
        let (inputs, targets) = batch?;
        let logits = model.forward(&inputs)?;
        let logits_softmax = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&logits_softmax, &targets)?;

        optimizer.backward_step(&loss)?;
        sum_loss += loss.to_vec0::<f32>()?;
        total += inputs.dims()[0];
    }
    // 'total' is a sample count, far below f32's 24-bit mantissa for any
    // dataset this trains on.
    #[allow(clippy::cast_precision_loss)]
    let avg_loss = sum_loss / total as f32;

    info!(loss = avg_loss, "completed training");

    Ok(varmap)
}

/// How a worker turns a coordinator request into updated model weights.
///
/// Exists so the request-handling path in `crate::handler` can be driven by
/// a stub in tests, instead of always downloading `FashionMNIST` from the
/// Hugging Face hub and running a full epoch.
pub trait Trainer: Send + Sync + 'static {
    /// Build a freshly initialized model and return its weights, for a
    /// `WeightsRequest`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to build.
    fn prepare_weights(&self, dev: &Device) -> Result<VarMap, Error>;

    /// Train on local data starting from `weights`, returning the updated
    /// weights, for a `FitRequest`.
    ///
    /// # Errors
    ///
    /// Returns an error if training fails.
    fn train(&self, weights: &SafeTensors, dev: &Device) -> Result<VarMap, Error>;
}

/// The real trainer used by the `worker` binary: `LeNet` on the full
/// `FashionMNIST` training set.
#[derive(Clone, Copy, Debug, Default)]
pub struct FashionMnistTrainer;

impl Trainer for FashionMnistTrainer {
    fn prepare_weights(&self, dev: &Device) -> Result<VarMap, Error> {
        prepare_model(dev).map(|(varmap, _model)| varmap)
    }

    fn train(&self, weights: &SafeTensors, dev: &Device) -> Result<VarMap, Error> {
        let data = prepare_data(dev)?;
        train(weights, &data, dev)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Tensor;

    use super::*;

    /// A small, synthetic stand-in for `prepare_data`'s `FashionMNIST`
    /// output, cheap enough to run in every test invocation.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "n is a tiny test sample count, never near u32::MAX"
    )]
    fn synthetic_data(n: usize, batch_size: usize) -> Dataloader {
        let dev = Device::Cpu;
        let inputs = Tensor::zeros((n, 784), DType::F32, &dev).unwrap();
        // LeNet has 10 output classes; targets must be valid class indices.
        let targets: Vec<u32> = (0..n as u32).map(|i| i % 10).collect();
        let targets = Tensor::new(targets, &dev).unwrap();
        Dataloader::new(inputs, targets, batch_size)
    }

    fn initial_weights_bytes() -> Vec<u8> {
        let (varmap, _model) = prepare_model(&Device::Cpu).unwrap();
        let data: Vec<_> = {
            let tensor_data = varmap.data().lock().unwrap();
            tensor_data
                .iter()
                .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
                .collect()
        };
        safetensors::serialize(data, None).unwrap()
    }

    #[test]
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the lock is only ever touched by this single-threaded test; holding it for the whole function isn't a contention risk"
    )]
    fn prepare_model_builds_the_expected_parameter_set() {
        let (varmap, _model) = prepare_model(&Device::Cpu).unwrap();

        let tensor_data = varmap.data().lock().unwrap();
        assert_eq!(tensor_data.len(), 10);
        assert!(tensor_data.contains_key("conv1.weight"));
        assert!(tensor_data.contains_key("dense3.bias"));
    }

    #[test]
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the lock is only ever touched by this single-threaded test; holding it for the whole function isn't a contention risk"
    )]
    fn train_on_synthetic_data_updates_weights() {
        let dev = Device::Cpu;
        let bytes = initial_weights_bytes();
        let initial = SafeTensors::deserialize(&bytes).unwrap();

        let before: Vec<(String, Tensor)> = initial
            .names()
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    initial.tensor(name).unwrap().load(&dev).unwrap(),
                )
            })
            .collect();

        let data = synthetic_data(16, 8);

        let result = train(&initial, &data, &dev).unwrap();

        let after_data = result.data().lock().unwrap();
        assert_eq!(after_data.len(), before.len());

        let mut total_diff = 0f32;
        for (name, before_tensor) in &before {
            let after_tensor = after_data.get(name).unwrap().as_tensor();
            total_diff += (after_tensor - before_tensor)
                .unwrap()
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
        }

        assert!(
            total_diff > 0.0,
            "expected at least one parameter to change after a training step"
        );
    }

    #[test]
    fn train_with_missing_weight_key_errors_instead_of_panicking() {
        let dev = Device::Cpu;
        let (varmap, _model) = prepare_model(&dev).unwrap();
        let tensor_data = varmap.data().lock().unwrap();
        let data: Vec<_> = tensor_data
            .iter()
            .filter(|(k, _)| k.as_str() != "dense3.bias")
            .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
            .collect();
        drop(tensor_data);
        let bytes = safetensors::serialize(data, None).unwrap();
        let weights = SafeTensors::deserialize(&bytes).unwrap();

        let loader = synthetic_data(4, 4);

        // 'train' must return an error for the missing key, not panic --
        // it's driven by untrusted, network-supplied weights.
        let result = train(&weights, &loader, &dev);

        assert!(result.is_err());
    }
}
