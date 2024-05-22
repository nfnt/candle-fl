use candle_core::{safetensors::Load, DType, Device, Error, D};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap, SGD};
use safetensors::SafeTensors;
use tracing::info;

use crate::ml::dataloader::Dataloader;
use crate::ml::model::Model;

mod dataloader;
mod model;

pub fn prepare_data(dev: &Device) -> Result<Dataloader, Error> {
    let dataset = candle_datasets::vision::mnist::load()?;

    let inputs = dataset.train_images.to_device(dev)?;
    let targets = dataset.train_labels.to_device(dev)?;

    Ok(Dataloader::new(inputs, targets, 32))
}

pub fn prepare_model(dev: &Device) -> Result<(VarMap, Model), Error> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);

    // Creating the model builds 'varmap' parameters
    let model = Model::new(&vs)?;

    Ok((varmap, model))
}

pub fn train(weights: &SafeTensors, data: &Dataloader, dev: &Device) -> Result<VarMap, Error> {
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

    let mut optimizer = SGD::new(varmap.all_vars(), 0.1)?;

    let mut sum_loss = 0f32;
    let mut total = 0;

    for (inputs, targets) in data.iter() {
        let logits = model.forward(&inputs)?;
        let logits_softmax = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&logits_softmax, &targets)?;

        optimizer.backward_step(&loss)?;
        sum_loss += loss.to_vec0::<f32>()?;
        total += inputs.dims()[0];
    }
    let avg_loss = sum_loss / total as f32;

    info!(loss = avg_loss, "completed training");

    Ok(varmap)
}
