use candle_core::{Error, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module, VarBuilder, conv2d, linear, ops::sigmoid};

/// `LeNet` CNN model implementation
/// Architecture based on the original LeNet-5 paper
/// See: <https://ieeexplore.ieee.org/document/726791>
pub struct LeNet {
    conv1: Conv2d,
    conv2: Conv2d,
    dense1: Linear,
    dense2: Linear,
    dense3: Linear,
}

impl LeNet {
    /// # Errors
    ///
    /// Returns an error if the model's parameters fail to initialize.
    pub fn try_new(vs: &VarBuilder) -> Result<Self, Error> {
        let conv1 = conv2d(
            1,
            6,
            5,
            Conv2dConfig {
                padding: 2,
                ..Default::default()
            },
            vs.push_prefix("conv1"),
        )?;
        let conv2 = conv2d(6, 16, 5, Conv2dConfig::default(), vs.push_prefix("conv2"))?;
        let dense1 = linear(400, 120, vs.push_prefix("dense1"))?;
        let dense2 = linear(120, 84, vs.push_prefix("dense2"))?;
        let dense3 = linear(84, 10, vs.push_prefix("dense3"))?;

        Ok(Self {
            conv1,
            conv2,
            dense1,
            dense2,
            dense3,
        })
    }
}

impl Module for LeNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let (batch_size, _) = xs.dims2()?;
        let xs = xs.reshape((batch_size, 1, 28, 28))?;

        let xs = sigmoid(&self.conv1.forward(&xs)?)?.avg_pool2d(2)?;
        let xs = sigmoid(&self.conv2.forward(&xs)?)?.avg_pool2d(2)?;
        let xs = sigmoid(&self.dense1.forward(&xs.flatten_from(1)?)?)?;
        let xs = sigmoid(&self.dense2.forward(&xs)?)?;
        self.dense3.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    use super::*;

    fn model() -> (VarMap, LeNet) {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = LeNet::try_new(&vs).unwrap();
        (varmap, model)
    }

    #[test]
    fn forward_produces_ten_class_logits() {
        let (_varmap, model) = model();
        let input = Tensor::zeros((4, 784), DType::F32, &Device::Cpu).unwrap();

        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[4, 10]);
    }

    #[test]
    fn forward_rejects_non_2d_input() {
        let (_varmap, model) = model();
        // 'dims2()' requires exactly two dimensions; this input has three.
        let input = Tensor::zeros((4, 28, 28), DType::F32, &Device::Cpu).unwrap();

        let err = model.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            candle_core::Error::UnexpectedNumberOfDims { .. }
        ));
    }

    #[test]
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the lock is only ever touched by this single-threaded test; holding it for the whole function isn't a contention risk"
    )]
    fn parameter_shapes_match_the_documented_architecture() {
        let (varmap, _model) = model();

        let tensor_data = varmap.data().lock().unwrap();
        let shape_of = |name: &str| tensor_data.get(name).unwrap().shape().dims().to_vec();

        assert_eq!(shape_of("conv1.weight"), vec![6, 1, 5, 5]);
        assert_eq!(shape_of("conv1.bias"), vec![6]);
        assert_eq!(shape_of("conv2.weight"), vec![16, 6, 5, 5]);
        assert_eq!(shape_of("conv2.bias"), vec![16]);
        assert_eq!(shape_of("dense1.weight"), vec![120, 400]);
        assert_eq!(shape_of("dense1.bias"), vec![120]);
        assert_eq!(shape_of("dense2.weight"), vec![84, 120]);
        assert_eq!(shape_of("dense2.bias"), vec![84]);
        assert_eq!(shape_of("dense3.weight"), vec![10, 84]);
        assert_eq!(shape_of("dense3.bias"), vec![10]);
        assert_eq!(tensor_data.len(), 10);
    }
}
