use candle_core::{Error, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module, VarBuilder, conv2d, linear, ops::sigmoid};

/// LeNet CNN model implementation
/// Architecture based on the original LeNet-5 paper
/// See: https://ieeexplore.ieee.org/document/726791
pub struct LeNet {
    conv1: Conv2d,
    conv2: Conv2d,
    dense1: Linear,
    dense2: Linear,
    dense3: Linear,
}

impl LeNet {
    pub fn try_new(vs: VarBuilder) -> Result<Self, Error> {
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
