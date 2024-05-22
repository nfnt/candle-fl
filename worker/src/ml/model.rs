use candle_core::{Error, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct Model {
    ln1: Linear,
    ln2: Linear,
}

impl Model {
    pub fn new(vs: &VarBuilder) -> Result<Self, Error> {
        let ln1 = linear(28 * 28, 100, vs.push_prefix("ln1"))?;
        let ln2 = linear(100, 10, vs.push_prefix("ln2"))?;

        Ok(Self { ln1, ln2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        self.ln2.forward(&xs)
    }
}
