use candle_core::Tensor;

pub struct Dataloader {
    inputs: Tensor,
    targets: Tensor,
    batch_size: usize,
}

impl Dataloader {
    pub fn new(inputs: Tensor, targets: Tensor, batch_size: usize) -> Self {
        Self {
            inputs,
            targets,
            batch_size,
        }
    }

    pub fn iter(&self) -> DataloaderIterator {
        DataloaderIterator {
            inputs: &self.inputs,
            targets: &self.targets,
            batch_size: self.batch_size,
            index: 0,
        }
    }
}

pub struct DataloaderIterator<'a> {
    inputs: &'a Tensor,
    targets: &'a Tensor,
    batch_size: usize,
    index: usize,
}

impl Iterator for DataloaderIterator<'_> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.inputs.dims()[0] {
            let start = self.index;
            let len = (self.batch_size).min(self.inputs.dims()[0] - start);

            let inputs = self.inputs.narrow(0, start, len).unwrap();
            let targets = self.targets.narrow(0, start, len).unwrap();

            self.index = start + len;

            Some((inputs, targets))
        } else {
            None
        }
    }
}
