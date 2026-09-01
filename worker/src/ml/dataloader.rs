use candle_core::Tensor;

/// A simple sequential batcher over `inputs`/`targets` tensor pairs.
///
/// # Examples
///
/// ```
/// use candle_core::{DType, Device, Tensor};
/// use worker::ml::Dataloader;
///
/// let dev = Device::Cpu;
/// let inputs = Tensor::zeros((10, 4), DType::F32, &dev)?;
/// let targets = Tensor::zeros((10,), DType::U32, &dev)?;
///
/// let loader = Dataloader::new(inputs, targets, 4);
/// let batch_sizes = loader
///     .iter()
///     .map(|batch| batch.map(|(inputs, _targets)| inputs.dims()[0]))
///     .collect::<Result<Vec<_>, _>>()?;
///
/// // 10 rows in batches of 4: sizes 4, 4, then a ragged final batch of 2.
/// assert_eq!(batch_sizes, vec![4, 4, 2]);
/// # Ok::<(), candle_core::Error>(())
/// ```
pub struct Dataloader {
    inputs: Tensor,
    targets: Tensor,
    batch_size: usize,
}

impl Dataloader {
    /// # Panics
    ///
    /// Panics if `batch_size` is zero, since a zero-sized batch would never
    /// advance the iterator and loop forever.
    #[must_use]
    pub const fn new(inputs: Tensor, targets: Tensor, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than zero");
        Self {
            inputs,
            targets,
            batch_size,
        }
    }

    #[must_use]
    pub const fn iter(&self) -> DataloaderIterator<'_> {
        DataloaderIterator {
            inputs: &self.inputs,
            targets: &self.targets,
            batch_size: self.batch_size,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a Dataloader {
    type IntoIter = DataloaderIterator<'a>;
    type Item = <DataloaderIterator<'a> as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct DataloaderIterator<'a> {
    inputs: &'a Tensor,
    targets: &'a Tensor,
    batch_size: usize,
    index: usize,
}

impl Iterator for DataloaderIterator<'_> {
    type Item = Result<(Tensor, Tensor), candle_core::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.inputs.dims()[0] {
            let start = self.index;
            let len = (self.batch_size).min(self.inputs.dims()[0] - start);

            let batch = self
                .inputs
                .narrow(0, start, len)
                .and_then(|inputs| Ok((inputs, self.targets.narrow(0, start, len)?)));

            self.index = start + len;

            Some(batch)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    // These casts are exact for the tiny row counts (<= 10) these tests use.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn inputs_with_rows(n: usize) -> Tensor {
        let data: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, i as f32]).collect();
        Tensor::new(data, &Device::Cpu).unwrap()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn targets_with_len(n: usize) -> Tensor {
        let data: Vec<u32> = (0..n as u32).collect();
        Tensor::new(data, &Device::Cpu).unwrap()
    }

    #[test]
    fn exact_multiple_batching() {
        let loader = Dataloader::new(inputs_with_rows(8), targets_with_len(8), 4);

        let batches: Vec<_> = loader.iter().map(Result::unwrap).collect();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].0.dims(), &[4, 2]);
        assert_eq!(batches[0].1.dims(), &[4]);
        assert_eq!(batches[1].0.dims(), &[4, 2]);
    }

    #[test]
    fn ragged_final_batch() {
        let loader = Dataloader::new(inputs_with_rows(10), targets_with_len(10), 4);

        let batches: Vec<_> = loader.iter().map(Result::unwrap).collect();

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].0.dims()[0], 4);
        assert_eq!(batches[1].0.dims()[0], 4);
        assert_eq!(batches[2].0.dims()[0], 2);
    }

    #[test]
    fn batch_size_larger_than_dataset_yields_one_batch() {
        let loader = Dataloader::new(inputs_with_rows(3), targets_with_len(3), 10);

        let batches: Vec<_> = loader.iter().map(Result::unwrap).collect();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].0.dims()[0], 3);
    }

    #[test]
    fn empty_dataset_yields_no_batches() {
        let inputs = Tensor::zeros((0, 2), DType::F32, &Device::Cpu).unwrap();
        let targets = Tensor::zeros((0,), DType::U32, &Device::Cpu).unwrap();
        let loader = Dataloader::new(inputs, targets, 4);

        assert_eq!(loader.iter().count(), 0);
    }

    #[test]
    #[should_panic(expected = "batch_size must be greater than zero")]
    fn batch_size_zero_panics() {
        // Without the assertion in 'new', this would silently construct a
        // Dataloader whose iterator never advances 'index' and loops
        // forever the first time it's driven.
        let _ = Dataloader::new(inputs_with_rows(4), targets_with_len(4), 0);
    }
}
