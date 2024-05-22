use std::collections::HashMap;

use candle_core::Tensor;
use tracing::info;

use crate::state::State;

/// [FederatedAveraging](https://arxiv.org/abs/1602.05629)
pub struct FedAvg {
    state: State,
}

impl FedAvg {
    pub fn new(state: State) -> Self {
        FedAvg { state }
    }

    /// Fit model weights using federated averaging by training on data provided
    /// by connected workers.
    pub async fn fit(&self, num_rounds: usize) -> Result<HashMap<String, Tensor>, anyhow::Error> {
        let job = self.state.add_job().await?;

        info!(job_id = %job.id(), "starting job");

        let mut weights = job.get_weights().await?;

        for round in 0..num_rounds {
            info!(job_id = %job.id(), "starting round {}", round + 1);
            let local_weights = job.fit_round(weights.clone()).await?;

            weights = average_weights(&local_weights)?;
        }

        info!(job_id = %job.id(), "finished job");

        Ok(weights)
    }
}

fn average_weights(
    tensors: &[HashMap<String, Tensor>],
) -> Result<HashMap<String, Tensor>, candle_core::Error> {
    let num_tensors = tensors.len() as f64;

    let result = tensors
        .iter()
        .fold(HashMap::new(), |result, tensor| {
            tensor.iter().fold(result, |mut result, (name, tensor)| {
                if let Some(existing) = result.get(name) {
                    result.insert(name.to_string(), (existing + tensor).unwrap());
                } else {
                    result.insert(name.to_string(), tensor.clone());
                }
                result
            })
        })
        .iter()
        .map(|(name, tensor)| (name.to_string(), (num_tensors.recip() * tensor).unwrap()))
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn test_average_weights_trivial() -> Result<(), candle_core::Error> {
        let dev = Device::Cpu;

        let tensor1 = Tensor::new(vec![1.0, 1.0], &dev).unwrap();
        let tensor2 = Tensor::new(vec![1.0, 1.0], &dev).unwrap();

        let mut tensors = Vec::new();
        let mut map = HashMap::new();
        map.insert("a".to_string(), tensor1);
        map.insert("b".to_string(), tensor2);
        tensors.push(map);

        let result = average_weights(&tensors)?;

        assert_eq!(result.len(), 2);
        assert_eq!(
            result.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
        assert_eq!(
            result.get("b").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );

        Ok(())
    }

    #[test]
    fn test_average_weights_complex() -> Result<(), candle_core::Error> {
        let dev = Device::Cpu;

        let tensor1 = Tensor::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]], &dev).unwrap();
        let tensor2 = Tensor::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]], &dev).unwrap();

        let mut tensors = Vec::with_capacity(2);
        let mut map = HashMap::new();
        map.insert("a".to_string(), tensor1);
        map.insert("b".to_string(), tensor2);
        tensors.push(map);

        let tensor3 = Tensor::new(vec![vec![2.0, 1.0], vec![1.0, 2.0]], &dev).unwrap();
        let tensor4 = Tensor::new(vec![vec![2.0, 1.0], vec![1.0, 2.0]], &dev).unwrap();

        let mut map = HashMap::new();
        map.insert("a".to_string(), tensor3);
        map.insert("b".to_string(), tensor4);
        tensors.push(map);

        let result = average_weights(&tensors)?;

        assert_eq!(result.len(), 2);
        assert_eq!(
            result.get("a").unwrap().to_vec2::<f64>().unwrap(),
            vec![vec![1.5, 1.5], vec![1.5, 1.5]],
        );
        assert_eq!(
            result.get("b").unwrap().to_vec2::<f64>().unwrap(),
            vec![vec![1.5, 1.5], vec![1.5, 1.5]],
        );

        Ok(())
    }
}
