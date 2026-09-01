use std::collections::HashMap;

use candle_core::Tensor;
use tracing::info;

use crate::{
    state::{Error as StateError, Job, State},
    strategy::{Error, Strategy},
};

/// [FederatedAveraging](https://arxiv.org/abs/1602.05629)
pub struct FedAvg {
    state: State,
}

impl FedAvg {
    #[must_use]
    pub const fn new(state: State) -> Self {
        Self { state }
    }
}

impl Strategy for FedAvg {
    /// Fit model weights using federated averaging by training on data provided
    /// by connected workers.
    ///
    /// # Examples
    ///
    /// `num_rounds = 0` skips straight to returning the initial weights
    /// fetched from the first connected worker -- useful here to show the
    /// full flow without a fake worker also having to answer `FitRequest`s.
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::collections::HashMap;
    ///
    /// use candle_core::{Device, Tensor};
    /// use coordinator::{
    ///     candlefl::coordinator_message,
    ///     state::State,
    ///     strategy::{FedAvg, Strategy},
    /// };
    /// use tokio::sync::mpsc;
    ///
    /// let state = State::new();
    ///
    /// let (sender, mut receiver) = mpsc::channel(4);
    /// let addr = "127.0.0.1:1".parse().unwrap();
    /// state.add_worker(addr, sender).await;
    ///
    /// let responder = {
    ///     let state = state.clone();
    ///     tokio::spawn(async move {
    ///         let message = receiver.recv().await.expect("a WeightsRequest");
    ///         let job_id = match message.message {
    ///             Some(coordinator_message::Message::WeightsRequest(req)) => {
    ///                 req.job_id.parse().unwrap()
    ///             }
    ///             _ => unreachable!("only a WeightsRequest is sent for 0 rounds"),
    ///         };
    ///         let mut weights = HashMap::new();
    ///         weights.insert("a".to_string(), Tensor::new(vec![1.0, 1.0], &Device::Cpu)?);
    ///         state.set_fit_result(job_id, addr, weights).await
    ///     })
    /// };
    ///
    /// let strategy = FedAvg::new(state);
    /// let weights = strategy.fit(0).await?;
    /// responder.await??;
    ///
    /// assert_eq!(weights.get("a").unwrap().to_vec1::<f64>()?, vec![1.0, 1.0]);
    /// # Ok(())
    /// # }
    /// ```
    async fn fit(&self, num_rounds: usize) -> Result<HashMap<String, Tensor>, Error> {
        let job = self.state.add_job().await;

        info!(job_id = %job.id(), "starting job");

        // Run the rounds in a helper so the job is removed from the
        // coordinator's state on every exit path, including an early
        // return from a failed round -- otherwise it leaks for the
        // lifetime of the coordinator process.
        let result = run_rounds(&job, num_rounds).await;

        job.remove().await;

        info!(job_id = %job.id(), "finished job");

        Ok(result?)
    }
}

async fn run_rounds(
    job: &Job<'_>,
    num_rounds: usize,
) -> Result<HashMap<String, Tensor>, StateError> {
    let mut weights = job.get_weights().await?;

    for round in 0..num_rounds {
        info!(job_id = %job.id(), "starting round {}", round + 1);
        let local_weights = job.fit_round(weights.clone()).await?;

        weights = average_weights(local_weights)?;
    }

    Ok(weights)
}

fn average_weights(
    tensors: Vec<HashMap<String, Tensor>>,
) -> Result<HashMap<String, Tensor>, StateError> {
    if tensors.is_empty() {
        return Err(StateError::NoResults);
    }

    // Sum tensors per key, tracking how many responses actually contributed
    // that key so keys missing from some workers are still averaged correctly.
    let mut sums: HashMap<String, (Tensor, usize)> = HashMap::new();

    for tensor_map in tensors {
        for (name, tensor) in tensor_map {
            match sums.remove(&name) {
                Some((existing, count)) => sums.insert(name, ((existing + tensor)?, count + 1)),
                None => sums.insert(name, (tensor, 1)),
            };
        }
    }

    let mut result = HashMap::with_capacity(sums.len());
    for (name, (tensor, count)) in sums {
        // `count` is a worker count, far below f64's 52-bit mantissa.
        #[allow(clippy::cast_precision_loss)]
        let divisor = count as f64;
        result.insert(name, (divisor.recip() * tensor)?);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, time::Duration};

    use candle_core::{DType, Device};
    use tokio::sync::mpsc;
    use uuid::Uuid;

    use super::*;
    use crate::candlefl::{CoordinatorMessage, coordinator_message::Message};

    #[test]
    fn test_average_weights_trivial() {
        let dev = Device::Cpu;

        let tensor1 = Tensor::new(vec![1.0, 1.0], &dev).unwrap();
        let tensor2 = Tensor::new(vec![1.0, 1.0], &dev).unwrap();

        let mut weights_per_worker = Vec::new();
        let mut map = HashMap::new();
        map.insert("a".to_string(), tensor1);
        map.insert("b".to_string(), tensor2);
        weights_per_worker.push(map);

        let result = average_weights(weights_per_worker).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(
            result.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
        assert_eq!(
            result.get("b").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn test_average_weights_complex() {
        let dev = Device::Cpu;

        let tensor1 = Tensor::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]], &dev).unwrap();
        let tensor2 = Tensor::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]], &dev).unwrap();

        let mut weights_per_worker = Vec::with_capacity(2);
        let mut map = HashMap::new();
        map.insert("a".to_string(), tensor1);
        map.insert("b".to_string(), tensor2);
        weights_per_worker.push(map);

        let tensor3 = Tensor::new(vec![vec![2.0, 1.0], vec![1.0, 2.0]], &dev).unwrap();
        let tensor4 = Tensor::new(vec![vec![2.0, 1.0], vec![1.0, 2.0]], &dev).unwrap();

        let mut map = HashMap::new();
        map.insert("a".to_string(), tensor3);
        map.insert("b".to_string(), tensor4);
        weights_per_worker.push(map);

        let result = average_weights(weights_per_worker).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(
            result.get("a").unwrap().to_vec2::<f64>().unwrap(),
            vec![vec![1.5, 1.5], vec![1.5, 1.5]],
        );
        assert_eq!(
            result.get("b").unwrap().to_vec2::<f64>().unwrap(),
            vec![vec![1.5, 1.5], vec![1.5, 1.5]],
        );
    }

    #[test]
    fn test_average_weights_missing_key() {
        let dev = Device::Cpu;

        let mut map1 = HashMap::new();
        map1.insert("a".to_string(), Tensor::new(vec![2.0, 2.0], &dev).unwrap());
        map1.insert("b".to_string(), Tensor::new(vec![4.0, 4.0], &dev).unwrap());

        let mut map2 = HashMap::new();
        map2.insert("a".to_string(), Tensor::new(vec![4.0, 4.0], &dev).unwrap());
        // "b" is missing from this worker's response.

        let result = average_weights(vec![map1, map2]).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(
            result.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![3.0, 3.0]
        );
        assert_eq!(
            result.get("b").unwrap().to_vec1::<f64>().unwrap(),
            vec![4.0, 4.0]
        );
    }

    #[test]
    fn test_average_weights_empty_input_errors() {
        let err = average_weights(vec![]).unwrap_err();

        assert!(matches!(err, StateError::NoResults));
    }

    #[test]
    fn test_average_weights_shape_mismatch_errors() {
        let dev = Device::Cpu;

        let mut map1 = HashMap::new();
        map1.insert("a".to_string(), Tensor::new(vec![1.0, 1.0], &dev).unwrap());

        let mut map2 = HashMap::new();
        // Same key, incompatible shape.
        map2.insert(
            "a".to_string(),
            Tensor::new(vec![1.0, 1.0, 1.0], &dev).unwrap(),
        );

        let err = average_weights(vec![map1, map2]).unwrap_err();

        assert!(matches!(err, StateError::Candle(_)));
    }

    #[test]
    fn test_average_weights_three_workers_differing_keys() {
        let dev = Device::Cpu;

        let mut map1 = HashMap::new();
        map1.insert("a".to_string(), Tensor::new(vec![3.0], &dev).unwrap());
        map1.insert("b".to_string(), Tensor::new(vec![6.0], &dev).unwrap());

        let mut map2 = HashMap::new();
        map2.insert("a".to_string(), Tensor::new(vec![3.0], &dev).unwrap());
        map2.insert("c".to_string(), Tensor::new(vec![9.0], &dev).unwrap());

        let mut map3 = HashMap::new();
        map3.insert("a".to_string(), Tensor::new(vec![3.0], &dev).unwrap());
        map3.insert("b".to_string(), Tensor::new(vec![12.0], &dev).unwrap());
        map3.insert("c".to_string(), Tensor::new(vec![21.0], &dev).unwrap());

        let result = average_weights(vec![map1, map2, map3]).unwrap();

        assert_eq!(result.len(), 3);
        // "a" present in all 3 -> divisor 3.
        assert_eq!(
            result.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![3.0]
        );
        // "b" present in 2 -> divisor 2.
        assert_eq!(
            result.get("b").unwrap().to_vec1::<f64>().unwrap(),
            vec![9.0]
        );
        // "c" present in 2 -> divisor 2.
        assert_eq!(
            result.get("c").unwrap().to_vec1::<f64>().unwrap(),
            vec![15.0]
        );
    }

    #[test]
    fn test_average_weights_preserves_dtype() {
        let dev = Device::Cpu;

        let mut map = HashMap::new();
        map.insert(
            "a".to_string(),
            Tensor::new(vec![1.0f32, 1.0], &dev)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap(),
        );

        let result = average_weights(vec![map]).unwrap();

        assert_eq!(result.get("a").unwrap().dtype(), DType::F32);
    }

    fn tensor_map(value: f64) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(
            "a".to_string(),
            Tensor::new(vec![value, value], &Device::Cpu).unwrap(),
        );
        map
    }

    fn job_id_of(message: &CoordinatorMessage) -> Uuid {
        match &message.message {
            Some(Message::WeightsRequest(req)) => Uuid::parse_str(&req.job_id).unwrap(),
            Some(Message::FitRequest(req)) => Uuid::parse_str(&req.job_id).unwrap(),
            None => panic!("empty coordinator message"),
        }
    }

    #[tokio::test]
    async fn fit_no_workers_errors() {
        let state = State::new();
        let strategy = FedAvg::new(state);

        let err = strategy.fit(1).await.unwrap_err();

        assert!(matches!(err, Error::State(StateError::NoWorkers(_))));
    }

    #[tokio::test]
    async fn fit_zero_rounds_returns_initial_weights_unchanged() {
        let state = State::new();
        let (sender, mut receiver) = mpsc::channel(4);
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        state.add_worker(addr, sender).await;

        let responder_state = state.clone();
        let responder = tokio::spawn(async move {
            let message = receiver.recv().await.expect("a WeightsRequest");
            let job_id = job_id_of(&message);
            responder_state
                .set_fit_result(job_id, addr, tensor_map(1.0))
                .await
                .unwrap();
        });

        let strategy = FedAvg::new(state);
        let weights = tokio::time::timeout(Duration::from_secs(5), strategy.fit(0))
            .await
            .expect("must not hang")
            .unwrap();
        responder.await.unwrap();

        assert_eq!(
            weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[tokio::test]
    async fn fit_two_rounds_averages_across_rounds() {
        let state = State::new();
        let (sender, mut receiver) = mpsc::channel(4);
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        state.add_worker(addr, sender).await;

        let responder_state = state.clone();
        let responder = tokio::spawn(async move {
            // 1 WeightsRequest + 2 FitRequests, one worker each time.
            for _ in 0..3 {
                let message = receiver.recv().await.expect("a request");
                let job_id = job_id_of(&message);
                responder_state
                    .set_fit_result(job_id, addr, tensor_map(2.0))
                    .await
                    .unwrap();
            }
        });

        let strategy = FedAvg::new(state);
        let weights = tokio::time::timeout(Duration::from_secs(5), strategy.fit(2))
            .await
            .expect("must not hang")
            .unwrap();
        responder.await.unwrap();

        // A single worker: averaging is the identity, each round.
        assert_eq!(
            weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![2.0, 2.0]
        );
    }
}
