use std::collections::HashMap;

use candle_core::Tensor;
use tokio::sync::mpsc;
use tracing::info;

use crate::{
    state::{Error as StateError, Job},
    strategy::{Error, RoundUpdate, Strategy},
};

/// [FederatedAveraging](https://arxiv.org/abs/1602.05629)
#[derive(Default)]
pub struct FedAvg;

impl FedAvg {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Strategy for FedAvg {
    /// Fit model weights using federated averaging by training on data
    /// provided by connected workers, reporting each round's aggregate on
    /// `updates` as it completes.
    ///
    /// # Examples
    ///
    /// `num_rounds = 0` skips straight to reporting the initial weights
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
    /// // In a real coordinator, 'Runner' creates and removes the job around
    /// // a strategy's 'fit' call; here there's no 'Runner' involved, so the
    /// // example does that itself.
    /// let job = state.add_job(uuid::Uuid::new_v4()).await;
    ///
    /// let responder = {
    ///     let state = state.clone();
    ///     let job_id = job.id();
    ///     tokio::spawn(async move {
    ///         let message = receiver.recv().await.expect("a WeightsRequest");
    ///         assert!(matches!(
    ///             message.message,
    ///             Some(coordinator_message::Message::WeightsRequest(_))
    ///         ));
    ///         let mut weights = HashMap::new();
    ///         weights.insert("a".to_string(), Tensor::new(vec![1.0, 1.0], &Device::Cpu)?);
    ///         state.set_fit_result(job_id, addr, weights).await
    ///     })
    /// };
    ///
    /// let strategy = FedAvg::new();
    /// let (updates_tx, mut updates_rx) = mpsc::channel(4);
    /// strategy.fit(&job, 0, updates_tx).await?;
    /// responder.await??;
    ///
    /// let update = updates_rx.recv().await.expect("the round 0 update");
    /// assert_eq!(update.round, 0);
    /// assert_eq!(update.weights.get("a").unwrap().to_vec1::<f64>()?, vec![1.0, 1.0]);
    ///
    /// job.remove().await;
    /// # Ok(())
    /// # }
    /// ```
    async fn fit(
        &self,
        job: &Job<'_>,
        num_rounds: usize,
        updates: mpsc::Sender<RoundUpdate>,
    ) -> Result<(), Error> {
        Ok(run_rounds(job, num_rounds, updates).await?)
    }
}

/// Send `weights` for `round` on `updates`, logging (rather than erroring)
/// if the receiver has hung up -- the caller (e.g. a disconnected gRPC
/// client) is simply no longer listening, which is not itself a failure of
/// the training run. Returns whether the update was actually delivered, so
/// callers can stop early once nobody is listening.
async fn send_update(
    updates: &mpsc::Sender<RoundUpdate>,
    job_id: uuid::Uuid,
    round: u64,
    weights: HashMap<String, Tensor>,
) -> bool {
    let delivered = updates
        .send(RoundUpdate {
            job_id,
            round,
            weights,
        })
        .await
        .is_ok();

    if !delivered {
        info!(job_id = %job_id, round, "no one is listening for round updates anymore; stopping early");
    }

    delivered
}

async fn run_rounds(
    job: &Job<'_>,
    num_rounds: usize,
    updates: mpsc::Sender<RoundUpdate>,
) -> Result<(), StateError> {
    let mut weights = job.get_weights().await?;

    if num_rounds == 0 {
        send_update(&updates, job.id(), 0, weights).await;
        return Ok(());
    }

    for round in 0..num_rounds {
        info!(job_id = %job.id(), "starting round {}", round + 1);
        let local_weights = job.fit_round(weights.clone()).await?;

        weights = average_weights(local_weights)?;

        // 'round' is a loop index over 'usize'; clamp rather than wrap on
        // 32-bit targets where 'usize' is narrower than the wire type.
        let round_number = u64::try_from(round + 1).unwrap_or(u64::MAX);
        if !send_update(&updates, job.id(), round_number, weights.clone()).await {
            break;
        }
    }

    Ok(())
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
    use crate::{
        candlefl::{CoordinatorMessage, coordinator_message::Message},
        state::State,
    };

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
        let job = state.add_job(Uuid::new_v4()).await;
        let strategy = FedAvg::new();
        let (updates_tx, _updates_rx) = mpsc::channel(4);

        let err = strategy.fit(&job, 1, updates_tx).await.unwrap_err();

        assert!(matches!(err, Error::State(StateError::NoWorkers(_))));
    }

    #[tokio::test]
    async fn fit_zero_rounds_reports_initial_weights_unchanged() {
        let state = State::new();
        let (sender, mut receiver) = mpsc::channel(4);
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        state.add_worker(addr, sender).await;

        let job = state.add_job(Uuid::new_v4()).await;
        let job_id = job.id();

        let responder_state = state.clone();
        let responder = tokio::spawn(async move {
            let message = receiver.recv().await.expect("a WeightsRequest");
            assert_eq!(job_id_of(&message), job_id);
            responder_state
                .set_fit_result(job_id, addr, tensor_map(1.0))
                .await
                .unwrap();
        });

        let strategy = FedAvg::new();
        let (updates_tx, mut updates_rx) = mpsc::channel(4);
        tokio::time::timeout(Duration::from_secs(5), strategy.fit(&job, 0, updates_tx))
            .await
            .expect("must not hang")
            .unwrap();
        responder.await.unwrap();

        let update = updates_rx.recv().await.expect("a round 0 update");
        assert_eq!(update.round, 0);
        assert_eq!(
            update.weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
        assert!(updates_rx.recv().await.is_none(), "only one update");
    }

    #[tokio::test]
    async fn fit_two_rounds_averages_across_rounds() {
        let state = State::new();
        let (sender, mut receiver) = mpsc::channel(4);
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        state.add_worker(addr, sender).await;

        let job = state.add_job(Uuid::new_v4()).await;
        let job_id = job.id();

        let responder_state = state.clone();
        let responder = tokio::spawn(async move {
            // 1 WeightsRequest + 2 FitRequests, one worker each time.
            for _ in 0..3 {
                let message = receiver.recv().await.expect("a request");
                assert_eq!(job_id_of(&message), job_id);
                responder_state
                    .set_fit_result(job_id, addr, tensor_map(2.0))
                    .await
                    .unwrap();
            }
        });

        let strategy = FedAvg::new();
        let (updates_tx, mut updates_rx) = mpsc::channel(4);
        tokio::time::timeout(Duration::from_secs(5), strategy.fit(&job, 2, updates_tx))
            .await
            .expect("must not hang")
            .unwrap();
        responder.await.unwrap();

        // A single worker: averaging is the identity, each round.
        let first = updates_rx.recv().await.expect("round 1 update");
        assert_eq!(first.round, 1);
        assert_eq!(
            first.weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![2.0, 2.0]
        );

        let second = updates_rx.recv().await.expect("round 2 update");
        assert_eq!(second.round, 2);
        assert_eq!(second.job_id, first.job_id);
        assert_eq!(
            second.weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![2.0, 2.0]
        );

        assert!(updates_rx.recv().await.is_none(), "only two updates");
    }

    #[tokio::test]
    async fn fit_stops_early_once_the_receiver_is_dropped() {
        let state = State::new();
        let (sender, mut receiver) = mpsc::channel(4);
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        state.add_worker(addr, sender).await;

        let job = state.add_job(Uuid::new_v4()).await;
        let job_id = job.id();

        let (updates_tx, updates_rx) = mpsc::channel(4);

        let responder_state = state.clone();
        let responder = tokio::spawn(async move {
            // Answer the initial WeightsRequest, then drop the updates
            // receiver *before* round 1's FitRequest is even answered, so
            // by the time the strategy tries to send round 1's update the
            // channel is deterministically already closed -- no race with
            // the strategy's own send. If the strategy didn't stop early,
            // it would go on to send a second FitRequest for round 2, which
            // nothing here ever answers, and the test would time out
            // instead of passing.
            let message = receiver.recv().await.expect("a WeightsRequest");
            assert_eq!(job_id_of(&message), job_id);
            responder_state
                .set_fit_result(job_id, addr, tensor_map(1.0))
                .await
                .unwrap();

            drop(updates_rx);

            let message = receiver.recv().await.expect("round 1's FitRequest");
            assert_eq!(job_id_of(&message), job_id);
            responder_state
                .set_fit_result(job_id, addr, tensor_map(1.0))
                .await
                .unwrap();
        });

        let strategy = FedAvg::new();

        // Ending early because nobody is listening is not itself a failure.
        tokio::time::timeout(Duration::from_secs(5), strategy.fit(&job, 5, updates_tx))
            .await
            .expect("must not hang")
            .unwrap();
        responder.await.unwrap();
    }
}
