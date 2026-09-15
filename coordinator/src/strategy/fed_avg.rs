use std::collections::HashMap;

use candle_core::Tensor;
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    state::{Error as StateError, Job, WorkerFitResult},
    strategy::{Error, RoundMetrics, RoundUpdate, Strategy, WorkerMetrics},
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
    ///     state::{State, WorkerUpdate},
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
    ///         let update = WorkerUpdate { weights, metrics: None };
    ///         state.set_fit_result(job_id, addr, update).await
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
    job_id: Uuid,
    round: u64,
    weights: HashMap<String, Tensor>,
    metrics: Option<RoundMetrics>,
) -> bool {
    let delivered = updates
        .send(RoundUpdate {
            job_id,
            round,
            weights,
            metrics,
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
        // No training happened, so there's nothing to report metrics on.
        send_update(&updates, job.id(), 0, weights, None).await;
        return Ok(());
    }

    for round in 0..num_rounds {
        info!(job_id = %job.id(), "starting round {}", round + 1);
        let results = job.fit_round(weights.clone()).await?;

        let metrics = round_metrics(&results);
        weights = average_weights(contributions(job.id(), results))?;

        // 'round' is a loop index over 'usize'; clamp rather than wrap on
        // 32-bit targets where 'usize' is narrower than the wire type.
        let round_number = u64::try_from(round + 1).unwrap_or(u64::MAX);
        if !send_update(
            &updates,
            job.id(),
            round_number,
            weights.clone(),
            Some(metrics),
        )
        .await
        {
            break;
        }
    }

    Ok(())
}

/// Build this round's `RoundMetrics` from its raw per-worker results: one
/// `WorkerMetrics` entry for each worker that reported metrics (a
/// `WeightsResponse` never does, and a worker can fail to), and a
/// sample-weighted mean loss across those.
///
/// Falls back to an unweighted mean of the reported losses if the reported
/// sample counts summed to zero -- dividing by zero examples otherwise.
fn round_metrics(results: &[WorkerFitResult]) -> RoundMetrics {
    let workers: Vec<WorkerMetrics> = results
        .iter()
        .filter_map(|result| {
            result.metrics.map(|metrics| WorkerMetrics {
                addr: result.addr,
                loss: metrics.loss,
                num_examples: metrics.num_examples,
            })
        })
        .collect();

    let total_examples: u64 = workers.iter().map(|w| w.num_examples).sum();

    let loss = if total_examples > 0 {
        // Both operands are sample/loss magnitudes far below f64's 52-bit
        // mantissa for any dataset this trains on.
        #[allow(clippy::cast_precision_loss)]
        let weighted: f64 = workers
            .iter()
            .map(|w| f64::from(w.loss) * w.num_examples as f64)
            .sum();
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let mean = (weighted / total_examples as f64) as f32;
        mean
    } else if workers.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let mean = workers.iter().map(|w| w.loss).sum::<f32>() / workers.len() as f32;
        mean
    };

    RoundMetrics {
        loss,
        num_examples: total_examples,
        workers,
    }
}

/// Turn a round's raw per-worker results into `average_weights` inputs,
/// weighting each worker's contribution by its reported sample count.
///
/// Falls back to an equal weight of 1 for every contribution -- an
/// unweighted mean, matching plain `FedAvg` -- if any worker didn't report
/// metrics, or the reported sample counts summed to zero. Mixing real
/// sample weights with a stand-in for a missing one would silently distort
/// the average rather than just degrade it, so the round backs off
/// uniformly instead.
fn contributions(
    job_id: Uuid,
    results: Vec<WorkerFitResult>,
) -> Vec<(HashMap<String, Tensor>, u64)> {
    let total_examples = results
        .iter()
        .map(|result| result.metrics.map(|metrics| metrics.num_examples))
        .sum::<Option<u64>>();

    match total_examples {
        Some(total) if total > 0 => results
            .into_iter()
            .map(|result| {
                let num_examples = result
                    .metrics
                    .expect("every result has metrics: checked above")
                    .num_examples;
                (result.weights, num_examples)
            })
            .collect(),
        Some(_) => {
            warn!(
                job_id = %job_id,
                "every worker reported zero training examples this round; falling back to an unweighted average"
            );
            results
                .into_iter()
                .map(|result| (result.weights, 1))
                .collect()
        }
        None => {
            warn!(
                job_id = %job_id,
                "not every worker reported training metrics this round; falling back to an unweighted average"
            );
            results
                .into_iter()
                .map(|result| (result.weights, 1))
                .collect()
        }
    }
}

fn average_weights(
    contributions: Vec<(HashMap<String, Tensor>, u64)>,
) -> Result<HashMap<String, Tensor>, StateError> {
    if contributions.is_empty() {
        return Err(StateError::NoResults);
    }

    // Sum tensors per key two ways: weighted by each contribution's sample
    // count (the normal path), and plainly alongside a contributor count
    // (the fallback for a key whose contributing weight sums to zero, e.g.
    // every worker that reported this key legitimately reported zero local
    // examples -- dividing by that zero weight makes no sense, but
    // discarding the actual reported values in favor of a zero tensor is
    // worse, so fall back to an unweighted mean of them instead).
    let mut sums: HashMap<String, (Tensor, u64, Tensor, u64)> = HashMap::new();

    for (tensor_map, weight) in contributions {
        // `weight` is a sample count, far below f64's 52-bit mantissa.
        #[allow(clippy::cast_precision_loss)]
        let weight_f64 = weight as f64;

        for (name, tensor) in tensor_map {
            let scaled = (weight_f64 * &tensor)?;
            match sums.remove(&name) {
                Some((weighted, weight_total, plain, count)) => {
                    sums.insert(
                        name,
                        (
                            (weighted + scaled)?,
                            weight_total + weight,
                            (plain + &tensor)?,
                            count + 1,
                        ),
                    );
                }
                None => {
                    sums.insert(name, (scaled, weight, tensor, 1));
                }
            };
        }
    }

    let mut result = HashMap::with_capacity(sums.len());
    for (name, (weighted, weight_total, plain, count)) in sums {
        // `weight_total` and `count` are sample/contributor counts, far
        // below f64's 52-bit mantissa.
        #[allow(clippy::cast_precision_loss)]
        let (numerator, divisor) = if weight_total > 0 {
            (weighted, weight_total as f64)
        } else {
            (plain, count as f64)
        };
        result.insert(name, (divisor.recip() * numerator)?);
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
        state::{FitMetrics, State, WorkerFitResult, WorkerUpdate},
    };

    /// Wrap each map with a weight of 1, matching plain (unweighted)
    /// federated averaging -- the behavior these older tests assert.
    fn equal_weight(maps: Vec<HashMap<String, Tensor>>) -> Vec<(HashMap<String, Tensor>, u64)> {
        maps.into_iter().map(|map| (map, 1)).collect()
    }

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

        let result = average_weights(equal_weight(weights_per_worker)).unwrap();

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

        let result = average_weights(equal_weight(weights_per_worker)).unwrap();

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

        let result = average_weights(equal_weight(vec![map1, map2])).unwrap();

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

        let err = average_weights(equal_weight(vec![map1, map2])).unwrap_err();

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

        let result = average_weights(equal_weight(vec![map1, map2, map3])).unwrap();

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

        let result = average_weights(equal_weight(vec![map])).unwrap();

        assert_eq!(result.get("a").unwrap().dtype(), DType::F32);
    }

    #[test]
    fn test_average_weights_is_sample_weighted() {
        let dev = Device::Cpu;

        let mut map1 = HashMap::new();
        map1.insert("a".to_string(), Tensor::new(vec![1.0], &dev).unwrap());

        let mut map2 = HashMap::new();
        map2.insert("a".to_string(), Tensor::new(vec![2.0], &dev).unwrap());

        // Weighted 1:3, so the average leans toward the second contribution:
        // (1.0 * 1 + 2.0 * 3) / 4 = 1.75, not the unweighted 1.5.
        let result = average_weights(vec![(map1, 1), (map2, 3)]).unwrap();

        assert_eq!(
            result.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.75]
        );
    }

    #[test]
    fn test_average_weights_falls_back_to_an_unweighted_mean_for_a_zero_weight_key() {
        let dev = Device::Cpu;

        // "a" is contributed by two workers with a real, nonzero weight
        // each. "b" is contributed only by a third worker reporting weight
        // 0 (e.g. it legitimately had zero local examples this round, but
        // still sent a tensor). "a" should still be weighted normally; "b"
        // can't be, so it should fall back to a plain mean of the values
        // actually reported for it rather than being silently zeroed.
        let mut map1 = HashMap::new();
        map1.insert("a".to_string(), Tensor::new(vec![1.0], &dev).unwrap());

        let mut map2 = HashMap::new();
        map2.insert("a".to_string(), Tensor::new(vec![2.0], &dev).unwrap());

        let mut map3 = HashMap::new();
        map3.insert("b".to_string(), Tensor::new(vec![5.0], &dev).unwrap());

        let result = average_weights(vec![(map1, 1), (map2, 3), (map3, 0)]).unwrap();

        // "a": sample-weighted as usual: (1.0 * 1 + 2.0 * 3) / 4 = 1.75.
        assert_eq!(
            result.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.75]
        );
        // "b": only one contributor, at weight 0 -- falls back to that
        // contributor's own value instead of being zeroed.
        assert_eq!(
            result.get("b").unwrap().to_vec1::<f64>().unwrap(),
            vec![5.0]
        );
    }

    fn worker_fit_result(addr: &str, value: f64, metrics: Option<FitMetrics>) -> WorkerFitResult {
        WorkerFitResult {
            addr: addr.parse().unwrap(),
            weights: tensor_map(value),
            metrics,
        }
    }

    #[test]
    fn contributions_weights_by_reported_sample_count() {
        let results = vec![
            worker_fit_result(
                "127.0.0.1:1",
                1.0,
                Some(FitMetrics {
                    loss: 0.1,
                    num_examples: 10,
                }),
            ),
            worker_fit_result(
                "127.0.0.1:2",
                2.0,
                Some(FitMetrics {
                    loss: 0.2,
                    num_examples: 30,
                }),
            ),
        ];

        let weighted = contributions(Uuid::new_v4(), results);

        let weights: Vec<u64> = weighted.iter().map(|(_, weight)| *weight).collect();
        assert_eq!(weights, vec![10, 30]);
    }

    #[test]
    fn contributions_falls_back_to_unweighted_when_metrics_are_missing() {
        let results = vec![
            worker_fit_result(
                "127.0.0.1:1",
                1.0,
                Some(FitMetrics {
                    loss: 0.1,
                    num_examples: 10,
                }),
            ),
            // This worker didn't report metrics at all.
            worker_fit_result("127.0.0.1:2", 2.0, None),
        ];

        let weighted = contributions(Uuid::new_v4(), results);

        let weights: Vec<u64> = weighted.iter().map(|(_, weight)| *weight).collect();
        assert_eq!(weights, vec![1, 1]);
    }

    #[test]
    fn contributions_falls_back_to_unweighted_when_total_examples_is_zero() {
        let results = vec![
            worker_fit_result(
                "127.0.0.1:1",
                1.0,
                Some(FitMetrics {
                    loss: 0.1,
                    num_examples: 0,
                }),
            ),
            worker_fit_result(
                "127.0.0.1:2",
                2.0,
                Some(FitMetrics {
                    loss: 0.2,
                    num_examples: 0,
                }),
            ),
        ];

        let weighted = contributions(Uuid::new_v4(), results);

        let weights: Vec<u64> = weighted.iter().map(|(_, weight)| *weight).collect();
        assert_eq!(weights, vec![1, 1]);
    }

    #[test]
    fn round_metrics_is_sample_weighted_and_lists_every_reporting_worker() {
        let addr_a: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:2".parse().unwrap();
        let results = vec![
            worker_fit_result(
                "127.0.0.1:1",
                1.0,
                Some(FitMetrics {
                    loss: 1.0,
                    num_examples: 10,
                }),
            ),
            worker_fit_result(
                "127.0.0.1:2",
                2.0,
                Some(FitMetrics {
                    loss: 3.0,
                    num_examples: 30,
                }),
            ),
        ];

        let metrics = round_metrics(&results);

        // (1.0 * 10 + 3.0 * 30) / 40 = 2.5, not the unweighted 2.0.
        assert_eq!(metrics.loss, 2.5);
        assert_eq!(metrics.num_examples, 40);
        assert_eq!(metrics.workers.len(), 2);
        assert_eq!(metrics.workers[0].addr, addr_a);
        assert_eq!(metrics.workers[1].addr, addr_b);
    }

    #[test]
    fn round_metrics_omits_workers_that_reported_no_metrics() {
        let reporting: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let results = vec![
            worker_fit_result(
                "127.0.0.1:1",
                1.0,
                Some(FitMetrics {
                    loss: 1.0,
                    num_examples: 10,
                }),
            ),
            worker_fit_result("127.0.0.1:2", 2.0, None),
        ];

        let metrics = round_metrics(&results);

        assert_eq!(metrics.workers.len(), 1);
        assert_eq!(metrics.workers[0].addr, reporting);
        assert_eq!(metrics.loss, 1.0);
        assert_eq!(metrics.num_examples, 10);
    }

    #[test]
    fn round_metrics_falls_back_to_a_plain_mean_when_total_examples_is_zero() {
        let results = vec![
            worker_fit_result(
                "127.0.0.1:1",
                1.0,
                Some(FitMetrics {
                    loss: 1.0,
                    num_examples: 0,
                }),
            ),
            worker_fit_result(
                "127.0.0.1:2",
                2.0,
                Some(FitMetrics {
                    loss: 3.0,
                    num_examples: 0,
                }),
            ),
        ];

        let metrics = round_metrics(&results);

        assert_eq!(metrics.loss, 2.0);
        assert_eq!(metrics.num_examples, 0);
    }

    fn tensor_map(value: f64) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(
            "a".to_string(),
            Tensor::new(vec![value, value], &Device::Cpu).unwrap(),
        );
        map
    }

    fn worker_update(value: f64) -> WorkerUpdate {
        WorkerUpdate {
            weights: tensor_map(value),
            metrics: None,
        }
    }

    fn worker_update_with_metrics(value: f64, loss: f32, num_examples: u64) -> WorkerUpdate {
        WorkerUpdate {
            weights: tensor_map(value),
            metrics: Some(FitMetrics { loss, num_examples }),
        }
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
                .set_fit_result(job_id, addr, worker_update(1.0))
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
        // No training happened for the round-0 update.
        assert!(update.metrics.is_none());
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
            // 1 WeightsRequest + 2 FitRequests, one worker each time. The
            // WeightsRequest's metrics are ignored (there's no training
            // behind it); the FitRequests' are what round 1 and 2's
            // metrics come from.
            for _ in 0..3 {
                let message = receiver.recv().await.expect("a request");
                assert_eq!(job_id_of(&message), job_id);
                responder_state
                    .set_fit_result(job_id, addr, worker_update_with_metrics(2.0, 0.5, 100))
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

        // A single worker: averaging is the identity, each round, and the
        // round's aggregate metrics are exactly that worker's own.
        let first = updates_rx.recv().await.expect("round 1 update");
        assert_eq!(first.round, 1);
        assert_eq!(
            first.weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![2.0, 2.0]
        );
        let first_metrics = first.metrics.as_ref().expect("round 1 has metrics");
        assert_eq!(first_metrics.loss, 0.5);
        assert_eq!(first_metrics.num_examples, 100);
        assert_eq!(first_metrics.workers.len(), 1);
        assert_eq!(first_metrics.workers[0].loss, 0.5);
        assert_eq!(first_metrics.workers[0].num_examples, 100);
        assert_eq!(first_metrics.workers[0].addr, addr);

        let second = updates_rx.recv().await.expect("round 2 update");
        assert_eq!(second.round, 2);
        assert_eq!(second.job_id, first.job_id);
        assert_eq!(
            second.weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![2.0, 2.0]
        );
        // Every round is labeled with the same worker address.
        assert_eq!(second.metrics.unwrap().workers[0].addr, addr);

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
                .set_fit_result(job_id, addr, worker_update(1.0))
                .await
                .unwrap();

            drop(updates_rx);

            let message = receiver.recv().await.expect("round 1's FitRequest");
            assert_eq!(job_id_of(&message), job_id);
            responder_state
                .set_fit_result(job_id, addr, worker_update(1.0))
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
