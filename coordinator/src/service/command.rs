use std::pin::Pin;

use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use crate::{
    candlefl::{self, TrainRequest, TrainResponse, command_server::Command},
    state::State,
    strategy::{RoundMetrics, RoundUpdate, Runner, Strategy},
};

pub struct CommandService<S> {
    runner: Runner<S>,
}

impl<S: Strategy> CommandService<S> {
    pub fn new(strategy: S, state: State) -> Self {
        Self {
            runner: Runner::new(strategy, state),
        }
    }
}

type TrainResponseStream = Pin<Box<dyn Stream<Item = Result<TrainResponse, Status>> + Send>>;

#[tonic::async_trait]
impl<S: Strategy> Command for CommandService<S> {
    type TrainStream = TrainResponseStream;

    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponseStream>, Status> {
        let request = request.into_inner();

        // 'rounds' is u64 over the wire; clamp rather than wrap on 32-bit
        // targets where 'usize' is narrower.
        let rounds = usize::try_from(request.rounds).unwrap_or(usize::MAX);

        let updates = self
            .runner
            .start(rounds)
            .map_err(|e| Status::failed_precondition(e.to_string()))?;

        // The error, if any, always arrives after every successful update:
        // 'fit' only returns (ending the updates side of the run) after it
        // has sent everything it's going to send.
        let responses = updates.map(|item| match item {
            Ok(update) => to_response(update),
            Err(e) => Err(Status::internal(format!("failed to train model: {e}"))),
        });

        Ok(Response::new(Box::pin(responses) as TrainResponseStream))
    }
}

/// Serialize one round's aggregate into a `TrainResponse`.
///
/// A serialization failure here ends the client's stream with this error;
/// the strategy task driving `fit` only notices once its next `send` hits
/// the now-closed channel, at which point it stops the same as it would for
/// any other disconnected client.
fn to_response(update: RoundUpdate) -> Result<TrainResponse, Status> {
    let weights = safetensors::serialize(update.weights, None)
        .map_err(|e| Status::internal(format!("invalid weights: {e}")))?;

    Ok(TrainResponse {
        job_id: update.job_id.to_string(),
        round: update.round,
        weights,
        metrics: update.metrics.map(to_metrics),
    })
}

fn to_metrics(metrics: RoundMetrics) -> candlefl::RoundMetrics {
    candlefl::RoundMetrics {
        loss: metrics.loss,
        num_examples: metrics.num_examples,
        workers: metrics
            .workers
            .into_iter()
            .map(|worker| candlefl::WorkerMetrics {
                address: worker.addr.to_string(),
                loss: worker.loss,
                num_examples: worker.num_examples,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use candle_core::{Device, Tensor};
    use tokio::sync::{Notify, mpsc};

    use super::*;
    use crate::{
        state::Job,
        strategy::{Error, WorkerMetrics},
    };

    /// A test double that emits `rounds` successful updates, sharing `job`'s
    /// id, and then either succeeds or fails depending on `fails`.
    #[derive(Clone, Copy)]
    struct ScriptedStrategy {
        rounds: u64,
        fails: bool,
    }

    impl Strategy for ScriptedStrategy {
        async fn fit(
            &self,
            job: &Job<'_>,
            _num_rounds: usize,
            updates: mpsc::Sender<RoundUpdate>,
        ) -> Result<(), Error> {
            for round in 1..=self.rounds {
                let mut weights = HashMap::new();
                #[allow(clippy::cast_precision_loss)]
                let value = round as f64;
                weights.insert(
                    "a".to_string(),
                    Tensor::new(vec![value, value], &Device::Cpu).unwrap(),
                );

                // A fixed, recognizable metrics value keyed off 'round', so
                // a test can assert it survived the trip through
                // 'to_response' unchanged.
                #[allow(clippy::cast_precision_loss)]
                let loss = round as f32;
                let metrics = RoundMetrics {
                    loss,
                    num_examples: round * 10,
                    workers: vec![WorkerMetrics {
                        addr: "127.0.0.1:1".parse().unwrap(),
                        loss,
                        num_examples: round * 10,
                    }],
                };

                if updates
                    .send(RoundUpdate {
                        job_id: job.id(),
                        round,
                        weights,
                        metrics: Some(metrics),
                    })
                    .await
                    .is_err()
                {
                    return Ok(());
                }
            }

            if self.fails {
                Err(Error::State(crate::state::Error::NoResults))
            } else {
                Ok(())
            }
        }
    }

    async fn collect(stream: TrainResponseStream) -> Vec<Result<TrainResponse, Status>> {
        stream.collect().await
    }

    #[tokio::test]
    async fn train_streams_each_round_in_order() {
        let service = CommandService::new(
            ScriptedStrategy {
                rounds: 3,
                fails: false,
            },
            State::new(),
        );

        let response = service
            .train(Request::new(TrainRequest { rounds: 3 }))
            .await
            .unwrap();

        let responses = collect(response.into_inner()).await;

        assert_eq!(responses.len(), 3);

        let job_id = responses[0].as_ref().unwrap().job_id.clone();
        for (i, response) in responses.iter().enumerate() {
            let response = response.as_ref().unwrap();
            assert_eq!(response.job_id, job_id);
            assert_eq!(response.round, u64::try_from(i + 1).unwrap());

            let weights =
                candle_core::safetensors::load_buffer(&response.weights, &Device::Cpu).unwrap();
            let expected = u64::try_from(i + 1).unwrap();
            #[allow(clippy::cast_precision_loss)]
            let expected_f64 = expected as f64;
            assert_eq!(
                weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
                vec![expected_f64, expected_f64]
            );

            let metrics = response.metrics.as_ref().expect("every round has metrics");
            #[allow(clippy::cast_precision_loss)]
            let expected_loss = expected as f32;
            assert_eq!(metrics.loss, expected_loss);
            assert_eq!(metrics.num_examples, expected * 10);
            assert_eq!(metrics.workers.len(), 1);
            assert_eq!(metrics.workers[0].loss, expected_loss);
            assert_eq!(metrics.workers[0].num_examples, expected * 10);
            assert_eq!(metrics.workers[0].address, "127.0.0.1:1");
        }
    }

    #[test]
    fn to_response_omits_metrics_for_the_round_0_update() {
        let response = to_response(RoundUpdate {
            job_id: uuid::Uuid::new_v4(),
            round: 0,
            weights: HashMap::new(),
            metrics: None,
        })
        .unwrap();

        assert_eq!(response.round, 0);
        assert!(response.metrics.is_none());
    }

    #[tokio::test]
    async fn train_maps_a_strategy_error_to_an_internal_status() {
        let service = CommandService::new(
            ScriptedStrategy {
                rounds: 0,
                fails: true,
            },
            State::new(),
        );

        let response = service
            .train(Request::new(TrainRequest { rounds: 1 }))
            .await
            .unwrap();

        let responses = collect(response.into_inner()).await;

        assert_eq!(responses.len(), 1);
        assert_eq!(
            responses[0].as_ref().unwrap_err().code(),
            tonic::Code::Internal
        );
    }

    #[tokio::test]
    async fn train_streams_rounds_before_reporting_a_failure() {
        let service = CommandService::new(
            ScriptedStrategy {
                rounds: 2,
                fails: true,
            },
            State::new(),
        );

        let response = service
            .train(Request::new(TrainRequest { rounds: 2 }))
            .await
            .unwrap();

        let responses = collect(response.into_inner()).await;

        assert_eq!(responses.len(), 3);
        assert!(responses[0].is_ok());
        assert!(responses[1].is_ok());
        assert_eq!(
            responses[2].as_ref().unwrap_err().code(),
            tonic::Code::Internal
        );
    }

    /// A test double that parks until `release` fires before completing,
    /// so a test can hold a run open for as long as it needs to.
    struct BlockingStrategy {
        release: Arc<Notify>,
    }

    impl Strategy for BlockingStrategy {
        async fn fit(
            &self,
            _job: &Job<'_>,
            _num_rounds: usize,
            _updates: mpsc::Sender<RoundUpdate>,
        ) -> Result<(), Error> {
            self.release.notified().await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn train_rejects_a_second_call_while_a_run_is_in_progress() {
        let release = Arc::new(Notify::new());
        let service = CommandService::new(
            BlockingStrategy {
                release: Arc::clone(&release),
            },
            State::new(),
        );

        let first = service
            .train(Request::new(TrainRequest { rounds: 1 }))
            .await
            .unwrap();

        // Give the spawned fit task a chance to start (and park on
        // 'release') before checking that a second call is rejected.
        tokio::task::yield_now().await;

        let second = service
            .train(Request::new(TrainRequest { rounds: 1 }))
            .await;
        let status = second.err().unwrap();
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);

        release.notify_one();
        let responses = collect(first.into_inner()).await;
        assert!(responses.is_empty());

        // The slot is released once the run ends, so a further call
        // succeeds.
        let third = service
            .train(Request::new(TrainRequest { rounds: 1 }))
            .await;
        assert!(third.is_ok());
    }
}
