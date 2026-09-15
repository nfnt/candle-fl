use std::collections::HashMap;

use candle_core::{Device, Tensor, safetensors::load_buffer};
use tonic::{Request, Response, Status};
use tracing::debug;
use uuid::Uuid;

use crate::{
    candlefl::{self, WorkerMessage, publisher_server::Publisher, worker_message},
    state::{FitMetrics, State, WorkerUpdate},
};

pub struct PublisherService {
    state: State,
}

impl PublisherService {
    #[must_use]
    pub const fn new(state: State) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl Publisher for PublisherService {
    async fn publish(&self, request: Request<WorkerMessage>) -> Result<Response<()>, Status> {
        let addr = request
            .remote_addr()
            .ok_or_else(|| Status::internal("missing remote address"))?;

        if let Some(message) = request.into_inner().message {
            match message {
                worker_message::Message::WeightsResponse(weights_response) => {
                    debug!(
                        addr = addr.to_string(),
                        job_id = weights_response.job_id,
                        "received WeightsResponse"
                    );
                    let job_id =
                        Uuid::parse_str(weights_response.job_id.as_str()).map_err(|_| {
                            Status::invalid_argument(format!(
                                "invalid job ID {}",
                                weights_response.job_id.as_str()
                            ))
                        })?;

                    let weights = deserialize(&weights_response.weights)
                        .map_err(|e| Status::invalid_argument(format!("invalid weights: {e}")))?;

                    self.state
                        .set_fit_result(
                            job_id,
                            addr,
                            WorkerUpdate {
                                weights,
                                metrics: None,
                            },
                        )
                        .await
                        .map_err(|e| Status::from_error(Box::new(e)))?;
                }
                worker_message::Message::FitResponse(fit_response) => {
                    debug!(
                        addr = addr.to_string(),
                        job_id = fit_response.job_id,
                        "received FitResponse"
                    );
                    let job_id = Uuid::parse_str(fit_response.job_id.as_str()).map_err(|_| {
                        Status::invalid_argument(format!(
                            "invalid job ID {}",
                            fit_response.job_id.as_str()
                        ))
                    })?;

                    let weights = deserialize(&fit_response.weights)
                        .map_err(|e| Status::invalid_argument(format!("invalid weights: {e}")))?;

                    let metrics = to_fit_metrics(fit_response.metrics)?;

                    self.state
                        .set_fit_result(job_id, addr, WorkerUpdate { weights, metrics })
                        .await
                        .map_err(|e| Status::from_error(Box::new(e)))?;
                }
            }
        }

        Ok(Response::new(()))
    }
}

fn deserialize(data: &[u8]) -> Result<HashMap<String, Tensor>, candle_core::Error> {
    load_buffer(data, &Device::Cpu)
}

/// Validate and convert a `FitResponse`'s wire metrics.
///
/// Rejects a non-finite or negative loss as invalid rather than letting it
/// through: this is untrusted network input, and a `NaN` in particular
/// would silently poison every average it's folded into downstream.
fn to_fit_metrics(metrics: Option<candlefl::FitMetrics>) -> Result<Option<FitMetrics>, Status> {
    let Some(metrics) = metrics else {
        return Ok(None);
    };

    if !metrics.loss.is_finite() || metrics.loss < 0.0 {
        return Err(Status::invalid_argument(format!(
            "invalid loss {}: must be finite and non-negative",
            metrics.loss
        )));
    }

    Ok(Some(FitMetrics {
        loss: metrics.loss,
        num_examples: metrics.num_examples,
    }))
}

#[cfg(test)]
mod tests {
    use tonic::transport::server::TcpConnectInfo;

    use super::*;
    use crate::candlefl::{FitResponse, WeightsResponse};

    fn tensor_map(value: f64) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(
            "a".to_string(),
            Tensor::new(vec![value, value], &Device::Cpu).unwrap(),
        );
        map
    }

    fn with_remote_addr<T>(message: T, addr: &str) -> Request<T> {
        let mut request = Request::new(message);
        request.extensions_mut().insert(TcpConnectInfo {
            local_addr: None,
            remote_addr: Some(addr.parse().unwrap()),
        });
        request
    }

    #[test]
    fn deserialize_round_trips_with_serialize() {
        let weights = tensor_map(1.0);
        let bytes = safetensors::serialize(&weights, None).unwrap();

        let result = deserialize(&bytes).unwrap();

        assert_eq!(
            result.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn deserialize_rejects_malformed_buffer() {
        let err = deserialize(b"not a safetensors buffer").unwrap_err();

        assert!(matches!(err, candle_core::Error::SafeTensor(_)));
    }

    #[tokio::test]
    async fn publish_missing_remote_addr_errors_instead_of_panicking() {
        let service = PublisherService::new(State::new());
        let request = Request::new(WorkerMessage {
            message: Some(worker_message::Message::WeightsResponse(WeightsResponse {
                job_id: Uuid::new_v4().to_string(),
                weights: vec![],
            })),
        });

        let status = service.publish(request).await.unwrap_err();

        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[tokio::test]
    async fn publish_invalid_job_id_is_invalid_argument() {
        let service = PublisherService::new(State::new());
        let request = with_remote_addr(
            WorkerMessage {
                message: Some(worker_message::Message::FitResponse(FitResponse {
                    job_id: "not-a-uuid".to_string(),
                    weights: vec![],
                    metrics: None,
                })),
            },
            "127.0.0.1:1",
        );

        let status = service.publish(request).await.unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn publish_invalid_weights_is_invalid_argument() {
        let service = PublisherService::new(State::new());
        let request = with_remote_addr(
            WorkerMessage {
                message: Some(worker_message::Message::WeightsResponse(WeightsResponse {
                    job_id: Uuid::new_v4().to_string(),
                    weights: b"not safetensors".to_vec(),
                })),
            },
            "127.0.0.1:1",
        );

        let status = service.publish(request).await.unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn publish_unknown_job_surfaces_as_status() {
        // The job ID parses fine but was never created, exercising the
        // 'set_fit_result' -> 'Status::from_error' path.
        let service = PublisherService::new(State::new());
        let request = with_remote_addr(
            WorkerMessage {
                message: Some(worker_message::Message::FitResponse(FitResponse {
                    job_id: Uuid::new_v4().to_string(),
                    weights: safetensors::serialize(&tensor_map(1.0), None).unwrap(),
                    metrics: None,
                })),
            },
            "127.0.0.1:1",
        );

        let status = service.publish(request).await.unwrap_err();

        assert!(status.message().contains("unknown job"));
    }

    #[tokio::test]
    async fn publish_nan_loss_is_invalid_argument() {
        let service = PublisherService::new(State::new());
        let request = with_remote_addr(
            WorkerMessage {
                message: Some(worker_message::Message::FitResponse(FitResponse {
                    job_id: Uuid::new_v4().to_string(),
                    weights: safetensors::serialize(&tensor_map(1.0), None).unwrap(),
                    metrics: Some(candlefl::FitMetrics {
                        loss: f32::NAN,
                        num_examples: 1,
                    }),
                })),
            },
            "127.0.0.1:1",
        );

        let status = service.publish(request).await.unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn publish_negative_loss_is_invalid_argument() {
        let service = PublisherService::new(State::new());
        let request = with_remote_addr(
            WorkerMessage {
                message: Some(worker_message::Message::FitResponse(FitResponse {
                    job_id: Uuid::new_v4().to_string(),
                    weights: safetensors::serialize(&tensor_map(1.0), None).unwrap(),
                    metrics: Some(candlefl::FitMetrics {
                        loss: -1.0,
                        num_examples: 1,
                    }),
                })),
            },
            "127.0.0.1:1",
        );

        let status = service.publish(request).await.unwrap_err();

        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }
}
