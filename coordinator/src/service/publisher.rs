use std::collections::HashMap;

use candle_core::{safetensors::load_buffer, Device, Tensor};
use tonic::{Request, Response, Status};
use tracing::debug;
use uuid::Uuid;

use crate::{
    candlefl::{publisher_server::Publisher, worker_message, WorkerMessage},
    state::State,
};

pub struct PublisherService {
    state: State,
}

impl PublisherService {
    pub fn new(state: State) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl Publisher for PublisherService {
    async fn publish(&self, request: Request<WorkerMessage>) -> Result<Response<()>, Status> {
        let addr = request.remote_addr().unwrap();

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
                        .set_fit_result(job_id, addr, weights)
                        .await
                        .unwrap();
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

                    self.state
                        .set_fit_result(job_id, addr, weights)
                        .await
                        .unwrap();
                }
            }
        }

        Ok(Response::new(()))
    }
}

fn deserialize(data: &[u8]) -> Result<HashMap<String, Tensor>, candle_core::Error> {
    load_buffer(data, &Device::Cpu)
}
