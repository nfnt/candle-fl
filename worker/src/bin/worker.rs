use candle_core::Device;
use candle_nn::VarMap;
use clap::Parser;
use safetensors::{SafeTensorError, SafeTensors};
use tokio::task;
use tonic::transport::{Channel, Uri};
use tracing::{debug, info};
use worker::{
    candlefl::{
        self, FitResponse, WeightsResponse, WorkerMessage, publisher_client::PublisherClient,
        subscriber_client::SubscriberClient, worker_message,
    },
    ml::{prepare_data, prepare_model, train},
};

#[derive(Parser)]
#[command(version)]
struct Args {
    #[arg(long, default_value_t = String::from("[::1]:50051"))]
    addr: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let uri: Uri = format!("http://{}", args.addr).parse()?;

    let channel = Channel::builder(uri.clone())
        .user_agent(format!("candle-fl-worker/{}", env!("CARGO_PKG_VERSION")))?
        .connect()
        .await?;
    let mut stream = SubscriberClient::new(channel.clone())
        .subscribe(())
        .await?
        .into_inner();

    info!(%uri, "connected to coordinator");

    let dev = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };

    // In production code we need to handle stream disconnections by retrying
    // if a connection is dropped. This isn't done here.
    while let Some(message) = stream.message().await? {
        if let Some(message) = message.message {
            match message {
                candlefl::coordinator_message::Message::WeightsRequest(weights_request) => {
                    debug!(job_id = weights_request.job_id, "received WeightsRequest");

                    let channel = channel.clone();

                    task::spawn(async move {
                        // This is a blocking operation, so we'll offload it
                        let result = task::spawn_blocking(move || {
                            let dev = Device::Cpu;
                            prepare_model(&dev).map(|(v, _)| v)
                        })
                        .await
                        .expect("task doesn't panic");

                        let result = match result {
                            Ok(result) => result,
                            Err(e) => {
                                debug!(
                                    job_id = weights_request.job_id,
                                    "failed to prepare model: {}", e
                                );
                                return;
                            }
                        };

                        let weights = match serialize(&result) {
                            Ok(weights) => weights,
                            Err(e) => {
                                debug!(
                                    job_id = weights_request.job_id,
                                    "failed to serialize weights: {}", e
                                );
                                return;
                            }
                        };

                        match PublisherClient::new(channel)
                            .publish(WorkerMessage {
                                message: Some(worker_message::Message::WeightsResponse(
                                    WeightsResponse {
                                        job_id: weights_request.job_id.clone(),
                                        weights,
                                    },
                                )),
                            })
                            .await
                        {
                            Ok(_) => {
                                debug!(job_id = weights_request.job_id, "sent WeightsResponse");
                            }
                            Err(status) => {
                                debug!(
                                    job_id = weights_request.job_id,
                                    "failed to send WeightsResponse: {}", status
                                );
                            }
                        }
                    });
                }
                candlefl::coordinator_message::Message::FitRequest(fit_request) => {
                    debug!(job_id = fit_request.job_id, "received FitRequest");

                    let channel = channel.clone();

                    task::spawn({
                        let dev = dev.clone();
                        async move {
                            // This is a blocking operation, so we'll offload it
                            let result = task::spawn_blocking(move || {
                                let data = prepare_data(&dev)?;

                                train(&deserialize(&fit_request.weights)?, &data, &dev)
                            })
                            .await
                            .expect("task doesn't panic");

                            let result = match result {
                                Ok(result) => result,
                                Err(e) => {
                                    debug!(
                                        job_id = fit_request.job_id,
                                        "failed to prepare model: {}", e
                                    );
                                    return;
                                }
                            };

                            let weights = match serialize(&result) {
                                Ok(weights) => weights,
                                Err(e) => {
                                    debug!(
                                        job_id = fit_request.job_id,
                                        "failed to serialize weights: {}", e
                                    );
                                    return;
                                }
                            };

                            match PublisherClient::new(channel)
                                .publish(WorkerMessage {
                                    message: Some(worker_message::Message::FitResponse(
                                        FitResponse {
                                            job_id: fit_request.job_id.clone(),
                                            weights,
                                        },
                                    )),
                                })
                                .await
                            {
                                Ok(_) => {
                                    debug!(job_id = fit_request.job_id, "sent FitResponse");
                                }
                                Err(status) => {
                                    debug!(
                                        job_id = fit_request.job_id,
                                        "failed to send FitResponse: {}", status
                                    );
                                }
                            }
                        }
                    });
                }
            }
        }
    }

    Ok(())
}

fn serialize(varmap: &VarMap) -> Result<Vec<u8>, Box<dyn std::error::Error + '_>> {
    let tensor_data = varmap.data().lock()?;

    let data = tensor_data.iter().map(|(k, v)| (k, v.as_tensor()));

    Ok(safetensors::serialize(data, None)?)
}

fn deserialize(data: &[u8]) -> Result<SafeTensors<'_>, SafeTensorError> {
    safetensors::SafeTensors::deserialize(data)
}
