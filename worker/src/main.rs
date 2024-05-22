use candle_core::{Device, Error};
use candle_nn::VarMap;
use clap::Parser;
use safetensors::{SafeTensorError, SafeTensors};
use tokio::{sync::oneshot, task};
use tonic::transport::{Channel, Uri};
use tracing::{debug, info};

use crate::candlefl::{
    publisher_client::PublisherClient, subscriber_client::SubscriberClient, worker_message,
    FitResponse, WeightsResponse, WorkerMessage,
};
use crate::ml::{prepare_data, prepare_model, train};

mod candlefl {
    tonic::include_proto!("candlefl.v1");
}
mod ml;

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
        .user_agent("candle-fl-worker/0.1.0")?
        .connect()
        .await?;
    let mut subscriber_client = SubscriberClient::new(channel.clone());

    let mut stream = subscriber_client.subscribe(()).await?.into_inner();

    info!(uri = uri.to_string(), "connected to coordinator");

    // In production code we need to handle stream disconnections by retrying
    // if a connection is dropped. This isn't done here.
    while let Some(message) = stream.message().await? {
        if let Some(message) = message.message {
            match message {
                candlefl::coordinator_message::Message::WeightsRequest(weights_request) => {
                    debug!(job_id = weights_request.job_id, "received WeightsRequest");

                    let channel = channel.clone();

                    let (sender, receiver) = oneshot::channel();

                    // This is a blocking operation, so we'll offload it
                    task::spawn_blocking(move || {
                        let result = || -> Result<_, Error> {
                            let dev = Device::Cpu;
                            let (varmap, _) = prepare_model(&dev)?;

                            Ok(varmap)
                        }();

                        let _ = sender.send(result);
                    });

                    task::spawn(async move {
                        let result = receiver.await.unwrap();

                        let mut publisher_client = PublisherClient::new(channel);

                        publisher_client
                            .publish(WorkerMessage {
                                message: Some(worker_message::Message::WeightsResponse(
                                    WeightsResponse {
                                        job_id: weights_request.job_id.clone(),
                                        weights: serialize(&result.unwrap()).unwrap(),
                                    },
                                )),
                            })
                            .await
                            .unwrap();

                        debug!(job_id = weights_request.job_id, "sent WeightsResponse");
                    });
                }
                candlefl::coordinator_message::Message::FitRequest(fit_request) => {
                    debug!(job_id = fit_request.job_id, "received FitRequest");

                    let channel = channel.clone();

                    let (sender, receiver) = oneshot::channel();

                    // This is a blocking operation, so we'll offload it
                    task::spawn_blocking(move || {
                        let result = || -> Result<_, Error> {
                            let dev = Device::Cpu;
                            let data = prepare_data(&dev)?;

                            train(&deserialize(&fit_request.weights)?, &data, &dev)
                        }();

                        let _ = sender.send(result);
                    });

                    task::spawn(async move {
                        let result = receiver.await.unwrap();

                        let mut publisher_client = PublisherClient::new(channel);

                        publisher_client
                            .publish(WorkerMessage {
                                message: Some(worker_message::Message::FitResponse(FitResponse {
                                    job_id: fit_request.job_id.clone(),
                                    weights: serialize(&result.unwrap()).unwrap(),
                                })),
                            })
                            .await
                            .unwrap();

                        debug!(job_id = fit_request.job_id, "sent FitResponse");
                    });
                }
            }
        }
    }

    Ok(())
}

fn serialize(varmap: &VarMap) -> Result<Vec<u8>, SafeTensorError> {
    let tensor_data = varmap.data().lock().unwrap();

    let data = tensor_data.iter().map(|(k, v)| (k, v.as_tensor()));

    safetensors::serialize(data, &None)
}

fn deserialize(data: &[u8]) -> Result<SafeTensors, SafeTensorError> {
    safetensors::SafeTensors::deserialize(data)
}
