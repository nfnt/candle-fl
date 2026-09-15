use clap::Parser;
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Uri};
use tracing::info;
use worker::candlefl::{TrainRequest, command_client::CommandClient};

#[derive(Parser)]
#[command(version)]
struct Args {
    #[arg(long, default_value_t = String::from("[::1]:50051"))]
    addr: String,

    rounds: u64,
}

/// Simple command to request the coordinator to start a federated learning training run.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let uri: Uri = format!("http://{}", args.addr).parse()?;

    let channel = Channel::builder(uri.clone())
        .user_agent("candle-fl-command/0.1.0")?
        .connect()
        .await?;

    info!(%uri, "connected to coordinator, sending training request");

    let mut response_stream = CommandClient::new(channel.clone())
        .train(TrainRequest {
            rounds: args.rounds,
        })
        .await?
        .into_inner();

    while let Some(response) = response_stream.next().await {
        let response = response?;

        match response.metrics {
            Some(metrics) => {
                // Per-worker metrics are a variable-length list, so they
                // don't fit as their own structured fields; fold them into
                // one field instead of one log line per worker, so a round
                // still logs exactly once.
                let workers = metrics
                    .workers
                    .iter()
                    .map(|w| {
                        format!(
                            "{}(loss={}, num_examples={})",
                            w.address, w.loss, w.num_examples
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                info!(
                    job_id = %response.job_id,
                    loss = metrics.loss,
                    num_examples = metrics.num_examples,
                    workers = %workers,
                    "training round {} completed", response.round
                );
            }
            None => {
                info!(
                    job_id = %response.job_id,
                    "training round {} completed", response.round
                );
            }
        }
    }

    info!(%uri, "training completed");

    Ok(())
}
