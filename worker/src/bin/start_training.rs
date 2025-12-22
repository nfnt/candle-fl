use clap::Parser;
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

    let _response = CommandClient::new(channel.clone())
        .train(TrainRequest {
            rounds: args.rounds,
        })
        .await?;

    info!(%uri, "training completed");

    Ok(())
}
