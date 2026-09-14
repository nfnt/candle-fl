use std::time::Duration;

use clap::Parser;
use tokio::task;
use tonic::transport::{Channel, Uri};
use tracing::{debug, info};
use worker::{
    candlefl::{
        self, WorkerMessage, publisher_client::PublisherClient, subscriber_client::SubscriberClient,
    },
    handler::{handle_fit_request, handle_weights_request},
    ml::FashionMnistTrainer,
    select_device,
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
        // Matches the coordinator's 'http2_keepalive_interval'/'_timeout':
        // ping a connection that's been idle (e.g. waiting for the next
        // training round) so a half-open connection is detected and torn
        // down rather than looking alive forever.
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(20))
        .keep_alive_while_idle(true)
        .connect()
        .await?;
    let mut stream = SubscriberClient::new(channel.clone())
        .subscribe(())
        .await?
        .into_inner();

    info!(%uri, "connected to coordinator");

    let dev = select_device()?;

    // In production code we need to handle stream disconnections by retrying
    // if a connection is dropped. This isn't done here.
    //
    // Related, deliberately out-of-scope gap: a worker that's still
    // connected but internally wedged (e.g. stuck in a training step) is
    // indistinguishable to the coordinator from one that's just slow --
    // there's no heartbeat in the protocol. See the doc comment on
    // 'coordinator::state::job' for what disconnects the coordinator *does*
    // detect, and why a fixed timeout isn't a safe substitute here.
    while let Some(message) = stream.message().await? {
        let Some(message) = message.message else {
            continue;
        };

        match message {
            candlefl::coordinator_message::Message::WeightsRequest(weights_request) => {
                info!(job_id = %weights_request.job_id, "received WeightsRequest");

                let channel = channel.clone();
                let dev = dev.clone();
                let job_id = weights_request.job_id;

                task::spawn(async move {
                    match handle_weights_request(FashionMnistTrainer, dev, job_id.clone()).await {
                        Ok(message) => send(channel, message, &job_id, "WeightsResponse").await,
                        Err(e) => debug!(%job_id, "failed to prepare model: {}", e),
                    }
                });
            }
            candlefl::coordinator_message::Message::FitRequest(fit_request) => {
                info!(job_id = %fit_request.job_id, "received FitRequest");

                let channel = channel.clone();
                let dev = dev.clone();
                let job_id = fit_request.job_id;
                let weights = fit_request.weights;

                task::spawn(async move {
                    match handle_fit_request(FashionMnistTrainer, dev, job_id.clone(), weights)
                        .await
                    {
                        Ok(message) => send(channel, message, &job_id, "FitResponse").await,
                        Err(e) => debug!(%job_id, "failed to train model: {}", e),
                    }
                });
            }
        }
    }

    Ok(())
}

async fn send(channel: Channel, message: WorkerMessage, job_id: &str, kind: &str) {
    match PublisherClient::new(channel).publish(message).await {
        Ok(_) => info!(%job_id, "sent {}", kind),
        Err(status) => debug!(%job_id, "failed to send {}: {}", kind, status),
    }
}
