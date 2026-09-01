use std::{net::SocketAddr, time::Duration};

use clap::Parser;
use coordinator::{
    candlefl::{
        command_server::CommandServer, publisher_server::PublisherServer,
        subscriber_server::SubscriberServer,
    },
    service::{CommandService, PublisherService, SubscriberService},
    state::State,
    strategy::FedAvg,
};
use tonic::transport::Server;
use tonic_health::server::health_reporter;
use tracing::info;

#[derive(Parser)]
#[command(version)]
struct Args {
    #[arg(long, default_value_t = String::from("[::1]:50051"))]
    addr: String,

    /// Maximum time to wait for a single worker response before giving up
    /// on it. Unset by default: the coordinator otherwise waits
    /// indefinitely for a worker that is still connected, since training
    /// round durations vary too widely (CPU vs. GPU, dataset size) to have
    /// a safe default. A disconnected worker is detected and does not hang
    /// a round regardless of this setting.
    #[arg(long)]
    worker_deadline_secs: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let addr: SocketAddr = args.addr.parse()?;
    let deadline = args.worker_deadline_secs.map(Duration::from_secs);

    let state = State::with_deadline(deadline);

    let command_service = CommandService::new(FedAvg::new(state.clone()));
    let publisher_service = PublisherService::new(state.clone());
    let subscriber_service = SubscriberService::new(state.clone());

    let (health_reporter, health_service) = health_reporter();
    health_reporter
        .set_serving::<PublisherServer<PublisherService>>()
        .await;
    health_reporter
        .set_serving::<SubscriberServer<SubscriberService>>()
        .await;

    info!(addr = %addr, "coordinator started");

    Server::builder()
        // Ping idle connections so a half-open one (frozen host, dropped
        // network) is torn down instead of leaving a worker looking
        // "connected" forever. This bounds ping latency, not training
        // time, so it's safe regardless of how long a round takes -- see
        // 'coordinator::state::job' for how that distinction matters.
        .http2_keepalive_interval(Some(Duration::from_secs(30)))
        .http2_keepalive_timeout(Some(Duration::from_secs(20)))
        .add_service(health_service)
        .add_service(CommandServer::new(command_service))
        .add_service(PublisherServer::new(publisher_service))
        .add_service(SubscriberServer::new(subscriber_service))
        .serve(addr)
        .await?;

    Ok(())
}
