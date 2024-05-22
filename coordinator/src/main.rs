use std::net::SocketAddr;

use clap::Parser;
use tonic::transport::Server;
use tonic_health::server::health_reporter;
use tracing::info;

use crate::{
    candlefl::{
        command_server::CommandServer, publisher_server::PublisherServer,
        subscriber_server::SubscriberServer,
    },
    service::{CommandService, PublisherService, SubscriberService},
    state::State,
};

mod candlefl {
    tonic::include_proto!("candlefl.v1");
}
mod service;
mod state;
mod strategy;

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

    let addr: SocketAddr = args.addr.parse()?;

    let state = State::new();

    let command_service = CommandService::new(state.clone());
    let publisher_service = PublisherService::new(state.clone());
    let subscriber_service = SubscriberService::new(state.clone());

    let (mut health_reporter, health_service) = health_reporter();
    health_reporter
        .set_serving::<PublisherServer<PublisherService>>()
        .await;
    health_reporter
        .set_serving::<SubscriberServer<SubscriberService>>()
        .await;

    info!(addr = %addr, "coordinator started");

    Server::builder()
        .add_service(health_service)
        .add_service(CommandServer::new(command_service))
        .add_service(PublisherServer::new(publisher_service))
        .add_service(SubscriberServer::new(subscriber_service))
        .serve(addr)
        .await?;

    Ok(())
}
