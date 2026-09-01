//! End-to-end tests driving the coordinator entirely over real gRPC, with
//! fake workers standing in for `worker::bin::worker` (no `LeNet`, no
//! `FashionMNIST` download -- see `common::FakeWorker`). This is the only
//! place the full `Subscribe` / `Publish` / `Train` protocol is exercised
//! together.

use std::time::Duration;

use candle_core::{Device, safetensors::load_buffer};
use coordinator::candlefl::{TrainRequest, command_client::CommandClient};
use tonic::transport::Channel;

mod common;

use common::{FakeWorker, TestCoordinator};

async fn train(addr: std::net::SocketAddr, rounds: u64) -> Result<Vec<u8>, tonic::Status> {
    let channel = Channel::builder(format!("http://{addr}").parse().unwrap())
        .connect()
        .await
        .expect("connect to the test coordinator");

    let response = CommandClient::new(channel)
        .train(TrainRequest { rounds })
        .await?;

    Ok(response.into_inner().weights)
}

#[tokio::test]
async fn two_workers_two_rounds_produces_the_expected_average() {
    let coordinator = TestCoordinator::start().await;
    let _worker_a = FakeWorker::connect(coordinator.addr, 2.0).await;
    let _worker_b = FakeWorker::connect(coordinator.addr, 4.0).await;

    // Both workers always respond with their own fixed value regardless of
    // what they're sent, so the average is the same every round:
    // (2.0 + 4.0) / 2 = 3.0.
    let weights = tokio::time::timeout(Duration::from_secs(10), train(coordinator.addr, 2))
        .await
        .expect("must not hang")
        .expect("training to succeed");

    let tensors = load_buffer(&weights, &Device::Cpu).unwrap();
    assert_eq!(
        tensors.get("a").unwrap().to_vec1::<f64>().unwrap(),
        vec![3.0, 3.0]
    );
}

#[tokio::test]
async fn train_with_zero_workers_errors_without_hanging() {
    let coordinator = TestCoordinator::start().await;

    let result = tokio::time::timeout(Duration::from_secs(10), train(coordinator.addr, 1))
        .await
        .expect("must not hang");

    assert!(result.is_err());
}

#[tokio::test]
async fn a_worker_disconnecting_mid_round_still_completes_with_the_survivor() {
    let coordinator = TestCoordinator::start().await;

    // 'get_weights' only ever asks the first-registered worker, so it must
    // be the one that actually responds -- otherwise 'fit' would fail
    // before any round starts, which is a different (already-covered)
    // scenario, not a mid-round disconnect.
    let _healthy = FakeWorker::connect(coordinator.addr, 5.0).await;
    let _flaky = FakeWorker::connect_and_disconnect_on_first_request(coordinator.addr).await;

    let weights = tokio::time::timeout(Duration::from_secs(10), train(coordinator.addr, 1))
        .await
        .expect("must not hang")
        .expect("training to succeed with the surviving worker");

    let tensors = load_buffer(&weights, &Device::Cpu).unwrap();
    // Only the healthy worker's response contributes; averaging over one
    // response is the identity.
    assert_eq!(
        tensors.get("a").unwrap().to_vec1::<f64>().unwrap(),
        vec![5.0, 5.0]
    );
}
