//! End-to-end tests driving the coordinator entirely over real gRPC, with
//! fake workers standing in for `worker::bin::worker` (no `LeNet`, no
//! `FashionMNIST` download -- see `common::FakeWorker`). This is the only
//! place the full `Subscribe` / `Publish` / `Train` protocol is exercised
//! together.

use std::time::Duration;

use candle_core::{Device, safetensors::load_buffer};
use coordinator::candlefl::{TrainRequest, TrainResponse, command_client::CommandClient};
use tokio_stream::StreamExt;
use tonic::transport::Channel;

mod common;

use common::{FakeWorker, TestCoordinator};

/// Drive a full training run over gRPC, collecting every streamed round.
async fn train(
    addr: std::net::SocketAddr,
    rounds: u64,
) -> Result<Vec<TrainResponse>, tonic::Status> {
    let channel = Channel::builder(format!("http://{addr}").parse().unwrap())
        .connect()
        .await
        .expect("connect to the test coordinator");

    let mut stream = CommandClient::new(channel)
        .train(TrainRequest { rounds })
        .await?
        .into_inner();

    let mut responses = Vec::new();
    while let Some(response) = stream.next().await {
        responses.push(response?);
    }
    Ok(responses)
}

#[tokio::test]
async fn two_workers_two_rounds_produces_the_expected_average() {
    let coordinator = TestCoordinator::start().await;
    let _worker_a = FakeWorker::connect(coordinator.addr, 2.0).await;
    let _worker_b = FakeWorker::connect(coordinator.addr, 4.0).await;

    // Both workers always respond with their own fixed value regardless of
    // what they're sent, so the average is the same every round:
    // (2.0 + 4.0) / 2 = 3.0.
    let responses = tokio::time::timeout(Duration::from_secs(10), train(coordinator.addr, 2))
        .await
        .expect("must not hang")
        .expect("training to succeed");

    assert_eq!(responses.len(), 2);
    let job_id = responses[0].job_id.clone();

    for (i, response) in responses.iter().enumerate() {
        assert_eq!(response.job_id, job_id);
        assert_eq!(response.round, u64::try_from(i + 1).unwrap());

        let tensors = load_buffer(&response.weights, &Device::Cpu).unwrap();
        assert_eq!(
            tensors.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![3.0, 3.0]
        );
    }
}

#[tokio::test]
async fn unequal_shards_produce_a_sample_weighted_average() {
    let coordinator = TestCoordinator::start().await;
    // Worker A holds a small shard, worker B a much larger one; both the
    // weight average and the aggregate loss must lean toward B rather than
    // split the difference evenly.
    let _worker_a = FakeWorker::connect_with_metrics(coordinator.addr, 2.0, 1.0, 10).await;
    let _worker_b = FakeWorker::connect_with_metrics(coordinator.addr, 4.0, 3.0, 30).await;

    let responses = tokio::time::timeout(Duration::from_secs(10), train(coordinator.addr, 2))
        .await
        .expect("must not hang")
        .expect("training to succeed");

    assert_eq!(responses.len(), 2);

    let mut worker_addrs_by_round = Vec::new();

    for response in &responses {
        let tensors = load_buffer(&response.weights, &Device::Cpu).unwrap();
        // (2.0 * 10 + 4.0 * 30) / 40 = 3.5, not the unweighted 3.0.
        assert_eq!(
            tensors.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![3.5, 3.5]
        );

        let metrics = response.metrics.as_ref().expect("every round has metrics");
        // (1.0 * 10 + 3.0 * 30) / 40 = 2.5, not the unweighted 2.0.
        assert_eq!(metrics.loss, 2.5);
        assert_eq!(metrics.num_examples, 40);

        assert_eq!(metrics.workers.len(), 2);
        let small = metrics
            .workers
            .iter()
            .find(|w| w.num_examples == 10)
            .expect("the small shard's worker");
        assert_eq!(small.loss, 1.0);
        let large = metrics
            .workers
            .iter()
            .find(|w| w.num_examples == 30)
            .expect("the large shard's worker");
        assert_eq!(large.loss, 3.0);

        let mut addrs: Vec<&str> = metrics.workers.iter().map(|w| w.address.as_str()).collect();
        addrs.sort_unstable();
        worker_addrs_by_round.push(addrs);
    }

    // The same two worker addresses label both rounds -- each worker keeps
    // the same connection (and so the same address) for the life of the
    // run.
    assert_eq!(worker_addrs_by_round[0], worker_addrs_by_round[1]);
}

#[tokio::test]
async fn a_second_train_is_rejected_while_one_is_already_running() {
    let coordinator = TestCoordinator::start().await;
    let _worker = FakeWorker::connect(coordinator.addr, 1.0).await;

    let channel = Channel::builder(format!("http://{}", coordinator.addr).parse().unwrap())
        .connect()
        .await
        .expect("connect to the test coordinator");

    // A round count large enough that the first run is certainly still in
    // progress by the time the second 'Train' call below reaches the
    // coordinator -- what this test needs is a run that's ongoing, not one
    // that ever finishes.
    let first = CommandClient::new(channel.clone())
        .train(TrainRequest { rounds: 1_000_000 })
        .await
        .expect("the first run to start")
        .into_inner();

    let second = tokio::time::timeout(
        Duration::from_secs(10),
        CommandClient::new(channel.clone()).train(TrainRequest { rounds: 1 }),
    )
    .await
    .expect("must not hang");

    assert_eq!(
        second.expect_err("a second run must be rejected").code(),
        tonic::Code::FailedPrecondition
    );

    // Walking away from the first run lets the coordinator notice (on its
    // next attempted send to the now-closed stream) and release the slot,
    // so a further call eventually succeeds.
    drop(first);

    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let result = CommandClient::new(channel.clone())
                .train(TrainRequest { rounds: 1 })
                .await;
            if result.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the slot to eventually be released");
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

    let responses = tokio::time::timeout(Duration::from_secs(10), train(coordinator.addr, 1))
        .await
        .expect("must not hang")
        .expect("training to succeed with the surviving worker");

    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].round, 1);

    let tensors = load_buffer(&responses[0].weights, &Device::Cpu).unwrap();
    // Only the healthy worker's response contributes; averaging over one
    // response is the identity.
    assert_eq!(
        tensors.get("a").unwrap().to_vec1::<f64>().unwrap(),
        vec![5.0, 5.0]
    );
}
