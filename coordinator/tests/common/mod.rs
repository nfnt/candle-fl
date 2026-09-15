//! Test harness for end-to-end tests: a real coordinator gRPC server on an
//! ephemeral local port, plus a fake worker that speaks the same protocol
//! as `worker::bin::worker` without any of the ML machinery (no `LeNet`, no
//! `FashionMNIST` download), so tests run in milliseconds.

use std::net::SocketAddr;

use candle_core::{Device, Tensor};
use coordinator::{
    candlefl::{
        FitMetrics, FitResponse, WeightsResponse, WorkerMessage, command_server::CommandServer,
        coordinator_message, publisher_client::PublisherClient, publisher_server::PublisherServer,
        subscriber_client::SubscriberClient, subscriber_server::SubscriberServer, worker_message,
    },
    service::{CommandService, PublisherService, SubscriberService},
    state::State,
    strategy::FedAvg,
};
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Channel, Server};

/// A coordinator running the real `Command`/`Publisher`/`Subscriber`
/// services on a real (loopback) socket, so tests exercise the full gRPC
/// protocol instead of calling `State`/`Job` directly in-process.
pub struct TestCoordinator {
    pub addr: SocketAddr,
    server_task: JoinHandle<()>,
}

impl TestCoordinator {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind an ephemeral port");
        let addr = listener.local_addr().expect("a local address");
        let incoming = TcpListenerStream::new(listener);

        let state = State::new();
        let command_service = CommandService::new(FedAvg::new(), state.clone());
        let publisher_service = PublisherService::new(state.clone());
        let subscriber_service = SubscriberService::new(state);

        let server_task = tokio::spawn(async move {
            Server::builder()
                .add_service(CommandServer::new(command_service))
                .add_service(PublisherServer::new(publisher_service))
                .add_service(SubscriberServer::new(subscriber_service))
                .serve_with_incoming(incoming)
                .await
                .expect("the server doesn't fail to run");
        });

        Self { addr, server_task }
    }
}

impl Drop for TestCoordinator {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

/// A fake worker: connects to a `TestCoordinator`, subscribes, and reacts
/// to whatever it's asked for according to `Behavior`.
pub struct FakeWorker {
    // Kept alive for the worker's lifetime; dropping it detaches the task
    // (it keeps running until the test's runtime is torn down), which is
    // fine for these short-lived tests.
    #[allow(dead_code)]
    task: JoinHandle<()>,
}

#[derive(Clone, Copy)]
enum Behavior {
    /// Reply to every request with a fixed tensor value under key "a", and
    /// a `FitResponse` includes `metrics` iff this is `Some`.
    Respond {
        value: f64,
        metrics: Option<(f32, u64)>,
    },
    /// Drop the connection instead of replying to the first request
    /// received, simulating a worker that dies mid-round.
    DisconnectOnFirstRequest,
}

impl FakeWorker {
    /// Connect and reply to every request with a fixed tensor value and no
    /// training metrics.
    pub async fn connect(addr: SocketAddr, value: f64) -> Self {
        Self::run(
            addr,
            Behavior::Respond {
                value,
                metrics: None,
            },
        )
        .await
    }

    /// Connect and reply to every `FitRequest` with a fixed tensor value
    /// and metrics of `(loss, num_examples)`, so a test can drive workers
    /// with unequal shards.
    pub async fn connect_with_metrics(
        addr: SocketAddr,
        value: f64,
        loss: f32,
        num_examples: u64,
    ) -> Self {
        Self::run(
            addr,
            Behavior::Respond {
                value,
                metrics: Some((loss, num_examples)),
            },
        )
        .await
    }

    /// Connect, but disconnect instead of replying to the first request --
    /// simulates a worker that dies partway through a round.
    pub async fn connect_and_disconnect_on_first_request(addr: SocketAddr) -> Self {
        Self::run(addr, Behavior::DisconnectOnFirstRequest).await
    }

    async fn run(addr: SocketAddr, behavior: Behavior) -> Self {
        let channel = Channel::builder(format!("http://{addr}").parse().unwrap())
            .connect()
            .await
            .expect("connect to the test coordinator");

        let mut stream = SubscriberClient::new(channel.clone())
            .subscribe(())
            .await
            .expect("subscribe")
            .into_inner();

        let task = tokio::spawn(async move {
            while let Ok(Some(message)) = stream.message().await {
                let Some(message) = message.message else {
                    continue;
                };

                if matches!(behavior, Behavior::DisconnectOnFirstRequest) {
                    // Dropping the stream and channel closes the
                    // connection, which is what the coordinator's
                    // disconnect detection watches for.
                    return;
                }

                let Behavior::Respond { value, metrics } = behavior else {
                    unreachable!("handled above");
                };
                let weights = fixed_weights(value);
                let bytes = safetensors::serialize(&weights, None).unwrap();

                let response = match message {
                    coordinator_message::Message::WeightsRequest(req) => WorkerMessage {
                        message: Some(worker_message::Message::WeightsResponse(WeightsResponse {
                            job_id: req.job_id,
                            weights: bytes,
                        })),
                    },
                    coordinator_message::Message::FitRequest(req) => WorkerMessage {
                        message: Some(worker_message::Message::FitResponse(FitResponse {
                            job_id: req.job_id,
                            weights: bytes,
                            metrics: metrics
                                .map(|(loss, num_examples)| FitMetrics { loss, num_examples }),
                        })),
                    },
                };

                let _ = PublisherClient::new(channel.clone())
                    .publish(response)
                    .await;
            }
        });

        Self { task }
    }
}

fn fixed_weights(value: f64) -> std::collections::HashMap<String, Tensor> {
    let mut map = std::collections::HashMap::new();
    map.insert(
        "a".to_string(),
        Tensor::new(vec![value, value], &Device::Cpu).unwrap(),
    );
    map
}
