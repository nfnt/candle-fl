use std::pin::Pin;

use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status};
use tracing::info;

use crate::{
    candlefl::{CoordinatorMessage, subscriber_server::Subscriber},
    state::State,
};

pub struct SubscriberService {
    state: State,
}

impl SubscriberService {
    pub const fn new(state: State) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl Subscriber for SubscriberService {
    type SubscribeStream = Pin<Box<dyn Stream<Item = Result<CoordinatorMessage, Status>> + Send>>;

    async fn subscribe(
        &self,
        request: Request<()>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let addr = request.remote_addr().expect("a remote address");

        info!(addr = addr.to_string(), "worker subscribing");

        let (sender, receiver) = mpsc::channel(32);

        self.state.add_worker(addr, sender).await;

        Ok(Response::new(
            Box::pin(ReceiverStream::new(receiver).map(Ok)) as Self::SubscribeStream,
        ))
    }
}
