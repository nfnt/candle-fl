use std::pin::Pin;

use futures_util::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::{
    candlefl::{subscriber_server::Subscriber, CoordinatorMessage},
    state::State,
};

pub struct SubscriberService {
    state: State,
}

impl SubscriberService {
    pub fn new(state: State) -> Self {
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
        let addr = request.remote_addr().unwrap();

        info!(addr = addr.to_string(), "worker subscribing");

        let (sender, receiver) = mpsc::channel(32);

        self.state
            .add_worker(addr, sender)
            .await
            .map_err(|e| Status::internal(format!("failed to add worker: {e}")))?;

        Ok(Response::new(
            Box::pin(ReceiverStream::new(receiver)) as Self::SubscribeStream
        ))
    }
}
