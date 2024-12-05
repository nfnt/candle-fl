use std::net::SocketAddr;

use tokio::sync::mpsc;
use tonic::Status;

use crate::candlefl::CoordinatorMessage;

#[derive(Clone)]
pub struct Worker {
    addr: SocketAddr,
    sender: mpsc::Sender<Result<CoordinatorMessage, Status>>,
}

impl Worker {
    pub const fn new(
        addr: SocketAddr,
        sender: mpsc::Sender<Result<CoordinatorMessage, Status>>,
    ) -> Self {
        Self { addr, sender }
    }

    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub const fn sender(&self) -> &mpsc::Sender<Result<CoordinatorMessage, Status>> {
        &self.sender
    }
}
