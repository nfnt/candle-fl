use std::net::SocketAddr;

use tokio::sync::mpsc;

use crate::candlefl::CoordinatorMessage;

#[derive(Clone)]
pub struct Worker {
    addr: SocketAddr,
    sender: mpsc::Sender<CoordinatorMessage>,
}

impl Worker {
    pub const fn new(addr: SocketAddr, sender: mpsc::Sender<CoordinatorMessage>) -> Self {
        Self { addr, sender }
    }

    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub const fn sender(&self) -> &mpsc::Sender<CoordinatorMessage> {
        &self.sender
    }
}
