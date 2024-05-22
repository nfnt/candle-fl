use std::{collections::HashMap, net::SocketAddr};

use candle_core::Tensor;
use tokio::sync::{mpsc, oneshot};
use tonic::Status;
use uuid::Uuid;

use crate::{candlefl::CoordinatorMessage, state::inmemory_state::InMemoryState};

mod inmemory_state;
mod job;
mod worker;

#[derive(Clone)]
pub struct Job<'a> {
    job_id: Uuid,
    state: &'a State,
}

impl<'a> Job<'a> {
    pub fn id(&self) -> Uuid {
        self.job_id
    }

    /// Get initial weights from a single worker.
    ///
    /// The initial weights can be used to ensure that each worker
    /// starts training with the same weights.
    pub async fn get_weights(&self) -> Result<HashMap<String, Tensor>, anyhow::Error> {
        let (response, receiver) = oneshot::channel();
        self.state
            .sender
            .send(Command::GetWeights {
                job_id: self.job_id,
                response,
            })
            .await?;
        receiver.await?
    }

    /// Perform a single round of training on all workers associated with this job.
    ///
    /// Each worker will use the provided weights to train a model and return
    /// the updated weights. The list of updated weights is then returned.
    pub async fn fit_round(
        &self,
        weights: HashMap<String, Tensor>,
    ) -> Result<Vec<HashMap<String, Tensor>>, anyhow::Error> {
        let (response, receiver) = oneshot::channel();
        self.state
            .sender
            .send(Command::FitRound {
                job_id: self.job_id,
                weights,
                response,
            })
            .await?;
        receiver.await?
    }
}

#[derive(Clone)]
pub struct State {
    sender: mpsc::Sender<Command>,
}

impl State {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel(32);
        tokio::spawn(handler(receiver));

        State { sender }
    }

    pub async fn add_worker(
        &self,
        addr: SocketAddr,
        sender: mpsc::Sender<Result<CoordinatorMessage, Status>>,
    ) -> Result<(), anyhow::Error> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(Command::AddWorker {
                addr,
                sender,
                response,
            })
            .await?;
        receiver.await?
    }

    pub async fn add_job(&self) -> Result<Job, anyhow::Error> {
        let (response, receiver) = oneshot::channel();
        self.sender.send(Command::AddJob { response }).await?;

        let job_id = receiver.await??;

        Ok(Job {
            job_id,
            state: self,
        })
    }

    pub async fn set_fit_result(
        &self,
        job_id: Uuid,
        addr: SocketAddr,
        weights: HashMap<String, Tensor>,
    ) -> Result<(), anyhow::Error> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(Command::SetFitResult {
                job_id,
                addr,
                weights,
                response,
            })
            .await?;
        receiver.await?
    }
}

#[derive(Debug)]
enum Command {
    AddWorker {
        addr: SocketAddr,
        sender: mpsc::Sender<Result<CoordinatorMessage, Status>>,
        response: CommandResponse<()>,
    },
    AddJob {
        response: CommandResponse<Uuid>,
    },
    GetWeights {
        job_id: Uuid,
        response: CommandResponse<HashMap<String, Tensor>>,
    },
    FitRound {
        job_id: Uuid,
        weights: HashMap<String, Tensor>,
        response: CommandResponse<Vec<HashMap<String, Tensor>>>,
    },
    SetFitResult {
        job_id: Uuid,
        addr: SocketAddr,
        weights: HashMap<String, Tensor>,
        response: CommandResponse<()>,
    },
}

type CommandResponse<T> = oneshot::Sender<Result<T, anyhow::Error>>;

async fn handler(mut receiver: mpsc::Receiver<Command>) {
    let mut state = InMemoryState::new();

    // To unblock the loop, functions return immediately and use
    // response handlers to set the result of the operation.
    while let Some(command) = receiver.recv().await {
        match command {
            Command::AddWorker {
                addr,
                sender,
                response,
            } => {
                state.add_worker(addr, sender, response);
            }
            Command::AddJob { response } => {
                state.add_job(response);
            }
            Command::GetWeights { job_id, response } => {
                state.get_weights(job_id, response);
            }
            Command::FitRound {
                job_id,
                weights,
                response,
            } => {
                state.fit_round(job_id, &weights, response);
            }
            Command::SetFitResult {
                job_id,
                addr,
                weights,
                response,
            } => {
                state.set_fit_result(job_id, addr, weights, response);
            }
        }
    }
}
