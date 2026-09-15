use std::{collections::HashMap, future::Future, net::SocketAddr};

use candle_core::Tensor;
use thiserror::Error as ThisError;
use tokio::sync::mpsc;
use uuid::Uuid;

pub use fed_avg::FedAvg;
pub use runner::{RunStream, Runner};

use crate::state::Job;

mod fed_avg;
mod runner;

/// Errors produced by a `Strategy`.
#[derive(Debug, ThisError)]
pub enum Error {
    #[error(transparent)]
    State(#[from] crate::state::Error),
    #[error("a training run is already in progress (job {0})")]
    AlreadyRunning(Uuid),
}

/// One worker's contribution to a round's aggregate metrics.
#[derive(Clone, Debug)]
pub struct WorkerMetrics {
    pub addr: SocketAddr,
    pub loss: f32,
    pub num_examples: u64,
}

/// A round's training metrics: a sample-weighted aggregate plus every
/// contributing worker's own numbers.
#[derive(Clone, Debug)]
pub struct RoundMetrics {
    pub loss: f32,
    pub num_examples: u64,
    pub workers: Vec<WorkerMetrics>,
}

/// One completed round's aggregate, reported while `fit` is still running.
#[derive(Debug)]
pub struct RoundUpdate {
    pub job_id: Uuid,
    pub round: u64,
    pub weights: HashMap<String, Tensor>,
    /// `None` for the round-0 update, which carries initial weights only --
    /// no training happened for it.
    pub metrics: Option<RoundMetrics>,
}

/// A federated learning strategy: how model weights are fit across
/// connected workers over a number of rounds.
pub trait Strategy: Send + Sync + 'static {
    /// Fit model weights over `num_rounds` rounds of training against
    /// `job`, reporting each round's aggregate on `updates` as soon as it
    /// completes.
    ///
    /// Rounds are numbered `1..=num_rounds`. `num_rounds == 0` still sends
    /// a single `round: 0` update carrying the initial weights, so `updates`
    /// always receives at least one item.
    ///
    /// A closed `updates` receiver (the caller is no longer listening, e.g.
    /// a disconnected gRPC client) ends the run early and is *not* itself an
    /// error -- there is nobody left to report a failure to.
    fn fit(
        &self,
        job: &Job<'_>,
        num_rounds: usize,
        updates: mpsc::Sender<RoundUpdate>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
}
