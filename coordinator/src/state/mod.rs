use std::{collections::HashMap, future::Future, net::SocketAddr, time::Duration};

use candle_core::Tensor;
use thiserror::Error;
use tokio::{
    sync::{mpsc, oneshot},
    task,
};
use tracing::warn;
use uuid::Uuid;

use crate::{
    candlefl::CoordinatorMessage,
    state::{inmemory_state::InMemoryState, store::Store, worker::Worker},
};

mod inmemory_state;
mod job;
mod store;
mod worker;

/// Metrics from one worker's local training run, mirroring
/// `candlefl::FitMetrics` on the wire but decoupled from it so the state
/// module doesn't depend on the generated proto types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitMetrics {
    pub loss: f32,
    pub num_examples: u64,
}

#[derive(Debug)]
pub struct WorkerUpdate {
    pub weights: HashMap<String, Tensor>,
    /// `None` for a `WeightsResponse`, which involves no training.
    pub metrics: Option<FitMetrics>,
}

#[derive(Debug)]
pub struct WorkerFitResult {
    pub addr: SocketAddr,
    pub weights: HashMap<String, Tensor>,
    pub metrics: Option<FitMetrics>,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("unknown job \"{0}\"")]
    UnknownJob(Uuid),
    #[error("job \"{0}\" has no connected workers")]
    NoWorkers(Uuid),
    #[error("failed to set result for worker \"{0}\"")]
    ResultNotSet(SocketAddr),
    #[error("unknown completer for worker \"{0}\"")]
    UnknownCompleter(SocketAddr),
    #[error("failed to receive worker response: {0}")]
    Receive(#[from] oneshot::error::RecvError),
    #[error("tensor operation failed: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("worker \"{0}\" is unreachable")]
    WorkerUnreachable(SocketAddr),
    #[error("worker \"{0}\" did not respond before the deadline")]
    Timeout(SocketAddr),
    #[error("job \"{0}\" failed: every worker disconnected, errored, or timed out")]
    AllWorkersFailed(Uuid),
    #[error("no results to average")]
    NoResults,
}

#[derive(Clone, Debug)]
pub struct Job<'a> {
    job_id: Uuid,
    state: &'a State,
}

impl<'a> Job<'a> {
    /// A handle for `job_id` without registering it, so tests elsewhere in
    /// the crate can check whether the actor still tracks a job -- e.g.
    /// that cleanup after a run actually removed it -- via `get_weights`'s
    /// `Error::UnknownJob` (removed/never added) vs. `Error::NoWorkers`
    /// (still tracked, just with no workers).
    #[cfg(test)]
    pub(crate) const fn probe(state: &'a State, job_id: Uuid) -> Self {
        Self { job_id, state }
    }
}

impl Job<'_> {
    #[must_use]
    pub const fn id(&self) -> Uuid {
        self.job_id
    }

    /// Get initial weights from a single worker.
    ///
    /// The initial weights can be used to ensure that each worker
    /// starts training with the same weights.
    ///
    /// # Errors
    ///
    /// Returns an error if the job is unknown, has no connected workers, or
    /// the worker disconnects or times out before responding.
    ///
    /// # Panics
    ///
    /// Panics if the coordinator's actor task has stopped running.
    pub async fn get_weights(&self) -> Result<HashMap<String, Tensor>, Error> {
        let (response, receiver) = oneshot::channel();
        self.state
            .sender
            .send(Command::GetWeights {
                job_id: self.job_id,
                response,
            })
            .await
            .expect("a running handler task");
        receiver.await.expect("a response from the handler task")
    }

    /// Perform a single round of training on all workers associated with this job.
    ///
    /// Each worker will use the provided weights to train a model and return
    /// the updated weights. The list of updated weights is then returned.
    ///
    /// # Errors
    ///
    /// Returns an error if the job is unknown, has no connected workers, or
    /// every worker disconnects, errors, or times out during the round.
    ///
    /// # Panics
    ///
    /// Panics if the coordinator's actor task has stopped running.
    pub async fn fit_round(
        &self,
        weights: HashMap<String, Tensor>,
    ) -> Result<Vec<WorkerFitResult>, Error> {
        let (response, receiver) = oneshot::channel();
        self.state
            .sender
            .send(Command::FitRound {
                job_id: self.job_id,
                weights,
                response,
            })
            .await
            .expect("a running handler task");
        receiver.await.expect("a response from the handler task")
    }

    /// Remove this job from the coordinator's state.
    ///
    /// Call this once done with the job, regardless of whether it
    /// succeeded or failed -- without it, every job started via `add_job`
    /// leaks for the lifetime of the coordinator process.
    ///
    /// # Panics
    ///
    /// Panics if the coordinator's actor task has stopped running.
    pub async fn remove(&self) {
        let (response, receiver) = oneshot::channel();
        self.state
            .sender
            .send(Command::RemoveJob {
                job_id: self.job_id,
                response,
            })
            .await
            .expect("a running handler task");
        receiver.await.expect("a response from the handler task");
    }
}

/// The coordinator's actor handle: connected workers and running jobs.
///
/// Cheap to clone -- every clone shares the same underlying actor task via
/// its `mpsc::Sender`.
///
/// # Examples
///
/// A worker never needs a real socket to be driven: `add_worker` accepts
/// any `mpsc::Sender`, so a plain channel stands in for one in tests (and
/// in this example).
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use std::collections::HashMap;
///
/// use candle_core::{Device, Tensor};
/// use coordinator::state::{State, WorkerUpdate};
/// use tokio::sync::mpsc;
///
/// let state = State::new();
///
/// let (sender, mut receiver) = mpsc::channel(4);
/// let addr = "127.0.0.1:1".parse().unwrap();
/// state.add_worker(addr, sender).await;
///
/// let job = state.add_job(uuid::Uuid::new_v4()).await;
///
/// // Answer the coordinator's request for this worker's initial weights.
/// let responder = {
///     let state = state.clone();
///     let job_id = job.id();
///     tokio::spawn(async move {
///         receiver.recv().await.expect("a WeightsRequest");
///         let mut weights = HashMap::new();
///         weights.insert("a".to_string(), Tensor::new(vec![1.0, 1.0], &Device::Cpu)?);
///         let update = WorkerUpdate { weights, metrics: None };
///         state.set_fit_result(job_id, addr, update).await
///     })
/// };
///
/// let weights = job.get_weights().await?;
/// responder.await??;
///
/// assert_eq!(weights.get("a").unwrap().to_vec1::<f64>()?, vec![1.0, 1.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct State {
    sender: mpsc::Sender<Command>,
}

impl State {
    /// A state backed by `InMemoryState`, with no per-request deadline: the
    /// coordinator waits indefinitely for a worker that is still connected.
    /// A disconnected worker is still detected and does not hang a round;
    /// see `state::job::send_and_await`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_deadline(None)
    }

    /// A state backed by `InMemoryState`, with an optional overall deadline
    /// for a single worker request.
    #[must_use]
    pub fn with_deadline(deadline: Option<Duration>) -> Self {
        Self::with_store(InMemoryState::new(), deadline)
    }

    /// A state backed by a custom `Store` (e.g. a fake used in tests).
    pub fn with_store<S: Store + Send + 'static>(store: S, deadline: Option<Duration>) -> Self {
        let (sender, receiver) = mpsc::channel(32);
        task::spawn(handler(store, receiver, deadline));

        Self { sender }
    }

    /// Register a newly connected worker.
    ///
    /// # Panics
    ///
    /// Panics if the coordinator's actor task has stopped running.
    pub async fn add_worker(&self, addr: SocketAddr, sender: mpsc::Sender<CoordinatorMessage>) {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(Command::AddWorker {
                addr,
                sender,
                response,
            })
            .await
            .expect("a running handler task");
        receiver.await.expect("a response from the handler task");
    }

    /// Start a new job under `job_id`, snapshotting the currently connected
    /// workers.
    ///
    /// # Panics
    ///
    /// Panics if the coordinator's actor task has stopped running.
    pub async fn add_job(&self, job_id: Uuid) -> Job<'_> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(Command::AddJob { job_id, response })
            .await
            .expect("a running handler task");

        receiver.await.expect("a response from the handler task");

        Job {
            job_id,
            state: self,
        }
    }

    /// Complete a pending worker request with its response.
    ///
    /// # Errors
    ///
    /// Returns an error if the job is unknown, nothing is pending for
    /// `addr` (e.g. the coordinator already gave up on it), or the pending
    /// request was already completed or dropped.
    ///
    /// # Panics
    ///
    /// Panics if the coordinator's actor task has stopped running.
    pub async fn set_fit_result(
        &self,
        job_id: Uuid,
        addr: SocketAddr,
        update: WorkerUpdate,
    ) -> Result<(), Error> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(Command::SetFitResult {
                job_id,
                addr,
                update,
                response,
            })
            .await
            .expect("a running handler task");
        receiver.await.expect("a response from the handler task")
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum Command {
    AddWorker {
        addr: SocketAddr,
        sender: mpsc::Sender<CoordinatorMessage>,
        response: oneshot::Sender<()>,
    },
    AddJob {
        job_id: Uuid,
        response: oneshot::Sender<()>,
    },
    RemoveJob {
        job_id: Uuid,
        response: oneshot::Sender<()>,
    },
    GetWeights {
        job_id: Uuid,
        response: CommandResponse<HashMap<String, Tensor>>,
    },
    FitRound {
        job_id: Uuid,
        weights: HashMap<String, Tensor>,
        response: CommandResponse<Vec<WorkerFitResult>>,
    },
    SetFitResult {
        job_id: Uuid,
        addr: SocketAddr,
        update: WorkerUpdate,
        response: CommandResponse<()>,
    },
}

type CommandResponse<T> = oneshot::Sender<Result<T, Error>>;

/// Send `value` on `response`, logging (rather than panicking) if the
/// caller already gave up waiting for it.
fn reply<T>(response: oneshot::Sender<T>, value: T) {
    if response.send(value).is_err() {
        warn!("failed to set response");
    }
}

/// Register `job_id`'s outcome future on the actor's own runtime and
/// forward its result to `response` once it resolves. `future` is
/// `'static` and owns everything it needs (see `job::Job::start_get_weights`
/// / `start_fit_round`), so this doesn't hold the store borrowed while it
/// runs -- the actor loop is free to handle the next command immediately.
fn spawn_reply<T: Send + 'static>(
    future: impl Future<Output = Result<T, Error>> + Send + 'static,
    response: CommandResponse<T>,
) {
    task::spawn(async move {
        reply(response, future.await);
    });
}

async fn handler<S: Store>(
    mut store: S,
    mut receiver: mpsc::Receiver<Command>,
    deadline: Option<Duration>,
) {
    // To unblock the loop, functions return immediately and use
    // response handlers to set the result of the operation.
    while let Some(command) = receiver.recv().await {
        match command {
            Command::AddWorker {
                addr,
                sender,
                response,
            } => {
                store.add_worker(Worker::new(addr, sender));
                reply(response, ());
            }
            Command::AddJob { job_id, response } => {
                store.prune_disconnected();
                store.insert_job(job::Job::new(job_id, store.workers()));
                reply(response, ());
            }
            Command::RemoveJob { job_id, response } => {
                store.remove_job(job_id);
                reply(response, ());
            }
            Command::GetWeights { job_id, response } => {
                let Some(job) = store.job_mut(job_id) else {
                    reply(response, Err(Error::UnknownJob(job_id)));
                    continue;
                };
                match job.start_get_weights(deadline) {
                    Ok(future) => spawn_reply(future, response),
                    Err(e) => reply(response, Err(e)),
                }
            }
            Command::FitRound {
                job_id,
                weights,
                response,
            } => {
                let Some(job) = store.job_mut(job_id) else {
                    reply(response, Err(Error::UnknownJob(job_id)));
                    continue;
                };
                match job.start_fit_round(&weights, deadline) {
                    Ok(future) => spawn_reply(future, response),
                    Err(e) => reply(response, Err(e)),
                }
            }
            Command::SetFitResult {
                job_id,
                addr,
                update,
                response,
            } => {
                let result = store.job_mut(job_id).map_or_else(
                    || Err(Error::UnknownJob(job_id)),
                    |job| job.set_result(addr, update),
                );
                reply(response, result);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use candle_core::Device;
    use tokio::sync::mpsc;

    use super::*;

    fn tensor_map(value: f64) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(
            "a".to_string(),
            Tensor::new(vec![value, value], &Device::Cpu).unwrap(),
        );
        map
    }

    fn worker_update(value: f64) -> WorkerUpdate {
        WorkerUpdate {
            weights: tensor_map(value),
            metrics: None,
        }
    }

    // A `Job` handle for a job_id the state doesn't know about, to exercise
    // the `UnknownJob` paths without needing a real job to already exist.
    fn unknown_job(state: &State) -> Job<'_> {
        Job {
            job_id: Uuid::new_v4(),
            state,
        }
    }

    #[tokio::test]
    async fn get_weights_unknown_job_errors() {
        let state = State::new();

        let err = unknown_job(&state).get_weights().await.unwrap_err();

        assert!(matches!(err, Error::UnknownJob(_)));
    }

    #[tokio::test]
    async fn fit_round_unknown_job_errors() {
        let state = State::new();

        let err = unknown_job(&state)
            .fit_round(tensor_map(1.0))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::UnknownJob(_)));
    }

    #[tokio::test]
    async fn set_fit_result_unknown_job_errors() {
        let state = State::new();
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();

        let err = state
            .set_fit_result(Uuid::new_v4(), addr, worker_update(1.0))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::UnknownJob(_)));
    }

    #[tokio::test]
    async fn set_fit_result_unknown_completer_errors() {
        let state = State::new();
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();

        // A job exists, but nothing is pending for this address (no worker
        // was ever asked to respond).
        let job = state.add_job(Uuid::new_v4()).await;

        let err = state
            .set_fit_result(job.id(), addr, worker_update(1.0))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::UnknownCompleter(a) if a == addr));
    }

    #[tokio::test]
    async fn remove_job_makes_it_unknown() {
        let state = State::new();
        let job = state.add_job(Uuid::new_v4()).await;

        job.remove().await;

        let err = job.get_weights().await.unwrap_err();
        assert!(matches!(err, Error::UnknownJob(id) if id == job.id()));
    }

    #[tokio::test]
    async fn get_weights_end_to_end_through_the_actor() {
        let state = State::new();

        let (sender, mut receiver) = mpsc::channel(4);
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        state.add_worker(addr, sender).await;

        let job = state.add_job(Uuid::new_v4()).await;

        let responder = tokio::spawn({
            let state = state.clone();
            let job_id = job.id();
            async move {
                let message = receiver.recv().await.expect("a WeightsRequest");
                assert!(matches!(
                    message.message,
                    Some(crate::candlefl::coordinator_message::Message::WeightsRequest(_))
                ));
                state
                    .set_fit_result(job_id, addr, worker_update(1.0))
                    .await
                    .unwrap();
            }
        });

        let weights = tokio::time::timeout(Duration::from_secs(5), job.get_weights())
            .await
            .expect("must not hang")
            .unwrap();
        responder.await.unwrap();

        assert_eq!(
            weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[tokio::test]
    async fn add_job_prunes_disconnected_workers_from_the_snapshot() {
        let state = State::new();

        // A worker that's already disconnected by the time it's added.
        let (dead_sender, dead_receiver) = mpsc::channel::<CoordinatorMessage>(1);
        drop(dead_receiver);
        let dead_addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        state.add_worker(dead_addr, dead_sender).await;

        // A live worker, added after the dead one.
        let (live_sender, mut live_receiver) = mpsc::channel(4);
        let live_addr: SocketAddr = "127.0.0.1:2".parse().unwrap();
        state.add_worker(live_addr, live_sender).await;

        let job = state.add_job(Uuid::new_v4()).await;

        // If the dead worker were still first in the snapshot,
        // 'get_weights' would target it and fail with
        // 'WorkerUnreachable' instead of ever reaching the live one.
        let responder = tokio::spawn({
            let state = state.clone();
            let job_id = job.id();
            async move {
                live_receiver.recv().await.expect("a WeightsRequest");
                state
                    .set_fit_result(job_id, live_addr, worker_update(1.0))
                    .await
                    .unwrap();
            }
        });

        let weights = tokio::time::timeout(Duration::from_secs(5), job.get_weights())
            .await
            .expect("must not hang")
            .unwrap();
        responder.await.unwrap();

        assert_eq!(
            weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
    }
}
