use std::{collections::HashMap, net::SocketAddr};

use candle_core::Tensor;
use tokio::sync::oneshot;
use tracing::warn;
use uuid::Uuid;

use crate::state::{Error, job::Job, worker::Worker};

/// In-memory state for the coordinator.
///
/// Keeps track of connected workers and running jobs.
/// Not suitable for production code as it doesn't persist data across restarts.
/// Furthermore, to scale the number of workers, you would need to provide a
/// shared state across multiple instances of the coordinator.
pub struct InMemoryState {
    workers: Vec<Worker>,
    jobs: HashMap<Uuid, Job>,
}

impl InMemoryState {
    pub fn new() -> Self {
        Self {
            workers: Vec::new(),
            jobs: HashMap::new(),
        }
    }

    pub fn add_worker(&mut self, worker: Worker, response: oneshot::Sender<()>) {
        self.workers.push(worker);

        if response.send(()).is_err() {
            warn!("failed to set response");
        }
    }

    pub fn add_job(&mut self, response: oneshot::Sender<Uuid>) {
        let job = Job::new(self.workers.clone());
        let job_id = job.id();
        self.jobs.insert(job_id, job);

        if response.send(job_id).is_err() {
            warn!("failed to set response");
        }
    }

    pub fn get_weights(
        &mut self,
        job_id: Uuid,
        response: oneshot::Sender<Result<HashMap<String, Tensor>, Error>>,
    ) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.get_weights(response);
        } else if response.send(Err(Error::UnknownJob(job_id))).is_err() {
            warn!("failed to set response");
        }
    }

    pub fn fit_round(
        &mut self,
        job_id: Uuid,
        weights: &HashMap<String, Tensor>,
        response: oneshot::Sender<Result<Vec<HashMap<String, Tensor>>, Error>>,
    ) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.fit_round(weights, response);
        } else if response.send(Err(Error::UnknownJob(job_id))).is_err() {
            warn!("failed to set response");
        }
    }

    pub fn set_fit_result(
        &mut self,
        job_id: Uuid,
        addr: SocketAddr,
        weight: HashMap<String, Tensor>,
        response: oneshot::Sender<Result<(), Error>>,
    ) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.set_result(addr, weight, response);
        } else if response.send(Err(Error::UnknownJob(job_id))).is_err() {
            warn!("failed to set response");
        }
    }
}
