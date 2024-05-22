use std::{collections::HashMap, net::SocketAddr};

use candle_core::Tensor;
use tokio::sync::{mpsc, oneshot};
use tonic::Status;
use tracing::warn;
use uuid::Uuid;

use crate::{
    candlefl::CoordinatorMessage,
    state::{job::Job, worker::Worker},
};

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
        InMemoryState {
            workers: Vec::new(),
            jobs: HashMap::new(),
        }
    }

    pub fn add_worker(
        &mut self,
        addr: SocketAddr,
        sender: mpsc::Sender<Result<CoordinatorMessage, Status>>,
        response: oneshot::Sender<Result<(), anyhow::Error>>,
    ) {
        self.workers.push(Worker::new(addr, sender));

        if response.send(Ok(())).is_err() {
            warn!("failed to set response");
        }
    }

    pub fn add_job(&mut self, response: oneshot::Sender<Result<Uuid, anyhow::Error>>) {
        let job = Job::new(self.workers.clone());
        let job_id = job.id();
        self.jobs.insert(job_id, job);

        if response.send(Ok(job_id)).is_err() {
            warn!("failed to set response");
        }
    }

    pub fn get_weights(
        &mut self,
        job_id: Uuid,
        response: oneshot::Sender<Result<HashMap<String, Tensor>, anyhow::Error>>,
    ) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.get_weights(response);
        } else if response
            .send(Err(anyhow::anyhow!("job {job_id} not found")))
            .is_err()
        {
            warn!("failed to set response");
        }
    }

    pub fn fit_round(
        &mut self,
        job_id: Uuid,
        weights: &HashMap<String, Tensor>,
        response: oneshot::Sender<Result<Vec<HashMap<String, Tensor>>, anyhow::Error>>,
    ) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.fit_round(weights, response);
        } else if response
            .send(Err(anyhow::anyhow!("job {job_id} not found")))
            .is_err()
        {
            warn!("failed to set response");
        }
    }

    pub fn set_fit_result(
        &mut self,
        job_id: Uuid,
        addr: SocketAddr,
        weight: HashMap<String, Tensor>,
        response: oneshot::Sender<Result<(), anyhow::Error>>,
    ) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.set_result(addr, weight, response);
        } else if response
            .send(Err(anyhow::anyhow!("job {job_id} not found")))
            .is_err()
        {
            warn!("failed to set response");
        }
    }
}
