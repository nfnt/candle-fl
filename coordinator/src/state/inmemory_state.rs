use std::{collections::HashMap, net::SocketAddr};

use uuid::Uuid;

use crate::state::{job::Job, store::Store, worker::Worker};

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
}

impl Default for InMemoryState {
    fn default() -> Self {
        Self::new()
    }
}

impl Store for InMemoryState {
    fn add_worker(&mut self, worker: Worker) {
        self.prune_disconnected();
        self.workers.push(worker);
    }

    fn remove_worker(&mut self, addr: &SocketAddr) {
        self.workers.retain(|worker| worker.addr() != *addr);
    }

    fn prune_disconnected(&mut self) {
        self.workers.retain(|worker| !worker.sender().is_closed());
    }

    fn workers(&self) -> Vec<Worker> {
        self.workers.clone()
    }

    fn insert_job(&mut self, job: Job) -> Uuid {
        let job_id = job.id();
        self.jobs.insert(job_id, job);
        job_id
    }

    fn job_mut(&mut self, job_id: Uuid) -> Option<&mut Job> {
        self.jobs.get_mut(&job_id)
    }

    fn remove_job(&mut self, job_id: Uuid) {
        self.jobs.remove(&job_id);
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;

    use super::*;
    use crate::candlefl::CoordinatorMessage;

    // Returns the worker along with the receiving half of its channel; the
    // caller must keep the receiver alive for as long as the worker should
    // be considered connected, since dropping it is what 'is_closed()'
    // (and therefore 'prune_disconnected') detects.
    fn worker(addr: &str) -> (Worker, mpsc::Receiver<CoordinatorMessage>) {
        let (sender, receiver) = mpsc::channel(1);
        (Worker::new(addr.parse().unwrap(), sender), receiver)
    }

    fn closed_worker(addr: &str) -> Worker {
        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        Worker::new(addr.parse().unwrap(), sender)
    }

    #[test]
    fn add_worker_prunes_disconnected_workers() {
        let mut state = InMemoryState::new();
        let (live, _receiver) = worker("127.0.0.1:2");
        state.add_worker(closed_worker("127.0.0.1:1"));
        state.add_worker(live);

        assert_eq!(state.workers().len(), 1);
        assert_eq!(state.workers()[0].addr(), "127.0.0.1:2".parse().unwrap());
    }

    #[test]
    fn prune_disconnected_removes_only_closed_workers() {
        let mut state = InMemoryState::new();
        let (live, _receiver) = worker("127.0.0.1:2");
        state.workers.push(closed_worker("127.0.0.1:1"));
        state.workers.push(live);

        state.prune_disconnected();

        assert_eq!(state.workers().len(), 1);
        assert_eq!(state.workers()[0].addr(), "127.0.0.1:2".parse().unwrap());
    }

    #[test]
    fn remove_worker_removes_by_address() {
        let mut state = InMemoryState::new();
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let (a, _rx_a) = worker("127.0.0.1:1");
        let (b, _rx_b) = worker("127.0.0.1:2");
        state.workers.push(a);
        state.workers.push(b);

        state.remove_worker(&addr);

        assert_eq!(state.workers().len(), 1);
        assert_eq!(state.workers()[0].addr(), "127.0.0.1:2".parse().unwrap());
    }

    #[test]
    fn insert_job_mut_remove_job_round_trip() {
        let mut state = InMemoryState::new();
        let job_id = state.insert_job(Job::new(vec![]));

        assert!(state.job_mut(job_id).is_some());
        assert!(state.job_mut(Uuid::new_v4()).is_none());

        state.remove_job(job_id);
        assert!(state.job_mut(job_id).is_none());
    }
}
