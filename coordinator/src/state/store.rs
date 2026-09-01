use std::net::SocketAddr;

use uuid::Uuid;

use crate::state::{job::Job, worker::Worker};

/// Backing store for the coordinator's state: connected workers and running
/// jobs.
///
/// Deliberately synchronous -- an async `Store` would make every command
/// passing through the actor in `state::handler` head-of-line-block behind
/// the slowest store call. `InMemoryState` is the only implementation
/// today; the trait exists so state can be faked in tests and, eventually,
/// backed by something other than an in-process store (see the doc comment
/// on `InMemoryState` for why that matters beyond a single coordinator
/// process).
pub trait Store {
    /// Register a newly connected worker.
    fn add_worker(&mut self, worker: Worker);

    /// Remove a specific worker, e.g. because it is known to have
    /// disconnected.
    fn remove_worker(&mut self, addr: &SocketAddr);

    /// Drop any worker whose outbound channel has already closed, so a
    /// worker that disconnected doesn't keep being included in every future
    /// job.
    fn prune_disconnected(&mut self);

    /// A snapshot of the currently known workers.
    fn workers(&self) -> Vec<Worker>;

    /// Store a new job, returning its id.
    fn insert_job(&mut self, job: Job) -> Uuid;

    /// Look up a running job by id.
    fn job_mut(&mut self, job_id: Uuid) -> Option<&mut Job>;

    /// Remove a job once it is no longer needed, e.g. because the training
    /// run it belongs to has finished. Without this, every `Train` call
    /// leaks its job for the lifetime of the coordinator process.
    fn remove_job(&mut self, job_id: Uuid);
}
