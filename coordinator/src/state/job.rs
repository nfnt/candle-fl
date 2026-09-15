//! Per-job orchestration of worker requests, and how the coordinator
//! notices a worker is no longer there.
//!
//! A round can't just wait forever for every worker: a disconnected or
//! killed worker would otherwise hang the whole job. But a fixed timeout
//! isn't a safe substitute, because training-round duration is genuinely
//! unbounded and unknowable in advance (CPU vs. GPU, dataset size), so any
//! hardcoded deadline is simultaneously too short (kills healthy slow
//! workers) and too long (doesn't actually keep the coordinator
//! responsive). Instead, liveness is bounded, not work:
//!
//! - **Disconnect detection** (primary, no configuration): each per-worker
//!   task races the worker's response against
//!   [`mpsc::Sender::closed`](tokio::sync::mpsc::Sender::closed), which
//!   resolves as soon as tonic drops the subscribe stream -- covering
//!   clean disconnects, crashes, and killed processes immediately.
//! - **Transport keepalive**, configured on the server/client in
//!   `bin/coordinator.rs` / `bin/worker.rs`: pings an idle connection so a
//!   half-open one (frozen host, dropped network) is torn down instead of
//!   looking alive forever. This bounds ping latency, not training time,
//!   so a fixed default is safe regardless of round length.
//! - **An optional overall deadline** (`Option<Duration>`, off by
//!   default), for anyone who explicitly wants a hard cap despite the
//!   above -- see the coordinator's `--worker-deadline-secs` flag.
//!
//! What none of this catches: a worker that's still connected but
//! internally wedged (e.g. stuck mid-epoch). That's indistinguishable from
//! "still training" without a heartbeat in the wire protocol, which is a
//! deliberately out-of-scope follow-up (see `worker::bin::worker`).
//!
//! See [`send_and_await`] for where these race together, and
//! [`Job::set_result`] for a residual race this design does not close.

use std::{
    collections::HashMap,
    future::Future,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use candle_core::Tensor;
use tokio::{sync::oneshot, task};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::{
    candlefl::{CoordinatorMessage, FitRequest, WeightsRequest, coordinator_message},
    state::{Error, WorkerFitResult, WorkerUpdate, worker::Worker},
};

type TaskMap = Arc<Mutex<HashMap<SocketAddr, oneshot::Sender<WorkerUpdate>>>>;
type FitOutcome = (SocketAddr, Result<WorkerUpdate, Error>);

pub struct Job {
    id: Uuid,
    workers: Vec<Worker>,
    // Each in-flight request registers a oneshot sender here, keyed by the
    // worker's address. It is removed either by 'set_result', once a
    // response arrives, or by the request's own task, if it gives up on the
    // worker (disconnect or deadline). The map is shared with those spawned
    // tasks via the 'Arc<Mutex<_>>' so a task that gives up can remove its
    // own entry, instead of leaving a stale sender behind for a later round
    // to silently insert over and a late response to resolve incorrectly.
    tasks: TaskMap,
}

impl Job {
    pub fn new(id: Uuid, workers: Vec<Worker>) -> Self {
        Self {
            id,
            workers,
            tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub const fn id(&self) -> Uuid {
        self.id
    }

    /// Request the initial weights from a single worker.
    ///
    /// The initial weights can be used to ensure that each worker starts
    /// training with the same weights.
    ///
    /// Returns a future that resolves once the worker responds,
    /// disconnects, or (if `timeout` is set) the deadline elapses. Fails
    /// immediately, without awaiting anything, if the job has no connected
    /// workers.
    pub fn start_get_weights(
        &self,
        timeout: Option<Duration>,
    ) -> Result<impl Future<Output = Result<HashMap<String, Tensor>, Error>> + Send + 'static, Error>
    {
        let job_id = self.id;

        let Some(worker) = self.workers.first().cloned() else {
            return Err(Error::NoWorkers(job_id));
        };

        let message = CoordinatorMessage {
            message: Some(coordinator_message::Message::WeightsRequest(
                WeightsRequest {
                    job_id: job_id.into(),
                },
            )),
        };

        let receiver = self.register(worker.addr());

        let future = send_and_await(
            worker,
            message,
            receiver,
            timeout,
            Arc::clone(&self.tasks),
            job_id,
        );

        // A 'WeightsRequest' involves no training, so only the weights
        // themselves are of interest here -- any metrics on the update are
        // meaningless for it and never populated by the caller anyway.
        Ok(async move { future.await.map(|update| update.weights) })
    }

    /// Perform a single round of training on all workers associated with
    /// this job.
    ///
    /// Each worker uses the provided weights to train a model and returns
    /// the updated weights. Returns a future that resolves once every
    /// worker has responded, disconnected, or (if `timeout` is set) had its
    /// deadline elapse. The round succeeds as long as at least one worker
    /// returns updated weights; workers that fail or drop out are logged
    /// and skipped. Fails immediately, without awaiting anything, if the
    /// job has no connected workers.
    pub fn start_fit_round(
        &self,
        weights: &HashMap<String, Tensor>,
        timeout: Option<Duration>,
    ) -> Result<impl Future<Output = Result<Vec<WorkerFitResult>, Error>> + Send + 'static, Error>
    {
        let job_id = self.id;

        if self.workers.is_empty() {
            return Err(Error::NoWorkers(job_id));
        }

        let weights = safetensors::serialize(weights, None)
            .map_err(|e| Error::Candle(candle_core::Error::SafeTensor(e)))?;

        let mut join_set = task::JoinSet::new();

        for worker in self.workers.clone() {
            let addr = worker.addr();
            let message = CoordinatorMessage {
                message: Some(coordinator_message::Message::FitRequest(FitRequest {
                    job_id: job_id.into(),
                    weights: weights.clone(),
                })),
            };

            let receiver = self.register(addr);
            let tasks = Arc::clone(&self.tasks);

            join_set.spawn(async move {
                let result =
                    send_and_await(worker, message, receiver, timeout, tasks, job_id).await;
                (addr, result)
            });
        }

        Ok(collect_fit_results(job_id, join_set))
    }

    /// Complete the pending request for `addr` with a worker's response.
    ///
    /// Note on a residual race this cannot close: if the coordinator gives
    /// up on a worker (deadline elapsed) and then registers a *new* request
    /// for the same address (e.g. the next training round starts) before
    /// that worker's original response arrives over the wire, the late
    /// response will resolve the new request instead of erroring. The
    /// worker-to-coordinator response path (`Publisher::publish`) is a
    /// separate RPC from the one used to detect disconnects here, and
    /// `FitResponse`/`WeightsResponse` carry no round or request token to
    /// disambiguate. Closing that gap needs a protocol change, not just
    /// bookkeeping. What this method does guarantee: a response for a
    /// worker nobody is currently waiting on (the common case, since giving
    /// up always removes the pending entry) is rejected rather than
    /// silently accepted.
    pub fn set_result(&self, addr: SocketAddr, update: WorkerUpdate) -> Result<(), Error> {
        let sender = self.tasks.lock().expect("tasks lock").remove(&addr);
        sender.map_or_else(
            || Err(Error::UnknownCompleter(addr)),
            |sender| sender.send(update).map_err(|_| Error::ResultNotSet(addr)),
        )
    }

    /// Register a pending request for `addr`, returning the receiving half.
    fn register(&self, addr: SocketAddr) -> oneshot::Receiver<WorkerUpdate> {
        let (sender, receiver) = oneshot::channel();
        if self
            .tasks
            .lock()
            .expect("tasks lock")
            .insert(addr, sender)
            .is_some()
        {
            warn!(
                job_id = %self.id,
                addr = %addr,
                "a request was already pending for this worker; the earlier one will never complete"
            );
        }
        receiver
    }
}

/// Send `message` to `worker` and wait for its response, whichever of these
/// happens first:
/// - the worker replies (successfully, or its sender was dropped),
/// - the worker disconnects, i.e. `worker`'s outbound channel closes, or
/// - `timeout` elapses, if set.
///
/// On any non-success outcome, removes the pending entry from `tasks` so a
/// later round doesn't silently reuse a stale sender, and a late response
/// from this worker correctly surfaces as `Error::UnknownCompleter` instead
/// of resolving the wrong round.
async fn send_and_await(
    worker: Worker,
    message: CoordinatorMessage,
    mut receiver: oneshot::Receiver<WorkerUpdate>,
    timeout: Option<Duration>,
    tasks: TaskMap,
    job_id: Uuid,
) -> Result<WorkerUpdate, Error> {
    let addr = worker.addr();

    debug!(job_id = %job_id, addr = %addr, "sending message to worker");

    if let Err(e) = worker.sender().send(message).await {
        warn!(
            job_id = %job_id,
            addr = %addr,
            error = %e,
            "failed to send message to worker"
        );
        tasks.lock().expect("tasks lock").remove(&addr);
        return Err(Error::WorkerUnreachable(addr));
    }

    let result = tokio::select! {
        biased;
        result = &mut receiver => result.map_err(Error::Receive),
        () = worker.sender().closed() => Err(Error::WorkerUnreachable(addr)),
        () = wait_deadline(timeout) => Err(Error::Timeout(addr)),
    };

    if result.is_err() {
        tasks.lock().expect("tasks lock").remove(&addr);
    }

    result
}

async fn wait_deadline(timeout: Option<Duration>) {
    match timeout {
        Some(duration) => tokio::time::sleep(duration).await,
        None => std::future::pending().await,
    }
}

/// Drain `join_set`, tolerating individual worker failures -- including a
/// worker task that panicked, which `JoinSet::join_all` would otherwise
/// propagate as a panic here, poisoning the whole round. Succeeds as long
/// as at least one worker returned weights.
async fn collect_fit_results(
    job_id: Uuid,
    mut join_set: task::JoinSet<FitOutcome>,
) -> Result<Vec<WorkerFitResult>, Error> {
    let mut succeeded = Vec::new();
    let mut failed = 0usize;

    while let Some(outcome) = join_set.join_next().await {
        match outcome {
            Ok((addr, Ok(update))) => succeeded.push(WorkerFitResult {
                addr,
                weights: update.weights,
                metrics: update.metrics,
            }),
            Ok((addr, Err(error))) => {
                warn!(
                    job_id = %job_id,
                    addr = %addr,
                    %error,
                    "worker did not complete the fit round"
                );
                failed += 1;
            }
            Err(join_error) => {
                warn!(job_id = %job_id, %join_error, "worker task panicked during fit round");
                failed += 1;
            }
        }
    }

    if succeeded.is_empty() {
        return Err(Error::AllWorkersFailed(job_id));
    }
    if failed > 0 {
        warn!(
            job_id = %job_id,
            failed,
            succeeded = succeeded.len(),
            "federated round completed with worker failures"
        );
    }

    Ok(succeeded)
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

    /// A worker whose channel is already closed, simulating one that
    /// disconnected before ever being sent a message.
    fn disconnected_worker(addr: &str) -> Worker {
        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        Worker::new(addr.parse().unwrap(), sender)
    }

    /// A worker paired with the receiving half of its channel, so a test can
    /// observe outbound messages and simulate a live connection.
    fn connected_worker(addr: &str) -> (Worker, mpsc::Receiver<CoordinatorMessage>) {
        let (sender, receiver) = mpsc::channel(4);
        (Worker::new(addr.parse().unwrap(), sender), receiver)
    }

    #[tokio::test]
    async fn get_weights_no_workers_errors_immediately() {
        let job = Job::new(Uuid::new_v4(), vec![]);
        let job_id = job.id();

        let Err(err) = job.start_get_weights(None) else {
            panic!("expected an error");
        };

        assert!(matches!(err, Error::NoWorkers(id) if id == job_id));
    }

    #[tokio::test]
    async fn get_weights_disconnected_worker_errors_instead_of_hanging() {
        let job = Job::new(Uuid::new_v4(), vec![disconnected_worker("127.0.0.1:1")]);

        let future = job.start_get_weights(None).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("must not hang");

        assert!(matches!(result, Err(Error::WorkerUnreachable(_))));
    }

    #[tokio::test]
    async fn get_weights_disconnect_after_send_errors_instead_of_hanging() {
        let (worker, mut receiver) = connected_worker("127.0.0.1:1");
        let addr = worker.addr();
        let job = Job::new(Uuid::new_v4(), vec![worker]);

        let future = job.start_get_weights(None).unwrap();

        // Receive the request, confirming the send succeeded, then drop the
        // receiving half to simulate the worker disconnecting before it
        // replies. This exercises the 'closed()' branch specifically,
        // rather than the immediate send failure covered by
        // 'get_weights_disconnected_worker_errors_instead_of_hanging'.
        let responder = task::spawn(async move {
            receiver.recv().await.expect("a request");
            drop(receiver);
        });

        let result = tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("must not hang");

        responder.await.unwrap();
        assert!(matches!(result, Err(Error::WorkerUnreachable(a)) if a == addr));
    }

    #[tokio::test]
    async fn get_weights_success() {
        let (worker, mut receiver) = connected_worker("127.0.0.1:1");
        let addr = worker.addr();
        let job = Job::new(Uuid::new_v4(), vec![worker]);

        let future = job.start_get_weights(None).unwrap();

        let responder = task::spawn(async move {
            receiver.recv().await.expect("a request");
            job.set_result(addr, worker_update(1.0)).unwrap();
            job
        });

        let result = tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("must not hang")
            .unwrap();

        responder.await.unwrap();
        assert_eq!(
            result.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[tokio::test]
    async fn get_weights_respects_deadline() {
        let (worker, _receiver) = connected_worker("127.0.0.1:1");
        // Keep the receiver alive (so the failure is a deadline, not a
        // disconnect) but never reply.
        let job = Job::new(Uuid::new_v4(), vec![worker]);

        let future = job
            .start_get_weights(Some(Duration::from_millis(20)))
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("must not hang");

        assert!(matches!(result, Err(Error::Timeout(_))));
    }

    #[tokio::test]
    async fn fit_round_no_workers_errors_immediately() {
        let job = Job::new(Uuid::new_v4(), vec![]);
        let job_id = job.id();

        let Err(err) = job.start_fit_round(&tensor_map(1.0), None) else {
            panic!("expected an error");
        };

        assert!(matches!(err, Error::NoWorkers(id) if id == job_id));
    }

    #[tokio::test]
    async fn fit_round_all_workers_disconnected_errors_instead_of_hanging() {
        let job = Job::new(
            Uuid::new_v4(),
            vec![
                disconnected_worker("127.0.0.1:1"),
                disconnected_worker("127.0.0.1:2"),
            ],
        );

        let future = job.start_fit_round(&tensor_map(1.0), None).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("must not hang");

        assert!(matches!(result, Err(Error::AllWorkersFailed(id)) if id == job.id()));
    }

    #[tokio::test]
    async fn fit_round_tolerates_partial_failure() {
        let (good, mut good_rx) = connected_worker("127.0.0.1:1");
        let bad = disconnected_worker("127.0.0.1:2");
        let good_addr = good.addr();

        let job = Job::new(Uuid::new_v4(), vec![good, bad]);

        let future = job.start_fit_round(&tensor_map(1.0), None).unwrap();

        let responder = task::spawn(async move {
            good_rx.recv().await.expect("a request");
            job.set_result(good_addr, worker_update(2.0)).unwrap();
            job
        });

        let result = tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("must not hang")
            .unwrap();

        responder.await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].addr, good_addr);
        assert_eq!(
            result[0]
                .weights
                .get("a")
                .unwrap()
                .to_vec1::<f64>()
                .unwrap(),
            vec![2.0, 2.0]
        );
    }

    #[tokio::test]
    async fn fit_round_tolerates_a_panicking_worker_task() {
        // A worker task panicking (e.g. a bug in 'send_and_await') must not
        // poison the whole round the way 'JoinSet::join_all' would -- it
        // panics on the caller's task if any spawned task panicked.
        let job_id = Uuid::new_v4();
        let good_addr: SocketAddr = "127.0.0.1:1".parse().unwrap();

        let mut join_set: task::JoinSet<FitOutcome> = task::JoinSet::new();
        join_set.spawn(async { panic!("simulated worker task panic") });
        join_set.spawn(async move { (good_addr, Ok(worker_update(3.0))) });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            collect_fit_results(job_id, join_set),
        )
        .await
        .expect("must not hang")
        .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].addr, good_addr);
        assert_eq!(
            result[0]
                .weights
                .get("a")
                .unwrap()
                .to_vec1::<f64>()
                .unwrap(),
            vec![3.0, 3.0]
        );
    }

    #[tokio::test]
    async fn late_response_after_giveup_is_unknown_completer() {
        // A worker's pending entry is removed as soon as the coordinator
        // gives up on it (deadline elapsed), rather than being left behind
        // for some unrelated future 'insert' to silently overwrite. A late
        // response arriving while nothing is pending for that worker is
        // correctly rejected instead of resolving to nothing.
        //
        // This does NOT cover every case: if a *new* request for the same
        // address is registered before the late response arrives (e.g. the
        // next round starts first), the late response resolves the new
        // request instead of erroring -- see the doc comment on
        // 'Job::set_result' for why that residual race is a protocol gap,
        // not something fixed here.
        let (worker, _receiver) = connected_worker("127.0.0.1:1");
        let addr = worker.addr();
        let job = Job::new(Uuid::new_v4(), vec![worker]);

        let future = job
            .start_get_weights(Some(Duration::from_millis(20)))
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("must not hang");
        assert!(matches!(result, Err(Error::Timeout(_))));

        let late = job.set_result(addr, worker_update(99.0));
        assert!(matches!(late, Err(Error::UnknownCompleter(a)) if a == addr));
    }

    #[tokio::test]
    async fn set_result_unknown_completer() {
        let job = Job::new(Uuid::new_v4(), vec![]);
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();

        let err = job.set_result(addr, worker_update(1.0)).unwrap_err();

        assert!(matches!(err, Error::UnknownCompleter(a) if a == addr));
    }
}
