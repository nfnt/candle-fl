use std::{
    panic::AssertUnwindSafe,
    pin::Pin,
    sync::{Arc, Mutex},
};

use futures_util::FutureExt as _;
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tracing::info;
use uuid::Uuid;

use crate::{
    state::State,
    strategy::{Error, RoundUpdate, Strategy},
};

/// A stream of one run's round updates, ending in an error if the run
/// failed. Mirrors `Strategy::fit`'s contract: every successful update
/// arrives before the final error, if any -- `fit` only returns (closing
/// the updates side) once it has sent everything it's going to send.
pub type RunStream = Pin<Box<dyn Stream<Item = Result<RoundUpdate, Error>> + Send>>;

/// The one run a `Runner` currently allows to be in flight, identified by
/// its job id. `None` means idle.
type Slot = Arc<Mutex<Option<Uuid>>>;

/// Runs at most one `Strategy::fit` at a time.
///
/// A running fit is assumed to occupy every connected worker's resources,
/// so a second `start` while a run is live is rejected outright rather
/// than allowed to run concurrently and interleave training on the same
/// workers.
pub struct Runner<S> {
    strategy: Arc<S>,
    state: State,
    slot: Slot,
}

impl<S: Strategy> Runner<S> {
    pub fn new(strategy: S, state: State) -> Self {
        Self {
            strategy: Arc::new(strategy),
            state,
            slot: Arc::new(Mutex::new(None)),
        }
    }

    /// Start a fit over `num_rounds`, or reject it if one is already
    /// running.
    ///
    /// Never blocks on the run itself: the slot is only ever held long
    /// enough to check or claim it, never across an `await`. Note that the
    /// slot is released when the *fit task* ends, not when the returned
    /// stream is dropped early by a disconnected caller -- `fit` only
    /// notices a closed `updates` channel on its next send, i.e. at the end
    /// of whatever round is currently in progress (see `RoundUpdate`'s
    /// producers), so a `start` arriving in that window is still rejected.
    pub fn start(&self, num_rounds: usize) -> Result<RunStream, Error> {
        let mut slot = self.slot.lock().expect("runner slot lock");

        if let Some(&job_id) = slot.as_ref() {
            return Err(Error::AlreadyRunning(job_id));
        }

        let job_id = Uuid::new_v4();
        *slot = Some(job_id);
        drop(slot);

        let guard = RunGuard {
            slot: Arc::clone(&self.slot),
        };

        // A buffer of one keeps at most one round of weights in flight
        // ahead of what the consumer has actually processed.
        let (updates_tx, updates_rx) = mpsc::channel(1);
        let (error_tx, error_rx) = mpsc::channel(1);

        let strategy = Arc::clone(&self.strategy);
        let state = self.state.clone();
        tokio::spawn(async move {
            let _guard = guard;

            let job = state.add_job(job_id).await;

            info!(job_id = %job.id(), "starting job");

            // The job is removed on every exit path from 'fit', including a
            // panic partway through a round -- otherwise it leaks for the
            // lifetime of the coordinator process. 'catch_unwind' only
            // catches the panic long enough to run that cleanup; it's then
            // resumed so the task still ends up panicking, same as before
            // this wrapping.
            let result = AssertUnwindSafe(strategy.fit(&job, num_rounds, updates_tx))
                .catch_unwind()
                .await;
            job.remove().await;

            info!(job_id = %job.id(), "finished job");

            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    // If the caller already walked away, 'updates_tx' was
                    // dropped along with 'error_tx' having nowhere to go
                    // either -- nobody is left to report the failure to.
                    let _ = error_tx.send(e).await;
                }
                Err(panic) => std::panic::resume_unwind(panic),
            }
        });

        let updates = ReceiverStream::new(updates_rx).map(Ok);
        let errors = ReceiverStream::new(error_rx).map(Err);

        Ok(Box::pin(updates.chain(errors)))
    }
}

/// Releases a `Runner`'s slot on drop, regardless of why the run's task
/// ended.
struct RunGuard {
    slot: Slot,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        *self.slot.lock().expect("runner slot lock") = None;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};
    use tokio::sync::Notify;

    use super::*;
    use crate::state::Job;

    /// A test double that emits `rounds` successful updates sharing `job`'s
    /// id, then waits for `release` before finishing -- successfully, with
    /// an error (`fails`), or by panicking (`panics`) -- so a test can
    /// control exactly when and how a run ends.
    struct GatedStrategy {
        rounds: u64,
        fails: bool,
        panics: bool,
        release: Arc<Notify>,
    }

    impl Strategy for GatedStrategy {
        async fn fit(
            &self,
            job: &Job<'_>,
            _num_rounds: usize,
            updates: mpsc::Sender<RoundUpdate>,
        ) -> Result<(), Error> {
            for round in 1..=self.rounds {
                let mut weights = HashMap::new();
                weights.insert(
                    "a".to_string(),
                    Tensor::new(vec![1.0], &Device::Cpu).unwrap(),
                );

                if updates
                    .send(RoundUpdate {
                        job_id: job.id(),
                        round,
                        weights,
                        metrics: None,
                    })
                    .await
                    .is_err()
                {
                    return Ok(());
                }
            }

            self.release.notified().await;

            if self.panics {
                panic!("GatedStrategy: intentional test panic");
            } else if self.fails {
                Err(Error::State(crate::state::Error::NoResults))
            } else {
                Ok(())
            }
        }
    }

    async fn collect(stream: RunStream) -> Vec<Result<RoundUpdate, Error>> {
        stream.collect().await
    }

    #[tokio::test]
    async fn a_second_start_is_rejected_while_a_run_is_in_progress() {
        let release = Arc::new(Notify::new());
        let runner = Runner::new(
            GatedStrategy {
                rounds: 0,
                fails: false,
                panics: false,
                release: Arc::clone(&release),
            },
            State::new(),
        );

        let _first = runner.start(1).unwrap();

        // Rounds: 0, so no update has been produced yet -- but the job id
        // is minted by 'start' itself, so it's already known.
        let busy = runner.start(1).err().unwrap();
        let Error::AlreadyRunning(busy_job_id) = busy else {
            panic!("expected Error::AlreadyRunning, got {busy:?}");
        };
        assert_ne!(busy_job_id, Uuid::nil());

        release.notify_one();
    }

    #[tokio::test]
    async fn already_running_reports_the_running_job_s_id() {
        let release = Arc::new(Notify::new());
        let runner = Runner::new(
            GatedStrategy {
                rounds: 1,
                fails: false,
                panics: false,
                release: Arc::clone(&release),
            },
            State::new(),
        );

        let mut first = runner.start(1).unwrap();

        let update = first.next().await.unwrap().unwrap();

        let busy = runner.start(1).err().unwrap();
        let Error::AlreadyRunning(busy_job_id) = busy else {
            panic!("expected Error::AlreadyRunning, got {busy:?}");
        };
        assert_eq!(busy_job_id, update.job_id);

        release.notify_one();
        drop(first);
    }

    #[tokio::test]
    async fn the_slot_is_released_after_a_successful_run() {
        let release = Arc::new(Notify::new());
        let runner = Runner::new(
            GatedStrategy {
                rounds: 0,
                fails: false,
                panics: false,
                release: Arc::clone(&release),
            },
            State::new(),
        );

        let first = runner.start(1).unwrap();
        release.notify_one();

        let responses = collect(first).await;
        assert!(responses.is_empty());

        assert!(runner.start(1).is_ok());
    }

    #[tokio::test]
    async fn the_slot_is_released_after_a_failed_run() {
        let release = Arc::new(Notify::new());
        let runner = Runner::new(
            GatedStrategy {
                rounds: 0,
                fails: true,
                panics: false,
                release: Arc::clone(&release),
            },
            State::new(),
        );

        let first = runner.start(1).unwrap();
        release.notify_one();

        let responses = collect(first).await;
        assert_eq!(responses.len(), 1);
        assert!(responses[0].is_err());

        assert!(runner.start(1).is_ok());
    }

    #[tokio::test]
    async fn the_slot_is_released_when_the_consumer_drops_the_stream_mid_run() {
        let release = Arc::new(Notify::new());
        let runner = Runner::new(
            GatedStrategy {
                rounds: 2,
                fails: false,
                panics: false,
                release: Arc::clone(&release),
            },
            State::new(),
        );

        let mut first = runner.start(1).unwrap();
        first.next().await.unwrap().unwrap(); // consume round 1

        // Walking away mid-run: 'fit' only notices once it tries to send
        // round 2 into the now-closed channel, at which point it returns
        // early without ever waiting on 'release'.
        drop(first);

        for _ in 0..100 {
            if runner.start(1).is_ok() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the slot was never released after the stream was dropped");
    }

    #[tokio::test]
    async fn a_run_streams_every_update_before_its_final_error() {
        let release = Arc::new(Notify::new());
        let runner = Runner::new(
            GatedStrategy {
                rounds: 2,
                fails: true,
                panics: false,
                release: Arc::clone(&release),
            },
            State::new(),
        );

        let stream = runner.start(2).unwrap();
        release.notify_one();

        let responses = collect(stream).await;

        assert_eq!(responses.len(), 3);
        assert!(responses[0].is_ok());
        assert!(responses[1].is_ok());
        assert!(responses[2].is_err());
    }

    #[tokio::test]
    async fn the_job_is_removed_from_state_even_if_fit_panics() {
        let release = Arc::new(Notify::new());
        let state = State::new();
        let runner = Runner::new(
            GatedStrategy {
                rounds: 1,
                fails: false,
                panics: true,
                release: Arc::clone(&release),
            },
            state.clone(),
        );

        let mut stream = runner.start(1).unwrap();
        let update = stream.next().await.unwrap().unwrap();
        let job_id = update.job_id;

        release.notify_one();

        // The panic unwinds past this without ever reaching 'error_tx', so
        // the stream just ends once both channels close -- which only
        // happens once the spawned task itself has finished, i.e. after its
        // (panic-safe) job cleanup has already run.
        let responses = collect(stream).await;
        assert!(responses.is_empty());

        let err = Job::probe(&state, job_id).get_weights().await.unwrap_err();
        assert!(
            matches!(err, crate::state::Error::UnknownJob(id) if id == job_id),
            "expected the job to have been removed, got {err:?}"
        );
    }
}
