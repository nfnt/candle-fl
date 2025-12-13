use std::{collections::HashMap, net::SocketAddr};

use candle_core::Tensor;
use tokio::{sync::oneshot, task};
use tonic::Status;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::{
    candlefl::{CoordinatorMessage, FitRequest, WeightsRequest, coordinator_message},
    state::worker::Worker,
};

pub struct Job {
    id: Uuid,
    workers: Vec<Worker>,
    // Tasks wait for responses from workers.
    // They are removed once the response is received in 'set_result'.
    tasks: HashMap<SocketAddr, Box<oneshot::Sender<HashMap<String, Tensor>>>>,
}

impl Job {
    pub fn new(workers: Vec<Worker>) -> Self {
        Self {
            id: Uuid::new_v4(),
            workers,
            tasks: HashMap::new(),
        }
    }

    pub const fn id(&self) -> Uuid {
        self.id
    }

    pub fn get_weights(
        &mut self,
        response: oneshot::Sender<Result<HashMap<String, Tensor>, anyhow::Error>>,
    ) {
        let job_id = self.id;

        let message = CoordinatorMessage {
            message: Some(coordinator_message::Message::WeightsRequest(
                WeightsRequest {
                    job_id: job_id.into(),
                },
            )),
        };

        let _task = self
            .workers
            .first()
            .map(|worker| {
                let worker = worker.clone();
                let message = message.clone();

                let (sender, receiver) = oneshot::channel();
                self.tasks.insert(worker.addr(), Box::new(sender));

                task::spawn(async move {
                    debug!(
                        job_id = %job_id,
                        addr = %worker.addr(),
                        "sending WeightsRequest"
                    );

                    if let Err(e) = worker
                        .sender()
                        .send(Result::<_, Status>::Ok(message.clone()))
                        .await
                    {
                        warn!(
                            job_id = %job_id,
                            addr = %worker.addr(),
                            error = %e,
                            "failed to send WeightsRequest"
                        );
                    }

                    let weights = receiver.await.map_err(|e| anyhow::anyhow!(e));
                    if response.send(weights).is_err() {
                        warn!("failed to set response");
                    }
                })
            })
            .unwrap();
    }

    pub fn fit_round(
        &mut self,
        weights: &HashMap<String, Tensor>,
        response: oneshot::Sender<Result<Vec<HashMap<String, Tensor>>, anyhow::Error>>,
    ) {
        let job_id = self.id;

        let message = CoordinatorMessage {
            message: Some(coordinator_message::Message::FitRequest(FitRequest {
                job_id: job_id.into(),
                weights: serialize(weights).unwrap(),
            })),
        };

        let mut join_set = task::JoinSet::new();

        for worker in &self.workers {
            let worker = worker.clone();
            let message = message.clone();

            let (sender, receiver) = oneshot::channel();
            self.tasks.insert(worker.addr(), Box::new(sender));

            join_set.spawn(async move {
                debug!(
                    job_id = %job_id,
                    addr = %worker.addr(),
                    "sending FitRequest"
                );

                if let Err(e) = worker.sender().send(Result::<_, Status>::Ok(message)).await {
                    warn!(
                        job_id = %job_id,
                        addr = %worker.addr(),
                        error = %e,
                        "failed to send FitRequest"
                    );
                }

                receiver.await
            });
        }

        let _task = task::spawn(async move {
            let results = join_set
                .join_all()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!(e));

            if response.send(results).is_err() {
                warn!("failed to set response");
            }
        });
    }

    pub fn set_result(
        &mut self,
        addr: SocketAddr,
        weights: HashMap<String, Tensor>,
        response: oneshot::Sender<Result<(), anyhow::Error>>,
    ) {
        if let Some(sender) = self.tasks.remove(&addr) {
            if response
                .send(
                    sender
                        .send(weights)
                        .map_err(|_| anyhow::anyhow!("failed to set result for {addr}")),
                )
                .is_err()
            {
                warn!("failed to set response");
            }
        } else if response
            .send(Err(anyhow::anyhow!("completer not found for {addr}")))
            .is_err()
        {
            warn!("failed to set response");
        }
    }
}

fn serialize(weights: &HashMap<String, Tensor>) -> Result<Vec<u8>, safetensors::SafeTensorError> {
    safetensors::serialize(weights, &None)
}
