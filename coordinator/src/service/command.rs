use tonic::{Request, Response, Status};

use crate::{
    candlefl::{command_server::Command, TrainRequest, TrainResponse},
    state::State,
    strategy::FedAvg,
};

pub struct CommandService {
    state: State,
}

impl CommandService {
    pub fn new(state: State) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl Command for CommandService {
    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let request = request.into_inner();

        let strategy = FedAvg::new(self.state.clone());

        let weights = strategy
            .fit(request.rounds as usize)
            .await
            .map_err(|e| Status::internal(format!("failed to train model: {e}")))?;

        let serialized_weights = safetensors::serialize(weights, &None)
            .map_err(|e| Status::internal(format!("invalid weights: {e}")))?;

        Ok(Response::new(TrainResponse {
            weights: serialized_weights,
        }))
    }
}
