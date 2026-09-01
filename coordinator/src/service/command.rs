use tonic::{Request, Response, Status};

use crate::{
    candlefl::{TrainRequest, TrainResponse, command_server::Command},
    strategy::Strategy,
};

pub struct CommandService<S> {
    strategy: S,
}

impl<S: Strategy> CommandService<S> {
    pub const fn new(strategy: S) -> Self {
        Self { strategy }
    }
}

#[tonic::async_trait]
impl<S: Strategy> Command for CommandService<S> {
    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let request = request.into_inner();

        // 'rounds' is u64 over the wire; clamp rather than wrap on 32-bit
        // targets where 'usize' is narrower.
        let rounds = usize::try_from(request.rounds).unwrap_or(usize::MAX);

        let weights = self
            .strategy
            .fit(rounds)
            .await
            .map_err(|e| Status::internal(format!("failed to train model: {e}")))?;

        let serialized_weights = safetensors::serialize(weights, None)
            .map_err(|e| Status::internal(format!("invalid weights: {e}")))?;

        Ok(Response::new(TrainResponse {
            weights: serialized_weights,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};

    use super::*;
    use crate::strategy::Error;

    #[derive(Clone, Copy)]
    struct FixedWeightsStrategy;

    impl Strategy for FixedWeightsStrategy {
        async fn fit(&self, _num_rounds: usize) -> Result<HashMap<String, Tensor>, Error> {
            let mut map = HashMap::new();
            map.insert(
                "a".to_string(),
                Tensor::new(vec![1.0, 1.0], &Device::Cpu).unwrap(),
            );
            Ok(map)
        }
    }

    #[derive(Clone, Copy)]
    struct FailingStrategy;

    impl Strategy for FailingStrategy {
        async fn fit(&self, _num_rounds: usize) -> Result<HashMap<String, Tensor>, Error> {
            Err(Error::State(crate::state::Error::NoResults))
        }
    }

    #[tokio::test]
    async fn train_returns_the_strategy_s_weights_serialized() {
        let service = CommandService::new(FixedWeightsStrategy);

        let response = service
            .train(Request::new(TrainRequest { rounds: 3 }))
            .await
            .unwrap();

        let weights =
            candle_core::safetensors::load_buffer(&response.into_inner().weights, &Device::Cpu)
                .unwrap();
        assert_eq!(
            weights.get("a").unwrap().to_vec1::<f64>().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[tokio::test]
    async fn train_maps_a_strategy_error_to_an_internal_status() {
        let service = CommandService::new(FailingStrategy);

        let status = service
            .train(Request::new(TrainRequest { rounds: 1 }))
            .await
            .unwrap_err();

        assert_eq!(status.code(), tonic::Code::Internal);
    }
}
