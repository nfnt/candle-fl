use candle_core::{Device, Error as CandleError};
use candle_nn::VarMap;
use safetensors::{SafeTensorError, SafeTensors};
use thiserror::Error;
use tokio::task;
use tracing::debug;

use crate::{
    candlefl::{FitResponse, WeightsResponse, WorkerMessage, worker_message},
    ml::Trainer,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("tensor operation failed: {0}")]
    Candle(#[from] CandleError),
    #[error("failed to (de)serialize weights: {0}")]
    SafeTensor(#[from] SafeTensorError),
    #[error("the model's parameter lock was poisoned")]
    LockPoisoned,
}

/// Build the `WeightsResponse` for a `WeightsRequest`: a freshly
/// initialized model's weights, serialized.
///
/// Offloads model construction to a blocking-pool thread, since it isn't
/// async-aware work.
///
/// # Errors
///
/// Returns an error if `trainer` fails to build the model or the
/// resulting weights fail to serialize.
///
/// # Panics
///
/// Panics if the blocking task panics.
pub async fn handle_weights_request<T: Trainer>(
    trainer: T,
    dev: Device,
    job_id: String,
) -> Result<WorkerMessage, Error> {
    debug!(job_id, "handling WeightsRequest");

    let varmap = task::spawn_blocking(move || trainer.prepare_weights(&dev))
        .await
        .expect("task doesn't panic")?;

    let weights = serialize(&varmap)?;

    Ok(WorkerMessage {
        message: Some(worker_message::Message::WeightsResponse(WeightsResponse {
            job_id,
            weights,
        })),
    })
}

/// Build the `FitResponse` for a `FitRequest`: train on local data starting
/// from `weights`, then serialize the result.
///
/// Offloads training to a blocking-pool thread, since it isn't async-aware
/// work.
///
/// # Errors
///
/// Returns an error if `weights` fails to deserialize, `trainer` fails to
/// train, or the resulting weights fail to serialize.
///
/// # Panics
///
/// Panics if the blocking task panics.
pub async fn handle_fit_request<T: Trainer>(
    trainer: T,
    dev: Device,
    job_id: String,
    weights: Vec<u8>,
) -> Result<WorkerMessage, Error> {
    debug!(job_id, "handling FitRequest");

    let varmap = task::spawn_blocking(move || {
        let weights = deserialize(&weights)?;
        trainer.train(&weights, &dev)
    })
    .await
    .expect("task doesn't panic")?;

    let weights = serialize(&varmap)?;

    Ok(WorkerMessage {
        message: Some(worker_message::Message::FitResponse(FitResponse {
            job_id,
            weights,
        })),
    })
}

/// Serialize a model's current parameters as safetensors bytes.
///
/// # Errors
///
/// Returns an error if the model's parameter lock is poisoned or the
/// tensors fail to serialize.
pub fn serialize(varmap: &VarMap) -> Result<Vec<u8>, Error> {
    // Collect into owned tensors and drop the lock before the
    // (comparatively slow) serialization step, rather than holding it for
    // the duration.
    let data: Vec<(String, candle_core::Tensor)> = {
        let tensor_data = varmap.data().lock().map_err(|_| Error::LockPoisoned)?;
        tensor_data
            .iter()
            .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
            .collect()
    };

    Ok(safetensors::serialize(data, None)?)
}

/// Deserialize safetensors bytes received from the coordinator.
///
/// # Errors
///
/// Returns an error if `data` isn't a valid safetensors buffer.
pub fn deserialize(data: &[u8]) -> Result<SafeTensors<'_>, SafeTensorError> {
    SafeTensors::deserialize(data)
}

#[cfg(test)]
mod tests {
    use candle_core::{Tensor, Var, safetensors::Load};

    use super::*;

    fn varmap_with(name: &str, values: Vec<f32>) -> VarMap {
        let varmap = VarMap::new();
        let mut data = varmap.data().lock().unwrap();
        data.insert(
            name.to_string(),
            Var::from_tensor(&Tensor::new(values, &Device::Cpu).unwrap()).unwrap(),
        );
        drop(data);
        varmap
    }

    /// A trainer that doesn't touch the network or a real dataset: it
    /// returns a fixed model for `WeightsRequest`s, and for `FitRequest`s
    /// echoes back each incoming tensor doubled, so a test can tell the
    /// two request kinds apart by their result.
    #[derive(Clone, Copy)]
    struct StubTrainer;

    impl Trainer for StubTrainer {
        fn prepare_weights(&self, _dev: &Device) -> Result<VarMap, CandleError> {
            Ok(varmap_with("a", vec![1.0, 1.0]))
        }

        fn train(&self, weights: &SafeTensors, dev: &Device) -> Result<VarMap, CandleError> {
            let varmap = VarMap::new();
            let mut data = varmap.data().lock().unwrap();
            for name in weights.names() {
                let tensor = weights.tensor(name)?.load(dev)?;
                let doubled = (&tensor + &tensor)?;
                data.insert(name.to_string(), Var::from_tensor(&doubled)?);
            }
            drop(data);
            Ok(varmap)
        }
    }

    #[derive(Clone, Copy)]
    struct FailingTrainer;

    impl Trainer for FailingTrainer {
        fn prepare_weights(&self, _dev: &Device) -> Result<VarMap, CandleError> {
            Err(CandleError::Msg("prepare_weights failed".to_string()))
        }

        fn train(&self, _weights: &SafeTensors, _dev: &Device) -> Result<VarMap, CandleError> {
            Err(CandleError::Msg("train failed".to_string()))
        }
    }

    #[test]
    fn serialize_deserialize_round_trip() {
        let varmap = varmap_with("a", vec![1.0, 2.0]);

        let bytes = serialize(&varmap).unwrap();
        let deserialized = deserialize(&bytes).unwrap();

        let tensor = deserialized
            .tensor("a")
            .unwrap()
            .load(&Device::Cpu)
            .unwrap();
        assert_eq!(tensor.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn deserialize_rejects_malformed_bytes() {
        assert!(deserialize(b"not a safetensors buffer").is_err());
    }

    #[tokio::test]
    async fn handle_weights_request_echoes_job_id_and_the_trainer_s_weights() {
        let message = handle_weights_request(StubTrainer, Device::Cpu, "job-1".to_string())
            .await
            .unwrap();

        let Some(worker_message::Message::WeightsResponse(resp)) = message.message else {
            panic!("expected a WeightsResponse variant, got {message:?}");
        };
        assert_eq!(resp.job_id, "job-1");

        let tensors = deserialize(&resp.weights).unwrap();
        let tensor = tensors.tensor("a").unwrap().load(&Device::Cpu).unwrap();
        assert_eq!(tensor.to_vec1::<f32>().unwrap(), vec![1.0, 1.0]);
    }

    #[tokio::test]
    async fn handle_fit_request_echoes_job_id_and_the_trainer_s_output() {
        let incoming = serialize(&varmap_with("a", vec![1.0, 1.0])).unwrap();

        let message = handle_fit_request(StubTrainer, Device::Cpu, "job-2".to_string(), incoming)
            .await
            .unwrap();

        let Some(worker_message::Message::FitResponse(resp)) = message.message else {
            panic!("expected a FitResponse variant, got {message:?}");
        };
        assert_eq!(resp.job_id, "job-2");

        let tensors = deserialize(&resp.weights).unwrap();
        let tensor = tensors.tensor("a").unwrap().load(&Device::Cpu).unwrap();
        // StubTrainer::train doubles each incoming tensor.
        assert_eq!(tensor.to_vec1::<f32>().unwrap(), vec![2.0, 2.0]);
    }

    #[tokio::test]
    async fn handle_weights_request_propagates_trainer_errors() {
        let err = handle_weights_request(FailingTrainer, Device::Cpu, "job-3".to_string())
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Candle(_)));
    }

    #[tokio::test]
    async fn handle_fit_request_propagates_trainer_errors() {
        let incoming = serialize(&VarMap::new()).unwrap();

        let err = handle_fit_request(FailingTrainer, Device::Cpu, "job-4".to_string(), incoming)
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Candle(_)));
    }

    #[tokio::test]
    async fn handle_fit_request_rejects_malformed_incoming_weights() {
        let err = handle_fit_request(
            StubTrainer,
            Device::Cpu,
            "job-5".to_string(),
            b"not safetensors".to_vec(),
        )
        .await
        .unwrap_err();

        // Deserializing inside 'train' goes through 'candle_core::Error's
        // own 'SafeTensorError' conversion before reaching this crate's
        // 'Error', so it surfaces as 'Candle', not 'SafeTensor' -- the
        // latter is only for a failure in this crate's own (de)serialize.
        assert!(matches!(err, Error::Candle(_)));
    }
}
