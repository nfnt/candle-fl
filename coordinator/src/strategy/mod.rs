use std::{collections::HashMap, future::Future};

use candle_core::Tensor;
use thiserror::Error as ThisError;

pub use fed_avg::FedAvg;

mod fed_avg;

/// Errors produced by a `Strategy`.
#[derive(Debug, ThisError)]
pub enum Error {
    #[error(transparent)]
    State(#[from] crate::state::Error),
}

/// A federated learning strategy: how model weights are fit across
/// connected workers over a number of rounds.
///
/// This uses return-position `impl Trait` with an explicit `+ Send` bound
/// rather than a plain `async fn` in the trait. `CommandService::train`
/// (see `crate::service::command`) is boxed as `dyn Future + Send` by
/// `#[tonic::async_trait]`; a bare `async fn` in a trait produces an opaque
/// return type with no `Send` bound, which would make that outer future
/// non-`Send` for a generic `CommandService<S>` and fail to compile.
///
/// There is deliberately no `dyn Strategy` use anywhere: `TrainRequest`
/// carries no strategy selector, so `CommandService` is monomorphized once
/// over `FedAvg`. If a strategy-by-name field is ever added to the proto,
/// switch this to `#[async_trait]` instead, which supports dynamic
/// dispatch at the cost of a `Box::pin` per call.
pub trait Strategy: Send + Sync + 'static {
    /// Fit model weights over `num_rounds` rounds of training.
    fn fit(
        &self,
        num_rounds: usize,
    ) -> impl Future<Output = Result<HashMap<String, Tensor>, Error>> + Send;
}
