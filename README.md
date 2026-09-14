# Federated Learning with Candle

This is a proof-of-concept of federated learning using the
[Candle](https://github.com/huggingface/candle) framework with Rust.
It implements the [FedAvg](https://arxiv.org/abs/1602.05629) algorithm for
horizontal federated learning with data provided by workers.

Multiple workers connect to a coordinator, which orchestrates them to train a
model on their local data. The focus of this code is on the distributed system
needed for federated learning, not on the machine learning model. As such, the
model is the classic [LeNet](https://ieeexplore.ieee.org/document/726791) CNN
and each worker trains on the same
[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

## Architecture

The components communicate using [gRPC](https://grpc.io/) in a client-server
model.

### Coordinator

A coordinator manages the training process.
It provides a publish-subscribe service for workers to connect to, send
training requests, and receive training results. These results are then
aggregated by the coordinator.

### Worker

Workers have access to their local training data. They connect to a coordinator
and wait for training requests. When they receive a training request, they train
a model on their local data and send the trained model back to the coordinator.

## Usage

Ensure that you have `protoc` installed and in `$PATH`.
Start the coordinator with `cargo run -r --bin coordinator` then connect one
or more workers with `cargo run -r --bin worker`. With the workers connected,
start a training run with `cargo run -r --bin start_training 10`. This will
train the model for 10 rounds. For example:

```shell
$ cargo run -r --bin coordinator &
$ cargo run -r --bin worker &
$ cargo run -r --bin worker &
$ cargo run -r --bin start_training 10
```

The coordinator waits indefinitely for a still-connected worker to respond
by default -- round durations vary too widely (CPU vs. GPU, dataset size)
for a safe default deadline. Pass `--worker-deadline-secs <N>` to the
coordinator to opt into a hard per-request cap instead; disconnected
workers are detected and don't hang a round regardless of this setting.

## Development

Ensure that you have `protoc` installed and in `$PATH` (needed by both crates'
`build.rs` to compile the protos in `api/proto/`).

```shell
$ cargo fmt --all --check                               # formatting
$ cargo clippy --workspace --all-targets -- -D warnings # lints
$ cargo test --workspace --all-targets                  # unit + integration tests
$ cargo test --workspace --doc                          # doc tests
```

### Known limitations

- A worker that's still connected but internally wedged (e.g. stuck
  mid-epoch) is indistinguishable from one that's simply still training.
  See the module doc on `coordinator::state::job` for what the coordinator
  *does* detect (disconnects, half-open connections) and why a fixed timeout
  isn't used as a substitute.
- `coordinator::state::job::Job::set_result` documents a narrow residual
  race: a worker's response arriving right as the coordinator moves on to
  the next round can, in rare timing, resolve the wrong round. Closing
  that gap needs a round/request token in the protocol, not more
  bookkeeping.
