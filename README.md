# Federated Learning with Candle

This is a proof-of-concept of federated learning using the
[Candle](https://github.com/huggingface/candle) framework with Rust.
It implements the [FedAvg](https://arxiv.org/abs/1602.05629) algorithm for
horizontal federated learning with data provided by workers.

Multiple workers connect to a coordinator, which orchestrates them to train a
model on their local data. The focus of this code is on the distributed system
needed for federated learning, not on the machine learning model. As such, the
model is a simple linear classification model and each worker trains on the same
MNIST dataset.

## Architecture

The components communicate using [gRPC](https://grpc.io/) in a client-server
model.

### Coordinator

A coordinator manages the training process.
It provides a publish-subscribe service for workers to connect to, send
training requests, and receive training results. These results are then
aggregated by the coordinator.
It also provides a service to start a training run.

### Worker

Workers have access to their local training data. They connect to a coordinator
and wait for training requests. When they receive a training request, they train
a model on their local data and send the trained model back to the coordinator.

## Usage

Build the project with `cargo build -r` and start the coordinator with
`cargo run -r --bin coordinator` then connect one or more workers with
`cargo run -r --bin worker`. With the workers connected, start a training run
with `cargo run -r --bin start_training 10`. This will train the model for 10
rounds. For example:

```shell
$ cargo run -r --bin coordinator &
$ cargo run -r --bin worker &
$ cargo run -r --bin worker &
$ cargo run -r --bin start_training 10
```
