# ConvNet Training for MNIST

## Technologies Used

- Tensorflow for abstraction of convolutional neural network implementation & training, as well as the retrieval of MNIST data
- RPyC for abstraction of RPC implementation

## Installation

Run:

```sh
pip install -r requirements.txt
```

## Running

To spawn the master process:

```sh
python master.py
```

To spawn a worker process:

```sh
python worker.py
```