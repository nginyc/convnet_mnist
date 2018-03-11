# ConvNet Training for MNIST

## Technologies Used

- Tensorflow for abstraction of convolutional neural network implementation & training, as well as the retrieval of MNIST data
- RPyC for abstraction of RPC implementation

## Installation

Tested on:

- Windows 10, with Python 3.6.4 64-bit
- MacOS X MacOS High Sierra, with Python 3.6.4

Run:

```sh
pip install -r requirements.txt
```

On Windows, ensure that you are running Python 3.6.x 64-bit for the installation of Tensorflow. If tensorflow installation fails, refer to the guide at https://www.tensorflow.org/install/.

## Running

To spawn the master process:

```sh
python master.py
```

After spawning the master process, to spawn a worker process:

```sh
python worker.py
```

Multiple worker processes can be spawned.