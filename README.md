
# TSC_VRAE Module

Time series clustering VRAE.

## Features

* Time-series clustering
* Variational recurrent auto-encoders
* Pytorch/Tensorflow

---

## Prerequisites

* **Python (>= v3.7.11)**
* For **NVIDIA GPU**:
    * **NVIDIA GPU driver (at least >= v418.39)** - [https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/nvidia-driver-linux.md](https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/nvidia-driver-linux.md)
    * **NVIDIA CUDA (v10.1)** and **cuDNN (v7.6.5)** - [https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/cuda-linux.md](https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/cuda-linux.md)

## Getting started

Clone the project

```bash
git clone https://github.com/bybatkhuu/tsc_vrae.git
cd tsc_vrae
```

Install dependencies

```bash
# For CPU:
cat requirements.txt | xargs -n 1 -L 1 pip3 install --timeout 60 --no-cache-dir

# For GPU:
cat requirements.gpu.txt | xargs -n 1 -L 1 pip3 install --timeout 60 --no-cache-dir
```

## Usage/Examples

```python
import os
import numpy as np
from tsc_vrae import TscVRAE

model_name = 'model_name'
model_dir = f"{os.getcwd()}/models"
vrae_kwargs = {
    'hidden_size': 200,
    'hidden_layer_depth': 2,
    'latent_length': 10,
    'batch_size': 16,
    'learning_rate': 1e-5,
    'dropout_rate': 0.2,
    'n_epochs': 200,
    'optimizer': 'Adam',
    'cuda': True,
    'print_every': 30,
    'clip': True,
    'max_grad_norm': 5,
    'loss': 'MSELoss',
    'block': 'LSTM'
}

X = np.array([[[1, 1], [1, 1]],
              [[2, 1], [1, 1]],
              [[3, 3], [3, 3]],
              [[3, 3], [3, 3]],
              [[10, 2], [1, 2]],
              [[10, 2], [1, 2]],
              [[10, 2], [1, 2]],
              [[3, 3], [3, 4.5]],
              [[1, 2], [1, 2]],
              [[1, 2], [1, 2]]])

tsc_vrae = TscVRAE(model_name=model_name, model_dir=model_dir, vrae_kwargs=vrae_kwargs)
if not tsc_vrae.is_trained:
    tsc_vrae.train(X)

cluster_ids = tsc_vrae.cluster(X)
print(f"Cluster IDs: {cluster_ids}")
```

---

## Running Tests

To run tests, run the following command:

```python
python -m tests/test*.py
```

## Environment Variables

You can use the following environment variables to your .env file:

```bash
ENV=development
DEBUG=true
APP_NAME=tsc_vrae
LOGS_DIR="/var/log/app"
```

## Documentation

* Hyper-parameters

## Roadmap

* Support Tensorflow
* Add module API documentation
* Add tests
* Add more integrations

---

## References

* [https://github.com/tejaslodaya/timeseries-clustering-vae](https://github.com/tejaslodaya/timeseries-clustering-vae)
* [https://github.com/RobRomijnders/AE_ts](https://github.com/RobRomijnders/AE_ts)
* [https://arxiv.org/pdf/1412.6581.pdf](https://arxiv.org/pdf/1412.6581.pdf)
