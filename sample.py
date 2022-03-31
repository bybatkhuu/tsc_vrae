#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

from tsc_vrae import TscVRAE


model_name = 'model_name'
model_dir = f"{os.getcwd()}/models"
vrae_kwargs = {
    'cuda': True,
    'gpu_id': 0,
    'batch_size': 16,
    'n_epochs': 200,
    'print_every': 10,
    'save_each_epoch': 10,
    'hidden_size': 200,
    'hidden_layer_depth': 2,
    'latent_length': 10,
    'optimizer': 'Adam',
    'learning_rate': 1e-5,
    'dropout_rate': 0.2,
    'max_grad_norm': 5,
    'loss': 'MSELoss',
    'block': 'LSTM',
    'clip': True
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
