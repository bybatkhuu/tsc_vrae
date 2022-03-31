
tsc_vrae = {
    'vrae':
    {
        'cuda': True,
        'gpu_id': 0,
        'batch_size': 16,
        'n_epochs': 200,
        'print_every': 1,
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
    },
    'dbscan':
    {
        'eps': 0.015,
        'min_samples': 3,
        'algorithm': 'auto',
        'leaf_size': 30,
        'n_jobs': None
    },
    'kmeans':
    {
        'n_clusters': 5,
        'n_init': 10,
        'max_iter': 300,
        'algorithm': 'auto',
        'verbose': 0,
        'random_state': None
    },
    'const':
    {
        'model_files_suffix': 'tsc_vrae',
        'low_lim_train_data': 10
    }
}
