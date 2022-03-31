# -*- coding: utf-8 -*-

from cerberus import Validator


_vrae_schema = {
    'cuda': { 'type': 'boolean', 'coerce': bool },
    'gpu_id': { 'type': 'integer', 'coerce': int },
    'batch_size': { 'type': 'integer', 'coerce': int },
    'n_epochs': { 'type': 'integer', 'coerce': int },
    'print_every': { 'type': 'integer', 'coerce': int },
    'save_each_epoch': { 'type': 'integer', 'coerce': int },
    'hidden_size': { 'type': 'integer', 'coerce': int },
    'hidden_layer_depth': { 'type': 'integer', 'coerce': int },
    'latent_length': { 'type': 'integer', 'coerce': int },
    'learning_rate': { 'type': 'number', 'coerce': float },
    'dropout_rate': { 'type': 'number', 'coerce': float },
    'optimizer': { 'type': 'string' },
    'clip': { 'type': 'boolean', 'coerce': bool },
    'max_grad_norm': { 'type': 'integer', 'coerce': int },
    'loss': { 'type': 'string' },
    'block': { 'type': 'string' }
}
vrae_validator = Validator(_vrae_schema)
vrae_validator.allow_unknown = True
vrae_validator.require_all = True


_dbscan_schema = {
    'eps': { 'type': 'number', 'coerce': float },
    'min_samples': { 'type': 'integer', 'coerce': int },
    'algorithm': { 'type': 'string' },
    'leaf_size': { 'type': 'integer', 'coerce': int },
    'n_jobs': { 'type': 'integer', 'coerce': int, 'nullable': True }
}
dbscan_validator = Validator(_dbscan_schema)
dbscan_validator.allow_unknown = True
dbscan_validator.require_all = True


_kmeans_schema = {
    'n_clusters': { 'type': 'integer', 'coerce': int },
    'n_init': { 'type': 'integer', 'coerce': int },
    'max_iter': { 'type': 'integer', 'coerce': int },
    'algorithm': { 'type': 'string' },
    'verbose': { 'type': 'integer', 'coerce': int },
    'random_state': { 'type': 'integer', 'coerce': int, 'nullable': True }
}
kmeans_validator = Validator(_kmeans_schema)
kmeans_validator.allow_unknown = True
kmeans_validator.require_all = True


_config_schema = {
    'tsc_vrae':
    {
        'type': 'dict',
        'schema':
        {
            'vrae':
            {
                'type': 'dict',
                'schema': _vrae_schema
            },
            'dbscan':
            {
                'type': 'dict',
                'schema': _dbscan_schema
            },
            'kmeans':
            {
                'type': 'dict',
                'schema': _kmeans_schema
            },
            'const':
            {
                'type': 'dict',
                'schema':
                {
                    'model_files_suffix': { 'type': 'string' },
                    'low_lim_train_data': { 'type': 'integer', 'coerce': int }
                }
            }
        }
    }
}
config_validator = Validator(_config_schema)
config_validator.allow_unknown = True
config_validator.require_all = True
