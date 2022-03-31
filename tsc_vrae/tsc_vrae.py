# -*- coding: utf-8 -*-

import os
import json
# import traceback
from collections import Counter

import torch
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from torch.utils.data import TensorDataset
from sklearn.cluster import DBSCAN, KMeans
from os import path
from scipy.spatial import distance
from datetime import datetime

from el_logging import logger
from el_validator import checkers

from .config import config
from .helpers import utils
from .helpers.validators import vrae_validator, dbscan_validator, kmeans_validator
from .helpers.evaluation import summary
from .vrae.vrae import VRAE


class TscVRAE:
    """Time-series clustering VRAE base class.

    Attributes:
        _MODEL_FILES_SUFFIX (str       ): Suffix name for model files. Defaults to 'tsc_vrae'.
        _LOW_LIM_TRAIN_DATA (int       ): Minimum data size to train. Defaults to 10.

        model_name          (str       ): Model name to save and load.
        model_dir           (str       ): Model directory to save and load.
        vrae_kwargs         (dict      ): VRAE model's arguments as a dictionary. Defaults to <config.tsc_vrae.vrae>.
        model_vrae          (VRAE      ): Main VRAE model.
        latent_features     (np.ndarray): Latent features extracted from trained VRAE model.
        cluster_method      (str       ): Clustering method's name. Defaults to 'DBSCAN'.
        cluster_kwargs      (dict      ): Clustering model's arguments as a dictionary. Defaults to <config.tsc_vrae.dbscan>.
        is_trained          (bool      ): Indicate TscVRAE model is trained or not.

    Methods:
        is_model_files_exist(): Check <model_name> model files exist in <model_dir> directory.
        _get_files_meta()     : Get TSC_VRAE model files metadata.

        load()                : Load <model_name> model files from <model_dir> directory.
        delete()              : Delete <model_name> model files from <model_dir> directory.
        save()                : Save <model_name> model files to <model_dir> directory.
        train()               : Train TSC_VRAE model and save to files.
        extract_features()    : Extract latent features from data input based on trained VRAE model.
        cluster()             : Clustering method for new data input.
        cluster_with_data_id(): Clustering method with data_id.
        cluster_with_label()  : Clustering method with label.
        inference()           : Clustering inference.
    """

    _MODEL_FILES_SUFFIX = config.tsc_vrae.const.model_files_suffix
    _LOW_LIM_TRAIN_DATA = config.tsc_vrae.const.low_lim_train_data

    def __init__(self, model_name: str, model_dir: str, vrae_kwargs: dict=None, cluster_method: str=None, cluster_kwargs: dict=None, auto_load: bool=True, cluster_id: str=None):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_files_meta = self._get_files_meta(self.model_name, self.model_dir)

        if auto_load and self.is_model_files_exist(self.model_name, self.model_dir):
            self.load()
            if vrae_kwargs:
                logger.warning("Can't change already trained model parameters!")
        else:
            if vrae_kwargs:
                self.vrae_kwargs = vrae_kwargs

        if cluster_method:
            self.cluster_method = cluster_method

        if cluster_kwargs:
            self.cluster_kwargs = cluster_kwargs

        if cluster_id:
            self.centroid_info = self.load_centroid_info(cluster_id)
            self.cluster_info = self.load_cluster_info(cluster_id)
        else:
            self.centroid_info = None
            self.cluster_info = None

    ### STATIC METHODS ###
    @staticmethod
    @validate_arguments
    def is_model_files_exist(model_name: str, model_dir: str, any_file: bool=False):
        """Check <model_name> model files exist in <model_dir> directory.

        Args:
            model_name (str , required): Model name to check.
            model_dir  (str , required): Model directory to check.
            any_file   (bool, optional): Indicate to check any model file exists or not. Defaults to False.

        Returns:
            bool: Indicate model files exists or not.
        """

        _model_files_meta = TscVRAE._get_files_meta(model_name, model_dir)
        if not os.path.isfile(_model_files_meta['model_vrae_file_path']):
            logger.debug(f"'{_model_files_meta['model_vrae_file_path']}' file doesn't exist!")
            return False
        else:
            if any_file:
                return True

        if not os.path.isfile(_model_files_meta['latent_features_file_path']):
            logger.debug(f"'{_model_files_meta['latent_features_file_path']}' file doesn't exist!")
            return False
        else:
            if any_file:
                return True

        if not TscVRAE._is_meta_file_exist(model_name, model_dir):
            return False
        else:
            if any_file:
                return True

        return True


    @staticmethod
    @validate_arguments
    def _is_meta_file_exist(model_name: str, model_dir: str):
        """Checker method for meta file exist or not.

        Args:
            model_name (str, required): Model name of meta files.
            model_dir  (str, required): Directory path to check meta files.

        Returns:
            bool: True when meta file exist, False when doesn't exist.
        """

        _model_files_meta = TscVRAE._get_files_meta(model_name, model_dir)
        if not os.path.isfile(_model_files_meta['meta_file_path']):
            logger.debug(f"'{_model_files_meta['meta_file_path']}' file doesn't exist!")
            return False

        return True


    @staticmethod
    @validate_arguments
    def _get_files_meta(model_name: str, model_dir: str):
        """Get TSC_VRAE model files metadata.

        Args:
            model_name (str, required): Model name of TSC_VRAE model files.
            model_dir  (str, required): Directory path to read and save TSC_VRAE model files.

        Raises:
            ValueError: Raise error if 'model_name' argument value is empty.
            ValueError: Raise error if 'model_dir' argument value is empty.

        Returns:
            dict: TSC_VRAE model files metadata as dictionary.
        """

        try:
            model_name = model_name.strip()
            if checkers.is_empty(model_name):
                raise ValueError("'model_name' argument value is empty!")
            model_dir = model_dir.strip()
            if checkers.is_empty(model_dir):
                raise ValueError("'model_dir' argument value is empty!")
        except ValueError as err:
            logger.error(err)
            raise

        _base_filename = os.path.splitext(model_name)[0]
        _model_vrae_filename = model_name

        _model_vrae_file_path = os.path.join(model_dir, _model_vrae_filename)
        _latent_features_filename = f"{_base_filename}.latent_faetures.npy"
        _latent_features_file_path = os.path.join(model_dir, _latent_features_filename)
        _meta_filename = f"{_base_filename}.meta.json"
        _meta_file_path = os.path.join(model_dir, _meta_filename)

        _model_files_meta = {
            'model_name': model_name,
            'model_dir': model_dir,
            'base_filename': _base_filename,
            'model_vrae_filename': _model_vrae_filename,
            'model_vrae_file_path': _model_vrae_file_path,
            'latent_features_filename': _latent_features_filename,
            'latent_features_file_path': _latent_features_file_path,
            'meta_filename': _meta_filename,
            'meta_file_path': _meta_file_path
        }
        return _model_files_meta
    ### STATIC METHODS ###


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def train(self, X: np.ndarray, save: bool=True):
        """Train TSC_VRAE model and save to files.

        Args:
            X    (np.ndarray, required): Input dataset as numpy (Should be 3D).
            save (bool      , optional): Save TSC_VRAE model to files after training finished. Defaults to True.

        Raises:
            ValueError: Raise error if 'X' argument numpy array is empty.
            ValueError: Raise error if 'X' argument numpy array dimension is not [3D].
            ValueError: Raise error if 'X' argument size is lower than 10.
            ValueError: Raise error if 'X' argument sequence_length is lower than 1.
            ValueError: Raise error if 'X' argument number_of_features is lower than 1.
        """

        try:
            if checkers.is_empty(X):
                raise ValueError("'X' argument numpy array is empty!")

            if X.ndim != 3:
                raise ValueError(f"'X' argument numpy array dimension [{X.ndim}D] is invalid, should be [3D]!")

            if X.shape[0] < TscVRAE._LOW_LIM_TRAIN_DATA:
                raise ValueError(f"'X' argument size '{X.shape[0]}' is invalid, should be higher than => '{TscVRAE._LOW_LIM_TRAIN_DATA}'!")

            if X.shape[1] <= 0:
                raise ValueError(f"'X' argument sequence_length '{X.shape[1]}' is invalid, should be higher than > '0'!")

            if X.shape[2] <= 0:
                raise ValueError(f"'X' argument number_of_features '{X.shape[2]}' is invalid, should be higher than > '0'!")
        except ValueError as err:
            logger.error(err)
            raise

        logger.info(f"Training '{self.model_name}' VRAE model...")
        if X.dtype != 'float32':
            X = X.astype('float32')

        if checkers.is_empty(self.vrae_kwargs):
            self.vrae_kwargs = config.tsc_vrae.vrae.to_dict()

        self.vrae_kwargs['sequence_length'] = X.shape[1]
        self.vrae_kwargs['number_of_features'] = X.shape[2]

        _tensor_X = TensorDataset(torch.from_numpy(X).to(device=self.device))

        self.model_vrae = VRAE(model_name=self.model_name, model_dir=self.model_dir, **self.vrae_kwargs).to(device=self.device)

        self.model_vrae.fit(_tensor_X)
        self.latent_features = self.extract_features(X)
        self.is_trained = True
        logger.success(f"Trained '{self.model_name}' VRAE model.")

        if save:
            self.save()


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def extract_features(self, X: np.ndarray):
        """Extract latent features from data input based on trained VRAE model.

        Args:
            X (np.ndarray, required): Extracting input dataset as numpy (Should be 3D).

        Raises:
            ValueError: Raise error if 'X' argument numpy array is empty.
            ValueError: Raise error if 'X' argument numpy array dimension is not [3D].
            ValueError: Raise error if 'X' argument size is lower than 1.
            ValueError: Raise error if 'X' argument sequence_length is lower than trained dataset's <sequence_length>.
            ValueError: Raise error if 'X' argument number_of_features is lower than trained dataset's <number_of_features>.

        Returns:
            np.ndarray: Returns extracted latent features.
        """

        try:
            if checkers.is_empty(X):
                raise ValueError("'X' argument numpy array is empty!")

            if X.ndim != 3:
                raise ValueError(f"'X' argument numpy array dimension [{X.ndim}D] is invalid, should be [3D]!")

            if X.shape[0] <= 0:
                raise ValueError(f"'X' argument size '{X.shape[0]}' is invalid, should be higher than > '0'!")

            if X.shape[1] != self.vrae_kwargs['sequence_length']:
                raise ValueError(f"'X' argument sequence_length '{X.shape[1]}' is invalid, should be equal to == '{self.vrae_kwargs['sequence_length']}'!")

            if X.shape[2] != self.vrae_kwargs['number_of_features']:
                raise ValueError(f"'X' argument number_of_features '{X.shape[2]}' is invalid, should be equal to == '{self.vrae_kwargs['number_of_features']}'!")
        except ValueError as err:
            logger.error(err)
            raise

        _tensor_X = TensorDataset(torch.from_numpy(X).to(device=self.device))
        _latent_features = self.model_vrae.transform(_tensor_X)
        return _latent_features


    def save(self):
        """Save <model_name> model files to <model_dir> directory.

        Raises:
            RuntimeError: Raise error if TSC_VRAE model is not trained.
            RuntimeError: Raise error if 'model_vrae' is empty.
            RuntimeError: Raise error if 'latent_features' is empty.
            RuntimeError: Raise error if 'model_files_meta' is empty.
            RuntimeError: Raise error if 'vrae_kwargs' is empty.
        """

        try:
            if not self.is_trained:
                raise RuntimeError(f"'{self.model_name}' TSC_VRAE model is not trained, can't save model files!")

            if checkers.is_empty(self.model_files_meta):
                raise RuntimeError(f"'{self.model_name}' TSC_VRAE model's `model_files_meta` is empty, can't save model files!")

            if checkers.is_empty(self.model_vrae):
                raise RuntimeError(f"'{self.model_name}' `model_vrae` is empty, can't save VRAE model file!")

            if checkers.is_empty(self.latent_features):
                raise RuntimeError(f"'{self.model_name}' TSC_VRAE model's `latent_features` is empty, can't save latent features file!")
        except RuntimeError as err:
            logger.error(err)
            raise

        if self.is_model_files_exist(self.model_name, self.model_dir, any_file=True):
            self.delete()

        logger.info(f"Saving '{self.model_name}' TSC_VRAE model files...")
        try:
            logger.debug(f"Saving '{self.model_files_meta['model_vrae_file_path']}' VRAE model file...")
            self.model_vrae.save(self.model_files_meta['model_vrae_file_path'])
            logger.debug(f"Successfully saved '{self.model_files_meta['model_vrae_file_path']}' VRAE model file.")

            logger.debug(f"Saving '{self.model_files_meta['latent_features_file_path']}' latent features numpy file...")
            with open(self.model_files_meta['latent_features_file_path'], 'wb') as _latent_features_file:
                np.save(_latent_features_file, self.latent_features)
            logger.debug(f"Successfully saved '{self.model_files_meta['latent_features_file_path']}' latent features numpy file.")

            self._save_meta_file()
        except Exception:
            logger.error(f"Failed to save '{self.model_name}' TSC_VRAE model files.")
            raise
        logger.success(f"Successfully saved '{self.model_name}' TSC_VRAE model files.")


    def _save_meta_file(self):
        try:
            if checkers.is_empty(self.model_files_meta):
                raise RuntimeError(f"'{self.model_name}' TSC_VRAE model's `model_files_meta` is empty, can't save meta json file!")

            if checkers.is_empty(self.vrae_kwargs):
                raise RuntimeError(f"'{self.model_name}' TSC_VRAE model's `vrae_kwargs` is empty, can't save meta json file!")
        except RuntimeError as err:
            logger.error(err)
            raise

        try:
            logger.debug(f"Saving '{self.model_files_meta['meta_file_path']}' meta json file...")
            _vrae_kwargs = self.vrae_kwargs.copy()
            # train_history is callback func --> cause error to json write. --> set train_history to None.
            if 'train_history' in _vrae_kwargs:
                del _vrae_kwargs['train_history']

            # learn_obj is object            --> cause error to json write. --> set learn_obj to None.
            if 'learn_obj' in _vrae_kwargs:
                del _vrae_kwargs['learn_obj']

            if 'cuda' in _vrae_kwargs:
                del _vrae_kwargs['cuda']

            if 'gpu_id' in _vrae_kwargs:
                del _vrae_kwargs['gpu_id']

            if 'batch_size' in _vrae_kwargs:
                del _vrae_kwargs['batch_size']

            _meta_json = {
                'model_files_meta': self.model_files_meta.copy(),
                'vrae_kwargs': _vrae_kwargs,
                'cluster_method': self.cluster_method,
                'cluster_kwargs': self.cluster_kwargs
            }

            with open(self.model_files_meta['meta_file_path'], 'w') as _meta_json_file:
                _meta_json_file.write(json.dumps(_meta_json, indent=4, ensure_ascii=False))
            logger.debug(f"Successfully saved '{self.model_files_meta['meta_file_path']}' meta json file.")
        except Exception:
            logger.error(f"Failed to save '{self.model_files_meta['meta_file_path']}' meta json file.")
            raise


    def delete(self):
        """Delete <model_name> model files from <model_dir> directory.
        """

        logger.debug(f"Deleting old '{self.model_name}' TSC_VRAE model files...")
        try:
            if os.path.isfile(self.model_files_meta['model_vrae_file_path']):
                logger.debug(f"Deleting '{self.model_files_meta['model_vrae_file_path']}' VRAE model file...")
                os.remove(self.model_files_meta['model_vrae_file_path'])
                logger.debug(f"Successfully deleted '{self.model_files_meta['model_vrae_file_path']}' VRAE model file.")

            if os.path.isfile(self.model_files_meta['latent_features_file_path']):
                logger.debug(f"Deleting '{self.model_files_meta['latent_features_file_path']}' latent features numpy file...")
                os.remove(self.model_files_meta['latent_features_file_path'])
                logger.debug(f"Successfully deleted '{self.model_files_meta['latent_features_file_path']}' latent features numpy file.")

            if os.path.isfile(self.model_files_meta['meta_file_path']):
                logger.debug(f"Deleting '{self.model_files_meta['meta_file_path']}' meta json file...")
                os.remove(self.model_files_meta['meta_file_path'])
                logger.debug(f"Successfully deleted '{self.model_files_meta['meta_file_path']}' meta json file.")
        except Exception:
            logger.error(f"Failed to delete '{self.model_name}' TSC_VRAE model files.")
            raise
        logger.debug(f"Successfully deleted old '{self.model_name}' TSC_VRAE model files.")


    def load(self, load_latent_features:bool=True):
        """Load <model_name> model files from <model_dir> directory.

        Raises:
            RuntimeError: Raise error if meta json file doesn't exist.
            RuntimeError: Raise error if latent features numpy file doesn't exist.
            RuntimeError: Raise error if VRAE model file doesn't exist.
        """

        try:
            if not os.path.isfile(self.model_files_meta['latent_features_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['latent_features_file_path']}' file doesn't exist!")

            if not os.path.isfile(self.model_files_meta['model_vrae_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['model_vrae_file_path']}' file doesn't exist!")
        except RuntimeError as err:
            logger.error(err)
            raise

        logger.info(f"Loading '{self.model_name}' TSC_VRAE model files...")
        try:
            self._load_meta_file()

            if load_latent_features is True:
                logger.debug(f"Loading '{self.model_files_meta['latent_features_file_path']}' latent features numpy file...")
                with open(self.model_files_meta['latent_features_file_path'], 'rb') as _latent_features_file:
                    self.latent_features = np.load(_latent_features_file)
                logger.debug(f"Successfully loaded '{self.model_files_meta['latent_features_file_path']}' latent features numpy file.")

            logger.debug(f"Loading '{self.model_files_meta['model_vrae_file_path']}' VRAE model file...")

            self.model_vrae = VRAE(model_name=self.model_name, model_dir=self.model_dir, **self.vrae_kwargs).to(device=self.device)
            self.model_vrae.load(self.model_files_meta['model_vrae_file_path'])
            self.is_trained = True
            logger.debug(f"Successfully loaded '{self.model_files_meta['model_vrae_file_path']}' VRAE model file.")
        except Exception:
            logger.exception(f"Failed to load '{self.model_name}' TSC_VRAE model files.")
            # logger.error(traceback.format_exc())
            raise
        logger.success(f"Successfully loaded '{self.model_name}' TSC_VRAE model files.")


    @validate_arguments
    def _load_meta_file(self):
        try:
            if not os.path.isfile(self.model_files_meta['meta_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['meta_file_path']}' file doesn't exist!")
        except RuntimeError as err:
            logger.error(err)
            raise

        logger.debug(f"Loading '{self.model_files_meta['meta_file_path']}' meta json file...")
        with open(self.model_files_meta['meta_file_path'], 'r') as _meta_json_file:
            _meta_json = json.load(_meta_json_file)
            self.vrae_kwargs = _meta_json['vrae_kwargs'].copy()

            if checkers.is_empty(self.cluster_method):
                if not checkers.is_attr_empty(_meta_json, 'cluster_method'):
                    self.cluster_method = _meta_json['cluster_method']

            if checkers.is_empty(self.cluster_kwargs):
                if not checkers.is_attr_empty(_meta_json, 'cluster_kwargs'):
                    self.cluster_kwargs = _meta_json['cluster_kwargs'].copy()
        logger.debug(f"Successfully loaded '{self.model_files_meta['meta_file_path']}' meta json file.")


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def cluster_latent_features(self, latenet_features: np.ndarray):
        """Clustering method for latent_features.

        Args:
            latenet_features (np.ndarray, required): Clustering latent_features as numpy (Should be 2D).

        Raises:
            ValueError         : Raise error if 'latenet_features' argument numpy array is empty.
            ValueError         : Raise error if 'latenet_features' argument numpy array dimension is not [2D].
            ValueError         : Raise error if 'latenet_features' argument size is lower than 1.

        Returns:
            np.ndarray: Infered cluster ids.
        """

        try:
            if checkers.is_empty(latenet_features):
                raise ValueError("'latenet_features' argument numpy array is empty!")

            if latenet_features.ndim != 2:
                raise ValueError(f"'latenet_features' argument numpy array dimension [{latenet_features.ndim}D] is invalid, should be [2D]!")

            if latenet_features.shape[0] <= 0:
                raise ValueError(f"'latenet_features' argument size '{latenet_features.shape[0]}' is invalid, should be higher than > '0'!")

        except ValueError as err:
            logger.error(err)
            raise

        logger.debug(f'cluster_kwargs: {self.cluster_kwargs}')

        if checkers.is_empty(self.cluster_method):
            self.cluster_method = 'DBSCAN'

        _model_cluster = None
        if self.cluster_method == 'DBSCAN':
            if checkers.is_empty(self.cluster_kwargs):
                self.cluster_kwargs = config.tsc_vrae.dbscan.to_dict()
            _model_cluster = DBSCAN(**self.cluster_kwargs)
        elif self.cluster_method == 'KMEANS':
            if checkers.is_empty(self.cluster_kwargs):
                self.cluster_kwargs = config.tsc_vrae.kmeans.to_dict()
            _model_cluster = KMeans(**self.cluster_kwargs)
        else:
            #TODO: Add more clustering method!
            try:
                raise NotImplementedError(f"Not implemented '{self.__cluster_method}' clustering method, yet!")
            except NotImplementedError as err:
                logger.error(err)
                raise

        _model_cluster.fit(latenet_features)
        _cluster_ids = _model_cluster.labels_
        return _cluster_ids


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def cluster(self, X: np.ndarray, use_train_data=True):
        """Clustering method for new data input.

        Args:
            X (np.ndarray, required): Clustering input dataset as numpy (Should be 3D).

        Raises:
            ValueError         : Raise error if 'X' argument numpy array is empty.
            ValueError         : Raise error if 'X' argument numpy array dimension is not [3D].
            ValueError         : Raise error if 'X' argument size is lower than 1.
            ValueError         : Raise error if 'X' argument sequence_length is lower than trained dataset's <sequence_length>.
            ValueError         : Raise error if 'X' argument number_of_features is lower than trained dataset's <number_of_features>.
            NotImplementedError: Raise error if 'cluster_method' is not 'DBSCAN' and not 'KMEANS'.

        Returns:
            np.ndarray: Infered cluster ids.
        """

        try:
            if checkers.is_empty(X):
                raise ValueError("'X' argument numpy array is empty!")

            if X.ndim != 3:
                raise ValueError(f"'X' argument numpy array dimension [{X.ndim}D] is invalid, should be [3D]!")

            if X.shape[0] <= 0:
                raise ValueError(f"'X' argument size '{X.shape[0]}' is invalid, should be higher than > '0'!")

            if X.shape[1] != self.vrae_kwargs['sequence_length']:
                raise ValueError(f"'X' argument sequence_length '{X.shape[1]}' is invalid, should be equal to == '{self.vrae_kwargs['sequence_length']}'!")

            if X.shape[2] != self.vrae_kwargs['number_of_features']:
                raise ValueError(f"'X' argument number_of_features '{X.shape[2]}' is invalid, should be equal to == '{self.vrae_kwargs['number_of_features']}'!")
        except ValueError as err:
            logger.error(err)
            raise

        _latent_features = self.extract_features(X)

        _n_train_data = 0
        if use_train_data is True:
            _n_train_data = self.latent_features.shape[0]
            logger.debug(f"train_data shape: {self.latent_features.shape}")
            _all_latent_features = np.concatenate((self.latent_features, _latent_features), axis=0)
        else:
            _all_latent_features = _latent_features

        logger.debug(f"_all_latent_features shape: {_all_latent_features.shape}, use_train_data: {use_train_data}")

        _cluster_ids = self.cluster_latent_features(_all_latent_features)[_n_train_data:]
        return _cluster_ids


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def gen_cluster_centroid_info(self, cluster_ids, latent_features: np.ndarray):
        """_summary_

        Args:
            cluster_ids (_type_): _description_
            latent_features (_type_): _description_
            max_cnt (int, optional): _description_. Defaults to 200.

        Returns:
            _type_: _description_
        """
        # gen count info and cluster file info
        latent_info = {}
        centroid_info = {}

        max_cnt = config.tsc_vrae.const.max_centroid_count
        if max_cnt <= 0:
            max_cnt = 200

        # ----------------------------
        # gen count info 
        cluster_count = dict(Counter(cluster_ids))
        logger.debug(f"Cluster count: {len(cluster_count)}, Cluster IDs count: {cluster_count}, result len: {len(cluster_ids)}")

        # 클러스터 id별로 latent 데이터 정보 저장을 위한 latent_info 생성
        for cluster_id in cluster_count:
            latent_info[str(cluster_id)] = []

        # latent_info의 cluster_id에 latent data 저장
        for idx, cluster_id in enumerate(cluster_ids):
            latent_info[str(cluster_id)].append(latent_features[idx])

        for cluster_id, latents in latent_info.items():
            # 클러스터의 centroid vector 생성
            mean_v = np.mean(latents, axis=0)
            # centroid와 각 latent와의 거리 계산
            vec_dist_lst = [self.get_vec_distance(mean_v, val, dis_type='euc') for val in latents]
            # centroid와 먼 순서로 정렬. . 
            _arg_lst = np.argsort(vec_dist_lst)[::-1]

            # _arg_lst에서 max_cnt개의 데이터 선택. 전체 거리 구간에서 골고루 가져옴. 
            steps = int(len(_arg_lst) // max_cnt)
            if steps < 1:
                steps = 1
            arg_lst = [_arg_lst[i] for i in range(0, len(_arg_lst), steps)]

            # centroid_info의 처음은 centroid값. 나머지는 가장 먼 값들. 
            centroid_info[str(cluster_id)] = [mean_v.tolist()]
            for idx in arg_lst:
                centroid_info[str(cluster_id)].append(latents[idx].tolist())

        return centroid_info

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def get_cluster_id(self, centroid_info: dict, latent_features: np.ndarray, cluster_ids=None):
        """
        latent features각 line에 대한 cluster id 목록 반

        Args:
            centroid_info (_type_): _description_
            latent_features (_type_): _description_
            cluster_ids (_type_, optional): _description_. Defaults to None.

        Returns:
            string list: ['0', '0', '-1', '1', ...]
        """

        result_cluster_id_lst = []

        start_t = datetime.now()

        # 비교할 latent 하나씩 읽어서 centroid와 비교하여 cluster 찾음. 
        success_cnt = 0
        fail_cnt = 0
        for idx, latent_data in enumerate(latent_features):
            # centroid info에 있는 각각의 cluster별 centroid와 거리를 dist_lst에 기록
            dist_lst = []  # cluster id 순서대로 distance 값 기록됨. 
            for cluster_id, centroid_vecs in centroid_info.items():
                vec_lst = []
                # cnetroid에 있는 모든 vec와 거리 측정 --> 목록화. 
                for cent_vec in centroid_vecs:
                    vec_lst.append(self.get_vec_distance(cent_vec, latent_data, dis_type='euc'))
                # 전체중 가장 가까운값 선택
                dist_lst.append(vec_lst[np.argmin(vec_lst)])

            # vec distance가 가장 작은곳의 index value. --> cluster id와 일치함. 주의 마지막은 -1이 될수 있음. 
            _cluster_id = np.argmin(dist_lst)

            result_cluster_id = list(centroid_info.keys())[_cluster_id]
            result_cluster_id_lst.append(int(result_cluster_id))

            if cluster_ids is not None:
                # accuracy 계산.
                assert len(cluster_ids) == len(latent_features)

                if str(cluster_ids[idx]) == result_cluster_id:
                    # logger.debug(f"{idx}. bingo! true:{str(cluster_ids[idx])}. check result: {result_cluster_id}. {dist_lst}")
                    success_cnt += 1
                else:
                    logger.error(f"{idx}. failed! true:{str(cluster_ids[idx])}. check result: {result_cluster_id}. {dist_lst}")
                    fail_cnt += 1
            else:
                logger.debug(f"{idx}. check result: {result_cluster_id}")

        if cluster_ids is not None:
            logger.debug(f"accuracy : {(success_cnt/len(cluster_ids)) * 100}. success: {success_cnt}, fail: {fail_cnt}. time: {(datetime.now() - start_t)/len(latent_features)}")
        
        return result_cluster_id_lst


    def get_vec_distance(self, vec1, vec2, dis_type='euc'):
        """
        return vector distance. Euclidean distance or Cosine distance

        :param vec1: input vector. 1D
        :param vec2: input vector. 1D
        :param dis_type: euc - Euclidean distance, cos - cosine distance
        :return: float
        """

        if dis_type == 'euc':
            return distance.euclidean(vec1, vec2)
        elif dis_type == 'cos':
            return distance.cosine(vec1, vec2)
        else:
            logger.error('invalid type for get_vec_distance')
        return

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def cluster_with_data_id(self, X: np.ndarray, data_ids=None, filename: str='cluster_output.csv', result_dir: str='results', save: bool=False):
        """Clustering method with data_id.

        Args:
            X          (np.ndarray        , required): Clustering input dataset as numpy (Should be 3D).
            data_ids   (list or np.ndarray, optional): Data ID or data file name as unique value to identify. Defaults to None.
            filename   (str               , optional): File name used to save result. Defaults to 'cluster_output.csv'.
            result_dir (str               , optional): Saving directory path. Defaults to 'results'.
            save       (bool              , optional): Indicate save output to file or not. Defaults to False.

        Raises:
            TypeError : Rase error if 'data_ids' is not list or np.ndarray.
            ValueError: Rase error if 'X' size is not matching with 'data_ids' length.
            ValueError: Rase error if 'filename' is empty and 'save' is True.
            ValueError: Rase error if 'result_dir' is empty and 'save' is True.

        Returns:
            tuple(list, pd.DataFrame): Clustered output data with data_ids as list and pd.DataFrame.
        """

        if data_ids is None:
            data_ids = list(range(X.shape[0]))

        if isinstance(data_ids, np.ndarray):
            data_ids = data_ids.tolist()

        try:
            if not isinstance(data_ids, list):
                raise TypeError(f"'data_ids' argument type <{type(data_ids).__name__}> is invalid, should be <list> or <np.ndarray>!")

            if X.shape[0] != len(data_ids):
                raise ValueError(f"'X' argument size '{len(X.shape[0])}' is invalid, should match with 'data_ids' length {len(data_ids)}!")

            if save:
                filename = filename.strip()
                if checkers.is_empty(filename):
                    raise ValueError("'filename' argument value is empty!")

                result_dir = result_dir.strip()
                if checkers.is_empty(result_dir):
                    raise ValueError("'result_dir' argument value is empty!")

                utils.create_dir(result_dir)
        except Exception as err:
            logger.error(err)
            raise

        _cluster_ids = self.cluster(X).tolist()

        _output_list = []
        for _i in range(len(_cluster_ids)):
            _item = {
                'data_id': data_ids[_i],
                'cluster_id': _cluster_ids[_i]
            }
            _output_list.append(_item)

        _output_df = pd.DataFrame({'data_id': data_ids, 'cluster_id': _cluster_ids})
        if save:
            _output_file_path = os.path.join(result_dir, filename)
            _output_df.to_csv(_output_file_path, index=False)

        return _output_list, _output_df


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def cluster_with_label(self, X: np.ndarray, label_map: dict, filename: str='cluster_label.csv', result_dir: str='results', save: bool=False):
        """Clustering method with label.

        Args:
            X          (np.ndarray, required): Clustering input dataset as numpy (Should be 3D).
            label_map  (dict      , required): Label mapping dictionary to label cluster_ids.
            filename   (str       , optional): File name used to save result. Defaults to 'cluster_label.csv'.
            result_dir (str       , optional): Saving directory path. Defaults to 'results'.
            save       (bool      , optional): Indicate save output to file or not. Defaults to False.

        Raises:
            ValueError: Rase error if 'filename' is empty and 'save' is True.
            ValueError: Rase error if 'result_dir' is empty and 'save' is True.

        Returns:
            tuple(list, pd.DataFrame): Clustered output data with labels as list and pd.DataFrame.
        """

        try:
            if save:
                filename = filename.strip()
                if checkers.is_empty(filename):
                    raise ValueError("'filename' argument value is empty!")

                result_dir = result_dir.strip()
                if checkers.is_empty(result_dir):
                    raise ValueError("'result_dir' argument value is empty!")

                utils.create_dir(result_dir)
        except Exception as err:
            logger.error(err)
            raise

        _cluster_ids = self.cluster(X).tolist()

        _labels = []
        for _cluster_id in _cluster_ids:
            if _cluster_id in label_map:
                _labels.append(label_map[_cluster_id])
            elif str(_cluster_id) in label_map:
                _labels.append(label_map[str(_cluster_id)])
            else:
                _labels.append(str(_cluster_id))

        _output_list = []
        for _i in range(len(_cluster_ids)):
            _item = {
                'cluster_id': _cluster_ids[_i],
                'label': _labels[_i]
            }
            _output_list.append(_item)

        _output_df = pd.DataFrame({ 'cluster_id': _cluster_ids, 'label': _labels })
        if save:
            _output_file_path = os.path.join(result_dir, filename)
            _output_df.to_csv(_output_file_path, index=False)

        return _output_list, _output_df


    def inference(self, dataset, **kwargs):
        """[summary]

        Args:
            dataset ([type]): [description]

        Returns:
            {
                'cluster_info': {
                        'cluster_count': ,
                        'cluster_count_info': ,
                        'cluster_ids': ,
                        'cluster_ids_len': ,
                        'summary_info':
                    }
            }

        """
        try:
            if checkers.is_empty(dataset):
                raise ValueError("'dataset' argument value is empty!")

            model_info = kwargs.get('model_info')
            if checkers.is_empty(model_info):
                raise ValueError("'model_info' value is empty!")

            cluster_id = model_info.get('cluster_id')
            if checkers.is_empty(cluster_id):
                raise ValueError("'cluster_id' value is empty!")

            testset_name = model_info.get('testset_name')
            if checkers.is_empty(testset_name):
                raise ValueError("'testset_name' value is empty!")

        except Exception as err:
            logger.error(err)
            raise

        # logger.debug(f'latent features: {self.latent_features.shape}, dataset: {dataset.shape}')
        logger.debug(f'dataset: {dataset.shape}')

        _latent_features = self.extract_features(dataset)
        logger.debug(f'latent_features: {_latent_features.shape}')

        cluster_ids = self.get_cluster_id(self.centroid_info, _latent_features)

        anomaly_list = []
        _cluster_id = None
        name_info = None
        description = ''
        min_normal_value = config.tsc_vrae.const.min_normal_value
        if self.cluster_info and self.cluster_info.get('cluster_name_info'):
            cluster_name_info = self.cluster_info.get('cluster_name_info')
            _cluster_id = cluster_ids[0]
            name_info = cluster_name_info.get(str(_cluster_id))

            if name_info is None:
                logger.warning(f"no name info of cluster_id: {_cluster_id}. keep it as normal")
            else:
                _level = name_info.get('level')
                if _level is not None:
                    if int(_level) > min_normal_value:
                        anomaly_list.append(_cluster_id)

                        description = f"{testset_name} is abanomal! {name_info.get('name')} as level {name_info.get('level')}"
                    else:
                        description = f"{testset_name} is normal. {name_info.get('name')} as level {name_info.get('level')}"

        cluster_info = {'anomaly_list': anomaly_list,
                        'cluster_id': _cluster_id,
                        'cluster_name_info': name_info,
                        'description': description}

        return {'diff': [],
                'diff_detail': [],
                'max': 0,
                'max_idx': 0,
                'mean': 0,
                'std': 0,
                'anomaly_list': anomaly_list,
                'anomaly_group_list': [],
                'forecaset_info': None,
                'inference_info': None,
                'cluster_info': cluster_info}

    def load_centroid_info(self, cluster_id: str) -> dict:
        """
        Load centroid info

        Args:
            cluster_id (str): cluster id

        Returns:
            dict: centroid info
        """

        if cluster_id is None:
            logger.warning('cluster_id is None')
            return 

        clustering_result_dir = path.join(self.model_dir, self.model_name + '_files', cluster_id)

        # read centroid info
        centroid_info_filename = path.join(clustering_result_dir, 'centroid_info.json')
        if path.exists(centroid_info_filename):
            with open(centroid_info_filename, 'r') as _centroid_info_filename:
                _centroid_info = json.load(_centroid_info_filename)
            return _centroid_info
        else:
            logger.warning(f'fail to load centroid info. {centroid_info_filename}')

    def load_cluster_info(self, cluster_id: str) -> dict:
        """
        Load cluster info

        Args:
            cluster_id (str): cluster id

        Returns:
            dict: cluster info
        """
        if cluster_id is None:
            logger.warning('cluster_id is None')
            return 

        clustering_result_dir = path.join(self.model_dir, self.model_name + '_files', cluster_id)

        # read centroid info
        cluster_info_filename = path.join(clustering_result_dir, 'cluster_info.json')
        if path.exists(cluster_info_filename):
            with open(cluster_info_filename, 'r') as _cluster_info_filename:
                _cluster_info = json.load(_cluster_info_filename)
            return _cluster_info
        else:
            logger.warning(f'fail to load cluster info. {cluster_info_filename}')

    def update_cluster_info(self, cluster_info, cluster_name_info):

        def get_cluster_name(cluster_id, cluster_name_info):
            for cluster_name in cluster_name_info:
                if int(cluster_id) in cluster_name_info[cluster_name]:
                    return cluster_name
            return 'normal'

        if cluster_name_info is None or len(cluster_name_info) <= 0:
            return cluster_info

        if 'cluster_count_info' in cluster_info:
            _cluster_count_info = {}
            for cluster_id in cluster_info['cluster_count_info']:
                cluster_name = get_cluster_name(cluster_id, cluster_name_info)
                if cluster_name in _cluster_count_info:
                    _cluster_count_info[cluster_name] += cluster_info['cluster_count_info'][cluster_id]
                else:
                    _cluster_count_info[cluster_name] = cluster_info['cluster_count_info'][cluster_id]
            cluster_info['cluster_count_info'] = _cluster_count_info

        if 'cluster_ids' in cluster_info:
            _cluster_ids = []
            for cluster_id in cluster_info['cluster_ids']:
                cluster_name = get_cluster_name(cluster_id, cluster_name_info)
                _cluster_ids.append(cluster_name)
            cluster_info['cluster_result'] = _cluster_ids

        return cluster_info


    ### ATTRIBUTES ###
    ## model_name ##
    @property
    def model_name(self):
        try:
            return self.__model_name
        except AttributeError:
            return None

    @model_name.setter
    def model_name(self, model_name):
        try:
            if not isinstance(model_name, str):
                raise TypeError(f"'model_name' argument type <{type(model_name).__name__}> is invalid, should be <str>!")

            model_name = model_name.strip()
            if checkers.is_empty(model_name):
                raise ValueError("'model_name' argument value is empty!")
        except Exception as err:
            logger.error(err)
            raise

        self.__model_name = model_name
        if not checkers.is_empty(self.model_files_meta):
            self.model_files_meta = self._get_files_meta(self.__model_name, self.model_dir)
    ## model_name ##


    ## model_dir ##
    @property
    def model_dir(self):
        try:
            return self.__model_dir
        except AttributeError:
            return None

    @model_dir.setter
    def model_dir(self, model_dir):
        try:
            if not isinstance(model_dir, str):
                raise TypeError(f"'model_dir' argument type <{type(model_dir).__name__}> is invalid, should be <str>!")

            model_dir = model_dir.strip()
            if checkers.is_empty(model_dir):
                raise ValueError("'model_dir' argument value is empty!")
        except Exception as err:
            logger.error(err)
            raise

        utils.create_dir(model_dir)
        self.__model_dir = model_dir
        if not checkers.is_empty(self.model_files_meta):
            self.model_files_meta = self._get_files_meta(self.model_name, self.__model_dir)
    ## model_dir ##


    ## model_files_meta ##
    @property
    def model_files_meta(self):
        try:
            return self.__model_files_meta
        except AttributeError:
            return None

    @model_files_meta.setter
    def model_files_meta(self, model_files_meta):
        try:
            if not isinstance(model_files_meta, dict):
                raise TypeError(f"'model_files_meta' argument type <{type(model_files_meta).__name__}> is invalid, should be <dict>!")
        except TypeError as err:
            logger.error(err)
            raise

        self.__model_files_meta = model_files_meta
    ## model_files_meta ##


    ## vrae_kwargs ##
    @property
    def vrae_kwargs(self):
        try:
            return self.__vrae_kwargs
        except AttributeError:
            return None

    @vrae_kwargs.setter
    def vrae_kwargs(self, vrae_kwargs):
        try:
            if not isinstance(vrae_kwargs, dict):
                raise TypeError(f"'vrae_kwargs' argument type <{type(vrae_kwargs).__name__}> is invalid, should be <dict>!")

            _vrae_kwargs = config.tsc_vrae.vrae.to_dict()
            _vrae_kwargs.update(vrae_kwargs)

            if not vrae_validator.validate(_vrae_kwargs):
                _err_str = "The 'vrae_kwargs' is invalid:\n"
                for _key, _value in vrae_validator.errors.items():
                    _err_str = _err_str + f"'{_key}' field is invalid => {_value}\n"
                raise ValueError(_err_str)

        except Exception as err:
            logger.error(err)
            raise

        if _vrae_kwargs['cuda']:
            if (not torch.cuda.is_available()) or (torch.cuda.device_count() == 0):
                _vrae_kwargs['cuda'] = False
            else:
                if (_vrae_kwargs['gpu_id'] < 0) or (torch.cuda.device_count() <= _vrae_kwargs['gpu_id']):
                    logger.warning(f"Not found GPU ID: {_vrae_kwargs['gpu_id']}, changing GPU ID to 0!")
                    _vrae_kwargs['gpu_id'] = 0

                self.device = torch.device(f"cuda:{_vrae_kwargs['gpu_id']}")
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        if not _vrae_kwargs['cuda']:
            self.device = torch.device('cpu')

        self.__vrae_kwargs = _vrae_kwargs
    ## vrae_kwargs ##


    ## model_vrae ##
    @property
    def model_vrae(self):
        try:
            return self.__model_vrae
        except AttributeError:
            return None

    @model_vrae.setter
    def model_vrae(self, model_vrae):
        try:
            if not isinstance(model_vrae, VRAE):
                raise TypeError(f"'model_vrae' argument type <{type(model_vrae).__name__}> is invalid, should be <VRAE>!")
        except Exception as err:
            logger.error(err)
            raise

        self.__model_vrae = model_vrae
    ## model_vrae ##


    ## device ##
    @property
    def device(self):
        try:
            return self.__device
        except AttributeError:
            if torch.cuda.is_available() and (0 < torch.cuda.device_count()):
                return torch.device('cuda:0')
            else:
                return torch.device('cpu')

    @device.setter
    def device(self, device):
        try:
            if not isinstance(device, torch.device):
                raise TypeError(f"'device' argument type <{type(device)}> is invalid, should be <torch.device>!")

            if device.type == 'cuda':
                if (not torch.cuda.is_available()) or (torch.cuda.device_count() == 0):
                    device = torch.device('cpu')
                else:
                    if (device.index < 0) or (torch.cuda.device_count() <= device.index):
                        logger.warning(f"Not found GPU ID: {device.index}, changing GPU ID to 0!")
                        device = torch.device('cuda:0')

        except TypeError as err:
            logger.error(err)
            raise

        self.__device = device
    ## device ##


    ## latent_features ##
    @property
    def latent_features(self):
        try:
            return self.__latent_features
        except AttributeError:
            return None

    @latent_features.setter
    def latent_features(self, latent_features):
        try:
            if not isinstance(latent_features, np.ndarray):
                raise TypeError(f"'latent_features' argument type <{type(latent_features).__name__}> is invalid, should be <np.ndarray>!")

            if checkers.is_empty(latent_features):
                raise ValueError("'latent_features' argument numpy array is empty!")

            if latent_features.ndim == 1:
                latent_features = np.expand_dims(latent_features, 0)
            elif latent_features.ndim != 2:
                raise ValueError(f"'latent_features' argument numpy array dimension [{latent_features.ndim}D] is invalid, should be [2D]!")
        except Exception as err:
            logger.error(err)
            raise

        if latent_features.dtype != 'float32':
            latent_features = latent_features.astype('float32')
        self.__latent_features = latent_features
    ## latent_features ##


    ## cluster_method ##
    @property
    def cluster_method(self):
        try:
            return self.__cluster_method
        except AttributeError:
            return None

    @cluster_method.setter
    def cluster_method(self, cluster_method):
        try:
            if not isinstance(cluster_method, str):
                raise TypeError(f"'cluster_method' argument type <{type(cluster_method).__name__}> is invalid, should be <str>!")

            cluster_method = cluster_method.strip().upper()
            if checkers.is_empty(cluster_method, trim_str=True):
                raise ValueError("'cluster_method' argument value is empty!")

            if (cluster_method != 'DBSCAN') and (cluster_method != 'KMEANS') and (cluster_method != 'FAISS'):
                raise ValueError(f"'cluster_method' argument value '{cluster_method}' is invalid, should be 'DBSCAN', 'KMEANS' or 'FAISS'!")

            #TODO: Remove this after implement other clustering methods!
            if (cluster_method == 'FAISS'):
                raise NotImplementedError(f"Not implemented '{cluster_method}' clustering method, yet!")
        except Exception as err:
            logger.error(err)
            raise

        self.__cluster_method = cluster_method
    ## cluster_method ##


    ## cluster_kwargs ##
    @property
    def cluster_kwargs(self):
        try:
            return self.__cluster_kwargs
        except AttributeError:
            return None

    @cluster_kwargs.setter
    def cluster_kwargs(self, cluster_kwargs):
        try:
            if not isinstance(cluster_kwargs, dict):
                raise TypeError(f"'cluster_kwargs' argument type <{type(cluster_kwargs).__name__}> is invalid, should be <dict>!")

            if self.cluster_method is None:
                self.cluster_method = 'DBSCAN'
                # raise RuntimeError("'cluster_method' is empty, first set 'cluster_method' value!")

            if self.cluster_method == 'DBSCAN':
                _cluster_kwargs = config.tsc_vrae.dbscan.to_dict()
                _cluster_kwargs.update(cluster_kwargs)

                if not dbscan_validator.validate(_cluster_kwargs):
                    _err_str = "The 'cluster_kwargs' is invalid:\n"
                    for _key, _value in dbscan_validator.errors.items():
                        _err_str = _err_str + f"'{_key}' field is invalid => {_value}\n"
                    raise ValueError(_err_str)

            elif self.cluster_method == 'KMEANS':
                _cluster_kwargs = config.tsc_vrae.kmeans.to_dict()
                _cluster_kwargs.update(cluster_kwargs)

                if not kmeans_validator.validate(_cluster_kwargs):
                    _err_str = "The 'cluster_kwargs' is invalid:\n"
                    for _key, _value in kmeans_validator.errors.items():
                        _err_str = _err_str + f"'{_key}' field is invalid => {_value}\n"
                    raise ValueError(_err_str)

            else:
                raise NotImplementedError(f"Not implemented '{self.cluster_method}' clustering method, yet!")

        except Exception as err:
            logger.error(err)
            raise

        self.__cluster_kwargs = _cluster_kwargs
    ## cluster_kwargs ##


    ## is_trained ##
    @property
    def is_trained(self):
        try:
            return self.__is_trained
        except AttributeError:
            return False

    @is_trained.setter
    def is_trained(self, is_trained):
        try:
            if not isinstance(is_trained, bool):
                raise TypeError(f"'is_trained' argument type <{type(is_trained).__name__}> is invalid, should be <bool>!")
        except TypeError as err:
            logger.error(err)
            raise

        if is_trained:
            if checkers.is_empty(self.model_vrae) or (not hasattr(self.model_vrae, 'is_fitted')) or (not self.model_vrae.is_fitted):
                logger.warning("'vrae' is not trained, changed 'is_trained' to 'False'!")
                is_trained = False
        else:
            if (not checkers.is_empty(self.model_vrae)) and hasattr(self.model_vrae, 'is_fitted') and self.model_vrae.is_fitted:
                is_trained = True
        self.__is_trained = is_trained
    ## is_trained ##

    ## cluster_id ##
    @property
    def cluster_id(self):
        try:
            return self.__cluster_id
        except AttributeError:
            return None

    @cluster_id.setter
    def cluster_id(self, cluster_id):
        try:
            if not isinstance(cluster_id, str):
                raise TypeError(f"'cluster_id' argument type <{type(cluster_id).__name__}> is invalid, should be <str>!")

            cluster_id = cluster_id.strip()
            if checkers.is_empty(cluster_id):
                raise ValueError("'cluster_id' argument value is empty!")
        except Exception as err:
            logger.error(err)
            raise

        self.__cluster_id = cluster_id
    ## model_name ##

    ### ATTRIBUTES ###


    ## METHOD OVERRIDING ##
    def __str__(self):
        _self_dict = utils.clean_obj_dict(self.__dict__, self.__class__.__name__)
        return f"{self.__class__.__name__}: {_self_dict}"

    def __repr__(self):
        return utils.obj_to_repr(self)
    ## METHOD OVERRIDING ##
0