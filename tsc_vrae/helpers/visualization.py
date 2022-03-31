# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import validate_arguments
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from el_logging import logger
from el_validator import checkers
import traceback

from . import utils


@validate_arguments
def gen_unique_cluster_colors(unique_cluster_ids):
    """Generate unique cluster colors.

    Args:
        unique_cluster_ids (np.ndarray or list, required): Unique cluster ids to colorize differently.

    Raises:
        ValueError: If `unique_cluster_ids` argument value is empty.
        TypeError: If `unique_cluster_ids` argument value is not a list or numpy array.

    Returns:
        dict: Color dictionary for each cluster.
    """

    try:
        if (not isinstance(unique_cluster_ids, list)) and (not isinstance(unique_cluster_ids, np.ndarray)):
            raise TypeError(f"`unique_cluster_ids` argument type <{type(unique_cluster_ids).__name__}> is invalid, should be <list> or <np.ndarray>!")

    except Exception as err:
        logger.error(err)
        raise

    _color_dict = {}
    _base_colors = ['#FF160C', '#5BBDFF', '#2ECC71', '#FFFF07', '#FF76FF', '#FF7F00']
    for _i, _cluster_id in enumerate(unique_cluster_ids):
        if _i < len(_base_colors):
            _color_dict[str(_cluster_id)] = _base_colors[_i]
        else:
            _tmp_color = ''
            while True:
                _tmp_color = utils.rand_color()
                if _tmp_color not in _color_dict.values():
                    break
            _color_dict[str(_cluster_id)] = _tmp_color

    return  _color_dict


@validate_arguments
def gen_cluster_colors(cluster_ids):
    """Generate cluster colors.

    Args:
        cluster_ids (np.ndarray or list, required): Cluster ids to group and colorize differently.

    Raises:
        ValueError: If `cluster_ids` argument value is empty.
        TypeError: If `cluster_ids` argument value is not a list or numpy array.

    Returns:
        tuple (list, dict): Tuple of cluster colors.
            list: Color list for `cluster_ids`.
            dict: Color dictionary for each cluster.
    """

    try:
        if (not isinstance(cluster_ids, list)) and (not isinstance(cluster_ids, np.ndarray)):
            raise TypeError(f"`cluster_ids` argument type <{type(cluster_ids).__name__}> is invalid, should be <list> or <np.ndarray>!")

    except Exception as err:
        logger.error(err)
        raise

    _unique_cluster_ids = np.unique(cluster_ids)
    _color_dict = gen_unique_cluster_colors(_unique_cluster_ids)
    _colors = [_color_dict[str(_cluster_id)] for _cluster_id in cluster_ids]
    return  _color_dict, _colors


@validate_arguments
def plot_latent(latent_features, cluster_ids, color_dict: dict={}, transform_method: str='SVD', filename: str='cluster_plot_svd.png', result_dir: str='results', save: bool=True, legend: bool=False, labelsize: int=6):
    """Visualization and plotting method for latent features.

    Args:
        latent_features  (np.ndarray or list, required): Extracted latent features.
        cluster_ids      (np.ndarray or list, required): Labels or cluster ids to group and colorize differently.
        color_dict       (dict              , optional): Color dictionary for each cluster. Default is {}.
        transform_method (str               , optional): Transform method for visualizing latent features. Defaults to 'SVD'.
        filename         (str               , optional): File name used to save result. Defaults to 'cluster_plot_svd.csv'.
        result_dir       (str               , optional): Saving directory path. Defaults to 'results'.
        save             (bool              , optional): Indicate save output to file or not. Defaults to True.

    Raises:
        TypeError : Rase error if `latent_features` is not list or np.ndarray.
        TypeError : Rase error if `cluster_ids` is not list or np.ndarray.
        ValueError: Rase error if `latent_features` length is not matching `cluster_ids` length.
        ValueError: Rase error if `latent_features` is not 'SVD', 'PCA' or 'TSNE'.
        ValueError: Rase error if `filename` is empty and `save` is True.
        ValueError: Rase error if `result_dir` is empty and `save` is True.
    """

    if isinstance(latent_features, np.ndarray):
        latent_features = latent_features.tolist()

    if isinstance(cluster_ids, np.ndarray):
        cluster_ids = cluster_ids.tolist()

    try:
        if not isinstance(latent_features, list):
            raise TypeError(f"`latent_features` argument type <{type(latent_features).__name__}> is invalid, should be <list> or <np.ndarray>!")

        if not isinstance(cluster_ids, list):
            raise TypeError(f"`cluster_ids` argument type <{type(cluster_ids).__name__}> is invalid, should be <list> or <np.ndarray>!")

        if len(latent_features) != len(cluster_ids):
            raise ValueError(f"`latent_features` length '{len(latent_features)}' is not matching with `cluster_ids` length {len(cluster_ids)}!")

        transform_method = transform_method.strip().upper()
        if (transform_method != 'SVD') and (transform_method != 'PCA') and (transform_method != 'TSNE') and (transform_method != 'LDA'):
                raise ValueError(f"`transform_method` argument value '{transform_method}' is invalid, should be 'SVD', 'PCA', 'TSNE' or 'LDA'!")

        if save:
            filename = filename.strip()
            if checkers.is_empty(filename):
                raise ValueError("`filename` argument value is empty!")

            result_dir = result_dir.strip()
            if checkers.is_empty(result_dir):
                raise ValueError("`result_dir` argument value is empty!")

            utils.create_dir(result_dir)
    except Exception as err:
        logger.error(err)
        raise

    plt.rc('xtick', labelsize=labelsize)
    plt.rc('ytick', labelsize=labelsize)

    _plot_file_path = ''
    _cluster_colors = []
    try:
        # cluster_ids = cluster_ids[:latent_features.shape[0]] # because of weird batch_size

        _point_size = 8
        _n_latent = len(latent_features)
        if _n_latent < 100:
            _point_size = 8
        elif _n_latent < 1000:
            _point_size = 5
        elif _n_latent < 10000:
            _point_size = 3
        else:
            _point_size = 1

        _legend_size = 8
        _unique_cluster_ids = np.unique(cluster_ids)
        _n_cluster_ids = len(_unique_cluster_ids)
        if _n_cluster_ids <= 3:
            _legend_size: 8
        elif _n_cluster_ids <= 5:
            _legend_size: 6
        elif _n_cluster_ids <= 10:
            _legend_size: 4
        else:
            _legend_size: 3

        if checkers.is_empty(color_dict):
            color_dict, _cluster_colors = gen_cluster_colors(cluster_ids)
        else:
            _cluster_colors = [color_dict[str(_cluster_id)] for _cluster_id in cluster_ids]

        _component_2 = 1
        _transform_features = None
        if transform_method == 'SVD':
            _transform_features = TruncatedSVD(n_components=2).fit_transform(latent_features)
        elif transform_method == 'PCA':
            _transform_features = PCA(n_components=2).fit_transform(latent_features)
        elif transform_method == 'TSNE':
            _transform_features = TSNE(n_components=2).fit_transform(latent_features)
        elif (transform_method == 'LDA'):
            if _n_cluster_ids < 2:
                logger.warning("'LDA' transform method is not applicable for cluster count is lower than < 2!")
                return None, None
            elif _n_cluster_ids == 2:
                _n_components = 1
                _component_2 = 0
            else:
                _n_components = 2
            _transform_features = LinearDiscriminantAnalysis(n_components=_n_components).fit_transform(latent_features, cluster_ids)
        else:
            raise ValueError(f"`transform_method` argument value '{transform_method}' is invalid, should be 'SVD', 'PCA', 'TSNE' or 'LDA'!")

        _scatter_ids_dict = {}
        for _color in color_dict.values():
            _scatter_ids_dict[_color] = [_i for _i, _cluster_color in enumerate(_cluster_colors) if _cluster_color == _color]

        for _color_name, _color in color_dict.items():
            plt.scatter(_transform_features[_scatter_ids_dict[_color], 0], _transform_features[_scatter_ids_dict[_color], _component_2], s=_point_size, c=_color, label=_color_name, alpha=0.8, linewidths=0.1)

        if legend is True:
            plt.legend(prop={ 'size': _legend_size })
        plt.title(f"{transform_method}")
        if save:
            _plot_file_path = os.path.join(result_dir, filename)
            plt.savefig(_plot_file_path, dpi=100, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    except Exception as err:
        logger.error('Failed to plot latent_features!')
        logger.error(traceback.format_exc())
        raise

    return _plot_file_path, _cluster_colors


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_overlap_columns(X: np.ndarray, column_names: list=None, cluster_ids=None, cluster_name_map: dict=None, result_dir: str='results', 
                            save: bool=True, labelsize: int=6, ylim:list=None):
    """Visualization and plotting method for each column by cluster_ids.

    Args:
        X                (np.ndarray        , required): 3D array of data.
        column_names     (list              , required): Each column names to visualize. Defaults to None.
        cluster_ids      (np.ndarray or list, required): Cluster ids to group. Defaults to None.
        cluster_name_map (dict              , optional): Cluster name map. Defaults to None.
        result_dir       (str               , optional): Saving directory path. Defaults to 'results'.
        save             (bool              , optional): Indicate save output to file or not. Defaults to True.

    Raises:
        ValueError: Raise error if `X` is empty.
        ValueError: Raise error if `X` is not [3D] np.ndarray.
        TypeError : Raise error if `cluster_ids` is not np.ndarray or not list.
        ValueError: Raise error if `X` size is not matching `cluster_ids` length.
        ValueError: Raise error if `X` number_of_features is not matching `column_names` length.
        ValueError: Raise error if `result_dir` is empty when `save` is True.
    """

    if checkers.is_empty(column_names):
        column_names = [f"{_i}" for _i in range(X.shape[2])]

    if checkers.is_empty(cluster_ids):
        cluster_ids = [0] * X.shape[0]
        cluster_name_map = { 0: 'all' }

    if isinstance(cluster_ids, np.ndarray):
        cluster_ids = cluster_ids.tolist()

    try:
        if checkers.is_empty(X):
            raise ValueError("`X` argument numpy array is empty!")

        if X.ndim != 3:
            raise ValueError(f"`X` argument numpy array dimension [{X.ndim}D] is invalid, should be [3D] (size, sequence_length, number_of_features)!")

        if not isinstance(cluster_ids, list):
            raise TypeError(f"`cluster_ids` argument type <{type(cluster_ids).__name__}> is invalid, should be <np.ndarray> or <list>!")

        if X.shape[0] != len(cluster_ids):
            raise ValueError(f"`X` argument size '{X.shape[0]}' is invalid, should be equal to == '{len(cluster_ids)}'!")

        if X.shape[2] != len(column_names):
            raise ValueError(f"`X` argument number_of_features '{X.shape[2]}' is invalid, should be equal to == '{len(column_names)}'!")

        if save:
            result_dir = result_dir.strip()
            if checkers.is_empty(result_dir):
                raise ValueError("`result_dir` argument value is empty!")

            utils.create_dir(result_dir)
    except Exception as err:
        logger.error(err)
        raise

    plt.rc('xtick', labelsize=labelsize)
    plt.rc('ytick', labelsize=labelsize)

    _plot_file_paths = []
    for _column_id, _column in enumerate(column_names):
        _col_max_value = np.max(X[:, :, _column_id])
        _col_max_value = _col_max_value + abs(_col_max_value * 0.01)
        _col_min_value = np.min(X[:, :, _column_id])

        _un_cluster_ids = np.unique(cluster_ids)
        for _cluster_id in _un_cluster_ids:
            _cluster_indices = np.where(cluster_ids == _cluster_id)[0]
            _X_cluster = X[_cluster_indices]

            if not checkers.is_empty(cluster_name_map) and (_cluster_id in cluster_name_map):
                _cluster_name = cluster_name_map[_cluster_id]
            else:
                _cluster_name = str(_cluster_id)

            plt.title(f"{_column}")
            if ylim is not None:
                plt.ylim(ylim)
            else:
                plt.ylim([_col_min_value, _col_max_value])

            for _i in range(_X_cluster.shape[0]):
                plt.plot(_X_cluster[_i, :, _column_id], color="black", linewidth=0.7, alpha=0.3)

            plt.plot(np.mean(_X_cluster[:, :, _column_id], axis=0), color="red", linewidth=0.7, alpha=0.8)

            _column_name = str(_column).strip().replace(' ', '_')
            if save:
                _plot_file_path = os.path.join(result_dir, f"plot_column_{_column_name}.cluster_{_cluster_name}.png")
                plt.savefig(_plot_file_path, dpi=100, bbox_inches='tight')
                _plot_file_paths.append(_plot_file_path)
            else:
                plt.show()
            plt.close()

    return _plot_file_paths


@validate_arguments
def plot_overlap_columns_files(file_paths: list, column_names: list=None, column_indices: list=None, cluster_id: int=0, cluster_name: str=None, 
                                result_dir: str='results', save: bool=True, ylim:list=None):
    """Visualization and plotting method for each column by cluster_id. Read data from files.

    Args:
        file_paths     (list, required): List of CSV file path.
        column_names   (list, optional): Each column names to visualize. Defaults to None.
        column_indices (list, optional): Each column indices to visualize. If `column_names` is empty this will be used. Defaults to None.
        cluster_id     (int , optional): Cluster id to group. Defaults to None.
        cluster_name   (str , optional): Cluster name to display. Defaults to None.
        result_dir     (str , optional): Saving directory path. Defaults to 'results'.
        save           (bool, optional): Indicate save output to file or not . Defaults to True.
        ylim           (list, optional): y limit

    Raises:
        ValueError: Raise error if `file_paths` is empty.
        ValueError: Raise error if not found some file in `file_paths`.
        ValueError: Raise error if not supported file extension in `file_paths`.
        ValueError: Raise error if `column_names` and `column_indices` both of them empty.
        ValueError: Raise error if `result_dir` is empty when `save` is True.
    """

    cluster_name_map = None
    cluster_ids = [cluster_id] * len(file_paths)
    if not checkers.is_empty(cluster_name):
        cluster_name_map = { cluster_id: cluster_name }

    try:
        if checkers.is_empty(file_paths):
            raise ValueError("`file_paths` argument list is empty!")

        for _file_path in file_paths:
            if not os.path.exists(_file_path):
                raise ValueError(f"Not found '{_file_path}' file!")

            _filename, _ext = os.path.splitext(_file_path)
            if _ext.lower() != '.csv':
                raise ValueError(f"Not supported '{_file_path}' file format, only 'CSV' files!")

        if checkers.is_empty(column_names) and checkers.is_empty(column_indices):
            raise ValueError("`column_names` and `column_indices` both of them empty, at least one of them should've assigned with values!")

        if save:
            result_dir = result_dir.strip()
            if checkers.is_empty(result_dir):
                raise ValueError("`result_dir` argument value is empty!")

            utils.create_dir(result_dir)

        if ylim is not None:
            if not isinstance(ylim, list):
                raise ValueError("`ylim` argument type have to list!")

            if len(ylim) != 2:
                raise ValueError("`ylim` argument length have to 2!")

    except Exception as err:
        logger.error(err)
        raise


    _header = 0
    if checkers.is_empty(column_names) and (not checkers.is_empty(column_indices)):
        _header = None

    _max_row_size = 0
    _use_cols = []
    if not checkers.is_empty(column_names):
        _use_cols = column_names
    elif not checkers.is_empty(column_indices):
        _use_cols = column_indices

    for _file_path in file_paths:
        _df = pd.read_csv(_file_path, header=_header)

        if checkers.is_empty(column_names) and checkers.is_empty(column_indices):
            _use_cols = list(range(len(_df.columns)))

        if _max_row_size < _df.shape[0]:
            _max_row_size = _df.shape[0]

    _X = []
    for _file_path in file_paths:
        _df = pd.read_csv(_file_path, header=_header, usecols=_use_cols)
        _df = _df.select_dtypes(include=['int64', 'float64'])

        if checkers.is_empty(column_names):
            if not checkers.is_empty(column_indices):
                column_names = [str(_i) for _i in column_indices]
            else:
                column_names = list(_df.columns.values)

        _df = _df.replace(np.nan, 0)

        _column_min_values = _df.min()
        if _df.shape[0] < _max_row_size:
            _df = _df.reindex(index=range(_max_row_size))
            _df = _df.fillna(_column_min_values)

        _X.append(_df.values)

    _X = np.stack(_X)

    _plot_file_paths = plot_overlap_columns(X=_X, column_names=column_names, cluster_ids=cluster_ids, cluster_name_map=cluster_name_map, 
                                            result_dir=result_dir, save=save, ylim=ylim)
    return _plot_file_paths
