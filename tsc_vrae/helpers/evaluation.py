# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from el_logging import logger
from el_validator import checkers

from . import utils


@validate_arguments
def evaluate_cluster(cluster_ids, true_labels, filename: str='cluster_scores.csv', result_dir: str='results', save: bool=False):
    """Evaluate cluster ids with true labels based on normalized mutual fnformation, adjusted mutual information and rand index adjusted methods.

    Args:
        cluster_ids (np.ndarray or list, required): Infered cluster ids to evaluate.
        true_labels (np.ndarray or list, required): Ground truth labels to evaluate cluster ids.
        filename    (str               , optional): File name used to save result. Defaults to 'cluster_scores.csv'.
        result_dir  (str               , optional): Saving directory path. Defaults to 'results'.
        save        (bool              , optional): Indicate save output to file or not. Defaults to False.

    Raises:
        TypeError : Rase error if 'cluster_ids' is not list or np.ndarray.
        TypeError : Rase error if 'true_labels' is not list or np.ndarray.
        ValueError: Rase error if 'cluster_ids' length is not matching 'labels' length.
        ValueError: Rase error if 'filename' is empty and 'save' is True.
        ValueError: Rase error if 'result_dir' is empty and 'save' is True.

    Returns:
        tuple(dict, pd.DataFrame): Cluster evaluation scores as dictionary and pd.DataFrame.
    """

    if isinstance(cluster_ids, np.ndarray):
        cluster_ids = cluster_ids.tolist()

    if isinstance(true_labels, np.ndarray):
        true_labels = true_labels.tolist()

    try:
        if not isinstance(cluster_ids, list):
            raise TypeError(f"'cluster_ids' argument type <{type(cluster_ids).__name__}> is invalid, should be <list> or <np.ndarray>!")

        if not isinstance(true_labels, list):
            raise TypeError(f"'true_labels' argument type <{type(true_labels).__name__}> is invalid, should be <list> or <np.ndarray>!")

        if len(cluster_ids) != len(true_labels):
            raise ValueError(f"'cluster_ids' length '{len(cluster_ids)}' is not matching with 'true_labels' length {len(true_labels)}!")

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

    _unique_labels = list(map(str, set(true_labels)))
    _label_id_dict = {}
    for _i, _val in enumerate(_unique_labels):
        _label_id_dict[_val] = _i

    _label_ids = []
    for _true_label in true_labels:
        _label_ids.append(_label_id_dict[str(_true_label)])

    _nmi_score = normalized_mutual_info_score(_label_ids, cluster_ids)
    _ami_score = adjusted_mutual_info_score(_label_ids, cluster_ids)
    _ar_score = adjusted_rand_score(_label_ids, cluster_ids)

    _eval_scores = {
        'normalized_mutual_info': _nmi_score,
        'adjusted_mutual_info': _ami_score,
        'adjusted_rand': _ar_score
    }

    _eval_scores_df = pd.DataFrame(_eval_scores.items(), columns=['eval_method', 'score'])
    if save:
        _eval_file_path = os.path.join(result_dir, filename)
        _eval_scores_df.to_csv(_eval_file_path, index=False)

    return _eval_scores, _eval_scores_df


@validate_arguments
def summary(cluster_ids, true_labels, filename: str='cluster_summary.csv', result_dir: str='results', save: bool=False):
    """Summarize cluster based on true labels.

    Args:
        cluster_ids (list or np.ndarray or pd.DataFrame, required): Infered cluster ids to summarize.
        true_labels (list or np.ndarray or pd.DataFrame, required): Ground truth labels to summarize cluster ids.
        filename    (str                               , optional): File name used to save result. Defaults to 'cluster_summary.csv'.
        result_dir  (str                               , optional): Saving directory path. Defaults to 'results'.
        save        (bool                              , optional): Indicate save output to file or not. Defaults to False.

    Raises:
        TypeError : Rase error if 'cluster_ids' is not list, np.ndarray or pd.DataFrame.
        TypeError : Rase error if 'true_labels' is not list, np.ndarray or pd.DataFrame.
        ValueError: Rase error if 'cluster_ids' length is not matching 'true_labels' length.
        TypeError : Rase error if 'cluster_ids' is pd.DataFrame and doesn't have 'cluster_id' column.
        TypeError : Rase error if 'true_labels' is pd.DataFrame and doesn't have 'label' column.
        ValueError: Rase error if 'filename' is empty and 'save' is True.
        ValueError: Rase error if 'result_dir' is empty and 'save' is True.

    Returns:
        tuple(dict, pd.DataFrame): Cluster summary based on true labels as dictionary and pd.DataFrame.
    """

    _cluster_ids_df = cluster_ids
    _true_labels_df = true_labels

    if isinstance(cluster_ids, list) or isinstance(cluster_ids, np.ndarray):
        _cluster_ids_df = pd.DataFrame(cluster_ids, columns=['cluster_id'])

    if isinstance(true_labels, list) or isinstance(true_labels, np.ndarray):
        _true_labels_df = pd.DataFrame(true_labels, columns=['label'])

    try:
        if not isinstance(_cluster_ids_df, pd.DataFrame):
            raise TypeError(f"'cluster_ids' argument type <{type(_cluster_ids_df).__name__}> is invalid, should be <pd.DataFrame>, <list> or <np.ndarray>!")

        if not isinstance(_true_labels_df, pd.DataFrame):
            raise TypeError(f"'true_labels' argument type <{type(_true_labels_df).__name__}> is invalid, should be <pd.DataFrame>, <list> or <np.ndarray>!")

        if len(_cluster_ids_df.index) != len(_true_labels_df.index):
            raise ValueError(f"'cluster_ids' length '{len(_cluster_ids_df.index)}' is not matching with 'true_labels' length {len(_true_labels_df.index)}!")

        if not 'cluster_id' in _cluster_ids_df.columns:
            raise TypeError(f"'cluster_ids' argument doesn't have 'cluster_id' column!")

        if not 'label' in _true_labels_df.columns:
            raise TypeError(f"'true_labels' argument doesn't have 'label' column!")

        if len(_true_labels_df.columns) > 1:
            _true_labels_df = _true_labels_df['label']

        if save:
            filename = filename.strip()
            if checkers.is_empty(filename, trim_str=True):
                raise ValueError("'filename' argument value is empty!")

            result_dir = result_dir.strip()
            if checkers.is_empty(result_dir, trim_str=True):
                raise ValueError("'result_dir' argument value is empty!")

            utils.create_dir(result_dir)
    except Exception as err:
        logger.error(err)
        raise

    _label_cluster_df = _cluster_ids_df.merge(_true_labels_df, left_index=True, right_index=True)
    _label_count_df = _label_cluster_df.groupby(['label']).size().reset_index(name='count')
    _summary_dict = { 'cluster_id': [] }

    for _i, _row in _label_count_df.iterrows():
        _summary_dict[_row['label']] = []
    _summary_dict['total_count'] = []

    _summary_count_df = _label_cluster_df.groupby(['cluster_id', 'label']).size().reset_index(name='count')
    _cluster_count_df = _label_cluster_df.groupby('cluster_id').size().reset_index(name='cluster_count')

    _summary_df = pd.DataFrame(_summary_dict)


    for _i, _row in _cluster_count_df.iterrows():
        _tmp_df = _summary_count_df.loc[_summary_count_df['cluster_id'] == _row['cluster_id']].drop(columns=['cluster_id'])

        _tmp_df = _tmp_df.T
        _tmp_df.reset_index(drop=True, inplace=True)
        _tmp_df.rename(columns=_tmp_df.iloc[0], inplace=True)
        _tmp_df.drop(_tmp_df.index[0], inplace=True)
        _tmp_df.reset_index(drop=True, inplace=True)
        _tmp_df.insert(0, 'cluster_id', [_row['cluster_id']])
        _tmp_df['total_count'] = [_row['cluster_count']]

        _summary_df = _summary_df.merge(_tmp_df, how='outer')

    _summary_df.fillna(0, inplace=True)
    _summary_df = _summary_df.astype('int64')

    _cluster_id_col = _summary_df['cluster_id']
    _summary_df.drop(columns=['cluster_id'], inplace=True)
    _summary_df.insert(0, 'cluster_id', _cluster_id_col)

    _eval_score_filename = f"{os.path.splitext(filename)[0]}.scores.csv"
    _eval_scores, _ = evaluate_cluster(_cluster_ids_df['cluster_id'].values, _true_labels_df['label'].values, _eval_score_filename, result_dir, save)
    _summary_dict = {
        'summary': json.loads(_summary_df.to_json(orient='records')),
        'eval_scores': _eval_scores
    }

    if save:
        _summary_file_path = os.path.join(result_dir, filename)
        _summary_df.to_csv(_summary_file_path, index=False)

        _summary_json_file_path = os.path.join(result_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(_summary_json_file_path, 'w') as _json_file:
            _json_file.write(json.dumps(_summary_dict, indent=4, ensure_ascii=False))

    return _summary_dict, _summary_df


@validate_arguments
def evaluate_conf_matrix(cluster_ids, true_labels, label_map: dict, filename: str='cluster_conf_matrix.csv', result_dir: str='results', save: bool=False):
    """Evaluate confusion matrix of clusters with true labels based on label_map.

    Args:
        cluster_ids (np.ndarray or list, required): Infered cluster ids to evaluate.
        labels      (np.ndarray or list, required): Ground truth labels to evaluate cluster ids.
        label_map   (dict              , required): Label map for cluster ids.
        filename    (str               , optional): File name used to save result. Defaults to 'cluster_conf_matrix.csv'.
        result_dir  (str               , optional): Saving directory path. Defaults to 'results'.
        save        (bool              , optional): Indicate save output to file or not. Defaults to False.

    Raises:
        TypeError : Rase error if 'cluster_ids' is not list or np.ndarray.
        TypeError : Rase error if 'true_labels' is not list or np.ndarray.
        ValueError: Rase error if 'cluster_ids' length is not matching 'true_labels' length.
        ValueError: Rase error if 'filename' is empty and 'save' is True.
        ValueError: Rase error if 'result_dir' is empty and 'save' is True.

    Returns:
        tuple(np.int64, np.int64, np.int64, np.int64): Cluster confusion matrix => TN, FP, FN, TP.
    """

    if isinstance(cluster_ids, np.ndarray):
        cluster_ids = cluster_ids.tolist()

    if isinstance(true_labels, np.ndarray):
        true_labels = true_labels.tolist()

    try:
        if not isinstance(cluster_ids, list):
            raise TypeError(f"'cluster_ids' argument type <{type(cluster_ids).__name__}> is invalid, should be <list> or <np.ndarray>!")

        if not isinstance(true_labels, list):
            raise TypeError(f"'true_labels' argument type <{type(true_labels).__name__}> is invalid, should be <list> or <np.ndarray>!")

        if len(cluster_ids) != len(true_labels):
            raise ValueError(f"'cluster_ids' length '{len(cluster_ids)}' is not matching with 'true_labels' length {len(true_labels)}!")

        if save:
            filename = filename.strip()
            if checkers.is_empty(filename, trim_str=True):
                raise ValueError("'filename' argument value is empty!")

            filename = result_dir.strip()
            if checkers.is_empty(result_dir, trim_str=True):
                raise ValueError("'result_dir' argument value is empty!")

            utils.create_dir(result_dir)
    except Exception as err:
        logger.error(err)
        raise

    _cluster_labels = []
    for _cluster_id in cluster_ids:
        if _cluster_id in label_map:
            _cluster_labels.append(label_map[_cluster_id])
        elif str(_cluster_id) in label_map:
            _cluster_labels.append(label_map[str(_cluster_id)])
        else:
            _cluster_labels.append(str(_cluster_id))

    _tn, _fp, _fn, _tp = confusion_matrix(true_labels, _cluster_labels).ravel()
    return _tn, _fp, _fn, _tp
