import os
import sys
import dill
import yaml
import numpy as np
import pandas as pd
from pandas import DataFrame

from water_quality.exception import WaterQualityException
from water_quality.logger import logger


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise WaterQualityException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as e:
        raise WaterQualityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info(f"Saving object to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise WaterQualityException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file path: {file_path} does not exist")
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise WaterQualityException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            np.save(f, array)
    except Exception as e:
        raise WaterQualityException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        raise WaterQualityException(e, sys) from e


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    try:
        df.drop(columns=cols, inplace=True)
        return df
    except Exception as e:
        raise WaterQualityException(e, sys) from e


def get_classification_score(y_true, y_pred) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from water_quality.entity.artifact_entity import ClassificationMetricArtifact
    try:
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
            roc_auc_score=roc_auc_score(y_true, y_pred),
        )
    except Exception as e:
        raise WaterQualityException(e, sys) from e
