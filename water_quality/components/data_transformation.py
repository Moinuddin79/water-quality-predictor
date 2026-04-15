import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTETomek

from water_quality.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from water_quality.entity.config_entity import DataTransformationConfig
from water_quality.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from water_quality.exception import WaterQualityException
from water_quality.logger import logger
from water_quality.utils.main_utils import (
    save_numpy_array_data,
    save_object,
    read_yaml_file,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise WaterQualityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise WaterQualityException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Returns a sklearn Pipeline with:
        - Median imputation for missing values (water data has ~15% NaN)
        - RobustScaler to handle outliers in chemical measurements
        """
        try:
            logger.info("Building transformation pipeline: Imputer → RobustScaler")
            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]
            )
            return pipeline
        except Exception as e:
            raise WaterQualityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info(">>> Starting Data Transformation <<<")
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Split features and target
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            logger.info(f"Train class distribution before resampling:\n{y_train.value_counts()}")

            # Fit and transform
            preprocessor = self.get_data_transformer_object()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # Handle class imbalance with SMOTETomek
            smt = SMOTETomek(random_state=42)
            X_train_resampled, y_train_resampled = smt.fit_resample(X_train_arr, y_train)

            logger.info(
                f"After SMOTETomek — train shape: {X_train_resampled.shape}, "
                f"class dist: {pd.Series(y_train_resampled).value_counts().to_dict()}"
            )

            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logger.info(f"Data Transformation Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise WaterQualityException(e, sys)
