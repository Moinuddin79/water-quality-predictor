import os
import sys
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from water_quality.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from water_quality.entity.config_entity import ModelTrainerConfig
from water_quality.exception import WaterQualityException
from water_quality.logger import logger
from water_quality.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object,
    get_classification_score,
)


class WaterQualityModel:
    """Wraps preprocessor + classifier for unified predict()."""
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, x):
        try:
            x_transformed = self.preprocessor.transform(x)
            return self.model.predict(x_transformed)
        except Exception as e:
            raise WaterQualityException(e, sys)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise WaterQualityException(e, sys)

    def train_model(self, X_train, y_train):
        try:
            xgb_clf = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
            xgb_clf.fit(X_train, y_train)
            return xgb_clf
        except Exception as e:
            raise WaterQualityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info(">>> Starting Model Trainer <<<")
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model = self.train_model(X_train, y_train)

            y_train_pred = model.predict(X_train)
            train_metric = get_classification_score(y_train, y_train_pred)
            logger.info(f"Train metrics: {train_metric}")

            y_test_pred = model.predict(X_test)
            test_metric = get_classification_score(y_test, y_test_pred)
            logger.info(f"Test metrics: {test_metric}")

            if test_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model F1 {test_metric.f1_score:.4f} is below "
                    f"threshold {self.model_trainer_config.expected_accuracy}"
                )

            # Wrap with preprocessor for prediction pipeline
            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )
            wq_model = WaterQualityModel(preprocessor=preprocessor, model=model)

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True
            )
            save_object(self.model_trainer_config.trained_model_file_path, wq_model)

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )
            logger.info(f"Model Trainer Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise WaterQualityException(e, sys)
