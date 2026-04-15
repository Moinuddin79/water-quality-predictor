import os
import sys
import pandas as pd
import numpy as np

from water_quality.constants import TARGET_COLUMN, MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
from water_quality.entity.artifact_entity import (
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from water_quality.entity.config_entity import ModelEvaluationConfig
from water_quality.exception import WaterQualityException
from water_quality.logger import logger
from water_quality.utils.main_utils import (
    load_object,
    write_yaml_file,
    get_classification_score,
)


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise WaterQualityException(e, sys)

    def get_best_model(self):
        """Load the best model from S3 if it exists."""
        try:
            import boto3
            bucket_name = self.model_evaluation_config.bucket_name
            s3_key = self.model_evaluation_config.s3_model_key_path + "/model.pkl"

            s3 = boto3.client("s3")
            local_path = os.path.join("artifact", "s3_best_model", "model.pkl")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            try:
                s3.download_file(bucket_name, s3_key, local_path)
                logger.info(f"Downloaded best model from S3: s3://{bucket_name}/{s3_key}")
                return load_object(local_path)
            except Exception:
                logger.info("No existing best model found in S3. This is the first run.")
                return None
        except Exception as e:
            raise WaterQualityException(e, sys)

    def evaluate_model(self):
        try:
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            trained_model_score = get_classification_score(y_test, trained_model.predict(X_test))

            best_model = self.get_best_model()

            if best_model is None:
                logger.info("No production model found — accepting trained model.")
                return ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=0.0,
                    best_model_path=None,
                    trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                    train_model_metric_artifact=trained_model_score,
                    best_model_metric_artifact=trained_model_score,
                )

            best_model_score = get_classification_score(y_test, best_model.predict(X_test))
            improved_accuracy = trained_model_score.f1_score - best_model_score.f1_score

            is_model_accepted = improved_accuracy > self.model_evaluation_config.change_threshold

            logger.info(
                f"Trained F1: {trained_model_score.f1_score:.4f} | "
                f"Best F1: {best_model_score.f1_score:.4f} | "
                f"Improvement: {improved_accuracy:.4f} | Accepted: {is_model_accepted}"
            )

            report = {
                "trained_model_f1": trained_model_score.f1_score,
                "best_model_f1": best_model_score.f1_score,
                "improvement": improved_accuracy,
                "is_accepted": is_model_accepted,
            }
            write_yaml_file(self.model_evaluation_config.report_file_path, report)

            return ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=None,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=trained_model_score,
                best_model_metric_artifact=best_model_score,
            )
        except Exception as e:
            raise WaterQualityException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info(">>> Starting Model Evaluation <<<")
            artifact = self.evaluate_model()
            logger.info(f"Model Evaluation Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise WaterQualityException(e, sys)
