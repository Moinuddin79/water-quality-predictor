import os
import sys
import shutil
import boto3

from water_quality.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from water_quality.entity.config_entity import ModelPusherConfig
from water_quality.exception import WaterQualityException
from water_quality.logger import logger
from water_quality.utils.main_utils import load_object, save_object


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig = ModelPusherConfig(),
    ):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise WaterQualityException(e, sys)

    def upload_model_to_s3(self, local_model_path: str) -> str:
        """Upload the trained model to AWS S3."""
        try:
            s3 = boto3.client("s3")
            s3_key = f"{self.model_pusher_config.s3_model_key_path}/model.pkl"
            s3.upload_file(
                local_model_path,
                self.model_pusher_config.bucket_name,
                s3_key,
            )
            s3_uri = f"s3://{self.model_pusher_config.bucket_name}/{s3_key}"
            logger.info(f"Model uploaded to: {s3_uri}")
            return s3_uri
        except Exception as e:
            raise WaterQualityException(e, sys)

    def save_model_locally(self, source: str) -> str:
        """Also copy model to saved_models/ for local prediction serving."""
        try:
            os.makedirs(self.model_pusher_config.trained_model_path, exist_ok=True)
            dest = os.path.join(self.model_pusher_config.trained_model_path, "model.pkl")
            shutil.copy(source, dest)
            logger.info(f"Model saved locally to: {dest}")
            return dest
        except Exception as e:
            raise WaterQualityException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logger.info(">>> Starting Model Pusher <<<")
            trained_model_path = self.model_evaluation_artifact.trained_model_path
            self.save_model_locally(trained_model_path)
            s3_path = self.upload_model_to_s3(trained_model_path)

            artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=s3_path,
            )
            logger.info(f"Model Pusher Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise WaterQualityException(e, sys)
