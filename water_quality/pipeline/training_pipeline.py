import sys

from water_quality.components.data_ingestion import DataIngestion
from water_quality.components.data_validation import DataValidation
from water_quality.components.data_transformation import DataTransformation
from water_quality.components.model_trainer import ModelTrainer
from water_quality.components.model_evaluation import ModelEvaluation
from water_quality.components.model_pusher import ModelPusher
from water_quality.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from water_quality.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)
from water_quality.exception import WaterQualityException
from water_quality.logger import logger


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion stage")
            ingestion = DataIngestion(self.data_ingestion_config)
            return ingestion.initiate_data_ingestion()
        except Exception as e:
            raise WaterQualityException(e, sys)

    def start_data_validation(self, ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation stage")
            validation = DataValidation(ingestion_artifact, self.data_validation_config)
            return validation.initiate_data_validation()
        except Exception as e:
            raise WaterQualityException(e, sys)

    def start_data_transformation(self, validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation stage")
            transformation = DataTransformation(validation_artifact, self.data_transformation_config)
            return transformation.initiate_data_transformation()
        except Exception as e:
            raise WaterQualityException(e, sys)

    def start_model_trainer(self, transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model trainer stage")
            trainer = ModelTrainer(transformation_artifact, self.model_trainer_config)
            return trainer.initiate_model_trainer()
        except Exception as e:
            raise WaterQualityException(e, sys)

    def start_model_evaluation(
        self,
        validation_artifact: DataValidationArtifact,
        trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation stage")
            evaluation = ModelEvaluation(
                self.model_evaluation_config, validation_artifact, trainer_artifact
            )
            return evaluation.initiate_model_evaluation()
        except Exception as e:
            raise WaterQualityException(e, sys)

    def start_model_pusher(self, evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            logger.info("Starting model pusher stage")
            pusher = ModelPusher(evaluation_artifact, self.model_pusher_config)
            return pusher.initiate_model_pusher()
        except Exception as e:
            raise WaterQualityException(e, sys)

    def run_pipeline(self):
        try:
            logger.info("=" * 60)
            logger.info("WATER QUALITY TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(ingestion_artifact)
            transformation_artifact = self.start_data_transformation(validation_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            evaluation_artifact = self.start_model_evaluation(validation_artifact, trainer_artifact)

            if evaluation_artifact.is_model_accepted:
                pusher_artifact = self.start_model_pusher(evaluation_artifact)
                logger.info(f"Model pushed to S3: {pusher_artifact.s3_model_path}")
            else:
                logger.info("Model not accepted — skipping model push.")

            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            return evaluation_artifact
        except Exception as e:
            raise WaterQualityException(e, sys)
