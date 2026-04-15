import os
from datetime import date

# ─── MongoDB ───────────────────────────────────────────────────────────────────
DATABASE_NAME = "water_quality"
COLLECTION_NAME = "water_data"
MONGODB_URL_KEY = "MONGODB_URL"

# ─── Common pipeline constants ─────────────────────────────────────────────────
PIPELINE_NAME: str = "WaterQualityPipeline"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "water.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
MODEL_FILE_NAME: str = "model.pkl"
PREPROCSSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
TARGET_COLUMN: str = "is_safe"
CURRENT_YEAR = date.today().year
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")
MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# ─── Data Ingestion ────────────────────────────────────────────────────────────
DATA_INGESTION_COLLECTION_NAME: str = "water_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# ─── Data Validation ──────────────────────────────────────────────────────────
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# ─── Data Transformation ──────────────────────────────────────────────────────
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# ─── Model Trainer ────────────────────────────────────────────────────────────
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.70
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# ─── Model Evaluation ─────────────────────────────────────────────────────────
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME: str = "report.yaml"

# ─── Model Pusher ─────────────────────────────────────────────────────────────
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR: str = os.path.join("saved_models")

# ─── AWS S3 ───────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID_ENV_KEY: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY: str = "AWS_SECRET_ACCESS_KEY"
AWS_DEFAULT_REGION_NAME: str = "us-east-1"
MODEL_BUCKET_NAME: str = "water-quality-model-bucket"
MODEL_PUSHER_S3_KEY: str = "model-registry"
