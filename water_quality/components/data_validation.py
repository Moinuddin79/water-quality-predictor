import os
import sys
import pandas as pd
from evidently.metric_preset import DataDriftPreset
try:
    from evidently.report import Report
except ImportError:
    Report = None

from water_quality.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from water_quality.entity.config_entity import DataValidationConfig
from water_quality.exception import WaterQualityException
from water_quality.logger import logger
from water_quality.utils.main_utils import read_yaml_file, write_yaml_file
from water_quality.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise WaterQualityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise WaterQualityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise WaterQualityException(e, sys)

    def is_column_exist(self, df: pd.DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            if missing_numerical_columns:
                logger.error(f"Missing columns: {missing_numerical_columns}")
                return False
            return True
        except Exception as e:
            raise WaterQualityException(e, sys)

    def detect_dataset_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """Run EvidentlyAI DataDrift report and save to YAML."""
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)
            report_dict = report.as_dict()

            drift_status = report_dict["metrics"][0]["result"]["dataset_drift"]
            drift_share = report_dict["metrics"][0]["result"]["drift_share"]

            drift_report = {
                "drift_detected": bool(drift_status),
                "drift_share": float(drift_share),
                "columns_drifted": int(
                    report_dict["metrics"][0]["result"]["number_of_drifted_columns"]
                ),
            }

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=drift_report,
            )
            logger.info(f"Drift report: {drift_report}")
            return not drift_status
        except Exception as e:
            raise WaterQualityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info(">>> Starting Data Validation <<<")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            error_message = ""
            if not self.validate_number_of_columns(train_df):
                error_message += "Train data is missing columns. "
            if not self.validate_number_of_columns(test_df):
                error_message += "Test data is missing columns. "
            if not self.is_column_exist(train_df):
                error_message += "Train data missing required numerical columns. "
            if not self.is_column_exist(test_df):
                error_message += "Test data missing required numerical columns. "

            validation_status = len(error_message) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logger.info("No significant data drift detected.")
                else:
                    logger.warning("Data drift detected in the dataset!")

            # Copy valid files
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)

            if validation_status:
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)
            else:
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                raise Exception(error_message)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logger.info(f"Data Validation Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise WaterQualityException(e, sys)
