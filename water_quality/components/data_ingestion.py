import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

from water_quality.constants import MONGODB_URL_KEY, DATABASE_NAME
from water_quality.entity.config_entity import DataIngestionConfig
from water_quality.entity.artifact_entity import DataIngestionArtifact
from water_quality.exception import WaterQualityException
from water_quality.logger import logger


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise WaterQualityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            logger.info("Fetching data from MongoDB collection")
            mongo_client = MongoClient(os.environ[MONGODB_URL_KEY])
            collection = mongo_client[DATABASE_NAME][self.data_ingestion_config.collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": pd.NA}, inplace=True)
            # Remove rows where is_safe has invalid values like '#NUM!'
            before = len(df)
            df = df[pd.to_numeric(df["is_safe"], errors="coerce").notna()].copy()
            df["is_safe"] = df["is_safe"].astype(int)
            removed = before - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} rows with invalid 'is_safe' values")
            logger.info(f"Fetched {len(df)} valid records. Shape: {df.shape}")
            return df
        except Exception as e:
            raise WaterQualityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(f"Saved data to feature store: {feature_store_file_path}")
            return dataframe
        except Exception as e:
            raise WaterQualityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=dataframe["is_safe"],
            )
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True
            )
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logger.info(f"Train shape: {train_set.shape} | Test shape: {test_set.shape}")
        except Exception as e:
            raise WaterQualityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info(">>> Starting Data Ingestion <<<")
            df = self.export_collection_as_dataframe()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)
            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logger.info(f"Data Ingestion Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise WaterQualityException(e, sys)
