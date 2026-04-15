import os
import sys
import numpy as np
import pandas as pd
import boto3

from water_quality.exception import WaterQualityException
from water_quality.logger import logger
from water_quality.utils.main_utils import load_object
from water_quality.constants import MODEL_BUCKET_NAME, MODEL_PUSHER_S3_KEY


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("saved_models", "model.pkl")
        self.bucket_name = MODEL_BUCKET_NAME
        self.s3_key = f"{MODEL_PUSHER_S3_KEY}/model.pkl"

    def get_model(self):
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from local path: {self.model_path}")
                return load_object(self.model_path)
            logger.info("Downloading model from S3...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            s3 = boto3.client("s3")
            s3.download_file(self.bucket_name, self.s3_key, self.model_path)
            return load_object(self.model_path)
        except Exception as e:
            raise WaterQualityException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        try:
            model = self.get_model()
            return model.predict(dataframe)
        except Exception as e:
            raise WaterQualityException(e, sys)


class WaterQualityData:
    """Handles form data -> DataFrame conversion for the 20-feature dataset."""
    def __init__(
        self,
        aluminium: float,
        ammonia: float,
        arsenic: float,
        barium: float,
        cadmium: float,
        chloramine: float,
        chromium: float,
        copper: float,
        flouride: float,
        bacteria: float,
        viruses: float,
        lead: float,
        nitrates: float,
        nitrites: float,
        mercury: float,
        perchlorate: float,
        radium: float,
        selenium: float,
        silver: float,
        uranium: float,
    ):
        self.aluminium = aluminium
        self.ammonia = ammonia
        self.arsenic = arsenic
        self.barium = barium
        self.cadmium = cadmium
        self.chloramine = chloramine
        self.chromium = chromium
        self.copper = copper
        self.flouride = flouride
        self.bacteria = bacteria
        self.viruses = viruses
        self.lead = lead
        self.nitrates = nitrates
        self.nitrites = nitrites
        self.mercury = mercury
        self.perchlorate = perchlorate
        self.radium = radium
        self.selenium = selenium
        self.silver = silver
        self.uranium = uranium

    def get_water_input_data_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame({
                "aluminium":   [self.aluminium],
                "ammonia":     [self.ammonia],
                "arsenic":     [self.arsenic],
                "barium":      [self.barium],
                "cadmium":     [self.cadmium],
                "chloramine":  [self.chloramine],
                "chromium":    [self.chromium],
                "copper":      [self.copper],
                "flouride":    [self.flouride],
                "bacteria":    [self.bacteria],
                "viruses":     [self.viruses],
                "lead":        [self.lead],
                "nitrates":    [self.nitrates],
                "nitrites":    [self.nitrites],
                "mercury":     [self.mercury],
                "perchlorate": [self.perchlorate],
                "radium":      [self.radium],
                "selenium":    [self.selenium],
                "silver":      [self.silver],
                "uranium":     [self.uranium],
            })
        except Exception as e:
            raise WaterQualityException(e, sys)
