import logging
import sys

# This makes logs print to terminal AND save to file
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),   # print to terminal
    ]
)

from water_quality.pipeline.training_pipeline import TrainingPipeline
from water_quality.exception import WaterQualityException

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("  WATER QUALITY TRAINING PIPELINE STARTING...")
        print("=" * 60)
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        print("=" * 60)
        print("  PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as e:
        raise WaterQualityException(e, sys)