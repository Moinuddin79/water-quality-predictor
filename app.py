from flask import Flask, request, render_template, jsonify
from water_quality.pipeline.prediction_pipeline import PredictionPipeline, WaterQualityData
from water_quality.pipeline.training_pipeline import TrainingPipeline
from water_quality.logger import logger
from water_quality.exception import WaterQualityException
import sys

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")
    try:
        water_data = WaterQualityData(
            aluminium=float(request.form.get("aluminium")),
            ammonia=float(request.form.get("ammonia")),
            arsenic=float(request.form.get("arsenic")),
            barium=float(request.form.get("barium")),
            cadmium=float(request.form.get("cadmium")),
            chloramine=float(request.form.get("chloramine")),
            chromium=float(request.form.get("chromium")),
            copper=float(request.form.get("copper")),
            flouride=float(request.form.get("flouride")),
            bacteria=float(request.form.get("bacteria")),
            viruses=float(request.form.get("viruses")),
            lead=float(request.form.get("lead")),
            nitrates=float(request.form.get("nitrates")),
            nitrites=float(request.form.get("nitrites")),
            mercury=float(request.form.get("mercury")),
            perchlorate=float(request.form.get("perchlorate")),
            radium=float(request.form.get("radium")),
            selenium=float(request.form.get("selenium")),
            silver=float(request.form.get("silver")),
            uranium=float(request.form.get("uranium")),
        )
        df = water_data.get_water_input_data_frame()
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(df)[0]
        result = "Safe to Drink" if prediction == 1 else "Not Safe to Drink"
        result_class = "safe" if prediction == 1 else "unsafe"
        return render_template("index.html", result=result, result_class=result_class)
    except Exception as e:
        raise WaterQualityException(e, sys)

@app.route("/train")
def train():
    try:
        logger.info("Training pipeline triggered via /train endpoint")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return jsonify({"status": "success", "message": "Training pipeline completed successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
