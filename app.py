from flask import Flask, request, jsonify, render_template
from pathlib import Path
import yaml
from src.pipeline.prediction_pipeline import ModelPredict   

app = Flask(__name__)
with open("config.yaml","r") as f:
    cfg=yaml.safe_load(f)

predictor = ModelPredict(
    model_path=Path(cfg["model_pred"]["model_path"]),
    encoder_path=Path(cfg["model_pred"]["encoder_path"]),
    tfidf_path=Path(cfg["model_pred"]["tfidf_path"])
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if data is None or "texts" not in data:
        return jsonify({"error": "JSON icinde 'texts' alani olmali"}), 400

    texts = data["texts"]

    if not isinstance(texts, list):
        return jsonify({"error": "'texts' list olmali"}), 400

    try:
        preds = predictor.predict(texts)

        return jsonify({
            "predictions": preds.tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)