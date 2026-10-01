"""
app.py
------
Flask web backend for the AI-Powered Real-Time Phishing Detection System.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import os
from flask import Flask, render_template, request, jsonify

from predict import predict_url

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    submitted_url = ""

    if request.method == "POST":
        submitted_url = request.form.get("url", "")
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            error = "Model not found. Please run 'python train.py' first."
        else:
            try:
                result = predict_url(submitted_url)
            except ValueError as e:
                error = str(e)
            except Exception:
                error = "Something went wrong while analysing this URL. Please try again."

    return render_template(
        "index.html", result=result, error=error, submitted_url=submitted_url
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """A small JSON API endpoint, e.g. for a future browser extension."""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")
    try:
        result = predict_url(url)
        return jsonify({"success": True, **result})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "Prediction failed."}), 500


if __name__ == "__main__":
    app.run(debug=True)
