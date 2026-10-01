"""
predict.py
----------
Loads the artefacts saved by train.py (models/model.pkl and
models/vectorizer.pkl) and exposes a single function, predict_url(), that
classifies one URL as "Phishing" or "Legitimate".

This module is imported by app.py (the Flask backend). It can also be run
directly for a quick command-line check:

    python predict.py https://example.com/login
"""

import sys
import joblib
from scipy.sparse import hstack

from features import clean_url

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

_model = None
_vectorizer_bundle = None


def _load_artifacts():
    """Load the saved model/vectorizer once and cache them (lazy singleton)."""
    global _model, _vectorizer_bundle
    if _model is None or _vectorizer_bundle is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer_bundle = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer_bundle


def _featurize(url: str):
    """Turn a single raw URL into the same combined feature vector used in training."""
    _, bundle = _load_artifacts()
    tfidf, lexical, scaler = bundle["tfidf"], bundle["lexical"], bundle["scaler"]
    u = clean_url(url)
    X_tfidf = tfidf.transform([u])
    X_lex = scaler.transform(lexical.transform([u]))
    return hstack([X_tfidf, X_lex]).tocsr()


def predict_url(url: str) -> dict:
    """
    Classify a single URL.

    Returns a dict:
        {
          "url": <original input, trimmed>,
          "label": "Phishing" | "Legitimate",
          "is_phishing": bool,
          "confidence": float in [0, 1]   # probability of the predicted class
        }
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("Please enter a URL.")

    model, _ = _load_artifacts()
    features = _featurize(url)

    pred = int(model.predict(features)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = float(proba[pred])
    else:
        confidence = None  # model type has no probability estimate

    return {
        "url": url,
        "label": "Phishing" if pred == 1 else "Legitimate",
        "is_phishing": bool(pred == 1),
        "confidence": confidence,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <url>")
        sys.exit(1)
    result = predict_url(sys.argv[1])
    conf = f"{result['confidence']*100:.1f}%" if result["confidence"] is not None else "n/a"
    print(f"URL       : {result['url']}")
    print(f"Prediction: {result['label']}  (confidence: {conf})")
