"""
train.py
--------
Trains the AI-powered phishing-URL detector described in the project report.

Pipeline:  URL text
              |
              v
   TF-IDF (word 1-2 grams)  +  hand-crafted lexical features   [features.py]
              |
              v
        classifier  (Logistic Regression is the primary/baseline model;
                      Naive Bayes / Linear SVM / Random Forest are trained
                      alongside it for comparison -- this is the
                      "Iteration 2: compare alternative classifiers"
                      refinement from Chapter 4 of the report)
              |
              v
   best model saved to models/model.pkl
   feature extractor saved to models/vectorizer.pkl

Run:
    python train.py
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from features import LexicalFeatures, clean_url

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset/phishing.csv"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
RANDOM_STATE = 42

# The model that gets deployed (saved to MODEL_PATH) at the end of training.
# Logistic Regression is the primary/baseline model documented in the project
# report -- fast, interpretable, and a tiny file to ship. The other
# classifiers below are trained purely for the comparison table (Table 6.1 /
# "Iteration 2 -- compare alternative classifiers"). Change this constant if
# you would rather deploy whichever model tests best (e.g. "Random Forest"),
# but note Random Forest in particular saves a much larger .pkl file.
DEPLOY_MODEL_NAME = "Logistic Regression (baseline / primary model)"


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).map(clean_url)
    df = df[df["url"].str.len() > 0]
    df = df.drop_duplicates(subset="url")
    # phishing = 1 (the "positive" / attack class), legitimate = 0
    df["target"] = df["label"].str.strip().str.lower().map(
        {"phishing": 1, "bad": 1, "legitimate": 0, "good": 0}
    )
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    return df


def build_feature_extractor():
    """TF-IDF (fit on training text) + hand-crafted lexical features."""
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
    )
    lexical = LexicalFeatures()
    return tfidf, lexical


def make_matrix(tfidf, lexical, urls, scaler=None, fit=False):
    """Fit (if requested) and transform raw URLs into one combined sparse matrix."""
    if fit:
        X_tfidf = tfidf.fit_transform(urls)
        X_lex = lexical.transform(urls)
        scaler = MinMaxScaler()
        X_lex = scaler.fit_transform(X_lex)
    else:
        X_tfidf = tfidf.transform(urls)
        X_lex = lexical.transform(urls)
        X_lex = scaler.transform(X_lex)
    X = hstack([X_tfidf, X_lex]).tocsr()
    return X, scaler


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{'-'*60}\n{name}\n{'-'*60}")
    print(f"Accuracy : {acc*100:6.2f}%")
    print(f"Precision: {prec*100:6.2f}%")
    print(f"Recall   : {rec*100:6.2f}%")
    print(f"F1-score : {f1*100:6.2f}%")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    return {"name": name, "model": model, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "confusion_matrix": cm}


def main():
    print("Loading dataset...")
    df = load_data(DATASET_PATH)
    print(f"Total usable rows: {len(df)}  "
          f"(phishing={int((df.target==1).sum())}, "
          f"legitimate={int((df.target==0).sum())})")

    X_train_urls, X_test_urls, y_train, y_test = train_test_split(
        df["url"].values, df["target"].values,
        test_size=0.20, random_state=RANDOM_STATE, stratify=df["target"].values,
    )
    print(f"Train size: {len(X_train_urls)}   Test size: {len(X_test_urls)}")

    print("\nExtracting TF-IDF + lexical features (fit on training set only)...")
    tfidf, lexical = build_feature_extractor()
    X_train, scaler = make_matrix(tfidf, lexical, X_train_urls, fit=True)
    X_test, _ = make_matrix(tfidf, lexical, X_test_urls, scaler=scaler, fit=False)
    print(f"Feature matrix shape -> train: {X_train.shape}, test: {X_test.shape}")

    candidates = {
        "Logistic Regression (baseline / primary model)":
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes":
            MultinomialNB(),
        "Linear SVM":
            CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=3),
        "Random Forest":
            RandomForestClassifier(n_estimators=150, max_depth=None,
                                    n_jobs=-1, random_state=RANDOM_STATE),
    }

    results = []
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        dt = time.time() - t0
        res = evaluate(name, clf, X_test, y_test)
        res["train_seconds"] = dt
        print(f"(trained in {dt:.1f}s)")
        results.append(res)

    # ---- best-by-F1 across all candidates, shown for information only ----
    best_by_f1 = max(results, key=lambda r: r["f1"])

    # ---- the model actually deployed (see DEPLOY_MODEL_NAME above) ----
    deployed = next((r for r in results if r["name"] == DEPLOY_MODEL_NAME), best_by_f1)

    print("\nFull classification report for the deployed model:")
    print(classification_report(
        y_test, deployed["model"].predict(X_test),
        target_names=["legitimate", "phishing"],
    ))

    # ---- comparison table (matches Table 6.1 in the project report) ----
    print(f"\n{'='*70}")
    print("Comparison across all models  (Table 6.1 in the report)")
    print(f"{'='*70}")
    print(f"{'Model':45s}{'Accuracy':>10s}{'Precision':>11s}{'Recall':>9s}{'F1':>8s}")
    for r in results:
        tag = ""
        if r["name"] == deployed["name"]:
            tag += "  <-- DEPLOYED"
        if r["name"] == best_by_f1["name"] and best_by_f1["name"] != deployed["name"]:
            tag += "  <-- highest F1 (not deployed by default)"
        print(f"{r['name']:45s}{r['accuracy']*100:9.2f}%{r['precision']*100:10.2f}%"
              f"{r['recall']*100:8.2f}%{r['f1']*100:7.2f}%{tag}")

    if deployed["name"] != best_by_f1["name"]:
        print(f"\nNote: '{best_by_f1['name']}' scored slightly higher on F1, but "
              f"'{deployed['name']}' is deployed by default because it is the "
              f"primary model documented in the project report, trains/loads "
              f"almost instantly, and produces a much smaller model file. "
              f"Change DEPLOY_MODEL_NAME at the top of this script to deploy "
              f"a different model instead.")

    # ---- persist the feature extractor and the deployed classifier ----
    joblib.dump({"tfidf": tfidf, "lexical": lexical, "scaler": scaler}, VECTORIZER_PATH)
    joblib.dump(deployed["model"], MODEL_PATH)
    print(f"\nSaved feature extractor -> {VECTORIZER_PATH}")
    print(f"Saved trained model     -> {MODEL_PATH}  ({deployed['name']})")
    print("\nDone. You can now run:  python app.py")


if __name__ == "__main__":
    main()
