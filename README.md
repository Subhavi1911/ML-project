# AI-Powered Real-Time Phishing Detection System

A machine-learning web app that classifies a URL as **Legitimate** or **Phishing**
in real time. Built with TF-IDF + Logistic Regression (with Naive Bayes,
Linear SVM and Random Forest trained alongside it for comparison), and served
through a Flask backend with a simple HTML/CSS interface.

**This copy already includes a trained model** in `models/` (Logistic
Regression, ~91% test accuracy — see below), so you can run `python app.py`
right away. Re-run `train.py` any time you change the dataset or code.

## Project structure

```
Phishing-Detection/
├── dataset/
│   └── phishing.csv        # labelled URL dataset (url, label)
├── models/                 # created after you run train.py
│   ├── model.pkl
│   └── vectorizer.pkl
├── templates/
│   └── index.html          # web UI
├── static/
│   └── style.css           # styling
├── features.py              # shared feature-extraction code
├── train.py                 # trains and saves the model
├── predict.py                # loads the model and classifies a URL
├── app.py                    # Flask web application
└── requirements.txt
```

## Setup (VS Code / any machine)

1. **Open this folder in VS Code.**

2. **Create a virtual environment** (recommended) and activate it:

   Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS / Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Train the model** (this reads `dataset/phishing.csv`, trains and
   compares 4 classifiers, and saves the best one to `models/`):
   ```
   python train.py
   ```
   You'll see accuracy/precision/recall/F1 for every model and a final
   comparison table — copy these numbers straight into your project report
   (Table 6.1).

5. **Run the web app:**
   ```
   python app.py
   ```
   Then open **http://127.0.0.1:5000** in your browser.

6. **(Optional) Command-line check without the browser:**
   ```
   python predict.py https://example.com/login
   ```

## About the dataset

`dataset/phishing.csv` (~40,300 rows) is a balanced sample of 20,000
phishing + 20,000 legitimate URLs drawn from a public malicious-URL
dataset, plus ~300 rows of well-known real domains (google.com, github.com,
wikipedia.org, ...) added because the original source happened to contain
*zero* legitimate "bare domain" URLs (e.g. `google.com` with no path) even
though such URLs are completely normal in real browsing — without this
addition the model would flag almost any homepage-only link as suspicious.
Feel free to replace/extend it with your own data — just keep the two
columns `url` and `label` (`phishing` / `legitimate`), and re-run `train.py`.

## Measured results (this run)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Logistic Regression (deployed)** | 89.97% | 93.58% | 85.67% | 89.45% |
| Multinomial Naive Bayes | 89.63% | 93.39% | 85.12% | 89.06% |
| Linear SVM | 90.53% | 94.16% | 86.27% | 90.04% |
| Random Forest | 93.17% | 94.50% | 91.57% | 93.01% |

These are real numbers from this exact dataset/code — safe to drop straight
into your report's Table 6.1. Random Forest scores highest but is not
deployed by default (large file, slower); see `DEPLOY_MODEL_NAME` in
`train.py` if you want to switch.

## Notes

- `features.py` combines TF-IDF (word 1–2 grams) with hand-crafted lexical
  features (URL length, digit/hyphen counts, presence of an IP address or
  `@` symbol, suspicious keywords, etc.) — this is the "additional lexical
  features" refinement described in the project report's Iteration 2.
- `train.py` trains Logistic Regression, Multinomial Naive Bayes, a
  (calibrated) Linear SVM, and Random Forest, then automatically keeps
  whichever scores highest on F1-score.
- This is a machine-learning prediction, not a guarantee — treat it as one
  signal, not a replacement for user caution.
