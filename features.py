"""
features.py
-----------
Shared feature-extraction code used by BOTH train.py and predict.py.

Keeping this in one shared module guarantees that a URL is turned into
numbers in EXACTLY the same way during training and during a live
prediction -- if the two ever drifted apart the model would be reading a
different kind of input than it was trained on.

Two kinds of features are combined (this is the "Iteration 2 -- additional
lexical features" refinement described in the project report):

1. TF-IDF over the URL text itself (word-level 1- and 2-grams).
2. A handful of hand-crafted lexical features that are known to correlate
   with phishing URLs (length, count of special characters, presence of
   an IP address instead of a domain name, suspicious keywords, etc).
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Keywords that phishing URLs commonly contain (imitating a login/verification
# step). This list is intentionally short and easy to extend.
SUSPICIOUS_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "update", "secure", "security", "account", "banking", "confirm",
    "password", "webscr", "ebayisapi", "paypal", "wp-admin", "admin",
]

IP_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


_SCHEME_PATTERN = re.compile(r"^\w+://")
_WWW_PATTERN = re.compile(r"^www\.")


def clean_url(url: str) -> str:
    """
    Light, consistent normalisation applied before feature extraction.

    The training dataset stores URLs WITHOUT a leading scheme and almost
    never with a "www." prefix (e.g. "example.com/login", not
    "https://www.example.com/login"). A user typing or pasting a URL into
    the web form will usually include both, so we strip them here to keep
    prediction-time input in the same format the model was trained on --
    "https://www.google.com" and "google.com" should be read identically.
    This function is used identically by train.py and predict.py.
    """
    url = str(url).strip().lower()
    url = _SCHEME_PATTERN.sub("", url)
    url = _WWW_PATTERN.sub("", url)
    return url


class LexicalFeatures(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-compatible transformer that converts a list of raw URL
    strings into a small dense matrix of hand-crafted numeric features.

    Being a proper BaseEstimator/TransformerMixin subclass (rather than a
    plain function) means it can live inside a Pipeline/FeatureUnion and be
    pickled and reloaded by predict.py without any extra glue code.
    """

    def fit(self, X, y=None):
        return self  # nothing to learn -- these features are rule-based

    def transform(self, X):
        rows = []
        for raw in X:
            u = clean_url(raw)
            length = len(u)
            n_dots = u.count(".")
            n_hyphens = u.count("-")
            n_digits = sum(c.isdigit() for c in u)
            n_special = sum(not c.isalnum() and c not in ".-/_" for c in u)
            has_at = int("@" in u)
            has_ip = int(bool(IP_PATTERN.search(u)))
            has_https = int(u.startswith("https"))
            n_subdirs = u.count("/")
            n_suspicious = sum(w in u for w in SUSPICIOUS_WORDS)
            rows.append([
                length, n_dots, n_hyphens, n_digits, n_special,
                has_at, has_ip, has_https, n_subdirs, n_suspicious,
            ])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "url_length", "num_dots", "num_hyphens", "num_digits",
            "num_special_chars", "has_at_symbol", "has_ip_address",
            "has_https", "num_subdirs", "num_suspicious_words",
        ])
