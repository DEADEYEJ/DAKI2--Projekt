from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack


_RULES = [
    (re.compile(r"^\[\s*-?\d+\.\d+\s*,\s*-?\d+\.?\d*\s*\]$"), "COORDINATES"),
    (re.compile(r"^[\w.%+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"), "EMAIL"),
    (re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,28}$"), "IBAN"),
    (re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$"), "IPV4"),
    (re.compile(r"^(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}$"), "IPV6"),
    (re.compile(r"^\d{3}-\d{2}-\d{4}$"), "SSN"),
    (re.compile(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$"), "SWIFT_CODE"),
]


def clean_text(text: str) -> str:
    """Standardiser tekst uden eksterne biblioteker."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<[^>]*>", "", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def build_hand_features(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        s = str(text)
        n = max(len(s), 1)
        digit_ratio = sum(char.isdigit() for char in s) / n
        alpha_ratio = sum(char.isalpha() for char in s) / n
        space_ratio = s.count(" ") / n
        upper_ratio = sum(char.isupper() for char in s) / n
        hex_chars = sum(char in "0123456789abcdefABCDEF" for char in s) / n
        special_cnt = sum(not char.isalnum() and not char.isspace() for char in s)

        rows.append([
            len(s),
            digit_ratio,
            alpha_ratio,
            space_ratio,
            upper_ratio,
            special_cnt,
            hex_chars,
            shannon_entropy(s),
            float("@" in s),
            float(s.startswith("+")),
            float(s.startswith("[") and s.endswith("]")),
            float(bool(re.match(r"^[A-Z]{2}\d{2}", s))),
            float(bool(re.match(r"^[A-Z]{6}[A-Z0-9]{2}", s))),
            float(bool(re.match(r"^\d{9}$", s.replace(" ", "")))),
            float(bool(re.match(r"^\d{4,6}$", s))),
            float(bool(re.match(r"^\d{13,19}$", s.replace(" ", "")))),
            float(bool(re.match(r"^[\w.+-]+@[\w-]+\.[a-z]{2,}$", s, re.I))),
            float(bool(re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", s))),
            float(bool(re.match(r"^(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}$", s))),
            float(bool(re.match(r"^\d{3}-\d{2}-\d{4}$", s))),
            float(":" in s),
            float("-" in s),
            float("." in s),
            float("/" in s),
            s.count(":"),
            s.count("-"),
            s.count("."),
            s.count(" "),
        ])
    return np.array(rows, dtype=float)


class PIIClassifier:
    def __init__(self, clf, tfidf, le, rules):
        self.clf = clf
        self.tfidf = tfidf
        self.le = le
        self.rules = rules

    def predict(self, texts: str | list[str]) -> list[dict]:
        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = [clean_text(text) for text in texts]
        X_tfidf = self.tfidf.transform(cleaned_texts)
        X_hand = csr_matrix(build_hand_features(cleaned_texts))
        X = hstack([X_tfidf, X_hand])
        probabilities = self.clf.predict_proba(X)

        results = []
        for text, proba in zip(texts, probabilities):
            rule_label = self._apply_rules(text)
            if rule_label is not None:
                results.append({"label": rule_label, "confidence": 1.0, "source": "rule"})
                continue

            idx = int(np.argmax(proba))
            results.append({
                "label": str(self.le.classes_[idx]),
                "confidence": round(float(proba[idx]), 4),
                "source": "svm",
            })
        return results

    def _apply_rules(self, text: str) -> str | None:
        stripped = text.strip()
        compact = stripped.replace(" ", "")
        for pattern, label in self.rules:
            if pattern.match(compact) or pattern.match(stripped):
                return label
        return None
