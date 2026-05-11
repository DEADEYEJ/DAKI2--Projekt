import re
import math
import joblib
import numpy as np
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

# Regler for mønstre der er entydige nok til at overstyre SVM - BLIVER IKKE BENYTTET LÆNGERE
_RULES = [
    # Koordinater: [tal, tal] – bracket + decimaltal + komma
    (re.compile(r'^\[\s*-?\d+\.\d+\s*,\s*-?\d+\.?\d*\s*\]$'), 'COORDINATES'),
    # E-mail: indeholder @ og domæne
    (re.compile(r'^[\w.%+\-]+@[\w\-]+\.[a-zA-Z]{2,}$'), 'EMAIL'),
    # IBAN: 2 bogstaver + 2 cifre + 11-28 alfanumeriske tegn (uden mellemrum)
    (re.compile(r'^[A-Z]{2}\d{2}[A-Z0-9]{11,28}$'), 'IBAN'),
    # SWIFT/BIC: 6 bogstaver + 2 bogstaver/cifre (+ valgfrie 3)
    (re.compile(r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$'), 'SWIFT_CODE'),
]

def clean_text(text: str) -> str:
    """Standardiserer tekst uden eksterne biblioteker."""
    if not isinstance(text, str):
        return ""
    
    # 1. Fjern HTML-tags vha. regex
    # Matcher alt mellem < og >
    text = re.sub(r'<[^>]*>', '', text)
    
    # 2. Håndter specialtegn
    # Vi fjerner ikke alle specialtegn (da @ og + er vigtige for PII),
    # men vi fjerner kontroltegn som \n, \t osv.
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # 3. Standardiser whitespace
    # Erstatter flere mellemrum med ét enkelt
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

class PIIClassifier:
    def __init__(self, clf, tfidf, le, rules):
        self.clf = clf
        self.tfidf = tfidf
        self.le = le
        self.rules = rules

    def predict(self, texts: str | list[str]) -> list[dict]:
        if isinstance(texts, str):
            texts = [texts]

        # 1. Rens teksten
        cleaned_texts = [clean_text(t) for t in texts]

        # 2. TF-IDF features
        X_tfidf = self.tfidf.transform(cleaned_texts)

        # 3. Hand-crafted features
        X_hand = self._build_hand_features(cleaned_texts)
        X_hand_sparse = csr_matrix(X_hand)

        # 4. Combine features
        X = hstack([X_tfidf, X_hand_sparse])

        # 5. Predict probabilities or derive confidence from decision scores
        proba = self._predict_scores(X)

        results = []
        for text, p in zip(texts, proba):
            rule_label = self._apply_rules(text)
            if rule_label is not None:
                results.append({'label': rule_label, 'confidence': 1.0, 'source': 'rule'})
            else:
                idx = int(np.argmax(p))
                results.append({
                    'label': self.le.classes_[idx],
                    'confidence': round(float(p[idx]), 4),
                    'source': 'svm',
                })
        return results

    def _predict_scores(self, X) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)

        scores = self.clf.decision_function(X)
        scores = np.asarray(scores)

        if scores.ndim == 1:
            scores = np.clip(scores, -50.0, 50.0)
            positive = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - positive, positive])

        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def _apply_rules(self, text: str) -> str | None:
        """Returnerer label hvis en regel matcher, ellers None."""
        s = text.strip().replace(' ', '')
        for pattern, label in self.rules:
            if pattern.match(s) or pattern.match(text.strip()):
                return label
        return None

    def _build_hand_features(self, texts: list[str]) -> np.ndarray:
        """
        Håndlavede features der fanger typiske mønstre for hver PII-kategori.
        Returnerer float-matrix af shape (n_samples, n_features).
        """
        rows = []
        for t in texts:
            s = str(t)
            n = max(len(s), 1)
            digit_ratio  = sum(c.isdigit() for c in s) / n
            alpha_ratio  = sum(c.isalpha() for c in s) / n
            space_ratio  = s.count(' ')                 / n
            upper_ratio  = sum(c.isupper() for c in s)  / n
            hex_chars    = sum(c in '0123456789abcdefABCDEF' for c in s) / n
            special_cnt  = sum(not c.isalnum() and not c.isspace() for c in s)

            rows.append([
                # --- generelle talkarakteristika ---
                len(s),                                                    # længde
                digit_ratio,                                               # andel cifre
                alpha_ratio,                                               # andel bogstaver
                space_ratio,                                               # andel mellemrum
                upper_ratio,                                               # andel store bogstaver
                special_cnt,                                               # antal specialtegn
                hex_chars,                                                 # andel hex-tegn (API-nøgler)
                self._shannon_entropy(s),                                       # entropi (adgangskoder)

                # --- strukturelle indikatorer ---
                float('@' in s),                                           # e-mail
                float(s.startswith('+')),                                  # telefon med landekode
                float(s.startswith('[') and s.endswith(']')),             # koordinater

                # --- mønster-matches ---
                float(bool(re.match(r'^[A-Z]{2}\d{2}', s))),             # IBAN-start
                float(bool(re.match(r'^[A-Z]{6}[A-Z0-9]{2}', s))),       # SWIFT (8-11 tegn)
                float(bool(re.match(r'^\d{9}$', s.replace(' ', '')))),    # routing-number (9 cifre)
                float(bool(re.match(r'^\d{4,6}$', s))),                   # PIN (4-6 cifre)
                float(bool(re.match(r'^\d{13,19}$', s.replace(' ', '')))), # kreditkort
                float(bool(re.match(r'^[\w.+-]+@[\w-]+\.[a-z]{2,}$', s, re.I))),  # e-mail-struktur

                # --- separatorer ---
                float('-' in s),
                float('.' in s),
                float('/' in s),
                s.count('-'),
                s.count('.'),
                s.count(' '),
            ])
        return np.array(rows, dtype=float)

    def _shannon_entropy(self, text: str) -> float:
        """Beregner Shannon-entropi for en streng."""
        if not text:
            return 0.0
        counts = Counter(text)
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())


class HybridPIIClassifier:
    def __init__(self, binary_clf, target_clf, tfidf, binary_label_encoder, target_label_encoder):
        self.binary_clf = binary_clf
        self.target_clf = target_clf
        self.tfidf = tfidf
        self.binary_label_encoder = binary_label_encoder
        self.target_label_encoder = target_label_encoder
        self.classes_ = np.array(["NON_PII", *list(self.target_label_encoder.classes_)], dtype=object)

    def predict_proba(self, texts: str | list[str]) -> np.ndarray:
        X = self._transform(texts)
        binary_proba = self._predict_scores(self.binary_clf, X)
        raw_target_proba = self._predict_scores(self.target_clf, X)

        binary_class_to_idx = {label: idx for idx, label in enumerate(self.binary_label_encoder.classes_)}
        non_pii_idx = binary_class_to_idx["NON_PII"]
        pii_idx = binary_class_to_idx["PII"]

        target_proba = np.zeros((X.shape[0], len(self.target_label_encoder.classes_)), dtype=float)
        observed_target_indices = np.asarray(self.target_clf.classes_, dtype=int)
        target_proba[:, observed_target_indices] = raw_target_proba

        final_proba = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        final_proba[:, 0] = binary_proba[:, non_pii_idx]
        final_proba[:, 1:] = binary_proba[:, [pii_idx]] * target_proba
        return final_proba

    def predict(self, texts: str | list[str]) -> list[dict]:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        final_proba = self.predict_proba(texts)
        results = []
        for text, probs in zip(texts, final_proba):
            idx = int(np.argmax(probs))
            label = str(self.classes_[idx])
            results.append({
                "label": label,
                "confidence": round(float(probs[idx]), 4),
                "probabilities": {
                    str(class_label): round(float(prob), 6)
                    for class_label, prob in zip(self.classes_, probs)
                },
                "source": "hybrid_svm",
            })
        return results

    def _transform(self, texts: str | list[str]):
        if isinstance(texts, str):
            texts = [texts]
        cleaned_texts = [clean_text(text) for text in texts]
        X_tfidf = self.tfidf.transform(cleaned_texts)
        X_hand = csr_matrix(self._build_hand_features(cleaned_texts))
        return hstack([X_tfidf, X_hand])

    def _predict_scores(self, clf, X) -> np.ndarray:
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)

        scores = clf.decision_function(X)
        scores = np.asarray(scores)

        if scores.ndim == 1:
            scores = np.clip(scores, -50.0, 50.0)
            positive = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - positive, positive])

        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def _build_hand_features(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            s = str(t)
            n = max(len(s), 1)
            digit_ratio = sum(c.isdigit() for c in s) / n
            alpha_ratio = sum(c.isalpha() for c in s) / n
            space_ratio = s.count(' ') / n
            upper_ratio = sum(c.isupper() for c in s) / n
            hex_chars = sum(c in '0123456789abcdefABCDEF' for c in s) / n
            special_cnt = sum(not c.isalnum() and not c.isspace() for c in s)

            rows.append([
                len(s),
                digit_ratio,
                alpha_ratio,
                space_ratio,
                upper_ratio,
                special_cnt,
                hex_chars,
                self._shannon_entropy(s),
                float('@' in s),
                float(s.startswith('+')),
                float(s.startswith('[') and s.endswith(']')),
                float(bool(re.match(r'^[A-Z]{2}\d{2}', s))),
                float(bool(re.match(r'^[A-Z]{6}[A-Z0-9]{2}', s))),
                float(bool(re.match(r'^\d{9}$', s.replace(' ', '')))),
                float(bool(re.match(r'^\d{4,6}$', s))),
                float(bool(re.match(r'^\d{13,19}$', s.replace(' ', '')))),
                float(bool(re.match(r'^[\w.+-]+@[\w-]+\.[a-z]{2,}$', s, re.I))),
                float('-' in s),
                float('.' in s),
                float('/' in s),
                s.count('-'),
                s.count('.'),
                s.count(' '),
            ])
        return np.array(rows, dtype=float)

    def _shannon_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())
