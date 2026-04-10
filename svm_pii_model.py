"""
SVM-baseret PII-klassifikator
-------------------------------
Klassificerer individuelle værdier i 20 kategorier af følsomme data:
  ACCOUNT_NUMBER, API_KEY, BANK_ACCOUNT_NUMBER, CREDIT_CARD_CVV,
  CREDIT_CARD_NUMBER, CUSTOMER_ID, DRIVER_LICENSE_NUMBER, EMPLOYEE_ID,
  IBAN, ID_CARD_NUMBER, PASSPORT_NUMBER, PASSWORD, PIN_NUMBER,
  ROUTING_NUMBER, SWIFT_CODE, TAX_NUMBER, EMAIL, PHONE_NUMBER,
  STREET_ADDRESS, COORDINATES
"""

import re
import math
import joblib
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# ─── Konfiguration ───────────────────────────────────────────────────────────

TARGET_LABELS = {
    'ACCOUNT_NUMBER', 'API_KEY', 'BANK_ACCOUNT_NUMBER', 'CREDIT_CARD_CVV',
    'CREDIT_CARD_NUMBER', 'CUSTOMER_ID', 'DRIVER_LICENSE_NUMBER', 'EMPLOYEE_ID',
    'IBAN', 'ID_CARD_NUMBER', 'PASSPORT_NUMBER', 'PASSWORD', 'PIN_NUMBER',
    'ROUTING_NUMBER', 'SWIFT_CODE', 'TAX_NUMBER', 'EMAIL', 'PHONE_NUMBER',
    'STREET_ADDRESS', 'COORDINATES',
}

# STREET i datasættet svarer til STREET_ADDRESS i målkategorierne
LABEL_MAP = {
    'STREET': 'STREET_ADDRESS',
}

CSV_PATH   = 'train-00000-of-00001.csv'
MODEL_PATH = 'svm_pii_classifier.joblib'
TEST_SIZE  = 0.20
RANDOM_STATE = 42

# ─── Parsing ─────────────────────────────────────────────────────────────────

# Matcher {'label': LABEL, 'value': 'quoted'} eller {'label': LABEL, 'value': unquoted}
_PAIR_RE = re.compile(
    r"\{'label': ([A-Z_]+), 'value': (?:'((?:[^'\\]|\\.)*)'|([^}]+?))\s*\}"
)


def extract_pairs(privacy_str: str) -> list[tuple[str, str]]:
    """Returnerer (label, value)-par fra én privacy-streng."""
    pairs = []
    for m in _PAIR_RE.finditer(str(privacy_str)):
        label = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)
        if value:
            value = value.strip()
            label = LABEL_MAP.get(label, label)
            if label in TARGET_LABELS:
                pairs.append((label, value))
    return pairs


# ─── Feature engineering ─────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    """Beregner Shannon-entropi for en streng."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def build_hand_features(texts: list[str]) -> np.ndarray:
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
            _shannon_entropy(s),                                       # entropi (adgangskoder)

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


# ─── Træning ─────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple[list[str], list[str]]:
    print(f"Indlæser datasæt fra {csv_path} …")
    df = pd.read_csv(csv_path)
    values, labels = [], []
    for privacy_str in df['privacy'].dropna():
        for label, value in extract_pairs(privacy_str):
            values.append(value)
            labels.append(label)
    return values, labels


def train(csv_path: str = CSV_PATH, model_path: str = MODEL_PATH):
    # --- 1. Indlæs data ---
    values, labels = load_data(csv_path)
    print(f"Samlede samples: {len(values):,}\n")

    dist = Counter(labels)
    print(f"{'Klasse':<30} {'Antal':>8}")
    print("-" * 40)
    for lbl, cnt in sorted(dist.items()):
        print(f"  {lbl:<28} {cnt:>8,}")

    # --- 2. Fjern klasser med for få eksempler ---
    min_samples = 5
    valid = {l for l, c in dist.items() if c >= min_samples}
    if len(valid) < len(dist):
        removed = set(dist) - valid
        print(f"\nFjerner klasser med < {min_samples} eksempler: {removed}")
        pairs   = [(v, l) for v, l in zip(values, labels) if l in valid]
        values, labels = map(list, zip(*pairs))

    # --- 3. Kodning ---
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"\nKlasser i modellen ({len(le.classes_)}):", list(le.classes_))

    # --- 4. Split ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTræning: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # --- 5. Features ---
    print("\nBygger features …")
    tfidf = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=80_000, sublinear_tf=True,
    )
    X_tr_tfidf = tfidf.fit_transform(X_tr)
    X_te_tfidf = tfidf.transform(X_te)

    X_tr_hand = csr_matrix(build_hand_features(X_tr))
    X_te_hand = csr_matrix(build_hand_features(X_te))

    X_tr_feat = hstack([X_tr_tfidf, X_tr_hand])
    X_te_feat = hstack([X_te_tfidf, X_te_hand])

    # --- 6. Træn SVM ---
    print("Træner LinearSVC …")
    base_clf = LinearSVC(C=1.0, max_iter=3000, class_weight='balanced')
    # CalibratedClassifierCV giver sandsynligheder (predict_proba)
    clf = CalibratedClassifierCV(base_clf, cv=3)
    clf.fit(X_tr_feat, y_tr)

    # --- 7. Evaluer ---
    print("\n" + "=" * 60)
    print("EVALUERINGSRAPPORT (test-sæt)")
    print("=" * 60)
    y_pred = clf.predict(X_te_feat)
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    # --- 8. Gem model ---
    bundle = {'clf': clf, 'tfidf': tfidf, 'label_encoder': le}
    joblib.dump(bundle, model_path)
    print(f"Model gemt → {model_path}")
    return bundle


# ─── Forudsigelse ─────────────────────────────────────────────────────────────

# Regler for mønstre der er entydige nok til at overstyre SVM
_RULES: list[tuple[re.Pattern, str]] = [
    # Koordinater: [tal, tal] – bracket + decimaltal + komma
    (re.compile(r'^\[\s*-?\d+\.\d+\s*,\s*-?\d+\.?\d*\s*\]$'), 'COORDINATES'),
    # E-mail: indeholder @ og domæne
    (re.compile(r'^[\w.%+\-]+@[\w\-]+\.[a-zA-Z]{2,}$'), 'EMAIL'),
    # IBAN: 2 bogstaver + 2 cifre + 11-28 alfanumeriske tegn (uden mellemrum)
    (re.compile(r'^[A-Z]{2}\d{2}[A-Z0-9]{11,28}$'), 'IBAN'),
    # SWIFT/BIC: 6 bogstaver + 2 bogstaver/cifre (+ valgfrie 3)
    (re.compile(r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$'), 'SWIFT_CODE'),
]


def _apply_rules(text: str) -> str | None:
    """Returnerer label hvis en regel matcher, ellers None."""
    s = text.strip().replace(' ', '')
    for pattern, label in _RULES:
        if pattern.match(s) or pattern.match(text.strip()):
            return label
    return None


def load_model(model_path: str = MODEL_PATH) -> dict:
    return joblib.load(model_path)


def predict(texts: str | list[str], bundle: dict | None = None) -> list[dict]:
    """
    Klassificér én eller flere strenge.
    Regel-baserede mønstre (IBAN, SWIFT, COORDINATES, EMAIL) har forrang.
    Returnerer liste af {'label': ..., 'confidence': ...}.
    """
    if bundle is None:
        bundle = load_model()
    clf, tfidf, le = bundle['clf'], bundle['tfidf'], bundle['label_encoder']

    if isinstance(texts, str):
        texts = [texts]

    X_tfidf = tfidf.transform(texts)
    X_hand  = csr_matrix(build_hand_features(texts))
    X       = hstack([X_tfidf, X_hand])

    proba   = clf.predict_proba(X)
    results = []
    for text, p in zip(texts, proba):
        rule_label = _apply_rules(text)
        if rule_label is not None:
            # Regel-match: sæt klasse til regellabel med konfidens 1.0
            results.append({'label': rule_label, 'confidence': 1.0, 'source': 'rule'})
        else:
            idx = int(np.argmax(p))
            results.append({
                'label': le.classes_[idx],
                'confidence': round(float(p[idx]), 4),
                'source': 'svm',
            })
    return results


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo(bundle: dict | None = None):
    if bundle is None:
        bundle = load_model()

    examples = [
        ("user@example.com",            "EMAIL"),
        ("+45 12 34 56 78",             "PHONE_NUMBER"),
        ("4532015112830366",             "CREDIT_CARD_NUMBER"),
        ("myS3cur3P@ssw0rd!",           "PASSWORD"),
        ("[40.7128, -74.0060]",         "COORDINATES"),
        ("GB29NWBK60161331926819",      "IBAN"),
        ("DEUTDEDB",                    "SWIFT_CODE"),
        ("123 Main Street",             "STREET_ADDRESS"),
        ("123456789",                   "ROUTING_NUMBER"),
        ("1234",                        "PIN_NUMBER"),
        ("sk-abc123DEF456ghi789jkl0",  "API_KEY"),
        ("A1234567",                    "PASSPORT_NUMBER"),
        ("9706611478335",               "ACCOUNT_NUMBER"),
        ("466-22-6318",                 "TAX_NUMBER"),
        ("343345698",                   "DRIVER_LICENSE_NUMBER"),
    ]

    print("\n" + "=" * 65)
    print("DEMO-FORUDSIGELSER")
    print("=" * 65)
    print(f"  {'Værdi':<35} {'Forventet':<25} {'Forudsagt':<25} {'Konfid.'}")
    print("-" * 100)
    texts = [e[0] for e in examples]
    preds = predict(texts, bundle)
    for (value, expected), res in zip(examples, preds):
        match = "✓" if res['label'] == expected else "✗"
        print(f"{match} {value:<35} {expected:<25} {res['label']:<25} {res['confidence']:.2%}")


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    bundle = train()
    demo(bundle)
