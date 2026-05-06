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
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import re
from pii_classifier import PIIClassifier, clean_text, _RULES
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
MODEL_PATH = 'svm_pii_classifier.pkl'
C_RANGE    = [0.01, 0.1, 1.0, 10.0, 100.0]
TEST_SIZE  = 0.20
RANDOM_STATE = 42

# ─── Parsing ─────────────────────────────────────────────────────────────────

# Matcher {'label': LABEL, 'value': 'quoted'} eller {'label': LABEL, 'value': unquoted}
_PAIR_RE = re.compile(
    r"\{'label': ([A-Z_]+), 'value': (?:'((?:[^'\\]|\\.)*)'|([^}]+?))\s*\}"
)


def parse_and_align(df: pd.DataFrame):
    """
    Parser annotationer og sikrer alignment.
    Håndterer fejlformaterede data og manglende labels.
    """
    cleaned_values = []
    cleaned_labels = []
    
    for idx, row in df.iterrows():
        try:
            # Brug din eksisterende _PAIR_RE til at udtrække pairs
            pairs = extract_pairs(row['privacy'])
            
            if not pairs:
                continue # Håndter manglende annotationer
                
            for label, value in pairs:
                # 1. Rens teksten
                clean_val = clean_text(value)
                
                # 2. Alignment tjek: Er værdien tom efter rensning?
                if clean_val:
                    cleaned_values.append(clean_val)
                    cleaned_labels.append(label)
                    
        except Exception as e:
            print(f"Log: Fejl i række {idx}: {e}")
            continue # Spring over fejlbehæftede rækker (Annotation errors)
            
    return cleaned_values, cleaned_labels

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
    print(f"Indlæser og renser datasæt fra {csv_path} …")
    df = pd.read_csv(csv_path)
    values, labels = [], []
    for privacy_str in df['privacy'].dropna():
        for label, value in extract_pairs(privacy_str):
            # Her kører vi rensningen med det samme
            cleaned_val = clean_text(value)
            if cleaned_val:
                values.append(cleaned_val)
                labels.append(label)
    return values, labels


def train(csv_path: str = CSV_PATH, model_path: str = MODEL_PATH):
    # --- 1. Indlæs data ---
    values, labels = load_data(csv_path)
    
    # --- 2. Label Encoding ---
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # --- 3. Split (Data Leakage beskyttelse) ---
    # Vi splitter de RÅ tekster før TF-IDF
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- 4. Features (Korrekt fit/transform rækkefølge) ---
    print("\nBygger features ...")
    tfidf = TfidfVectorizer(
        analyzer='char_wb', 
        ngram_range=(2, 4), 
        max_features=80_000, 
        sublinear_tf=True
    )

    # Lær kun fra træningsdata
    X_tr_tfidf = tfidf.fit_transform(X_train_raw)
    X_tr_hand = build_hand_features(X_train_raw)
    X_tr_hand_sparse = csr_matrix(X_tr_hand)
    X_tr = hstack([X_tr_tfidf, X_tr_hand_sparse])

    # Transformér testdata
    X_te_tfidf = tfidf.transform(X_test_raw)
    X_te_hand = build_hand_features(X_test_raw)
    X_te_hand_sparse = csr_matrix(X_te_hand)
    X_te = hstack([X_te_tfidf, X_te_hand_sparse])

    # --- 5. Find bedste C via Grid Search ---
    print(f"Tuner C-værdien på LinearSVC for {len(le.classes_)} klasser...")
    base_clf = LinearSVC(max_iter=3000, class_weight='balanced', dual='auto')
    grid = GridSearchCV(
        base_clf,
        param_grid={'C': C_RANGE},
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_tr, y_train)
    best_C = grid.best_params_['C']
    print(f"Bedste C fundet: {best_C}")

    # Kalibrér den bedste SVM for at få sandsynligheder
    best_clf = CalibratedClassifierCV(grid.best_estimator_, cv=3)
    best_clf.fit(X_tr, y_train)
    clf = best_clf

    # --- 6. Evaluering (Macro-F1 og Fejllogs) ---
    print("\n" + "="*60)
    print("EVALUERINGSRAPPORT (Macro-F1)")
    print("="*60)
    y_pred = clf.predict(X_te)
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

    # Logning af potentielle annotation errors
    print("\n--- Tjek for inkonsistente labels (Modellens største fejl) ---")
    probs = clf.predict_proba(X_te)
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            confidence = np.max(probs[i])
            if confidence > 0.85: # Flag hvis modellen er meget sikker, men uenig
                print(f"FLAG: '{X_test_raw[i]}' er mærket {le.classes_[y_test[i]]}, men gættet er {le.classes_[y_pred[i]]} ({confidence:.2%})")

    # --- 7. Gem model ---
    classifier = PIIClassifier(clf, tfidf, le, _RULES)
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"\nModel gemt → {model_path}")
    return classifier


# ─── Forudsigelse ─────────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH) -> PIIClassifier:
    with open(model_path, 'rb') as f:
        return pickle.load(f)


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo(classifier: PIIClassifier | None = None):
    if classifier is None:
        classifier = load_model()

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
    preds = classifier.predict(texts)
    for (value, expected), res in zip(examples, preds):
        match = "✓" if res['label'] == expected else "✗"
        print(f"{match} {value:<35} {expected:<25} {res['label']:<25} {res['confidence']:.2%}")


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    classifier = train()
    demo(classifier)
