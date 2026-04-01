import ast
import os
import re
import sys
import pickle
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import time
import threading

import pandas as pd
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics


# ============================================================
# KONFIGURATION
# ============================================================

PARQUET_PATH = "0000.parquet"   # ændr til din filsti
MODEL_OUTPUT_PATH = "crf_pii_model.pkl"
RANDOM_STATE = 42

# Hvis du vil teste hurtigere under udvikling, kan du sætte en grænse.
# Sæt til None for at bruge hele datasættet.
MAX_ROWS = 100000

# Hvor mange eksempel-forudsigelser der skal vises til sidst
N_EXAMPLES_TO_SHOW = 5


# ============================================================
# HJÆLPEFUNKTIONER
# ============================================================

def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokeniser tekst og returnér tokens med offset:
    [(token, start_index, end_index), ...]
    """
    pattern = r"\w+|[^\w\s]"
    return [(m.group(), m.start(), m.end()) for m in re.finditer(pattern, text)]


def normalize_privacy_column(value):
    """
    Gør privacy om til en liste af almindelige Python-dicts.
    Håndterer bl.a. numpy.ndarray, list, tuple og str.
    """
    if value is None:
        return []

    # hvis det allerede er en liste
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append(item)
            else:
                try:
                    result.append(dict(item))
                except Exception:
                    pass
        return result

    # hvis det er tuple
    if isinstance(value, tuple):
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append(item)
            else:
                try:
                    result.append(dict(item))
                except Exception:
                    pass
        return result

    # hvis det er numpy array eller anden itererbar struktur
    try:
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append(item)
            else:
                try:
                    result.append(dict(item))
                except Exception:
                    pass
        if result:
            return result
    except Exception:
        pass

    # hvis det er en strengrepræsentation
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict):
                        result.append(item)
                    else:
                        try:
                            result.append(dict(item))
                        except Exception:
                            pass
                return result
        except Exception:
            return []

    return []


def find_all_occurrences(text: str, substring: str) -> List[Tuple[int, int]]:
    """
    Finder alle forekomster af substring i text.
    Returnerer liste af (start, end).
    """
    if not substring:
        return []

    matches = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(substring)))
        start = idx + 1
    return matches


def char_span_overlaps(token_start: int, token_end: int, entity_start: int, entity_end: int) -> bool:
    """
    True hvis token og entity overlapper i tegnpositioner.
    """
    return not (token_end <= entity_start or token_start >= entity_end)


def create_bio_labels_from_privacy(
    text: str,
    privacy_items: List[Dict[str, Any]]
) -> Tuple[List[str], List[Tuple[int, int]], List[str]]:
    """
    Omdan source_text + privacy til:
    - tokens
    - token_spans
    - BIO-labels

    Strategi:
    1. Tokenisér source_text med offsets.
    2. Find hver privacy-værdi i teksten via exact string match.
    3. Marker alle overlappende tokens med B-/I-labels.

    Bemærk:
    - Hvis samme værdi optræder flere gange i teksten, markeres alle matches.
    - Hvis entities overlapper, beholdes den første label der rammer et token.
    """
    token_data = tokenize_with_spans(text)
    tokens = [tok for tok, _, _ in token_data]
    spans = [(start, end) for _, start, end in token_data]
    labels = ["O"] * len(tokens)

    for item in privacy_items:
        if not isinstance(item, dict):
            continue

        entity_label = item.get("label")
        entity_value = item.get("value")

        if entity_label is None or entity_value is None:
            continue

        entity_label = str(entity_label).strip()
        entity_value = str(entity_value)

        if not entity_label or not entity_value.strip():
            continue

        occurrences = find_all_occurrences(text, entity_value)

        for entity_start, entity_end in occurrences:
            overlapping_token_indices = []

            for i, (tok_start, tok_end) in enumerate(spans):
                if char_span_overlaps(tok_start, tok_end, entity_start, entity_end):
                    overlapping_token_indices.append(i)

            if not overlapping_token_indices:
                continue

            first_idx = overlapping_token_indices[0]
            if labels[first_idx] == "O":
                labels[first_idx] = f"B-{entity_label}"

            for idx in overlapping_token_indices[1:]:
                if labels[idx] == "O":
                    labels[idx] = f"I-{entity_label}"

    return tokens, spans, labels


def word_shape(word: str) -> str:
    """
    Omdan et token til et simpelt shape-pattern.
    Eksempler:
    - 'Anna' -> 'Xxxx'
    - '1234' -> 'dddd'
    - 'A-12' -> 'X-dd'
    """
    shape = []
    for ch in word:
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    return "".join(shape)


def word2features(tokens: List[str], i: int) -> Dict[str, Any]:
    """
    Features til CRF for token nr. i.
    """
    word = tokens[i]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.isalpha()": word.isalpha(),
        "word.isalnum()": word.isalnum(),
        "len(word)": len(word),
        "has_digit": any(c.isdigit() for c in word),
        "has_hyphen": "-" in word,
        "has_slash": "/" in word,
        "has_dot": "." in word,
        "has_at": "@" in word,
        "shape": word_shape(word),
    }

    if i > 0:
        prev_word = tokens[i - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
            "-1:word.isdigit()": prev_word.isdigit(),
            "-1:shape": word_shape(prev_word),
        })
    else:
        features["BOS"] = True

    if i > 1:
        prev2_word = tokens[i - 2]
        features.update({
            "-2:word.lower()": prev2_word.lower(),
            "-2:shape": word_shape(prev2_word),
        })

    if i < len(tokens) - 1:
        next_word = tokens[i + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
            "+1:word.isdigit()": next_word.isdigit(),
            "+1:shape": word_shape(next_word),
        })
    else:
        features["EOS"] = True

    if i < len(tokens) - 2:
        next2_word = tokens[i + 2]
        features.update({
            "+2:word.lower()": next2_word.lower(),
            "+2:shape": word_shape(next2_word),
        })

    return features


def sent2features(tokens: List[str]) -> List[Dict[str, Any]]:
    return [word2features(tokens, i) for i in range(len(tokens))]


def labels_to_entities(tokens: List[str], labels: List[str]) -> List[Dict[str, str]]:
    """
    Gør BIO-labels om til entiteter.
    Returnerer fx:
    [{"label": "FIRST_NAME", "value": "Anna Louise"}, ...]
    """
    entities = []
    current_label = None
    current_tokens = []

    for token, label in zip(tokens, labels):
        if label == "O":
            if current_tokens:
                entities.append({
                    "label": current_label,
                    "value": " ".join(current_tokens)
                })
                current_label = None
                current_tokens = []
            continue

        if label.startswith("B-"):
            if current_tokens:
                entities.append({
                    "label": current_label,
                    "value": " ".join(current_tokens)
                })
            current_label = label[2:]
            current_tokens = [token]

        elif label.startswith("I-"):
            label_type = label[2:]
            if current_label == label_type:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append({
                        "label": current_label,
                        "value": " ".join(current_tokens)
                    })
                current_label = label_type
                current_tokens = [token]

    if current_tokens:
        entities.append({
            "label": current_label,
            "value": " ".join(current_tokens)
        })

    return entities


def normalize_entity_value(text: str) -> str:
    """
    Simpel normalisering til entity-sammenligning.
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def entity_set_from_privacy_items(privacy_items: List[Dict[str, Any]]) -> set:
    """
    Gør ground-truth privacy-liste om til et set af:
    (label, normaliseret værdi)
    """
    result = set()
    for item in privacy_items:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        value = item.get("value")
        if label is None or value is None:
            continue
        result.add((str(label).strip(), normalize_entity_value(str(value))))
    return result


def entity_set_from_predicted_entities(predicted_entities: List[Dict[str, str]]) -> set:
    """
    Gør modellens entities om til et set af:
    (label, normaliseret værdi)
    """
    result = set()
    for item in predicted_entities:
        label = item.get("label")
        value = item.get("value")
        if label is None or value is None:
            continue
        result.add((str(label).strip(), normalize_entity_value(str(value))))
    return result


def evaluate_entity_level(
    df_subset: pd.DataFrame,
    tokens_list: List[List[str]],
    pred_labels_list: List[List[str]]
) -> Dict[str, float]:
    """
    Entity-level precision / recall / F1.
    Ground truth tages direkte fra privacy-kolonnen.
    Prediction tages fra modellens BIO-labels.
    """
    tp = 0
    fp = 0
    fn = 0

    for (_, row), tokens, pred_labels in zip(df_subset.iterrows(), tokens_list, pred_labels_list):
        true_set = entity_set_from_privacy_items(row["privacy"])
        pred_entities = labels_to_entities(tokens, pred_labels)
        pred_set = entity_set_from_predicted_entities(pred_entities)

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_crf_dataset(dataframe: pd.DataFrame) -> Tuple[List[List[Dict[str, Any]]], List[List[str]], List[List[str]]]:
    """
    Byg X, y og rå tokens til CRF.
    """
    X = []
    y = []
    tokens_all = []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Building dataset"):
        text = row["source_text"]
        privacy_items = row["privacy"]

        tokens, _, labels = create_bio_labels_from_privacy(text, privacy_items)

        if not tokens:
            continue

        X.append(sent2features(tokens))
        y.append(labels)
        tokens_all.append(tokens)

    return X, y, tokens_all


def print_split_sizes(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    print("\nDatasplit:")
    print(f"  Train: {len(train_df)} ({len(train_df)/total:.2%})")
    print(f"  Val:   {len(val_df)} ({len(val_df)/total:.2%})")
    print(f"  Test:  {len(test_df)} ({len(test_df)/total:.2%})")


def print_example_predictions(
    df_subset: pd.DataFrame,
    tokens_list: List[List[str]],
    true_labels_list: List[List[str]],
    pred_labels_list: List[List[str]],
    n_examples: int = 5
) -> None:
    print("\nEksempel-forudsigelser:")
    print("=" * 80)

    shown = 0
    for (_, row), tokens, true_labels, pred_labels in zip(
        df_subset.iterrows(), tokens_list, true_labels_list, pred_labels_list
    ):
        true_entities = labels_to_entities(tokens, true_labels)
        pred_entities = labels_to_entities(tokens, pred_labels)

        # Vis helst kun eksempler hvor der faktisk er entities
        if not true_entities and not pred_entities:
            continue

        print(f"source_text: {row['source_text']}")
        print(f"ground truth privacy: {row['privacy']}")
        print(f"true entities: {true_entities}")
        print(f"pred entities: {pred_entities}")
        print("-" * 80)

        shown += 1
        if shown >= n_examples:
            break

def show_timer(stop_event):
    start_time = time.time()

    while not stop_event.is_set():
        elapsed = int(time.time() - start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        print(f"\rTræning i gang: {minutes:02d}:{seconds:02d}", end="", flush=True)
        time.sleep(1)

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not os.path.exists(PARQUET_PATH):
        print(f"Fejl: Filen findes ikke: {PARQUET_PATH}")
        sys.exit(1)

    print(f"Indlæser data fra: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)

    print("Kolonner:", df.columns.tolist())
    print()

    for i in range(3):
        print(f"Række {i}")
        print("source_text:", repr(df.iloc[i]["source_text"]))
        print("privacy type:", type(df.iloc[i]["privacy"]))
        print("privacy value:", repr(df.iloc[i]["privacy"]))
        print("-" * 80)

    required_columns = {"source_text", "privacy"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Fejl: Manglende kolonner i datasættet: {missing_columns}")
        print(f"Fundne kolonner: {list(df.columns)}")
        sys.exit(1)

    # behold kun relevante kolonner
    df = df[["source_text", "privacy"]].copy()

    # fjern rækker uden tekst
    df = df.dropna(subset=["source_text"]).copy()
    df["source_text"] = df["source_text"].astype(str)

    # normaliser privacy-kolonnen
    df["privacy"] = df["privacy"].apply(normalize_privacy_column)

    print("\nEfter normalize_privacy_column():")
    for i in range(3):
        print(f"Række {i} normalized privacy:", df.iloc[i]["privacy"])

    # evt. begræns datamængde under udvikling
    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    # fjern tomme tekster
    df = df[df["source_text"].str.strip() != ""].copy()
    df = df.reset_index(drop=True)

    if len(df) < 10:
        print("Fejl: For få rækker i datasættet efter filtrering.")
        sys.exit(1)

    print(f"Antal rækker efter filtrering: {len(df)}")

    # 80 / 10 / 10 split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print_split_sizes(train_df, val_df, test_df)

    print("\nBygger CRF-datasæt...")
    X_train, y_train, train_tokens = build_crf_dataset(train_df)
    X_val, y_val, val_tokens = build_crf_dataset(val_df)
    X_test, y_test, test_tokens = build_crf_dataset(test_df)

    print(f"Train-sekvenser: {len(X_train)}")
    print(f"Val-sekvenser:   {len(X_val)}")
    print(f"Test-sekvenser:  {len(X_test)}")

    if not X_train or not X_val or not X_test:
        print("Fejl: Et af splittene gav ingen brugbare sekvenser.")
        sys.exit(1)

    print("\nTræner CRF-model...", flush=True)
    print("Lige før crf.fit()", flush=True)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )

    # start timer
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=show_timer, args=(stop_event,), daemon=True)
    timer_thread.start()

    start_time = time.time()

    crf.fit(X_train, y_train)

    end_time = time.time()

    # stop timer
    stop_event.set()
    timer_thread.join()

    print("\nTræning færdig!")
    print(f"Total tid: {end_time - start_time:.2f} sekunder")

    print("\nKører forudsigelser...")
    y_val_pred = crf.predict(X_val)
    y_test_pred = crf.predict(X_test)

    # Token-level evaluering
    print("\n" + "=" * 80)
    print("TOKEN-LEVEL EVALUERING - VALIDATION")
    print("=" * 80)
    print(metrics.flat_classification_report(y_val, y_val_pred, digits=3))

    print("\n" + "=" * 80)
    print("TOKEN-LEVEL EVALUERING - TEST")
    print("=" * 80)
    print(metrics.flat_classification_report(y_test, y_test_pred, digits=3))

    # Entity-level evaluering
    val_entity_scores = evaluate_entity_level(val_df, val_tokens, y_val_pred)
    test_entity_scores = evaluate_entity_level(test_df, test_tokens, y_test_pred)

    print("\n" + "=" * 80)
    print("ENTITY-LEVEL EVALUERING - VALIDATION")
    print("=" * 80)
    for k, v in val_entity_scores.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("ENTITY-LEVEL EVALUERING - TEST")
    print("=" * 80)
    for k, v in test_entity_scores.items():
        print(f"{k}: {v}")

    # Gem modellen
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(crf, f)

    print(f"\nModel gemt til: {MODEL_OUTPUT_PATH}")

    # Vis nogle eksempel-forudsigelser fra test
    print_example_predictions(
        df_subset=test_df,
        tokens_list=test_tokens,
        true_labels_list=y_test,
        pred_labels_list=y_test_pred,
        n_examples=N_EXAMPLES_TO_SHOW,
    )


if __name__ == "__main__":
    main()