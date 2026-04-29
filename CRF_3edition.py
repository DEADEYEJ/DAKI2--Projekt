from __future__ import annotations
#ikke nyeste version
import argparse
import ast
import gc
import html
import json
import os
import pickle
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import sklearn_crfsuite
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is only for nicer progress output.
    tqdm = None


# =========================================================
# Configuration
# =========================================================
PARQUET_PATH = "0000.parquet"
MODEL_OUTPUT_PATH = "crf_pii_model.pkl"

TEXT_COL = "source_text"
ANNOTATION_COL = "privacy"
RANDOM_STATE = 42

# Default values match the old CRF_3edition setup, while folds run
# sequentially by default to avoid multiplying memory use.
DEFAULT_MAX_ROWS = 50000
DEFAULT_DEV_MAX_ROWS = 50000
DEFAULT_TEST_SIZE = 0.10
DEFAULT_N_SPLITS = 9
DEFAULT_DEV_N_SPLITS = 3
DEFAULT_N_JOBS = 1

DEFAULT_CRF_CONFIG = {
    "algorithm": "lbfgs",
    "c1": 0.1,
    "c2": 0.1,
    "max_iterations": 100,
    "all_possible_transitions": True,
}

DEV_CRF_CONFIG = {
    "algorithm": "lbfgs",
    "c1": 0.2,
    "c2": 0.2,
    "max_iterations": 50,
    "all_possible_transitions": False,
}


# =========================================================
# Data structures
# =========================================================
@dataclass
class EntitySpan:
    start: int
    end: int
    label: str
    value: Optional[str] = None


@dataclass
class Token:
    text: str
    start: int
    end: int


@dataclass
class SequenceSample:
    sample_id: Any
    raw_text: str
    sequence_text: str
    entities: List[EntitySpan] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


# =========================================================
# Text and annotation preprocessing
# =========================================================
def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return bool(pd.isna(value))
    return False


def normalize_text(value: Any) -> str:
    """Normalize text used by the CRF while keeping entity values searchable."""
    if is_missing(value):
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)

    replacements = {
        "\u00A0": " ",
        "\u200B": "",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\t": " ",
        "\r": " ",
        "\n": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()


def normalize_privacy_items(value: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Convert the privacy column to a list of dictionaries."""
    issues: List[str] = []

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return [], ["missing_annotation"]

    parsed: Any = value

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return [], ["empty_annotation"]
        try:
            parsed = json.loads(stripped)
        except Exception:
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                return [], ["malformed_annotation"]
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
        try:
            parsed = value.tolist()
        except Exception:
            return [], ["unsupported_annotation_type"]

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, (list, tuple)):
        return [], ["annotation_not_list"]

    items: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed):
        if isinstance(item, dict):
            items.append(item)
            continue
        try:
            items.append(dict(item))
        except Exception:
            issues.append(f"annotation_item_{idx}_not_dict")

    return items, issues


def find_all_occurrences(text: str, substring: str, ignore_case: bool = False) -> List[Tuple[int, int]]:
    if not text or not substring:
        return []

    search_text = text.lower() if ignore_case else text
    search_substring = substring.lower() if ignore_case else substring

    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = search_text.find(search_substring, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(substring)))
        start = idx + max(len(substring), 1)
    return spans


def int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def get_annotation_span(item: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    start = item.get("start")
    end = item.get("end")

    if start is None:
        start = item.get("begin_offset") or item.get("offset_start")
    if end is None:
        end = item.get("stop") or item.get("offset_end")

    return int_or_none(start), int_or_none(end)


def spans_from_annotation_item(
    item: Dict[str, Any],
    raw_text: str,
    sequence_text: str,
    item_idx: int,
) -> Tuple[List[EntitySpan], List[str]]:
    issues: List[str] = []
    label = item.get("label")
    value = item.get("value")

    if label is None or not str(label).strip():
        return [], [f"annotation_item_{item_idx}_missing_label"]

    label_str = str(label).strip()
    value_text = normalize_text(value) if value is not None else ""
    start, end = get_annotation_span(item)

    spans: List[Tuple[int, int]] = []

    # Most rows only contain label/value pairs. Marking every occurrence avoids
    # teaching the CRF that repeated PII values are non-PII.
    if value_text:
        spans = find_all_occurrences(sequence_text, value_text)
        if not spans:
            spans = find_all_occurrences(sequence_text, value_text, ignore_case=True)
            if spans:
                issues.append(f"annotation_item_{item_idx}_case_insensitive_value_match")

    # If value matching fails and offsets are already aligned to sequence_text,
    # keep them as a fallback.
    if not spans and start is not None and end is not None:
        if 0 <= start < end <= len(sequence_text):
            extracted = normalize_text(sequence_text[start:end])
            if not value_text or extracted == value_text:
                spans = [(start, end)]
            else:
                issues.append(f"annotation_item_{item_idx}_span_value_mismatch")
        elif 0 <= start < end <= len(raw_text):
            raw_extracted = normalize_text(raw_text[start:end])
            raw_matches = find_all_occurrences(sequence_text, raw_extracted)
            if raw_matches:
                spans = raw_matches
                issues.append(f"annotation_item_{item_idx}_raw_span_remapped")
            else:
                issues.append(f"annotation_item_{item_idx}_span_not_mappable")
        else:
            issues.append(f"annotation_item_{item_idx}_invalid_span_range")

    if not spans:
        issues.append(f"annotation_item_{item_idx}_missing_span")
        return [], issues

    entities = [
        EntitySpan(start=s, end=e, label=label_str, value=value_text or None)
        for s, e in spans
        if 0 <= s < e <= len(sequence_text)
    ]
    return entities, issues


def resolve_overlapping_entities(entities: Sequence[EntitySpan]) -> Tuple[List[EntitySpan], List[str]]:
    """Keep the longest entity when value-based matching creates overlaps."""
    selected: List[EntitySpan] = []
    issues: List[str] = []

    by_priority = sorted(
        entities,
        key=lambda entity: (-(entity.end - entity.start), entity.start, entity.end, entity.label),
    )

    for entity in by_priority:
        overlaps_existing = any(entity.start < kept.end and entity.end > kept.start for kept in selected)
        if overlaps_existing:
            issues.append("overlapping_entity_removed")
            continue
        selected.append(entity)

    return sorted(selected, key=lambda entity: (entity.start, entity.end, entity.label)), issues


def parse_entities(raw_annotations: Any, raw_text: str, sequence_text: str) -> Tuple[List[EntitySpan], List[str]]:
    items, issues = normalize_privacy_items(raw_annotations)
    entities: List[EntitySpan] = []

    for idx, item in enumerate(items):
        item_entities, item_issues = spans_from_annotation_item(
            item=item,
            raw_text=raw_text,
            sequence_text=sequence_text,
            item_idx=idx,
        )
        issues.extend(item_issues)
        entities.extend(item_entities)

    deduped: List[EntitySpan] = []
    seen = set()
    for ent in sorted(entities, key=lambda e: (e.start, e.end, e.label)):
        key = (ent.start, ent.end, ent.label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ent)

    resolved, overlap_issues = resolve_overlapping_entities(deduped)
    issues.extend(overlap_issues)

    return resolved, issues


# =========================================================
# Tokenization, alignment, and feature extraction
# =========================================================
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_with_offsets(text: str) -> List[Token]:
    return [Token(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def align_entities_to_tokens(tokens: Sequence[Token], entities: Sequence[EntitySpan]) -> Tuple[List[str], List[str]]:
    labels = ["O"] * len(tokens)
    issues: List[str] = []

    for ent_idx, ent in enumerate(sorted(entities, key=lambda e: (e.start, e.end))):
        token_idxs = [
            i
            for i, token in enumerate(tokens)
            if token.start < ent.end and token.end > ent.start
        ]

        if not token_idxs:
            issues.append(f"entity_{ent_idx}_no_token_overlap")
            continue

        for offset, token_idx in enumerate(token_idxs):
            token = tokens[token_idx]
            if token.start < ent.start or token.end > ent.end:
                issues.append(f"entity_{ent_idx}_partial_token_overlap")

            if labels[token_idx] != "O":
                issues.append(f"token_{token_idx}_overlapping_entities")
                continue

            prefix = "B-" if offset == 0 else "I-"
            labels[token_idx] = f"{prefix}{ent.label}"

    return labels, issues


def build_sequence_sample(row: pd.Series, sample_id: Any) -> SequenceSample:
    raw_text = "" if is_missing(row.get(TEXT_COL, "")) else str(row.get(TEXT_COL, ""))
    sequence_text = normalize_text(raw_text)
    entities, parse_issues = parse_entities(row.get(ANNOTATION_COL, None), raw_text, sequence_text)
    tokens = tokenize_with_offsets(sequence_text)
    labels, alignment_issues = align_entities_to_tokens(tokens, entities)

    return SequenceSample(
        sample_id=sample_id,
        raw_text=raw_text,
        sequence_text=sequence_text,
        entities=entities,
        tokens=tokens,
        labels=labels,
        issues=parse_issues + alignment_issues,
    )


def word_shape(word: str, compress: bool = False) -> str:
    chars: List[str] = []
    for char in word:
        if char.isupper():
            code = "X"
        elif char.islower():
            code = "x"
        elif char.isdigit():
            code = "d"
        else:
            code = char

        if not compress or not chars or chars[-1] != code:
            chars.append(code)
    return "".join(chars)


def token_window(tokens: Sequence[Token], i: int, radius: int = 2) -> str:
    start = max(0, i - radius)
    end = min(len(tokens), i + radius + 1)
    return "".join(token.text for token in tokens[start:end])


def simple_token_flags(word: str) -> Dict[str, Any]:
    has_alpha = any(ch.isalpha() for ch in word)
    has_digit = any(ch.isdigit() for ch in word)

    return {
        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_digit": word.isdigit(),
        "is_alpha": word.isalpha(),
        "is_alnum": word.isalnum(),
        "is_punct": bool(re.fullmatch(r"[^\w\s]+", word)),
        "has_alpha": has_alpha,
        "has_digit": has_digit,
        "has_hyphen": "-" in word,
        "has_slash": "/" in word,
        "has_dot": "." in word,
        "has_at": "@" in word,
        "has_colon": ":" in word,
        "has_plus": "+" in word,
        "has_underscore": "_" in word,
        "alpha_digit_mix": has_alpha and has_digit,
        "length": len(word),
        "length_bucket": min(len(word), 20),
    }


def add_neighbor_features(features: Dict[str, Any], prefix: str, token: Token) -> None:
    word = token.text
    features.update({
        f"{prefix}:lower": word.lower(),
        f"{prefix}:shape": word_shape(word),
        f"{prefix}:compressed_shape": word_shape(word, compress=True),
        f"{prefix}:is_title": word.istitle(),
        f"{prefix}:is_upper": word.isupper(),
        f"{prefix}:is_digit": word.isdigit(),
        f"{prefix}:has_digit": any(ch.isdigit() for ch in word),
    })


def token_to_features(tokens: Sequence[Token], i: int) -> Dict[str, Any]:
    word = tokens[i].text
    lower = word.lower()
    window = token_window(tokens, i, radius=2)

    features: Dict[str, Any] = {
        "bias": 1.0,
        "word": word,
        "lower": lower,
        "prefix_1": word[:1],
        "prefix_2": word[:2],
        "prefix_3": word[:3],
        "prefix_4": word[:4],
        "suffix_1": word[-1:],
        "suffix_2": word[-2:],
        "suffix_3": word[-3:],
        "suffix_4": word[-4:],
        "shape": word_shape(word),
        "compressed_shape": word_shape(word, compress=True),
        "window_has_at": "@" in window,
        "window_has_dot": "." in window,
        "window_has_slash": "/" in window,
        "window_has_hyphen": "-" in window,
        "window_digit_count": sum(ch.isdigit() for ch in window),
    }
    features.update(simple_token_flags(word))

    if i == 0:
        features["BOS"] = True
    else:
        add_neighbor_features(features, "-1", tokens[i - 1])
        features["-1:lower+lower"] = f"{tokens[i - 1].text.lower()}|{lower}"

    if i > 1:
        add_neighbor_features(features, "-2", tokens[i - 2])

    if i == len(tokens) - 1:
        features["EOS"] = True
    else:
        add_neighbor_features(features, "+1", tokens[i + 1])
        features["lower+1:lower"] = f"{lower}|{tokens[i + 1].text.lower()}"

    if i < len(tokens) - 2:
        add_neighbor_features(features, "+2", tokens[i + 2])

    return features


def sample_to_features(sample: SequenceSample) -> List[Dict[str, Any]]:
    return [token_to_features(sample.tokens, i) for i in range(len(sample.tokens))]


def build_crf_dataset(
    dataframe: pd.DataFrame,
    description: str,
) -> Tuple[List[List[Dict[str, Any]]], List[List[str]], List[SequenceSample], Counter]:
    iterator: Iterable[Tuple[Any, pd.Series]] = dataframe.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(dataframe), desc=description, file=sys.stdout)

    X: List[List[Dict[str, Any]]] = []
    y: List[List[str]] = []
    samples: List[SequenceSample] = []
    issue_counter: Counter = Counter()

    for idx, row in iterator:
        sample = build_sequence_sample(row, sample_id=idx)
        issue_counter.update(sample.issues)

        if not sample.tokens:
            issue_counter.update(["empty_token_sequence"])
            continue

        X.append(sample_to_features(sample))
        y.append(sample.labels)
        samples.append(sample)

    return X, y, samples, issue_counter


# =========================================================
# Evaluation
# =========================================================
def flatten_label_sequences(label_sequences: Sequence[Sequence[str]]) -> List[str]:
    return [label for sequence in label_sequences for label in sequence]


def evaluate_token_labels(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]) -> Dict[str, Any]:
    y_true_flat = flatten_label_sequences(y_true)
    y_pred_flat = flatten_label_sequences(y_pred)

    labels = sorted(set(y_true_flat) | set(y_pred_flat))
    eval_labels = [label for label in labels if label != "O"]

    if not eval_labels:
        return {
            "per_label": {},
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "classification_report": "No entity labels found.",
        }

    per_label_p, per_label_r, per_label_f1, support = precision_recall_fscore_support(
        y_true_flat,
        y_pred_flat,
        labels=eval_labels,
        average=None,
        zero_division=0,
    )

    per_label = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": int(s),
        }
        for label, p, r, f1, s in zip(eval_labels, per_label_p, per_label_r, per_label_f1, support)
    }

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_flat,
        y_pred_flat,
        labels=eval_labels,
        average="macro",
        zero_division=0,
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_flat,
        y_pred_flat,
        labels=eval_labels,
        average="micro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_flat,
        y_pred_flat,
        labels=eval_labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "per_label": per_label,
        "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)},
        "micro": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f1)},
        "weighted": {"precision": float(weighted_p), "recall": float(weighted_r), "f1": float(weighted_f1)},
        "classification_report": classification_report(
            y_true_flat,
            y_pred_flat,
            labels=eval_labels,
            zero_division=0,
        ),
    }


def bio_labels_to_entities(tokens: Sequence[Token], labels: Sequence[str], text: str) -> List[EntitySpan]:
    entities: List[EntitySpan] = []
    current_label: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None

    def close_entity() -> None:
        nonlocal current_label, start, end
        if current_label is not None and start is not None and end is not None:
            entities.append(EntitySpan(start=start, end=end, label=current_label, value=text[start:end]))
        current_label = None
        start = None
        end = None

    for token, label in zip(tokens, labels):
        if label == "O":
            close_entity()
            continue

        if "-" not in label:
            prefix, label_type = "B", label
        else:
            prefix, label_type = label.split("-", 1)

        if prefix == "B" or current_label != label_type:
            close_entity()
            current_label = label_type
            start = token.start
            end = token.end
        else:
            end = token.end

    close_entity()
    return entities


def normalized_entity_key(entity: EntitySpan) -> Tuple[str, str]:
    value = entity.value if entity.value is not None else ""
    return entity.label, normalize_text(value).lower()


def entity_counter(entities: Sequence[EntitySpan]) -> Counter:
    return Counter(normalized_entity_key(entity) for entity in entities if normalized_entity_key(entity)[1])


def score_entity_counters(true_counter: Counter, pred_counter: Counter) -> Tuple[int, int, int]:
    tp = 0
    fp = 0
    fn = 0

    for key in set(true_counter) | set(pred_counter):
        true_count = true_counter.get(key, 0)
        pred_count = pred_counter.get(key, 0)
        tp += min(true_count, pred_count)
        fp += max(0, pred_count - true_count)
        fn += max(0, true_count - pred_count)

    return tp, fp, fn


def evaluate_entities(samples: Sequence[SequenceSample], pred_labels: Sequence[Sequence[str]]) -> Dict[str, float]:
    tp = fp = fn = 0

    for sample, labels in zip(samples, pred_labels):
        true_counter = entity_counter(sample.entities)
        pred_entities = bio_labels_to_entities(sample.tokens, labels, sample.sequence_text)
        pred_counter = entity_counter(pred_entities)
        sample_tp, sample_fp, sample_fn = score_entity_counters(true_counter, pred_counter)
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_cv_results(cv_results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    macro_f1s = [fold["token"]["macro"]["f1"] for fold in cv_results]
    macro_ps = [fold["token"]["macro"]["precision"] for fold in cv_results]
    macro_rs = [fold["token"]["macro"]["recall"] for fold in cv_results]
    entity_f1s = [fold["entity"]["f1"] for fold in cv_results]

    return {
        "macro_precision_mean": sum(macro_ps) / len(macro_ps),
        "macro_recall_mean": sum(macro_rs) / len(macro_rs),
        "macro_f1_mean": sum(macro_f1s) / len(macro_f1s),
        "entity_f1_mean": sum(entity_f1s) / len(entity_f1s),
        "n_folds": float(len(cv_results)),
    }


# =========================================================
# Training helpers
# =========================================================
def show_timer(stop_event: threading.Event, message: str) -> None:
    start_time = time.time()

    while not stop_event.is_set():
        elapsed = int(time.time() - start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        print(f"\r{message}: {minutes:02d}:{seconds:02d}", end="", flush=True)
        time.sleep(1)


def train_crf_model(
    X_train: Sequence[Sequence[Dict[str, Any]]],
    y_train: Sequence[Sequence[str]],
    crf_config: Dict[str, Any],
    timer_message: Optional[str] = None,
) -> sklearn_crfsuite.CRF:
    crf = sklearn_crfsuite.CRF(**crf_config)

    stop_event: Optional[threading.Event] = None
    timer_thread: Optional[threading.Thread] = None

    if timer_message:
        stop_event = threading.Event()
        timer_thread = threading.Thread(target=show_timer, args=(stop_event, timer_message), daemon=True)
        timer_thread.start()

    start_time = time.time()
    crf.fit(X_train, y_train)
    elapsed = time.time() - start_time

    if stop_event is not None and timer_thread is not None:
        stop_event.set()
        timer_thread.join()
        print()

    print(f"Training finished in {elapsed:.2f} seconds")
    return crf


def select_items(items: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    return [items[i] for i in indices]


def run_single_fold(
    fold_no: int,
    total_folds: int,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    X: Sequence[Sequence[Dict[str, Any]]],
    y: Sequence[Sequence[str]],
    samples: Sequence[SequenceSample],
    crf_config: Dict[str, Any],
    use_timer: bool,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"FOLD {fold_no}/{total_folds}")
    print("=" * 80)

    fold_start = time.time()
    X_train = select_items(X, train_idx)
    y_train = select_items(y, train_idx)
    X_val = select_items(X, val_idx)
    y_val = select_items(y, val_idx)
    val_samples = select_items(samples, val_idx)

    model = train_crf_model(
        X_train,
        y_train,
        crf_config,
        timer_message=f"Fold {fold_no} training" if use_timer else None,
    )
    y_val_pred = model.predict(X_val)

    token_metrics = evaluate_token_labels(y_val, y_val_pred)
    entity_metrics = evaluate_entities(val_samples, y_val_pred)

    print("\nTOKEN-LEVEL VALIDATION")
    print(token_metrics["classification_report"])
    print(f"Fold {fold_no} token macro-F1:  {token_metrics['macro']['f1']:.4f}")
    print(f"Fold {fold_no} entity-level F1: {entity_metrics['f1']:.4f}")
    print(f"Fold finished in {time.time() - fold_start:.2f} seconds")

    del model, y_val_pred, X_train, y_train, X_val, y_val, val_samples
    gc.collect()

    return {"token": token_metrics, "entity": entity_metrics}


def run_cross_validation(
    X_dev: Sequence[Sequence[Dict[str, Any]]],
    y_dev: Sequence[Sequence[str]],
    dev_samples: Sequence[SequenceSample],
    n_splits: int,
    random_state: int,
    crf_config: Dict[str, Any],
    n_jobs: int,
) -> List[Dict[str, Any]]:
    n_items = len(X_dev)
    if n_items < 2:
        raise ValueError("Need at least two sequences for cross-validation.")

    actual_splits = min(n_splits, n_items)
    if actual_splits < n_splits:
        print(f"Reducing n_splits from {n_splits} to {actual_splits} because the dataset is small.")

    kfold = KFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    folds = list(kfold.split(range(n_items)))

    if n_jobs == 1:
        return [
            run_single_fold(
                fold_no=fold_no,
                total_folds=actual_splits,
                train_idx=train_idx,
                val_idx=val_idx,
                X=X_dev,
                y=y_dev,
                samples=dev_samples,
                crf_config=crf_config,
                use_timer=True,
            )
            for fold_no, (train_idx, val_idx) in enumerate(folds, start=1)
        ]

    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_single_fold)(
            fold_no,
            actual_splits,
            train_idx,
            val_idx,
            X_dev,
            y_dev,
            dev_samples,
            crf_config,
            False,
        )
        for fold_no, (train_idx, val_idx) in enumerate(folds, start=1)
    )


# =========================================================
# Reporting helpers
# =========================================================
def print_issue_summary(name: str, issues: Counter, limit: int = 15) -> None:
    print(f"\n{name} preprocessing issues:")
    if not issues:
        print("  None")
        return
    for issue, count in issues.most_common(limit):
        print(f"  {issue}: {count}")


def print_entity_distribution(samples: Sequence[SequenceSample], limit: int = 40) -> None:
    counter = Counter()
    for sample in samples:
        for entity in sample.entities:
            counter[entity.label] += 1

    print("\nEntity distribution after alignment:")
    for label, count in counter.most_common(limit):
        print(f"  {label}: {count}")


def print_example_predictions(
    samples: Sequence[SequenceSample],
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    n_examples: int,
) -> None:
    if n_examples <= 0:
        return

    print("\nExample predictions")
    print("=" * 80)
    shown = 0

    for sample, true_labels, pred_labels in zip(samples, y_true, y_pred):
        true_entities = bio_labels_to_entities(sample.tokens, true_labels, sample.sequence_text)
        pred_entities = bio_labels_to_entities(sample.tokens, pred_labels, sample.sequence_text)

        if not true_entities and not pred_entities:
            continue

        print(f"sample_id: {sample.sample_id}")
        print(f"text: {sample.sequence_text[:500]}")
        print("true:", [{"label": e.label, "value": e.value} for e in true_entities])
        print("pred:", [{"label": e.label, "value": e.value} for e in pred_entities])
        print("-" * 80)

        shown += 1
        if shown >= n_examples:
            break


def print_metrics_block(title: str, token_metrics: Dict[str, Any], entity_metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("\nTOKEN-LEVEL")
    print(token_metrics["classification_report"])
    print("Token macro:", token_metrics["macro"])
    print("Token micro:", token_metrics["micro"])
    print("Token weighted:", token_metrics["weighted"])
    print("\nENTITY-LEVEL")
    for key, value in entity_metrics.items():
        print(f"{key}: {value}")


# =========================================================
# CLI and main workflow
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-contained CRF model for PII detection in log-like text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet-path", default=PARQUET_PATH)
    parser.add_argument("--model-output-path", default=MODEL_OUTPUT_PATH)
    parser.add_argument("--dev-mode", action="store_true", help="Use faster development defaults.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use <=0 for all rows.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-final-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--c1", type=float, default=None)
    parser.add_argument("--c2", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--all-possible-transitions", action="store_true", default=None)
    parser.add_argument("--no-all-possible-transitions", action="store_false", dest="all_possible_transitions")
    return parser.parse_args()


def make_runtime_config(args: argparse.Namespace) -> Tuple[Optional[int], int, Dict[str, Any]]:
    max_rows = args.max_rows
    if max_rows is None:
        max_rows = DEFAULT_DEV_MAX_ROWS if args.dev_mode else DEFAULT_MAX_ROWS
    if max_rows is not None and max_rows <= 0:
        max_rows = None

    n_splits = args.n_splits
    if n_splits is None:
        n_splits = DEFAULT_DEV_N_SPLITS if args.dev_mode else DEFAULT_N_SPLITS

    crf_config = dict(DEV_CRF_CONFIG if args.dev_mode else DEFAULT_CRF_CONFIG)
    if args.c1 is not None:
        crf_config["c1"] = args.c1
    if args.c2 is not None:
        crf_config["c2"] = args.c2
    if args.max_iterations is not None:
        crf_config["max_iterations"] = args.max_iterations
    if args.all_possible_transitions is not None:
        crf_config["all_possible_transitions"] = args.all_possible_transitions

    return max_rows, n_splits, crf_config


def load_dataframe(parquet_path: str, max_rows: Optional[int]) -> pd.DataFrame:
    if not os.path.exists(parquet_path):
        print(f"Error: file not found: {parquet_path}")
        sys.exit(1)

    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    missing_columns = {TEXT_COL, ANNOTATION_COL} - set(df.columns)
    if missing_columns:
        print(f"Error: missing columns: {sorted(missing_columns)}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    df = df[[TEXT_COL, ANNOTATION_COL]].dropna(subset=[TEXT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip() != ""].reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    if len(df) < 10:
        print("Error: too few rows after filtering.")
        sys.exit(1)

    return df


def save_model(model: sklearn_crfsuite.CRF, output_path: str, metadata: Dict[str, Any]) -> None:
    setattr(model, "pii_metadata_", metadata)
    with open(output_path, "wb") as file:
        pickle.dump(model, file)
    print(f"\nModel saved to: {output_path}")


def main() -> None:
    args = parse_args()
    max_rows, n_splits, crf_config = make_runtime_config(args)

    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"dev_mode: {args.dev_mode}")
    print(f"max_rows: {max_rows if max_rows is not None else 'all'}")
    print(f"test_size: {args.test_size}")
    print(f"n_splits: {n_splits}")
    print(f"n_jobs: {args.n_jobs}")
    print(f"run_cv: {not args.skip_cv}")
    print(f"train_final: {not args.skip_final_train}")
    print(f"run_test: {not args.skip_test}")
    print(f"save_model: {not args.no_save}")
    print(f"crf_config: {crf_config}")

    df = load_dataframe(args.parquet_path, max_rows=max_rows)
    print(f"\nRows after filtering: {len(df)}")

    dev_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nData split:")
    print(f"  Dev:  {len(dev_df)} ({len(dev_df) / len(df):.2%})")
    print(f"  Test: {len(test_df)} ({len(test_df) / len(df):.2%})")

    del df
    gc.collect()

    print("\nBuilding dev dataset once for CV and final training...")
    X_dev, y_dev, dev_samples, dev_issues = build_crf_dataset(dev_df, description="Dev dataset")
    print(f"Dev sequences: {len(X_dev)}")
    print_issue_summary("Dev", dev_issues)
    print_entity_distribution(dev_samples)

    if not X_dev:
        print("Error: no usable dev sequences.")
        sys.exit(1)

    if not args.skip_cv:
        print("\nStarting cross-validation...")
        cv_results = run_cross_validation(
            X_dev=X_dev,
            y_dev=y_dev,
            dev_samples=dev_samples,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
            crf_config=crf_config,
            n_jobs=args.n_jobs,
        )
        summary = summarize_cv_results(cv_results)

        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)
        for key, value in summary.items():
            print(f"{key}: {value}")

    if args.skip_final_train:
        print("\nFinal training skipped.")
        return

    print("\nTraining final model on full dev set...")
    final_model = train_crf_model(
        X_dev,
        y_dev,
        crf_config,
        timer_message="Final training",
    )

    metadata = {
        "text_col": TEXT_COL,
        "annotation_col": ANNOTATION_COL,
        "random_state": RANDOM_STATE,
        "max_rows": max_rows,
        "test_size": args.test_size,
        "n_splits": n_splits,
        "crf_config": crf_config,
        "feature_version": "CRF_3edition_self_contained_v1",
    }

    if not args.no_save:
        save_model(final_model, args.model_output_path, metadata)

    if args.skip_test:
        print("\nHold-out test skipped.")
        return

    print("\nBuilding hold-out test dataset...")
    X_test, y_test, test_samples, test_issues = build_crf_dataset(test_df, description="Test dataset")
    print(f"Test sequences: {len(X_test)}")
    print_issue_summary("Test", test_issues)

    if not X_test:
        print("Error: no usable test sequences.")
        sys.exit(1)

    print("\nPredicting hold-out test set...")
    y_test_pred = final_model.predict(X_test)

    token_metrics = evaluate_token_labels(y_test, y_test_pred)
    entity_metrics = evaluate_entities(test_samples, y_test_pred)
    print_metrics_block("HOLD-OUT TEST RESULT", token_metrics, entity_metrics)
    print_example_predictions(test_samples, y_test, y_test_pred, n_examples=args.examples)

    del X_dev, y_dev, dev_samples, X_test, y_test, test_samples, y_test_pred, final_model
    gc.collect()


if __name__ == "__main__":
    main()
