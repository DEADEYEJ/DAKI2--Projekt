from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.metrics import classification_report, precision_recall_fscore_support

from .crf5_dataset import EntitySpan, SequenceSample, Token, normalize_text


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
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(label_support),
        }
        for label, precision, recall, f1, label_support in zip(
            eval_labels,
            per_label_p,
            per_label_r,
            per_label_f1,
            support,
        )
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

        prefix, label_type = label.split("-", 1) if "-" in label else ("B", label)

        if prefix == "B" or current_label != label_type:
            close_entity()
            current_label = label_type
            start = token.start
            end = token.end
        else:
            end = token.end

    close_entity()
    return entities


def normalize_entity_value(value: str) -> str:
    return normalize_text(value).lower()


def entity_counter(entities: Sequence[EntitySpan]) -> Counter:
    counter: Counter = Counter()
    for entity in entities:
        if entity.value is None:
            continue
        normalized_value = normalize_entity_value(entity.value)
        if normalized_value:
            counter[(entity.label, normalized_value)] += 1
    return counter


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

