from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Sequence

import pandas as pd

from .crf5_config import ANNOTATION_COL, TEXT_COL
from .crf5_dataset import SequenceSample, has_severe_alignment_issues, normalize_annotation_items, raw_label_counter
from .crf5_metrics import bio_labels_to_entities


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
        print("true:", [{"label": entity.label, "value": entity.value} for entity in true_entities])
        print("pred:", [{"label": entity.label, "value": entity.value} for entity in pred_entities])
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


def print_dataframe_profile(name: str, dataframe: pd.DataFrame, top_k: int = 10) -> None:
    lengths = dataframe[TEXT_COL].astype(str).str.len()
    annotation_counts = [len(normalize_annotation_items(value)) for value in dataframe[ANNOTATION_COL]]
    labels = raw_label_counter(dataframe[ANNOTATION_COL])

    avg_annotations = sum(annotation_counts) / len(annotation_counts) if annotation_counts else 0.0

    print("\n" + "=" * 80)
    print(f"{name.upper()} PROFILE")
    print("=" * 80)
    print(f"Rows: {len(dataframe)}")
    print(f"Avg text length: {lengths.mean():.2f}")
    print(f"Median text length: {lengths.median():.2f}")
    print(f"P95 text length: {lengths.quantile(0.95):.2f}")
    print(f"Avg annotations per row: {avg_annotations:.2f}")
    print("Top raw labels:")
    if not labels:
        print("  None")
    else:
        for label, count in labels.most_common(top_k):
            print(f"  {label}: {count}")


def print_sample_quality(name: str, samples: Sequence[SequenceSample]) -> None:
    severe_count = sum(1 for sample in samples if has_severe_alignment_issues(sample.issues))
    token_lengths = [len(sample.tokens) for sample in samples]

    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    max_tokens = max(token_lengths) if token_lengths else 0

    print(f"\n{name} sample quality:")
    print(f"  Sequences: {len(samples)}")
    print(f"  Severe alignment issue samples: {severe_count} ({severe_count / max(len(samples), 1):.2%})")
    print(f"  Avg tokens per sequence: {avg_tokens:.2f}")
    print(f"  Max tokens in a sequence: {max_tokens}")
