from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from joblib import Parallel, delayed

from .crf5_dataset import SequenceSample
from .crf5_training import run_single_fold


def build_sample_label_sets(dev_samples: Sequence[SequenceSample]) -> List[set[str]]:
    label_sets: List[set[str]] = []
    for sample in dev_samples:
        label_sets.append({entity.label for entity in sample.entities})
    return label_sets


def compute_label_counts(label_sets: Sequence[set[str]]) -> Counter:
    counts: Counter = Counter()
    for labels in label_sets:
        counts.update(labels)
    return counts


def print_label_coverage_warnings(label_counts: Counter, n_splits: int) -> None:
    rare_labels = sorted(label for label, count in label_counts.items() if count < n_splits)
    if rare_labels:
        print("\nWarning: these labels are too rare to appear in every validation fold:")
        print("  " + ", ".join(rare_labels))


def sample_priority(index: int, label_sets: Sequence[set[str]], label_counts: Counter) -> Tuple[float, int, int]:
    labels = label_sets[index]
    rarity_score = sum(1.0 / label_counts[label] for label in labels) if labels else 0.0
    return (-rarity_score, -len(labels), index)


def choose_fold_for_sample(
    labels: set[str],
    fold_label_counts: Sequence[Counter],
    fold_sizes: Sequence[int],
    target_fold_size: float,
) -> int:
    best_fold = 0
    best_score: Tuple[float, float, int] | None = None

    for fold_idx in range(len(fold_label_counts)):
        if labels:
            label_balance = sum(fold_label_counts[fold_idx][label] for label in labels) / len(labels)
        else:
            label_balance = 0.0

        size_balance = fold_sizes[fold_idx] / target_fold_size if target_fold_size else fold_sizes[fold_idx]
        score = (label_balance, size_balance, fold_sizes[fold_idx])

        if best_score is None or score < best_score:
            best_score = score
            best_fold = fold_idx

    return best_fold


def build_multilabel_stratified_folds(
    label_sets: Sequence[set[str]],
    n_splits: int,
) -> List[Tuple[List[int], List[int]]]:
    n_items = len(label_sets)
    indices = list(range(n_items))
    label_counts = compute_label_counts(label_sets)
    sorted_indices = sorted(indices, key=lambda index: sample_priority(index, label_sets, label_counts))

    fold_assignments: List[List[int]] = [[] for _ in range(n_splits)]
    fold_label_counts: List[Counter] = [Counter() for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    target_fold_size = n_items / n_splits

    for index in sorted_indices:
        labels = label_sets[index]
        fold_idx = choose_fold_for_sample(labels, fold_label_counts, fold_sizes, target_fold_size)
        fold_assignments[fold_idx].append(index)
        fold_sizes[fold_idx] += 1
        fold_label_counts[fold_idx].update(labels)

    folds: List[Tuple[List[int], List[int]]] = []
    all_indices = set(indices)
    for val_indices in fold_assignments:
        val_indices = sorted(val_indices)
        train_indices = sorted(all_indices - set(val_indices))
        folds.append((train_indices, val_indices))

    return folds


def print_fold_label_coverage(folds: Sequence[Tuple[Sequence[int], Sequence[int]]], label_sets: Sequence[set[str]]) -> None:
    all_labels = sorted({label for labels in label_sets for label in labels})
    print("\nFold label coverage:")
    for fold_no, (_, val_indices) in enumerate(folds, start=1):
        fold_labels = set()
        for index in val_indices:
            fold_labels.update(label_sets[index])
        missing = [label for label in all_labels if label not in fold_labels]
        print(
            f"  Fold {fold_no}: {len(fold_labels)}/{len(all_labels)} labels covered, "
            f"{len(missing)} missing"
        )


def run_multilabel_cross_validation(
    X_dev: Sequence[Sequence[Dict[str, Any]]],
    y_dev: Sequence[Sequence[str]],
    dev_samples: Sequence[SequenceSample],
    n_splits: int,
    crf_config: Dict[str, Any],
    n_jobs: int,
) -> List[Dict[str, Any]]:
    if len(X_dev) < 2:
        raise ValueError("Need at least two sequences for cross-validation.")

    actual_splits = min(n_splits, len(X_dev))
    if actual_splits < n_splits:
        print(f"Reducing n_splits from {n_splits} to {actual_splits} because the dataset is small.")

    label_sets = build_sample_label_sets(dev_samples)
    label_counts = compute_label_counts(label_sets)
    print_label_coverage_warnings(label_counts, actual_splits)

    folds = build_multilabel_stratified_folds(label_sets, actual_splits)
    print_fold_label_coverage(folds, label_sets)

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

