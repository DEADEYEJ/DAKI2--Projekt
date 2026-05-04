from __future__ import annotations

import gc
import os
import pickle
import sys
import threading
import time
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import sklearn_crfsuite

from .crf5_dataset import SequenceSample, build_crf_dataset
from .crf5_metrics import evaluate_entities, evaluate_token_labels


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


def train_crf_silent(
    X_train: Sequence[Sequence[Dict[str, Any]]],
    y_train: Sequence[Sequence[str]],
    crf_config: Dict[str, Any],
) -> sklearn_crfsuite.CRF:
    model = sklearn_crfsuite.CRF(**crf_config)
    model.fit(X_train, y_train)
    return model


def select_items(items: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [items[index] for index in indices]


def deep_getsizeof(obj: Any, seen: Optional[set[int]] = None) -> int:
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(deep_getsizeof(key, seen) + deep_getsizeof(value, seen) for key, value in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_getsizeof(item, seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += deep_getsizeof(vars(obj), seen)
    elif hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                size += deep_getsizeof(getattr(obj, slot), seen)

    return size


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit_index = 0

    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1

    return f"{value:.2f} {units[unit_index]}"


def estimate_dataset_build_bytes(
    dataframe: pd.DataFrame,
    sample_rows: int,
    skip_noisy_samples: bool = False,
) -> Dict[str, float]:
    if dataframe.empty:
        return {
            "sample_rows": 0.0,
            "sample_total_bytes": 0.0,
            "bytes_per_row": 0.0,
            "estimated_total_bytes": 0.0,
        }

    actual_sample_rows = min(sample_rows, len(dataframe))
    sample_df = dataframe.head(actual_sample_rows).copy()
    X_sample, y_sample, samples_sample, _ = build_crf_dataset(
        sample_df,
        description=f"Memory estimate ({actual_sample_rows} rows)",
        skip_noisy_samples=skip_noisy_samples,
    )
    sample_total_bytes = (
        deep_getsizeof(X_sample)
        + deep_getsizeof(y_sample)
        + deep_getsizeof(samples_sample)
    )
    bytes_per_row = sample_total_bytes / max(actual_sample_rows, 1)
    estimated_total_bytes = bytes_per_row * len(dataframe)

    del sample_df, X_sample, y_sample, samples_sample
    gc.collect()

    return {
        "sample_rows": float(actual_sample_rows),
        "sample_total_bytes": float(sample_total_bytes),
        "bytes_per_row": float(bytes_per_row),
        "estimated_total_bytes": float(estimated_total_bytes),
    }


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


def resolve_n_jobs(n_jobs: int, task_count: int) -> int:
    if task_count <= 0:
        return 1

    cpu_count = os.cpu_count() or 1
    if n_jobs == -1:
        return max(1, min(cpu_count, task_count))
    return max(1, min(n_jobs, task_count))


def save_model(model: sklearn_crfsuite.CRF, output_path: str, metadata: Dict[str, Any]) -> None:
    setattr(model, "pii_metadata_", metadata)
    with open(output_path, "wb") as file:
        pickle.dump(model, file)
    print(f"\nModel saved to: {output_path}")

