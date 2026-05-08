from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:  # pragma: no cover
    resource = None
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

DEFAULT_INNER_THREADS = os.getenv("CRF5_HPC_INNER_THREADS", "1")
for _env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_name, DEFAULT_INNER_THREADS)

SUPERCOMPUTER_ROOT = Path(__file__).resolve().parent.parent

from sklearn.model_selection import train_test_split

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "crf5_sc_pipeline"

from .crf5_config import (
    ANNOTATION_COL,
    DEFAULT_CRF_CONFIG,
    DEFAULT_DEV_MAX_ROWS,
    DEFAULT_DEV_N_SPLITS,
    DEFAULT_MAX_ESTIMATED_BUILD_GB,
    DEFAULT_MAX_ROWS,
    DEFAULT_MEMORY_ESTIMATE_SAMPLE_ROWS,
    DEFAULT_N_SPLITS,
    DEFAULT_TEST_SIZE,
    DEV_CRF_CONFIG,
    PARQUET_PATH,
    RANDOM_STATE,
    TEXT_COL,
)
from .crf5_dataset import build_crf_dataset, has_severe_alignment_issues, load_filtered_dataframe, select_dataframe_rows
from .crf5_metrics import evaluate_entities, evaluate_token_labels, summarize_cv_results
from .crf5_reporting import (
    print_dataframe_profile,
    print_entity_distribution,
    print_example_predictions,
    print_issue_summary,
    print_metrics_block,
    print_sample_quality,
)
from .crf5_training import estimate_dataset_build_bytes, format_bytes, save_model, train_crf_model
from .crf5_validation import run_multilabel_cross_validation


DEFAULT_SAMPLE_MODE = "random"
DEFAULT_OUTPUT_DIR = str(SUPERCOMPUTER_ROOT / "crf5_hpc_runs")
DEFAULT_MODEL_FILENAME = "crf_pii_model_v5.pkl"
DEFAULT_MAX_WORKERS = 16
DEFAULT_MEMORY_LIMIT_GB = 60.0


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def default_n_jobs() -> int:
    for env_name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.getenv(env_name)
        if not value:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return os.cpu_count() or 1


def resolve_max_workers(default_jobs: int) -> int:
    env_limit = env_int("CRF5_HPC_MAX_WORKERS", DEFAULT_MAX_WORKERS)
    return max(1, min(default_jobs, env_limit))


def resolve_memory_limit_gb() -> float:
    return env_float("CRF5_HPC_MEMORY_LIMIT_GB", DEFAULT_MEMORY_LIMIT_GB)


def apply_memory_limit(memory_limit_gb: float) -> str:
    if memory_limit_gb <= 0:
        return "disabled"
    if resource is None or not hasattr(resource, "RLIMIT_AS"):
        return "unsupported_on_this_platform"

    limit_bytes = int(memory_limit_gb * (1024 ** 3))
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    infinite_values = {-1}
    if hasattr(resource, "RLIM_INFINITY"):
        infinite_values.add(resource.RLIM_INFINITY)

    effective_limit = limit_bytes
    if hard_limit not in infinite_values:
        effective_limit = min(limit_bytes, hard_limit)

    resource.setrlimit(resource.RLIMIT_AS, (effective_limit, hard_limit))
    return f"active:{effective_limit}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HPC-oriented CRF_5 training workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet-path", default=PARQUET_PATH)
    parser.add_argument(
        "--model-output-path",
        default="",
        help="Optional explicit model path. If omitted, the model is saved inside the HPC output directory.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="", help="Optional run label. If omitted, a timestamp-based name is generated.")
    parser.add_argument("--dev-mode", action="store_true", help="Use faster development defaults.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use <=0 for all rows.")
    parser.add_argument(
        "--sample-mode",
        choices=["random", "head"],
        default=DEFAULT_SAMPLE_MODE,
        help="How max_rows is selected from the filtered parquet data.",
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed used when sample-mode=random.",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=default_n_jobs(),
        help="Worker processes for cross-validation on the current node.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=resolve_max_workers(default_n_jobs()),
        help="Hard cap on worker processes to avoid RAM pressure per node.",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=resolve_memory_limit_gb(),
        help="Hard per-process memory limit in GB on Linux. Use <=0 to disable.",
    )
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-final-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true", help="Stop after dataset diagnostics.")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--memory-estimate-sample-rows", type=int, default=DEFAULT_MEMORY_ESTIMATE_SAMPLE_ROWS)
    parser.add_argument(
        "--max-estimated-build-gb",
        type=float,
        default=min(DEFAULT_MAX_ESTIMATED_BUILD_GB, resolve_memory_limit_gb()),
    )
    parser.add_argument("--allow-large-memory-build", action="store_true")
    parser.add_argument("--drop-noisy-dev-samples", action="store_true")
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


def make_run_name(run_name: str) -> str:
    if run_name:
        return run_name
    return f"crf5_hpc_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def prepare_output_paths(output_dir: str, run_name: str, explicit_model_path: str) -> Dict[str, Path]:
    base_dir = Path(output_dir).resolve() / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    if explicit_model_path:
        model_path = Path(explicit_model_path).resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        model_path = base_dir / DEFAULT_MODEL_FILENAME

    return {
        "base_dir": base_dir,
        "model_path": model_path,
        "meta_json": base_dir / "crf5_training_run_meta.json",
    }


def save_metadata(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def filter_clean_subset(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
    samples: Sequence[Any],
) -> Tuple[list[Sequence[str]], list[Sequence[str]], list[Any], list[int]]:
    clean_indices = [index for index, sample in enumerate(samples) if not has_severe_alignment_issues(sample.issues)]
    return (
        [y_true[index] for index in clean_indices],
        [y_pred[index] for index in clean_indices],
        [samples[index] for index in clean_indices],
        clean_indices,
    )


def main() -> None:
    args = parse_args()
    memory_limit_status = apply_memory_limit(args.memory_limit_gb)
    max_rows, n_splits, crf_config = make_runtime_config(args)
    run_name = make_run_name(args.run_name)
    paths = prepare_output_paths(args.output_dir, run_name, args.model_output_path)
    effective_worker_cap = max(1, args.max_workers)
    requested_n_jobs = max(1, args.n_jobs)
    capped_n_jobs = min(requested_n_jobs, effective_worker_cap)

    print("=" * 80)
    print("CRF_5 HPC TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"run_name: {run_name}")
    print(f"output_dir: {paths['base_dir']}")
    print(f"model_output_path: {paths['model_path']}")
    print(f"dev_mode: {args.dev_mode}")
    print(f"max_rows: {max_rows if max_rows is not None else 'all'}")
    print(f"sample_mode: {args.sample_mode}")
    print(f"sample_random_state: {args.sample_random_state}")
    print(f"test_size: {args.test_size}")
    print(f"n_splits: {n_splits}")
    print(f"n_jobs_requested: {requested_n_jobs}")
    print(f"max_workers: {effective_worker_cap}")
    print(f"n_jobs_after_cap: {capped_n_jobs}")
    print(f"inner_threads_per_worker: {DEFAULT_INNER_THREADS}")
    print(f"memory_limit_gb: {args.memory_limit_gb}")
    print(f"memory_limit_status: {memory_limit_status}")
    print(f"run_cv: {not args.skip_cv}")
    print(f"train_final: {not args.skip_final_train}")
    print(f"run_test: {not args.skip_test}")
    print(f"save_model: {not args.no_save}")
    print(f"diagnostics_only: {args.diagnostics_only}")
    print(f"drop_noisy_dev_samples: {args.drop_noisy_dev_samples}")
    print(f"crf_config: {crf_config}")
    if capped_n_jobs < requested_n_jobs:
        print(
            f"Worker count capped from {requested_n_jobs} to {capped_n_jobs} "
            "to reduce RAM pressure and worker crashes on shared nodes."
        )

    full_df = load_filtered_dataframe(args.parquet_path)
    print(f"\nRows after filtering: {len(full_df)}")
    print_dataframe_profile("Full filtered dataframe", full_df)

    selected_df = select_dataframe_rows(
        full_df,
        max_rows=max_rows,
        sample_mode=args.sample_mode,
        sample_random_state=args.sample_random_state,
    )
    print_dataframe_profile("Selected training pool", selected_df)

    dev_df, test_df = train_test_split(
        selected_df,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("\nData split:")
    print(f"  Dev:  {len(dev_df)} ({len(dev_df) / len(selected_df):.2%})")
    print(f"  Test: {len(test_df)} ({len(test_df) / len(selected_df):.2%})")
    print_dataframe_profile("Dev split", dev_df)
    print_dataframe_profile("Test split", test_df)

    if args.diagnostics_only:
        save_metadata(
            paths["meta_json"],
            {
                "run_name": run_name,
                "args": vars(args),
                "max_rows_resolved": max_rows,
                "n_splits_resolved": n_splits,
                "crf_config": crf_config,
                "n_jobs_requested": requested_n_jobs,
                "max_workers": effective_worker_cap,
                "n_jobs_after_cap": capped_n_jobs,
                "inner_threads_per_worker": int(DEFAULT_INNER_THREADS),
                "memory_limit_gb": args.memory_limit_gb,
                "memory_limit_status": memory_limit_status,
                "status": "diagnostics_only",
            },
        )
        print("\nDiagnostics-only mode: stopping before dataset build and training.")
        print(f"Saved run metadata to: {paths['meta_json']}")
        return

    if args.memory_estimate_sample_rows > 0:
        print("\nEstimating RAM needed for dev dataset build...")
        estimate = estimate_dataset_build_bytes(
            dev_df,
            sample_rows=args.memory_estimate_sample_rows,
            skip_noisy_samples=args.drop_noisy_dev_samples,
        )
        estimated_total_gb = estimate["estimated_total_bytes"] / (1024 ** 3)
        print(f"  Sample rows: {int(estimate['sample_rows'])}")
        print(f"  Sample build size: {format_bytes(estimate['sample_total_bytes'])}")
        print(f"  Approx bytes per row: {format_bytes(estimate['bytes_per_row'])}")
        print(f"  Estimated full dev build size: {format_bytes(estimate['estimated_total_bytes'])}")

        if estimated_total_gb > args.max_estimated_build_gb and not args.allow_large_memory_build:
            print("\nError: estimated dev dataset build is too large for a safe in-memory run.")
            print(
                f"Estimated build size is about {estimated_total_gb:.1f} GB, which exceeds the "
                f"configured limit of {args.max_estimated_build_gb:.1f} GB."
            )
            print("This usually causes the process to stall or die during 'Building dev dataset once...'.")
            print("Try one of these:")
            print("  1. Use a smaller --max-rows value.")
            print("  2. Use --dev-mode for faster iteration.")
            print("  3. Raise --max-estimated-build-gb if you intentionally want to risk a very large run.")
            print("  4. Use --allow-large-memory-build to continue anyway.")
            sys.exit(1)

    print("\nBuilding dev dataset once for CV and final training...")
    X_dev, y_dev, dev_samples, dev_issues = build_crf_dataset(
        dev_df,
        description="Dev dataset",
        skip_noisy_samples=args.drop_noisy_dev_samples,
    )
    print(f"Dev sequences: {len(X_dev)}")
    print_issue_summary("Dev", dev_issues)
    print_entity_distribution(dev_samples)
    print_sample_quality("Dev", dev_samples)

    if not X_dev:
        print("Error: no usable dev sequences.")
        sys.exit(1)

    cv_summary = None
    if not args.skip_cv:
        print("\nStarting multilabel-aware cross-validation...")
        cv_results = run_multilabel_cross_validation(
            X_dev=X_dev,
            y_dev=y_dev,
            dev_samples=dev_samples,
            n_splits=n_splits,
            crf_config=crf_config,
            n_jobs=capped_n_jobs,
        )
        cv_summary = summarize_cv_results(cv_results)

        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)
        for key, value in cv_summary.items():
            print(f"{key}: {value}")

    if args.skip_final_train:
        save_metadata(
            paths["meta_json"],
            {
                "run_name": run_name,
                "args": vars(args),
                "max_rows_resolved": max_rows,
                "n_splits_resolved": n_splits,
                "crf_config": crf_config,
                "n_jobs_requested": requested_n_jobs,
                "max_workers": effective_worker_cap,
                "n_jobs_after_cap": capped_n_jobs,
                "inner_threads_per_worker": int(DEFAULT_INNER_THREADS),
                "memory_limit_gb": args.memory_limit_gb,
                "memory_limit_status": memory_limit_status,
                "cv_summary": cv_summary,
                "status": "skip_final_train",
            },
        )
        print("\nFinal training skipped.")
        print(f"Saved run metadata to: {paths['meta_json']}")
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
        "sample_mode": args.sample_mode,
        "sample_random_state": args.sample_random_state,
        "test_size": args.test_size,
        "n_splits": n_splits,
        "n_jobs_requested": requested_n_jobs,
        "max_workers": effective_worker_cap,
        "n_jobs_after_cap": capped_n_jobs,
        "inner_threads_per_worker": int(DEFAULT_INNER_THREADS),
        "memory_limit_gb": args.memory_limit_gb,
        "memory_limit_status": memory_limit_status,
        "crf_config": crf_config,
        "feature_version": "CRF_5edition_hpc_v1",
        "run_name": run_name,
    }

    if not args.no_save:
        save_model(final_model, str(paths["model_path"]), metadata)

    test_summary: Dict[str, Any] | None = None
    clean_test_summary: Dict[str, Any] | None = None
    if not args.skip_test:
        print("\nBuilding hold-out test dataset...")
        X_test, y_test, test_samples, test_issues = build_crf_dataset(test_df, description="Test dataset")
        print(f"Test sequences: {len(X_test)}")
        print_issue_summary("Test", test_issues)
        print_entity_distribution(test_samples)
        print_sample_quality("Test", test_samples)

        if X_test:
            print("\nRunning hold-out test...")
            y_test_pred = final_model.predict(X_test)
            token_metrics = evaluate_token_labels(y_test, y_test_pred)
            entity_metrics = evaluate_entities(test_samples, y_test_pred)
            print_metrics_block("HOLD-OUT TEST (ALL SAMPLES)", token_metrics, entity_metrics)
            print_example_predictions(test_samples, y_test, y_test_pred, args.examples)
            test_summary = {"token": token_metrics, "entity": entity_metrics}

            y_test_clean, y_pred_clean, clean_samples, clean_indices = filter_clean_subset(y_test, y_test_pred, test_samples)
            if clean_indices and len(clean_indices) < len(test_samples):
                print(
                    f"\nClean-only evaluation keeps {len(clean_indices)}/{len(test_samples)} "
                    "test samples without severe alignment issues."
                )
                clean_token_metrics = evaluate_token_labels(y_test_clean, y_pred_clean)
                clean_entity_metrics = evaluate_entities(clean_samples, y_pred_clean)
                print_metrics_block("HOLD-OUT TEST (CLEAN SAMPLES ONLY)", clean_token_metrics, clean_entity_metrics)
                clean_test_summary = {"token": clean_token_metrics, "entity": clean_entity_metrics}
            elif not clean_indices:
                print("\nClean-only evaluation skipped because every test sample has severe alignment issues.")
            else:
                print("\nClean-only evaluation skipped because all test samples are already clean.")

            del X_test, y_test, test_samples, y_test_pred
            gc.collect()
        else:
            print("Error: no usable test sequences.")
    else:
        print("\nHold-out test skipped.")

    save_metadata(
        paths["meta_json"],
        {
            "run_name": run_name,
            "args": vars(args),
            "max_rows_resolved": max_rows,
            "n_splits_resolved": n_splits,
            "crf_config": crf_config,
            "n_jobs_requested": requested_n_jobs,
            "max_workers": effective_worker_cap,
            "n_jobs_after_cap": capped_n_jobs,
            "inner_threads_per_worker": int(DEFAULT_INNER_THREADS),
            "memory_limit_gb": args.memory_limit_gb,
            "memory_limit_status": memory_limit_status,
            "model_output_path": str(paths["model_path"]),
            "saved_model": not args.no_save,
            "cv_summary": cv_summary,
            "test_summary": test_summary,
            "clean_test_summary": clean_test_summary,
            "status": "completed",
        },
    )
    print(f"Saved run metadata to: {paths['meta_json']}")

    del final_model, X_dev, y_dev, dev_samples, full_df, selected_df
    gc.collect()


if __name__ == "__main__":
    main()
