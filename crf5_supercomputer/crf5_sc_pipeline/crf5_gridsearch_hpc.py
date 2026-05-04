from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:  # pragma: no cover
    resource = None
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_INNER_THREADS = os.getenv("CRF5_HPC_INNER_THREADS", "1")
for _env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_name, DEFAULT_INNER_THREADS)

SUPERCOMPUTER_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import sklearn_crfsuite
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "crf5_sc_pipeline"

from .crf5_config import DEFAULT_TEST_SIZE, PARQUET_PATH, RANDOM_STATE
from .crf5_dataset import build_crf_dataset, load_filtered_dataframe, select_dataframe_rows
from .crf5_metrics import evaluate_entities, evaluate_token_labels
from .crf5_reporting import print_issue_summary
from .crf5_training import resolve_n_jobs, select_items
from .crf5_validation import (
    build_multilabel_stratified_folds,
    build_sample_label_sets,
    compute_label_counts,
    print_fold_label_coverage,
    print_label_coverage_warnings,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# =========================================================
# HPC grid search configuration
# =========================================================
C1_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]
C2_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]
MAX_ITERATION_VALUES = [50, 100, 150]
ALL_POSSIBLE_TRANSITIONS_VALUES = [True, False]

DEFAULT_MAX_ROWS = 2662
DEFAULT_SAMPLE_MODE = "random"
DEFAULT_FINAL_FOLDS = 3
DEFAULT_STAGE_FOLDS = [1, 2, 3]
DEFAULT_STAGE_TOP_K = [20, 6, 2]
DEFAULT_PRIMARY_METRIC = "entity_f1"
DEFAULT_OUTPUT_DIR = str(SUPERCOMPUTER_ROOT / "crf5_hpc_runs")
DEFAULT_PARALLEL_BACKEND = "loky"
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


def default_num_chunks() -> int:
    return max(1, env_int("SLURM_ARRAY_TASK_COUNT", 1))


def default_chunk_id() -> int:
    return max(0, env_int("SLURM_ARRAY_TASK_ID", 0))


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
        description="HPC-oriented grid search for CRF_5edition hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet-path", default=PARQUET_PATH)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Rows loaded before dev/test split.")
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
    parser.add_argument("--final-folds", type=int, default=DEFAULT_FINAL_FOLDS, help="Folds for final stage.")
    parser.add_argument(
        "--stage-folds",
        default="1,2,3",
        help="Comma-separated folds used in each search stage.",
    )
    parser.add_argument(
        "--stage-top-k",
        default="20,6,2",
        help="Comma-separated survivor counts after each stage.",
    )
    parser.add_argument(
        "--primary-metric",
        choices=["entity_f1", "token_macro_f1"],
        default=DEFAULT_PRIMARY_METRIC,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=default_n_jobs(),
        help="Worker processes for config evaluation on the current node.",
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
    parser.add_argument(
        "--parallel-backend",
        choices=["threading", "loky"],
        default=DEFAULT_PARALLEL_BACKEND,
        help="Use loky on HPC unless you have a specific reason not to.",
    )
    parser.add_argument(
        "--drop-noisy-dev-samples",
        action="store_true",
        help="Drop dev samples with severe alignment issues before grid search.",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=default_chunk_id(),
        help="Chunk index for job-array execution.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=default_num_chunks(),
        help="Total chunk count for job-array execution.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where run-specific output files are written.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run label. If omitted, a timestamp-based name is generated.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_param_grid() -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []
    for c1, c2, max_iterations, transitions in itertools.product(
        C1_VALUES,
        C2_VALUES,
        MAX_ITERATION_VALUES,
        ALL_POSSIBLE_TRANSITIONS_VALUES,
    ):
        grid.append({
            "algorithm": "lbfgs",
            "c1": c1,
            "c2": c2,
            "max_iterations": max_iterations,
            "all_possible_transitions": transitions,
        })
    return grid


def select_chunk(configs: Sequence[Dict[str, Any]], chunk_id: int, num_chunks: int) -> List[Dict[str, Any]]:
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1.")
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError("chunk_id must be between 0 and num_chunks-1.")
    return [config for index, config in enumerate(configs) if index % num_chunks == chunk_id]


def config_id(config: Dict[str, Any]) -> str:
    return (
        f"c1={config['c1']}_"
        f"c2={config['c2']}_"
        f"iters={config['max_iterations']}_"
        f"transitions={config['all_possible_transitions']}"
    )


def train_crf_silent(
    X_train: Sequence[Sequence[Dict[str, Any]]],
    y_train: Sequence[Sequence[str]],
    crf_config: Dict[str, Any],
) -> sklearn_crfsuite.CRF:
    model = sklearn_crfsuite.CRF(**crf_config)
    model.fit(X_train, y_train)
    return model


def evaluate_config_on_folds(
    config: Dict[str, Any],
    fold_indices: Sequence[int],
    folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
    X_dev: Sequence[Sequence[Dict[str, Any]]],
    y_dev: Sequence[Sequence[str]],
    dev_samples: Sequence[Any],
) -> Dict[str, Any]:
    token_macro_f1s: List[float] = []
    entity_f1s: List[float] = []
    fold_times: List[float] = []

    for fold_idx in fold_indices:
        train_idx, val_idx = folds[fold_idx]
        X_train = select_items(X_dev, train_idx)
        y_train = select_items(y_dev, train_idx)
        X_val = select_items(X_dev, val_idx)
        y_val = select_items(y_dev, val_idx)
        val_samples = select_items(dev_samples, val_idx)

        fold_start = time.time()
        model = train_crf_silent(X_train, y_train, config)
        y_val_pred = model.predict(X_val)
        fold_elapsed = time.time() - fold_start

        token_metrics = evaluate_token_labels(y_val, y_val_pred)
        entity_metrics = evaluate_entities(val_samples, y_val_pred)

        token_macro_f1s.append(token_metrics["macro"]["f1"])
        entity_f1s.append(entity_metrics["f1"])
        fold_times.append(fold_elapsed)

    return {
        "config": dict(config),
        "config_id": config_id(config),
        "folds_evaluated": len(fold_indices),
        "token_macro_f1_mean": sum(token_macro_f1s) / len(token_macro_f1s),
        "entity_f1_mean": sum(entity_f1s) / len(entity_f1s),
        "runtime_seconds": sum(fold_times),
    }


def evaluate_stage_configs(
    stage_no: int,
    configs: Sequence[Dict[str, Any]],
    fold_indices: Sequence[int],
    folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
    X_dev: Sequence[Sequence[Dict[str, Any]]],
    y_dev: Sequence[Sequence[str]],
    dev_samples: Sequence[Any],
    n_jobs: int,
    max_workers: int,
    parallel_backend: str,
) -> List[Dict[str, Any]]:
    resolved_n_jobs = min(resolve_n_jobs(n_jobs, len(configs)), max_workers)

    if resolved_n_jobs == 1:
        iterator: Iterable[Dict[str, Any]] = configs
        if tqdm is not None:
            iterator = tqdm(configs, desc=f"Stage {stage_no}", file=sys.stdout)

        results: List[Dict[str, Any]] = []
        for config in iterator:
            result = evaluate_config_on_folds(
                config=config,
                fold_indices=fold_indices,
                folds=folds,
                X_dev=X_dev,
                y_dev=y_dev,
                dev_samples=dev_samples,
            )
            result["stage"] = stage_no
            results.append(result)
        return results

    print(
        f"Running stage {stage_no} in parallel with {resolved_n_jobs} worker(s) "
        f"using backend='{parallel_backend}'."
    )

    if parallel_backend == "threading":
        futures = []
        stage_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=resolved_n_jobs) as executor:
            for config in configs:
                futures.append(
                    executor.submit(
                        evaluate_config_on_folds,
                        config=config,
                        fold_indices=fold_indices,
                        folds=folds,
                        X_dev=X_dev,
                        y_dev=y_dev,
                        dev_samples=dev_samples,
                    )
                )

            completed = as_completed(futures)
            if tqdm is not None:
                completed = tqdm(completed, total=len(futures), desc=f"Stage {stage_no}", file=sys.stdout)

            for future in completed:
                result = future.result()
                result["stage"] = stage_no
                stage_results.append(result)
        return stage_results

    stage_results = Parallel(
        n_jobs=resolved_n_jobs,
        backend=parallel_backend,
        batch_size=1,
        pre_dispatch=resolved_n_jobs,
    )(
        delayed(evaluate_config_on_folds)(
            config=config,
            fold_indices=fold_indices,
            folds=folds,
            X_dev=X_dev,
            y_dev=y_dev,
            dev_samples=dev_samples,
        )
        for config in configs
    )

    for result in stage_results:
        result["stage"] = stage_no
    return stage_results


def ranking_key(result: Dict[str, Any], primary_metric: str) -> Tuple[float, float, float]:
    if primary_metric == "entity_f1":
        primary = result["entity_f1_mean"]
        secondary = result["token_macro_f1_mean"]
    else:
        primary = result["token_macro_f1_mean"]
        secondary = result["entity_f1_mean"]
    return (primary, secondary, -result["runtime_seconds"])


def stage_summary(stage_no: int, folds_used: int, top_k: int, results: Sequence[Dict[str, Any]], primary_metric: str) -> None:
    print("\n" + "=" * 80)
    print(f"STAGE {stage_no}")
    print("=" * 80)
    print(f"Folds used: {folds_used}")
    print(f"Survivors after stage: {top_k}")
    print("Top 5 configurations:")

    ranked = sorted(results, key=lambda result: ranking_key(result, primary_metric), reverse=True)
    for result in ranked[:5]:
        print(
            f"  {result['config_id']} | "
            f"entity_f1={result['entity_f1_mean']:.4f} | "
            f"token_macro_f1={result['token_macro_f1_mean']:.4f} | "
            f"runtime={result['runtime_seconds']:.2f}s"
        )


def save_results(results: Sequence[Dict[str, Any]], results_csv: Path, best_json: Path, primary_metric: str) -> None:
    rows: List[Dict[str, Any]] = []
    for result in results:
        row = {
            "stage": result["stage"],
            "config_id": result["config_id"],
            "folds_evaluated": result["folds_evaluated"],
            "token_macro_f1_mean": result["token_macro_f1_mean"],
            "entity_f1_mean": result["entity_f1_mean"],
            "runtime_seconds": result["runtime_seconds"],
        }
        row.update(result["config"])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        by=[primary_metric.replace("_f1", "_f1_mean"), "runtime_seconds"],
        ascending=[False, True],
    )
    df.to_csv(results_csv, index=False)

    best_row = df.iloc[0].to_dict()
    with open(best_json, "w", encoding="utf-8") as handle:
        json.dump(best_row, handle, indent=2)

    print(f"\nSaved all stage results to: {results_csv}")
    print(f"Saved best configuration to: {best_json}")


def make_run_name(run_name: str) -> str:
    if run_name:
        return run_name
    return f"crf5_hpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def prepare_output_paths(output_dir: str, run_name: str, chunk_id: int) -> Dict[str, Path]:
    base_dir = Path(output_dir).resolve() / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": base_dir,
        "results_csv": base_dir / f"crf5_gridsearch_chunk_{chunk_id}_results.csv",
        "best_json": base_dir / f"crf5_gridsearch_chunk_{chunk_id}_best.json",
        "meta_json": base_dir / f"crf5_gridsearch_chunk_{chunk_id}_meta.json",
    }


def save_metadata(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    memory_limit_status = apply_memory_limit(args.memory_limit_gb)
    if args.max_rows is not None and args.max_rows <= 0:
        args.max_rows = None
    stage_folds = parse_int_list(args.stage_folds)
    stage_top_k = parse_int_list(args.stage_top_k)

    if len(stage_folds) != len(stage_top_k):
        raise ValueError("stage-folds and stage-top-k must have the same length.")

    if stage_folds[-1] != args.final_folds:
        raise ValueError("The last stage-folds value must match --final-folds.")

    run_name = make_run_name(args.run_name)
    paths = prepare_output_paths(args.output_dir, run_name, args.chunk_id)
    effective_worker_cap = max(1, args.max_workers)
    requested_n_jobs = max(1, args.n_jobs)
    capped_n_jobs = min(requested_n_jobs, effective_worker_cap)

    full_grid = build_param_grid()
    current_pool = select_chunk(full_grid, chunk_id=args.chunk_id, num_chunks=args.num_chunks)
    if not current_pool:
        raise ValueError("This chunk contains no configurations.")

    print("=" * 80)
    print("CRF_5 HPC GRID SEARCH CONFIGURATION")
    print("=" * 80)
    print(f"max_rows: {args.max_rows}")
    print(f"sample_mode: {args.sample_mode}")
    print(f"sample_random_state: {args.sample_random_state}")
    print(f"test_size: {args.test_size}")
    print(f"final_folds: {args.final_folds}")
    print(f"stage_folds: {stage_folds}")
    print(f"stage_top_k: {stage_top_k}")
    print(f"primary_metric: {args.primary_metric}")
    print(f"n_jobs_requested: {requested_n_jobs}")
    print(f"max_workers: {effective_worker_cap}")
    print(f"n_jobs_after_cap: {capped_n_jobs}")
    print(f"inner_threads_per_worker: {DEFAULT_INNER_THREADS}")
    print(f"memory_limit_gb: {args.memory_limit_gb}")
    print(f"memory_limit_status: {memory_limit_status}")
    print(f"parallel_backend: {args.parallel_backend}")
    print(f"drop_noisy_dev_samples: {args.drop_noisy_dev_samples}")
    print(f"chunk_id: {args.chunk_id}")
    print(f"num_chunks: {args.num_chunks}")
    print(f"chunk_grid_size: {len(current_pool)}")
    print(f"output_dir: {paths['base_dir']}")
    if capped_n_jobs < requested_n_jobs:
        print(
            f"Worker count capped from {requested_n_jobs} to {capped_n_jobs} "
            "to reduce RAM pressure and worker crashes on shared nodes."
        )

    full_df = load_filtered_dataframe(args.parquet_path)
    selected_df = select_dataframe_rows(
        full_df,
        max_rows=args.max_rows,
        sample_mode=args.sample_mode,
        sample_random_state=args.sample_random_state,
    )
    dev_df, _ = train_test_split(
        selected_df,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    dev_df = dev_df.reset_index(drop=True)

    print(f"\nRows after filtering: {len(full_df)}")
    print(f"Rows selected for search pool: {len(selected_df)}")
    print(f"Dev rows used for search: {len(dev_df)}")

    X_dev, y_dev, dev_samples, dev_issues = build_crf_dataset(
        dev_df,
        description="CRF_5 HPC gridsearch dev dataset",
        skip_noisy_samples=args.drop_noisy_dev_samples,
    )
    print(f"Dev sequences built: {len(X_dev)}")
    print_issue_summary("HPC gridsearch dev", dev_issues, limit=10)

    if len(X_dev) < args.final_folds:
        raise ValueError("Not enough sequences for the requested number of folds.")

    label_sets = build_sample_label_sets(dev_samples)
    label_counts = compute_label_counts(label_sets)
    print_label_coverage_warnings(label_counts, args.final_folds)

    folds = build_multilabel_stratified_folds(label_sets, n_splits=args.final_folds)
    print_fold_label_coverage(folds, label_sets)

    all_results: List[Dict[str, Any]] = []
    started = time.time()

    for stage_no, (fold_count, survivor_count) in enumerate(zip(stage_folds, stage_top_k), start=1):
        print("\n" + "-" * 80)
        print(f"Evaluating stage {stage_no}: {len(current_pool)} configs, {fold_count} fold(s)")
        print("-" * 80)

        fold_indices = list(range(fold_count))
        stage_results = evaluate_stage_configs(
            stage_no=stage_no,
            configs=current_pool,
            fold_indices=fold_indices,
            folds=folds,
            X_dev=X_dev,
            y_dev=y_dev,
            dev_samples=dev_samples,
            n_jobs=capped_n_jobs,
            max_workers=effective_worker_cap,
            parallel_backend=args.parallel_backend,
        )

        all_results.extend(stage_results)
        ranked = sorted(stage_results, key=lambda result: ranking_key(result, args.primary_metric), reverse=True)
        current_pool = [result["config"] for result in ranked[:survivor_count]]
        stage_summary(stage_no, fold_count, survivor_count, stage_results, args.primary_metric)

    final_stage = [result for result in all_results if result["stage"] == len(stage_folds)]
    final_ranked = sorted(final_stage, key=lambda result: ranking_key(result, args.primary_metric), reverse=True)
    best_result = final_ranked[0]

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION IN CHUNK")
    print("=" * 80)
    print(best_result["config_id"])
    print(f"entity_f1_mean: {best_result['entity_f1_mean']:.4f}")
    print(f"token_macro_f1_mean: {best_result['token_macro_f1_mean']:.4f}")
    print(f"runtime_seconds: {best_result['runtime_seconds']:.2f}")
    print(f"total_search_time_seconds: {time.time() - started:.2f}")

    save_results(all_results, paths["results_csv"], paths["best_json"], args.primary_metric)
    save_metadata(
        paths["meta_json"],
        {
            "run_name": run_name,
            "chunk_id": args.chunk_id,
            "num_chunks": args.num_chunks,
            "chunk_grid_size": len(select_chunk(full_grid, args.chunk_id, args.num_chunks)),
            "args": vars(args),
            "inner_threads_per_worker": int(DEFAULT_INNER_THREADS),
            "n_jobs_requested": requested_n_jobs,
            "max_workers": effective_worker_cap,
            "n_jobs_after_cap": capped_n_jobs,
            "memory_limit_gb": args.memory_limit_gb,
            "memory_limit_status": memory_limit_status,
            "best_result": best_result,
            "total_search_time_seconds": time.time() - started,
        },
    )
    print(f"Saved chunk metadata to: {paths['meta_json']}")


if __name__ == "__main__":
    main()
