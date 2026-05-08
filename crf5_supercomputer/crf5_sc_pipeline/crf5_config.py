from __future__ import annotations


PARQUET_PATH = "0000.parquet"
TEXT_COL = "source_text"
ANNOTATION_COL = "privacy"
RANDOM_STATE = 42

DEFAULT_MAX_ROWS = 30000
DEFAULT_DEV_MAX_ROWS = 100000
DEFAULT_TEST_SIZE = 0.1
DEFAULT_N_SPLITS = 9
DEFAULT_DEV_N_SPLITS = 3
DEFAULT_N_JOBS = -1
DEFAULT_MEMORY_ESTIMATE_SAMPLE_ROWS = 200
DEFAULT_MAX_ESTIMATED_BUILD_GB = 24.0

DEFAULT_CRF_CONFIG = {
    "algorithm": "lbfgs",
    "c1": 0.2,
    "c2": 0.1,
    "max_iterations": 150,
    "all_possible_transitions": False,
}

DEV_CRF_CONFIG = {
    "algorithm": "lbfgs",
    "c1": 0.2,
    "c2": 0.2,
    "max_iterations": 50,
    "all_possible_transitions": False,
}

