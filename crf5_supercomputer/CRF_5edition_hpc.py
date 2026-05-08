from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from crf5_sc_pipeline.crf5_runner_hpc import main


if __name__ == "__main__":
    main()
