"""Run a script once per dataset, each in a child process with the BLAS thread count in configs/threads.json.

The thread count changes the floating-point results of some fits, so the count must be set before numpy is imported.
This module does not import numpy.
"""

import json
import os
import subprocess
import sys

CHILD = "TEA_CHILD"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def threads(dataset):
    """Return the BLAS thread count of a dataset from configs/threads.json."""
    with open(os.path.join(ROOT, "configs", "threads.json")) as f:
        return int(json.load(f)[dataset])


def is_child():
    """Return whether this process was started by run_per_dataset."""
    return os.environ.get(CHILD) == "1"


def run_per_dataset(script, datasets, args=()):
    """Run the script once per dataset, with the dataset as the first argument, and return the exit codes."""
    codes = {}
    for ds in datasets:
        n = str(threads(ds))
        env = dict(os.environ, **{CHILD: "1", "OMP_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
                                  "VECLIB_MAXIMUM_THREADS": n, "MKL_NUM_THREADS": n})
        codes[ds] = subprocess.run([sys.executable, script, ds, *args], env=env).returncode
    return codes
