"""Compare the outputs of the scripts with the files in the reference folder.

Run the scripts of the README first. Each check prints one line, which starts with ok, FAIL, or not run.
The printed rows of Tables 1, 2, and 3 must equal the rows of the paper.
The splits and the per-user NDCG@10 and recall@10 values are compared bit for bit, and TQ is compared exactly.
The data of Figures 1 and 2, the effective numbers of users, and the perplexity curves of K are compared within a
tolerance. The values that the text of the paper quotes must satisfy the quoted bounds.
The script exits with an error if any check fails. A check whose results are missing is reported as not run.
Usage: python scripts/verify.py
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from tea import experiment, render, registry
from tea.data import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, "reference")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
OK, FAIL, NOT_RUN = "ok", "FAIL", "not run"


def sha(array):
    """Return the SHA-256 digest of the bytes of an array."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def reference(name):
    """Load a JSON file from the reference folder."""
    with open(os.path.join(REFERENCE, name)) as f:
        return json.load(f)


def missing_cells():
    """Return the cells of Table 1 whose stored results are missing."""
    return [f"{ds} {m.name}" for ds in render.DATASETS for m in registry.METHODS
            if not os.path.exists(experiment.result_path(os.path.join(RESULTS, "cells"), ds, m.name)[:-4] + ".json")]


def missing_splits():
    """Return the datasets whose split is missing."""
    return [ds for ds in render.DATASETS if not os.path.exists(build.split_path(DATA, ds))]


def run_script(name, *args):
    """Run a script and return its exit code, its standard output, and the last line of its error output."""
    done = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", name), *args], capture_output=True, text=True)
    errors = done.stderr.strip().splitlines()
    return done.returncode, done.stdout, errors[-1] if errors else ""


def check_rows(table):
    """Compare the rows that a table script prints with the rows of the paper."""
    if table == "table1" and missing_cells():
        return NOT_RUN, "run scripts/fit.py first"
    if table == "table2" and missing_splits():
        return NOT_RUN, "run scripts/prepare_data.py first"
    code, out, error = run_script(f"{table}.py")
    if code:
        return FAIL, error
    with open(os.path.join(REFERENCE, f"{table}.tex")) as f:
        return (OK, "rows equal to the paper") if out == f.read() else (FAIL, "rows different from the paper")


def check_splits():
    """Compare the digest of every split array with reference/splits.json."""
    if missing_splits():
        return NOT_RUN, "run scripts/prepare_data.py first"
    ref = reference("splits.json")
    same = all(sha(build.load(ds, DATA)[k]) == digest for ds, arrays in ref.items() for k, digest in arrays.items())
    return (OK, "bitwise equal") if same else (FAIL, "not bitwise equal")


def check_cells():
    """Compare the per-user arrays and TQ of every cell with reference/cells.json."""
    if missing_cells():
        return NOT_RUN, "run scripts/fit.py first"
    ref = reference("cells.json")
    different = []
    for ds, methods in ref.items():
        for method, values in methods.items():
            r = experiment.load(os.path.join(RESULTS, "cells"), ds, method)
            same = sha(r["ndcg"]) == values["ndcg"] and sha(r["recall"]) == values["recall"]
            if "tq" in values:
                same = same and float(r["tq"]) == values["tq"]
            if not same:
                different.append(f"{ds} {method}")
    total = sum(len(m) for m in ref.values())
    message = f"{total - len(different)} of {total} bitwise equal" + (f", different: {', '.join(different)}" if different else "")
    return (OK if not different else FAIL), message


def _numbers(value):
    if isinstance(value, dict):
        return [x for k in sorted(value) for x in _numbers(value[k])]
    if isinstance(value, list):
        return [x for v in value for x in _numbers(v)]
    return [value] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def check_close(result, name, tolerance, script):
    """Compare every number of a stored result with its reference within a relative tolerance."""
    path = os.path.join(RESULTS, result)
    if not os.path.exists(path):
        return NOT_RUN, f"run {script} first"
    with open(path) as f:
        a = np.array(_numbers(json.load(f)), dtype=float)
    b = np.array(_numbers(reference(name)), dtype=float)
    if a.shape != b.shape:
        return FAIL, f"computed at other points than the reference, so run {script} with its defaults"
    d = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if a.size else 0.0
    return (OK if d <= tolerance else FAIL), f"largest relative difference {d:.1e} (tolerance {tolerance:g})"


def check_k(tolerance=1e-12):
    """Compare the selected K and the perplexity curves with reference/k.json."""
    K, ref = experiment.config("k"), reference("k.json")
    if not all(os.path.exists(os.path.join(RESULTS, "k", f"{ds}.json")) for ds in render.DATASETS):
        return NOT_RUN, "run scripts/select_k.py first"
    worst, same = 0.0, True
    for ds in render.DATASETS:
        with open(os.path.join(RESULTS, "k", f"{ds}.json")) as f:
            new = json.load(f)
        same &= new["K"] == K[ds] == ref[ds]["K"]
        for row_new, row_old in zip(new["per_build_set"], ref[ds]["per_build_set"]):
            for key, value in row_old.items():
                worst = max(worst, abs(row_new[key] - value) / value)
    ok = same and worst <= tolerance
    return (OK if ok else FAIL), (f"K {'equal to' if same else 'different from'} the paper, perplexity curves within "
                                  f"{worst:.1e} (relative tolerance {tolerance:g})")


def check_figure2(tolerance=1e-6):
    """Compare the search of Figure 2 with reference/figure2.json, using the full search where it exists."""
    ref = reference("figure2.json")
    worst, same, kinds = 0.0, True, set()
    for ds in render.DATASETS:
        paths = [os.path.join(RESULTS, "search", f"{ds}_TEA{suffix}.json") for suffix in ("", "_subset")]
        path = next((p for p in paths if os.path.exists(p)), None)
        if path is None:
            return NOT_RUN, "run scripts/figure2.py first"
        kinds.add("subset" if path.endswith("_subset.json") else "full search")
        with open(path) as f:
            new = json.load(f)
        stored = {json.dumps(c["params"], sort_keys=True): c["ndcg"] for c in ref[ds]["curve"]}
        for c in new["curve"]:
            worst = max(worst, abs(c["ndcg"] - stored[json.dumps(c["params"], sort_keys=True)]))
        same &= new["selected"] == ref[ds]["selected"]
    ok = same and worst <= tolerance
    return (OK if ok else FAIL), (f"{' and '.join(sorted(kinds))}: {'the same' if same else 'a different'} selected "
                                  f"setting, largest difference in validation NDCG@10 {worst:.1e} "
                                  f"(tolerance {tolerance:g})")


def check_text_numbers():
    """Check the values that scripts/text_numbers.py computes against the bounds in reference/text_numbers.json."""
    if missing_cells():
        return NOT_RUN, "run scripts/fit.py first"
    if not all(os.path.exists(os.path.join(RESULTS, "search", f"{ds}_TEA.json")) for ds in ("ml-1m", "steam")):
        return NOT_RUN, "run scripts/figure2.py first"
    code, out, error = run_script("text_numbers.py", "--json")
    if code:
        return FAIL, error
    values, bounds = json.loads(out), reference("text_numbers.json")
    wrong = []
    for key, bound in bounds.items():
        if "at_most" in bound and not values[key] <= bound["at_most"]:
            wrong.append(f"{key} {values[key]:.4f} above {bound['at_most']}")
        if "percent" in bound and round(100 * values[key]) != bound["percent"]:
            wrong.append(f"{key} {values[key]:.1%} not {bound['percent']}%")
    return (OK, "the quoted values hold") if not wrong else (FAIL, ", ".join(wrong))


def main():
    """Run every check, print one line per check as it finishes, and exit with an error if any check fails."""
    checks = [("table1", lambda: check_rows("table1")), ("table2", lambda: check_rows("table2")),
              ("table3", lambda: check_rows("table3")), ("splits", check_splits), ("cells", check_cells),
              ("figure1", lambda: check_close("figure1.json", "figure1.json", 1e-12, "scripts/figure1.py")),
              ("figure2", check_figure2),
              ("neff", lambda: check_close("neff.json", "neff.json", 1e-12, "scripts/neff.py")),
              ("K", check_k), ("text numbers", check_text_numbers)]
    failed = False
    for name, check in checks:
        status, message = check()
        failed |= status == FAIL
        print(f"{status:7s} {name}: {message}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
