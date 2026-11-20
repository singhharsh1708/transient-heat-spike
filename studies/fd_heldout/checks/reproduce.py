"""Run the original case generator and compare every field to the archived CSV in order."""
import csv
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter

study = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(study))
import sweep


class Compare:
    def __init__(self, rows):
        self.rows = rows
        self.n = 0
        self.differences = Counter()
        self.max_abs = Counter()

    def writerow(self, row):
        expected = next(self.rows)
        assert expected.keys() == row.keys()
        self.n += 1
        for k, v in row.items():
            if str(v) == expected[k]:
                continue
            self.differences[k] += 1
            try:
                self.max_abs[k] = max(self.max_abs[k], abs(float(v) - float(expected[k])))
            except ValueError:
                print("categorical mismatch", self.n, k, v, expected[k], flush=True)


for filename, fd_tol, tolerances in [
    ("results.csv", None, sweep.RTOLS),
    ("results_fd_tight.csv", 1e-12, sweep.RTOLS[:2]),
]:
    if len(sys.argv) > 1 and filename not in sys.argv[1:]:
        continue
    start = perf_counter()
    with (study / filename).open(newline="") as f:
        check = Compare(csv.DictReader(f))
        for (nx, ny), field, (steps, frac), rtol in product(
            ((12, 6), (16, 8)), sweep.FIELDS, sweep.SCHEDULES, tolerances
        ):
            sweep.run_case(nx, ny, field, steps, frac, rtol, check, fd_tol)
        assert next(check.rows, None) is None
    print(filename, "rows", check.n, "seconds", round(perf_counter() - start, 1),
          "different fields", dict(check.differences), "max abs", dict(check.max_abs), flush=True)
