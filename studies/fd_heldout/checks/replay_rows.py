"""CI form of reproduce.py for two cases: rerun them and compare with the archived rows of results.csv by tolerance.

Catches a change to the spike (physics, mesh, element matrices) that alters the studies' numbers, which the gradient
self-consistency check cannot see. Counts and categorical fields must match exactly; floats to a relative 1e-6
(absolute 1e-8; FD medians, which carry
finite-difference noise that differs between machines by about 1e-8, absolute 1e-6, far below the 1e-2 pass threshold).
"""
import csv
import math
import sys
from pathlib import Path

study = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(study))
import sweep  # noqa: E402

CASES = [(12, 6, 100, 1, 0.01, 1e-3), (16, 8, 102, 10, 0.1, 1e-4)]  # (nx, ny, field, steps, dt_frac, rtol)
archive = list(csv.DictReader((study / "results.csv").open()))
bad = 0
for nx, ny, field, steps, frac, rtol in CASES:
    want = [r for r in archive if (r["nx"], r["field"], r["steps"], r["dt_frac"], r["rtol"])
            == (str(nx), str(field), str(steps), str(frac), str(rtol))]
    got = []

    class Rows:
        def writerow(self, row):
            got.append(row)
    sweep.run_case(nx, ny, field, steps, frac, rtol, Rows())
    assert len(got) == len(want) > 0, (len(got), len(want))
    for g, w in zip(got, want):
        for k, v in g.items():
            if isinstance(v, float):
                ok = math.isclose(v, float(w[k]), rel_tol=1e-6, abs_tol=1e-6 if k.endswith("_med") else 1e-8)
            else:
                ok = str(v) == w[k]
            if not ok:
                bad += 1
                print("differs", (nx, field, steps, frac, rtol), k, v, w[k])
    print(f"case {(nx, field, steps, frac, rtol)}: {len(got)} rows compared")
assert bad == 0, f"{bad} fields differ from the archive"
