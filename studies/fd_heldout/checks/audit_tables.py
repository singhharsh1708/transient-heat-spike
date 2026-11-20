"""Independently score both archived tables and count paired inversions."""
import csv
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path

study = Path(__file__).resolve().parent.parent
names = ("unrolled", "implicit", "reference")
meshes = ((12, 6), (16, 8))
schedules = ((1, 1.0), (1, .01), (10, .1))
eps_grid = {.1, .01, .001, .0001}


def audit(filename, rtols):
    with (study / filename).open(newline="") as f:
        records = list(csv.DictReader(f))
    per_probe = defaultdict(dict)
    for row in records:
        k = ((int(row["nx"]), int(row["ny"])), int(row["field"]),
             (int(row["steps"]), float(row["dt_frac"])), float(row["rtol"]), row["obs"], int(row["probe"]))
        eps = float(row["eps"])
        assert eps not in per_probe[k]
        for col in row:
            if col.endswith(("_E", "_cos", "_med")) or col == "state_err":
                row[col] = float(row[col])
                assert math.isfinite(row[col])
        per_probe[k][eps] = row
    assert set(per_probe) == set(itertools.product(meshes, range(100, 105), schedules, rtols,
                                                  ("energy", "right_mean"), range(4)))
    assert all(set(v) == eps_grid for v in per_probe.values())
    E, passed, changes = {}, {}, Counter()
    for k, by_eps in per_probe.items():
        vv = list(by_eps.values())
        for g in names:
            ee = {r[g + "_E"] for r in vv}
            assert len(ee) == 1
            ek = k[:-1], g
            v = ee.pop()
            assert E.setdefault(ek, v) == v
            med = [r[g + "_med"] for r in vv]
            cos = [r[g + "_cos"] for r in vv]
            same = any(m <= .01 and c >= .99 for m, c in zip(med, cos))
            split = min(med) <= .01 and max(cos) >= .99
            changes[g] += same != split
            passed[k, g] = same
    print(filename, "rows", len(records), "probes", len(per_probe), "cases", len(E)//3)
    print("classification differences same vs separate epsilon", dict(changes))
    for g in names:
        good = [k for k in per_probe if E[k[:-1], g] < .01]
        bad = [k for k in per_probe if E[k[:-1], g] > .10]
        print(g, "accurate rejected", sum(not passed[k, g] for k in good), "/", len(good),
              "wrong passed", sum(passed[k, g] for k in bad), "/", len(bad),
              "max E", max(v for (_, name), v in E.items() if name == g))
        print("accurate rejects by rtol", sorted(Counter(k[3] for k in good if not passed[k, g]).items()))
    wrong = sorted(k for k in per_probe if E[k[:-1], "unrolled"] > .10)
    print("paired inversions", sum(passed[k, "unrolled"] and not passed[k, "reference"] for k in wrong),
          "/", len(wrong))
    for k in wrong[::4]:
        print("wrong", k[:-1], E[k[:-1], "unrolled"], "implicit E", E[k[:-1], "implicit"],
              "state error", per_probe[k][.1]["state_err"],
              "base/fd count ranges", sorted({(r["base_iters"], r["fd_iters"])
              for p in range(4) for r in per_probe[k[:-1]+(p,)].values()}))
    for t in rtols:
        vals = [v for (k, g), v in E.items() if k[3] == t and g == "unrolled"]
        print("rtol, max unrolled E", t, max(vals))
    for g in names:
        focus = [k for k in per_probe if k[2:5] == ((1, .01), .001, "energy")]
        print("short, loose energy", g, "accurate",
              sum(E[k[:-1], g] < .01 for k in focus), "failures", sum(not passed[k, g] for k in focus))
    if filename == "results.csv":
        each_eps = sum(r["unrolled_med"] <= .01 and r["unrolled_cos"] >= .99
                       for k in wrong for r in per_probe[k].values())
        cosine_only = sum(max(r["reference_cos"] for r in per_probe[k].values()) >= .99 for k in wrong)
        print("wrong unrolled passing at individual epsilons", each_eps, "/", 4*len(wrong))
        print("reference accepted by cosine alone on those probes", cosine_only, "/", len(wrong))
        for p in range(4):
            k = ((16, 8), 102, (1, .01), .001, "energy", p)
            for eps, r in per_probe[k].items():
                print("example", p, eps, r["unrolled_med"], r["reference_med"], r["reference_cos"])
        for s in schedules:
            for obs in ("energy", "right_mean"):
                ells = [v for (k, g), v in E.items() if k[2:5] == (s, .001, obs) and g == "unrolled"]
                print("rtol 1e-3 schedule/obs E range", s, obs, min(ells), max(ells))
    return E


if __name__ == "__main__":
    loose = audit("results.csv", (.001, .0001, .00001, 1e-6, 1e-8))
    tight = audit("results_fd_tight.csv", (.001, .0001))
    assert all(loose[k] == v for k, v in tight.items())
