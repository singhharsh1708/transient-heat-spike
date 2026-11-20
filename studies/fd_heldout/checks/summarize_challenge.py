"""Report every stress-test stratum and check that the paired design is complete."""
import csv
from collections import defaultdict
from pathlib import Path

review = Path(__file__).resolve().parent
with (review/"challenge.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 140
assert all(float(r["min_perturbed_k"]) > 0 and float(r["replay_error"]) < 1e-11 for r in rows)
fresh = defaultdict(list)
for r in rows:
    if int(r["seed"]) >= 200:
        fresh[tuple(r[k] for k in ("nx","start","precond","steps","obs"))].append(r)
assert len(fresh) == 12
print("nx start preconditioner steps output | min/max E (%) | fields over 10% | paired inversions/40 | iterations (min/max)")
for k, rs in fresh.items():
    assert sorted(int(r["seed"]) for r in rs) == list(range(200,210))
    e = [float(r["E"])*100 for r in rs]
    print(*k, f"| {min(e):.3f} / {max(e):.3f} | {sum(x>10 for x in e)}/10 |",
          f"{sum(int(r['inversion']) for r in rs)}/40 |",
          min(int(r["min_iter"]) for r in rs), max(int(r["max_iter"]) for r in rs))
for scale in ("rms","l2"):
    original = [r for r in rows if int(r["seed"])<200 and r["scale"]==scale and r["obs"]=="energy"]
    print("Original five, scale",scale,"epsilons per case",[int(r["eps_checked"]) for r in original],
          "paired inversions",sum(int(r["inversion"]) for r in original),
          "strict at smallest epsilon: unrolled/exact passed",
          sum(int(r["strict_fixed_smallest_unrolled"]) for r in original),
          sum(int(r["strict_fixed_smallest_exact"]) for r in original))
with (review/"mosaic_loss.csv").open(newline="") as f:
    results = list(csv.DictReader(f))
assert len(results) == 160
groups = defaultdict(lambda: defaultdict(dict))
for r in results:
    k = (r["nx"], r["seed"], r["probe"])
    item = groups[r["scalar_readout"],r["io"]][k]
    item[r["candidate"]] = int(r["passed"])
    item["E"] = float(r["unrolled_full_E"])
for key, by_probe in groups.items():
    vs = list(by_probe.values())
    assert len(vs) == 20
    print("Actual Mosaic FD functions with local adapter", *key, "E (%)",
          f"{100*min(v['E'] for v in vs):.3f} / {100*max(v['E'] for v in vs):.3f}",
          "pass unrolled/exact", sum(v["unrolled"] for v in vs), sum(v["reference"] for v in vs),
          "paired >10%",sum(v["E"]>.1 and v["unrolled"] and not v["reference"] for v in vs))
