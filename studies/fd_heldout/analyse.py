"""Score results.csv against PREREG.md."""
import csv
import collections
rows = list(csv.DictReader(open("results.csv")))
for r in rows:
    for k in r:
        if k.endswith(("_E", "_med", "_cos")) or k in ("rtol", "eps", "state_err"):
            r[k] = float(r[k])
def passes(r, g): return r[f"{g}_med"] <= 1e-2 and r[f"{g}_cos"] >= 0.99
# one verdict per (case, probe, gradient): pass if any eps passes
key = lambda r: (r["nx"], r["field"], r["steps"], r["dt_frac"], r["rtol"], r["obs"], r["probe"])
verdict = collections.defaultdict(lambda: collections.defaultdict(bool)); E = {}
for r in rows:
    for g in ("unrolled", "implicit", "reference"):
        verdict[key(r)][g] |= passes(r, g); E[key(r) + (g,)] = r[f"{g}_E"]
print("H1/H3: wrong gradients (E>0.10) and how many probe seeds pass them, by mesh/rtol")
for g in ("unrolled", "implicit"):
    tab = collections.Counter(); tot = collections.Counter()
    for k, v in verdict.items():
        if E[k + (g,)] > 0.10:
            tot[(k[0], k[4])] += 1; tab[(k[0], k[4])] += v[g]
    for kk in sorted(tot): print(f"  {g:9} nx={kk[0]:>2} rtol={kk[1]:g}: {tab[kk]}/{tot[kk]} wrong-gradient probes pass")
print("H2: accurate gradients (E<0.01) rejected")
for g in ("unrolled", "implicit", "reference"):
    rej = sum(1 for k, v in verdict.items() if E[k + (g,)] < 0.01 and not v[g]); tot = sum(1 for k in verdict if E[k + (g,)] < 0.01)
    print(f"  {g:9}: {rej}/{tot} rejected")
print("Largest unrolled error that passes, per mesh:")
for nx in ("12", "16"):
    c = [(E[k + ("unrolled",)], k) for k, v in verdict.items() if k[0] == nx and v["unrolled"]]
    print(" ", nx, max(c))
