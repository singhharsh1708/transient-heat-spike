"""Tables and plots from a sweep: python spike/summarise.py <results_dir> <out_dir>

Reads <backend>-<strategy>-<nx>x<ny>-<steps>.json (searched recursively; each name counted once) and
fdcheck-<backend>-<strategy>-*.json. Writes summary.md and memory.png / gradient_error.png / time.png to out_dir
and prints summary.md.
"""

import json
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from spike.problem import Problem, discrete_exact  # noqa: E402

root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
CASE = re.compile(r"^(?P<backend>jax|julia|firedrake)-(?P<strategy>[a-z-]+)-(?P<nx>\d+)x(?P<ny>\d+)-(?P<steps>\d+)\.json$")

cases, fdchecks, seen = {}, {}, set()
for f in sorted(root.rglob("*.json")):
    if f.name in seen:
        continue
    seen.add(f.name)
    d = json.loads(f.read_text())
    m = CASE.match(f.name)
    if m:
        key = (m["backend"], m["strategy"], int(m["nx"]), int(m["ny"]))
        cases.setdefault(key, {})[int(m["steps"])] = d
    elif (m := re.match(r"^fdcheck-(jax|julia|firedrake)-([a-z-]+)-\d+x\d+\.json$", f.name)):
        fdchecks[(m.group(1), m.group(2))] = d  # from the name: a killed run's stub has neither field

steps_all = sorted({s for v in cases.values() for s in v})
meshes = sorted({(k[2], k[3]) for k in cases})


def peak(d):
    return d.get("external_peak_rss_mb") or d.get("peak_rss_mb") or d.get("sampled_peak_rss_mb")


_exact = {}


def discrete(nx, ny, steps):
    """dJ_h/deps of the discrete scheme, exact on every mesh (spike/problem.py: discrete_exact)."""
    if (nx, ny, steps) not in _exact:
        _exact[(nx, ny, steps)] = discrete_exact(Problem(nx=nx, ny=ny), steps)[1]
    return _exact[(nx, ny, steps)]


def refused_mb(d):
    """The size XLA asked for when it refused a run up front (it never reached that peak), else None.

    CPU: "Out of memory allocating 26216674456 bytes"; GPU: "Out of memory while trying to allocate 24.42GiB".
    """
    m = re.search(r"Out of memory (?:allocating|while trying to allocate) ([\d.]+) ?(bytes|B|KiB|MiB|GiB|TiB)",
                  str(d.get("error", "")))
    shift = {"bytes": 0, "B": 0, "KiB": 10, "MiB": 20, "GiB": 30, "TiB": 40}
    return float(m.group(1)) * 2 ** shift[m.group(2)] / 2**20 if m else None


def mem_cell(d):
    if d is None:
        return "-"
    pk = peak(d)
    if not d.get("ok"):
        e = str(d.get("error", ""))
        if (need := refused_mb(d)) is not None:
            return f"out of memory: needs {need / 1024:.1f} GB"
        why = "killed" if "killed" in e else "failed"
        return f"{why}{f' at {pk / 1024:.1f} GB' if pk else ''}"
    base = d.get("baseline_rss_mb")
    return f"{pk:.0f}" + (f" (+{pk - base:.0f})" if base else "")


def err(d):
    if not d or not d.get("ok") or "dJ_uniform" not in d:
        return None
    return abs(d["dJ_uniform"] / d["exact_dJ_uniform"] - 1)


def dev_cell(d):
    """Peak accelerator memory (MB) and its growth over the warm-up; failures with what was reached or asked for."""
    if d is None:
        return "-"
    if not d.get("ok"):
        if (need := refused_mb(d)) is not None:
            return f"out of memory: needs {need / 1024:.1f} GB"
        if "device_peak_mb" in d:
            return f"failed at {d['device_peak_mb'] / 1024:.1f} GB"
        return "killed" if "killed" in str(d.get("error", "")) else "failed"  # a run_case stub
    if "device_peak_mb" not in d:
        return "-"
    def mb(v):  # three figures for the small cases, whole MB above 100
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"

    base = d.get("baseline_device_peak_mb")
    return mb(d["device_peak_mb"]) + (f" (+{mb(d['device_peak_mb'] - base)})" if base is not None else "")


accel = sorted({d["device"] for v in cases.values() for d in v.values() if d.get("device", "cpu") != "cpu"})
lines = [f"# mosaic#375 idea 4: transient heat, {', '.join(accel) if accel else 'CPU'} sweep\n"]
lines.append("Peak resident memory in MB of the whole process, and in brackets its growth above the peak of a 10-step")
lines.append("warm-up in the same process (what the measured run needed on top of a compiled process); failures with")
lines.append("the peak reached. Peaks are high-water marks: growth below the warm-up peak shows as +0.\n")
for nx, ny in meshes:
    keys = [k for k in sorted(cases) if (k[2], k[3]) == (nx, ny)]
    n_dofs = next((d.get("n_dofs") for k in keys for d in cases[k].values() if d.get("n_dofs")), "?")
    lines.append(f"\n## {nx} x {ny} mesh ({n_dofs} unknowns)\n")
    head = "| backend | strategy | " + " | ".join(f"{s} steps" for s in steps_all) + " |"
    sep = "|---|---|" + "---|" * len(steps_all)
    lines += ["**Peak memory (MB)**\n", head, sep]
    for k in keys:
        lines.append(f"| {k[0]} | {k[1]} | " + " | ".join(mem_cell(cases[k].get(s)) for s in steps_all) + " |")
    if any("device_peak_mb" in d for k in keys for d in cases[k].values()):
        limits = [d["device_limit_mb"] for k in keys for d in cases[k].values() if d.get("device_limit_mb")]
        cap = f"; the allocator's limit is {max(limits) / 1024:.1f} GB" if limits else ""
        lines += ["\n**Peak device memory (MB)** (the accelerator allocator's peak bytes in use; in brackets its "
                  f"growth over the warm-up{cap})\n", head, sep]
        for k in keys:
            lines.append(f"| {k[0]} | {k[1]} | " + " | ".join(dev_cell(cases[k].get(s)) for s in steps_all) + " |")
    lines += ["\n**Gradient time (s)**\n", head, sep]
    for k in keys:
        cells = []
        for s in steps_all:
            d = cases[k].get(s)
            cells.append(f"{d['warm_s']:.3g}" if d and d.get("ok") and "warm_s" in d else "-")
        lines.append(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + " |")
    lines += ["\n**Gradient along uniform conductivity, relative error against the exact gradient of the discrete "
              "scheme** (the differentiation error alone)\n", head, sep]
    for k in keys:
        cells = []
        for s in steps_all:
            d = cases[k].get(s)
            ex = discrete(nx, ny, s) if d and d.get("ok") and "dJ_uniform" in d else None
            cells.append(f"{abs(d['dJ_uniform'] / ex - 1):.1e}" if ex else "-")
        lines.append(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + " |")
    lines += ["\n**The same against the closed form of the continuous problem** (includes the mesh's spatial error)\n",
              head, sep]
    for k in keys:
        lines.append(f"| {k[0]} | {k[1]} | " + " | ".join(
            f"{e:.1e}" if (e := err(cases[k].get(s))) is not None else "-" for s in steps_all) + " |")

if fdchecks:
    runs = {(d["nx"], d["ny"], d["steps"]) for d in fdchecks.values() if "nx" in d}  # a killed run's stub has none
    where = "; ".join(f"{nx} x {ny} mesh, {st} steps" for nx, ny, st in sorted(runs))
    lines += [f"\n## Gradient against finite differences of the same backend's forward ({where})\n",
              "| backend | strategy | grad . d | central FD | relative difference |", "|---|---|---|---|---|"]
    for (b, s), d in sorted(fdchecks.items()):
        if d.get("ok") and d.get("fd_dir"):
            lines.append(f"| {b} | {s} | {d['grad_dir']:.10g} | {d['fd_dir']:.10g} | "
                         f"{abs(d['grad_dir'] / d['fd_dir'] - 1):.1e} |")
        else:
            lines.append(f"| {b} | {s} | - | - | {d.get('error', 'no result')[:60]} |")

KT = re.compile(r"^kt-(?P<diff>unrolled|implicit)-n(?P<n>\d+)-(?P<steps>\d+)\.json$")
kt, kt_ref = {}, {}
for f in sorted(root.rglob("kt-*.json")):
    if (m := KT.match(f.name)):
        kt[(int(m["steps"]), int(m["n"]), m["diff"])] = json.loads(f.read_text())
    elif (m := re.match(r"^kt-reference-(\d+)\.json$", f.name)):
        kt_ref[int(m.group(1))] = json.loads(f.read_text())
if kt:
    refcheck = "; ".join(f"{s} steps {abs(r['grad_dir'] / r['fd_dir'] - 1):.0e}" for s, r in sorted(kt_ref.items())
                         if r.get("ok"))
    kt_mesh = "; ".join(f"{nx} x {ny} mesh" for nx, ny in sorted(
        {(d["nx"], d["ny"]) for d in [*kt.values(), *kt_ref.values()] if "nx" in d}))
    lines += [f"\n## k(T) follow-on: Newton per step, unrolled vs implicit differentiation (JAX, {kt_mesh})\n",
              "Errors against the converged implicit gradient (30 Newton iterations, its own process; against central",
              f"differences: {refcheck}). Peak MB is the whole process's (dense Newton solves).\n",
              "| steps | Newton iterations | forward J error | unrolled: gradient error, peak MB, s | "
              "implicit: gradient error, peak MB, s |", "|---|---|---|---|---|"]
    for st, n in sorted({(k[0], k[1]) for k in kt}):
        u, i = kt.get((st, n, "unrolled"), {}), kt.get((st, n, "implicit"), {})
        ref = kt_ref.get(st, {})

        def cell(d):
            if not d.get("ok") or not ref.get("ok"):
                return "-"
            return f"{abs(d['grad_dir'] / ref['grad_dir'] - 1):.1e}, {peak(d):.0f}, {d['warm_s']:.3f}"

        fj = f"{abs(u['J'] / ref['J'] - 1):.1e}" if u.get("ok") and ref.get("ok") else "-"
        lines.append(f"| {st} | {n} | {fj} | {cell(u)} | {cell(i)} |")

KTF = re.compile(r"^kt-firedrake-(?P<diff>unrolled|implicit|reference)(?:-n(?P<n>\d+))?-(?P<steps>\d+)\.json$")
ktf = {}
for f in sorted(root.rglob("kt-firedrake-*.json")):
    if (m := KTF.match(f.name)):
        ktf[(int(m["steps"]), int(m["n"] or 30), m["diff"])] = json.loads(f.read_text())
if ktf:
    pairs = []
    for (st, n, df), d in ktf.items():
        twin = kt_ref.get(st, {}) if df == "reference" else kt.get((st, n, df), {})
        if d.get("ok") and twin.get("ok") and "grad_dir" in d and "grad_dir" in twin:
            pairs.append(max(abs(d[k] / twin[k] - 1) for k in ("J", "dJ_uniform", "grad_dir")))
    lines += ["\n## k(T) on Firedrake (pyadjoint): the same scheme, against the JAX records\n",
              "Errors of the gradient along the JAX table's random direction against Firedrake's own converged "
              "gradient (against central differences: " + "; ".join(
                  f"{st} steps {abs(d['grad_dir'] / d['fd_dir'] - 1):.0e}" for (st, _, df), d in sorted(ktf.items())
                  if df == "reference" and d.get("ok") and "fd_dir" in d) + ")." + (f" Over the {len(pairs)} cases both completed, J, the gradient along uniform "
                             f"conductivity and along that direction agree with JAX's to {max(pairs):.1e} relative."
                             if pairs else "") + "\n",
              "| steps | Newton iterations | unrolled | implicit | implicit / unrolled |", "|---|---|---|---|---|"]
    for st, n in sorted({(k[0], k[1]) for k in ktf if k[2] != "reference"}):
        ref = ktf.get((st, 30, "reference"), {})
        errs = []
        for df in ("unrolled", "implicit"):
            d = ktf.get((st, n, df), {})
            ok = d.get("ok") and ref.get("ok") and "grad_dir" in d and "grad_dir" in ref
            errs.append(abs(d["grad_dir"] / ref["grad_dir"] - 1) if ok else None)
        ratio = f"{errs[1] / errs[0]:.3g}" if None not in errs and errs[0] > 0 else "-"
        lines.append(f"| {st} | {n} | " + " | ".join(f"{e:.1e}" if e is not None else "-" for e in errs)
                     + f" | {ratio} |")

text = "\n".join(lines) + "\n"
(out / "summary.md").write_text(text)
print(text)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CAP = 1e3  # largest value drawn; anything above (or non-finite) is a triangle at the cap
    on_device = any("device_peak_mb" in d for v in cases.values() for d in v.values())  # GPU run: plot device memory
    plots = (("mem", "peak device memory (MB)" if on_device else "peak resident memory (MB)", "memory.png"),
             ("err", "relative error vs discrete exact dJ (triangle: > 1e3)", "gradient_error.png"),
             ("time", "gradient wall time (s)", "time.png"))
    for what, ylabel, fname in plots if meshes else ():  # a folder of k(T) records only has no sweep to plot
        fig, axes = plt.subplots(1, len(meshes), figsize=(5 * len(meshes) + 2, 4), squeeze=False,
                                 layout="constrained")
        for ax, (nx, ny) in zip(axes[0], meshes):
            for k in sorted(cases):
                if (k[2], k[3]) != (nx, ny):
                    continue
                xs, ys, xk, yk = [], [], [], []
                for s, d in sorted(cases[k].items()):
                    if what == "mem":
                        # a refusal is drawn at the size it asked for
                        v = refused_mb(d) or (d.get("device_peak_mb") if on_device else peak(d))
                        (xs if d.get("ok") else xk).append(s)
                        (ys if d.get("ok") else yk).append(v if v else math.nan)
                    elif what == "err" and d.get("ok") and "dJ_uniform" in d:
                        xs.append(s); ys.append(max(abs(d["dJ_uniform"] / discrete(nx, ny, s) - 1), 1e-16))
                    elif what == "time" and d.get("ok") and "warm_s" in d:
                        xs.append(s); ys.append(d["warm_s"])
                # A blown-up error (backsolve's 5.3e289) breaks log-axis ticks: draw it at the cap
                over = [i for i, y in enumerate(ys) if what == "err" and not (y <= CAP)]
                ys = [CAP if i in over else y for i, y in enumerate(ys)]
                line, = ax.loglog(xs, ys, "o-", label=f"{k[0]} {k[1]}")
                if over:
                    ax.loglog([xs[i] for i in over], [CAP] * len(over), "^", color=line.get_color(),
                              markersize=10)
                if xk:
                    ax.loglog(xk, yk, "x", color=line.get_color(), markersize=10)
            ax.set_title(f"{nx} x {ny} mesh")
            ax.set_xlabel("time steps")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
        fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside right upper", fontsize=7)
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)
except ImportError:
    pass
