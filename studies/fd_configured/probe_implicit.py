"""Probe: (a) implicit-adjoint-on-loose-forward candidate, (b) mosaic's J^2 loss. f64, rtol 1e-3."""
import numpy as np
import run as R
p, KE, M, DT, STEPS = R.p, R.KE, R.M, R.DT, R.STEPS

def implicit_grad(rho, states, power):
    a = R.assemble(rho)
    JT = states[-1] @ states[-1]
    cot, g = 2 * states[-1], np.zeros(p.n_cells)
    for i in range(STEPS, 0, -1):
        lam = np.linalg.solve(a.T, cot)
        g -= DT * np.einsum("i,eij,j->e", lam, KE, states[i])
        cot = M.T @ lam
    g = g * 3 * (1 - 1e-3) * np.clip(rho, 0, 1) ** 2
    return g if power == 1 else 2 * JT * g

dirs = R.directions()
res = {}
for seed in [0, *range(1000, 1030)]:
    rng = np.random.default_rng(seed)
    rho = np.clip(0.5 + 0.3 * rng.standard_normal(p.n_cells).astype(np.float32), 0.05, 0.95).astype(np.float32).astype(np.float64)
    rtol = 1e-3
    states, counts = R.forward(rho, rtol)
    ex_states = [R.T0]
    a = R.assemble(rho)
    for _ in range(STEPS): ex_states.append(np.linalg.solve(a, M @ ex_states[-1]))
    Jex = ex_states[-1] @ ex_states[-1]
    Jtr = states[-1] @ states[-1]
    ref = R.exact_grad(rho); unr = R.unrolled_grad(rho, tuple(counts))
    fds = {1: [], 2: []}
    for eps in R.EPS:
        h = eps * np.sqrt(np.mean(rho**2) + 1e-30)
        f1, f2 = [], []
        for d in dirs:
            sp, _ = R.forward(rho + h * d, rtol); sm, _ = R.forward(rho - h * d, rtol)
            jp, jm = sp[-1] @ sp[-1], sm[-1] @ sm[-1]
            f1.append((jp - jm) / (2 * h)); f2.append((jp**2 - jm**2) / (2 * h))
        fds[1].append(np.array(f1)); fds[2].append(np.array(f2))
    for power in (1, 2):
        cands = {"unrolled": unr if power == 1 else 2 * Jtr * unr,
                 "exact": ref if power == 1 else 2 * Jex * ref,
                 "implicit": implicit_grad(rho, states, power)}
        for c, g in cands.items():
            pr = dirs @ g
            meds = [np.median(np.abs(fd - pr) / np.maximum(np.maximum(abs(fd), abs(pr)), 1e-30)) for fd in fds[power]]
            coss = [fd @ pr / (np.linalg.norm(fd) * np.linalg.norm(pr) + 1e-30) for fd in fds[power]]
            res[(seed, power, c)] = (min(meds) <= 1e-2 and max(coss) >= 0.99, min(meds))
for power in (1, 2):
    for c in ("unrolled", "exact", "implicit"):
        fresh = [res[(s, power, c)][0] for s in range(1000, 1030)]
        print(f"loss J^{power} {c:9} passes {sum(fresh)}/30  seed0 pass={res[(0, power, c)][0]} minmed={res[(0, power, c)][1]:.3g}")
changes = sum(res[(s,1,c)][0] != res[(s,2,c)][0] for s in range(1000,1030) for c in ("unrolled","exact"))
print("verdict changes J vs J^2 (unrolled, exact):", changes)
