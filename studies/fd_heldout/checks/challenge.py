"""Sparse Q1 cross-check and paired stress tests. Run with python -B from any directory."""
import csv
import sys
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, splu

review = Path(__file__).resolve().parent
sys.path.insert(0, str(review.parents[2]))
from spike.problem import Problem, element_matrices, mesh

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)
EPS = (.1, .01, .001, .0001)
OBS = ("energy", "right_mean")


class Grid:
    """Keep the nine-point assembled matrix and its map from local entries."""
    def __init__(self, nx):
        p = self.p = Problem(nx=nx, ny=nx//2)
        coords, cells, free = mesh(p)
        self.x = coords[free, 0]
        self.n = n = len(free)
        self.right = np.isclose(self.x, p.Lx).astype(float)
        self.right /= self.right.sum()
        idx = np.full(len(coords), -1)
        idx[free] = np.arange(n)
        conn = idx[cells]
        rr = np.repeat(conn, 4, axis=1).ravel()
        cc = np.tile(conn, (1, 4)).ravel()
        keep = (rr >= 0) & (cc >= 0)
        self.lr, self.lc = rr[keep], cc[keep]
        self.cell = np.repeat(np.arange(p.n_cells), 16)[keep]
        ke, me = element_matrices(p.Lx/nx, p.Ly/p.ny)
        self.kw = np.tile(ke.ravel(), p.n_cells)[keep]
        mw = np.tile(me.ravel(), p.n_cells)[keep]
        locations, self.to_entry = np.unique(self.lr*n + self.lc, return_inverse=True)
        self.row, self.col = np.divmod(locations, n)
        self.diag = np.flatnonzero(self.row == self.col)
        self.indptr = np.r_[0, np.cumsum(np.bincount(self.row))]
        self.mv = np.bincount(self.to_entry, weights=mw)
        self.M = self.matrix(self.mv)

    def matrix(self, values):
        return csr_matrix((values, self.col, self.indptr), shape=(self.n, self.n))

    def a(self, k, dt):
        return self.matrix(self.mv + dt*np.bincount(self.to_entry, weights=self.kw*k[self.cell]))

    def inputs(self, seed, start):
        rng = np.random.default_rng(seed)
        k = np.exp(.5*rng.normal(size=self.p.n_cells))
        mode = np.sin(np.pi*self.x/(2*self.p.Lx))
        mode /= np.sqrt(mode @ (self.M @ mode))
        noise = rng.normal(size=mode.size)
        noise /= np.sqrt(noise @ (self.M @ noise))
        u0 = mode + .3*noise if start == "mixed" else noise
        return k, u0

    def run_forward(self, k, u0, dt, steps, jacobi, rtol=1e-3):
        a = self.a(k, dt)
        prec = diags(1/a.diagonal(), format="csr") if jacobi else None
        u, counts, states = u0, [], [u0]
        for _ in range(steps):
            n = 0
            def count(_):
                nonlocal n
                n += 1
            u, info = cg(a, self.M @ u, x0=u, M=prec, atol=0., rtol=rtol, callback=count, maxiter=5000)
            assert info == 0
            states.append(u)
            counts.append(n)
        return np.array(states), np.array(counts)

    def exact(self, k, u0, dt, steps):
        solve = splu(self.a(k, dt).tocsc()).solve
        states = [u0]
        for _ in range(steps):
            states.append(solve(self.M @ states[-1]))
        return self.adjoint(states, solve, dt), states[-1]

    def adjoint(self, states, solve, dt):
        cot = np.stack([2*(self.M @ states[-1]), self.right], axis=1)
        g = np.zeros((2, self.p.n_cells))
        for u in reversed(states[1:]):
            lam = solve(cot, trans="T")
            for j in range(2):
                g[j] -= dt*np.bincount(self.cell, weights=self.kw*lam[self.lr, j]*u[self.lc])
            cot = self.M.T @ lam
        return g

    @lru_cache(None)
    def ad(self, cap, jacobi):
        row, col = jnp.asarray(self.row), jnp.asarray(self.col)
        m = jnp.asarray(self.mv)
        def mv(a, u):
            return jnp.bincount(row, weights=a*u[col], length=self.n)

        def outputs(k, u0, dt, counts):
            a = m + dt*jnp.bincount(self.to_entry, weights=k[self.cell]*self.kw, length=len(self.mv))
            dinv = 1/a[self.diag] if jacobi else 1.
            def advance(u, count):
                r = mv(m, u) - mv(a, u)
                z = dinv*r
                def update(carry):
                    x, r, p, rho = carry
                    q = mv(a, p)
                    alpha = rho/(p @ q)
                    x, r = x + alpha*p, r - alpha*q
                    z = dinv*r
                    nxt = r @ z
                    return x, r, z + (nxt/rho)*p, nxt
                def body(i, carry):
                    return jax.lax.cond(i < count, update, lambda v: v, carry)
                out = jax.lax.fori_loop(0, cap, body, (u, r, z, r @ z))[0]
                return out, None
            last, _ = jax.lax.scan(advance, u0, counts)
            return jnp.stack([last @ mv(m, last), self.right @ last]), last
        return jax.jit(jax.jacrev(outputs, argnums=0, has_aux=True))

    def observable(self, u):
        return np.array([u @ (self.M @ u), self.right @ u])


@lru_cache(None)
def grid(nx):
    return Grid(nx)


@lru_cache(None)
def directions(cells):
    def unit(key):
        v = jax.random.normal(key, (cells,), dtype=jnp.float32)
        return v/(jnp.linalg.norm(v) + 1e-30)
    return np.vstack([np.array(jax.vmap(unit)(jax.random.split(jax.random.PRNGKey(p), 6)), dtype=float)
                      for p in range(4)])


def numerical(g, k, u0, dt, steps, jacobi, dirs, h, rtol=1e-3):
    fd, counts = [], []
    for d in dirs:
        out = []
        for sign in (1, -1):
            u, nn = g.run_forward(k + sign*h*d, u0, dt, steps, jacobi, rtol)
            out.append(g.observable(u[-1]))
            counts.append(nn)
        fd.append((out[0] - out[1])/(2*h))
    return np.array(fd), np.array(counts)


def case(nx, seed, start, jacobi, steps, audit_scale=False):
    g = grid(nx)
    k, u0 = g.inputs(seed, start)
    dt = .01*g.p.t_end
    states, counts = g.run_forward(k, u0, dt, steps, jacobi)
    # Round up the static scan length for JIT reuse. Inactive iterations use the identity.
    cap = 2**(int(counts.max())-1).bit_length()
    unrolled, replay = g.ad(cap, jacobi)(k, u0, dt, counts)
    unrolled, replay = np.asarray(unrolled), np.asarray(replay)
    replay_error = np.linalg.norm(replay-states[-1])/np.linalg.norm(states[-1])
    reference, exact_state = g.exact(k, u0, dt, steps)
    implicit = g.adjoint(states, splu(g.a(k, dt).tocsc()).solve, dt)
    E = np.linalg.norm(unrolled-reference, axis=1)/np.linalg.norm(reference, axis=1)
    Ei = np.linalg.norm(implicit-reference, axis=1)/np.linalg.norm(reference, axis=1)
    result = []
    dirs = directions(k.size)
    for scale in (("rms", "l2") if audit_scale else ("rms",)):
        stepscale = np.sqrt(np.mean(k*k)) if scale == "rms" else np.linalg.norm(k)
        metrics = np.full((4, 2, 2, 4, 3), np.nan)  # eps, observable, gradient, probe, (median, cosine, max)
        match, minpert = [], float("inf")
        for i, eps in enumerate(EPS):
            h = eps*stepscale
            smallest = np.min(k-h*np.abs(dirs))
            if smallest <= 0:  # discard the entire epsilon if it leaves the physical domain
                continue
            minpert = min(minpert, smallest)
            fd, nn = numerical(g, k, u0, dt, steps, jacobi, dirs, h)
            match.append(np.all(nn == counts[None, :]))
            for j in range(2):
                for c, grad in enumerate((unrolled[j], reference[j])):
                    pred = (dirs @ grad).reshape(4, 6)
                    slope = fd[:, j].reshape(4, 6)
                    rel = np.abs(slope-pred)/np.maximum(np.maximum(np.abs(slope), np.abs(pred)), 1e-30)
                    cos = np.sum(slope*pred, axis=1)/(np.linalg.norm(slope, axis=1)*np.linalg.norm(pred, axis=1)+1e-30)
                    metrics[i, j, c] = np.stack([np.median(rel, axis=1), cos, np.max(rel, axis=1)], axis=1)
        for j in range(2):
            score = metrics[:, j]
            passed = (np.nanmin(score[..., 0], axis=0) <= .01) & (np.nanmax(score[..., 1], axis=0) >= .99)
            strict = (score[..., 2] <= .01) & (score[..., 1] >= .99)
            result.append(dict(nx=nx, seed=seed, start=start, precond="jacobi" if jacobi else "identity",
                               steps=steps, obs=OBS[j], scale=scale, min_perturbed_k=minpert, E=E[j], implicit_E=Ei[j],
                               state_error=np.linalg.norm(states[-1]-exact_state)/np.linalg.norm(exact_state),
                               replay_error=replay_error, min_iter=counts.min(), max_iter=counts.max(),
                               eps_checked=len(match), branch_stable_eps=sum(match),
                               unrolled_pass=passed[0].sum(), exact_pass=passed[1].sum(),
                               inversion=((E[j] > .1) & passed[0] & ~passed[1]).sum(),
                               strict_fixed_smallest_unrolled=strict[-1,0].sum(),
                               strict_fixed_smallest_exact=strict[-1,1].sum()))
    if audit_scale:
        # All coordinate directions give an FD gradient in the original parameter space.
        fd, nn = numerical(g, k, u0, dt, steps, jacobi, np.eye(k.size), .0001*np.sqrt(np.mean(k*k)))
        print("coordinate", nx, seed, "all same count", np.all(nn == counts[None,:]),
              "energy FD/unrolled", np.linalg.norm(fd[:,0]-unrolled[0])/np.linalg.norm(unrolled[0]),
              "FD/exact", np.linalg.norm(fd[:,0]-reference[0])/np.linalg.norm(reference[0]), flush=True)
    return result


if __name__ == "__main__":
    original = [(nx, s, "mixed", True, 1, True) for nx, s in
                ((12,100), (12,104), (16,101), (16,102), (16,103))]
    variants = [(16,"mixed",True,1), (16,"noise",True,1), (16,"mixed",False,1),
                (32,"mixed",True,1), (64,"mixed",True,1), (16,"mixed",True,100)]
    fresh = [(nx, s, ic, pc, steps, False) for nx, ic, pc, steps in variants for s in range(200,210)]
    with (review/"challenge.csv").open("w", newline="") as fh:
        writer = None
        for args in original+fresh:
            rows = case(*args)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            print("completed", args, flush=True)
