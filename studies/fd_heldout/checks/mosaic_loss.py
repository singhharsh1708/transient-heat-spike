"""Exercise Mosaic's original FD functions on a local SciPy adapter, without Tesseract transport."""
import ast
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from challenge import grid, review

sys.path.insert(0, str(review.parent))
import sweep

# A checkout of github.com/singhharsh1708/mosaic at 749d775 (fork-ci/thermal-transient), via MOSAIC_CHECKOUT.
if "MOSAIC_CHECKOUT" not in os.environ:
    sys.exit("Set MOSAIC_CHECKOUT to a checkout of github.com/singhharsh1708/mosaic at 749d775.")
fork = Path(os.environ["MOSAIC_CHECKOUT"]) / "mosaic" / "benchmarks"


def kernel_functions():
    # Load only these numerical functions verbatim. The spike environment and
    # the fork's full benchmark environment have different Python ABIs.
    ns = dict(jax=jax, jnp=jnp, np=np, Any=object, KernelContext=object,
              apply_tesseract=lambda adapter, inputs: {"q": adapter(inputs["k"])})
    wanted = {"core/experiment.py": {"random_direction"},
              "problems/shared/gradient.py": {"_fd_cosine", "fd_eps_sweep", "_vjp_grad", "_fd_vjp_arrays"}}
    for name, names in wanted.items():
        src = fork/name
        defs = [node for node in ast.parse(src.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {f.name for f in defs} == names
        exec(compile(ast.Module(body=defs, type_ignores=[]), str(src), "exec"), ns)
    return ns


def test(ns, nx, seed, dtype, observable):
    g = grid(nx)
    k0, u0 = g.inputs(seed, "mixed")
    k0 = k0.astype(dtype)
    dt = .01*g.p.t_end
    _, ke, m, *_ = sweep.setup(nx,nx//2)
    a = lambda k: m+dt*jnp.einsum("e,eij->ij", k, ke)
    u0j = jnp.asarray(u0)
    rhs = np.array(m) @ u0
    A = np.asarray(a(k0.astype(float)))
    _, count = sweep.solve_cg(A, rhs, u0, .001)
    def q(u):
        return u @ m @ u if observable == "mass" else u @ u
    def branch(k):
        aj = a(k)
        return q(sweep.pcg(aj, m@u0j, u0j, 1/jnp.diag(aj), count))
    def direct(k):
        u = jnp.linalg.solve(a(k), m@u0j)
        return q(u)**2
    q0, branch_grad = jax.value_and_grad(branch)(jnp.array(k0,dtype=jnp.float64))
    exact = np.array(jax.grad(direct)(jnp.array(k0,dtype=jnp.float64)))
    unrolled = np.array(2*q0*branch_grad)

    def call(k):
        u, _ = g.run_forward(np.array(k,dtype=float), u0, dt, 1, True)
        val = u[-1] @ (g.M @ u[-1]) if observable == "mass" else u[-1] @ u[-1]
        return np.array(val, dtype=dtype)
    @jax.custom_jvp
    def adapter(k):
        return jax.pure_callback(call, jax.ShapeDtypeStruct((), dtype), k)
    @adapter.defjvp
    def adapter_jvp(primals, tangents):
        # The kernel requests its only VJP at k0. All +/- forwards re-stop.
        return adapter(primals[0]), jnp.asarray(branch_grad, dtype=dtype) @ tangents[0]

    error = np.linalg.norm(unrolled-exact)/np.linalg.norm(exact)
    for p in range(4):
        ctx = SimpleNamespace(name="local", ic=k0, phys={}, ic_key="k", output_key="q", seed=p,
                              make_inputs=lambda name, field: {"k":field})
        _, _, _, branch_dirs, fd = ns["_fd_vjp_arrays"](adapter, ctx, sweep.EPS, 6)
        keys = jax.random.split(jax.random.PRNGKey(p), 6)
        dirs = np.stack([np.asarray(ns["random_direction"](k0.shape,key)) for key in keys])
        for name, pred in (("unrolled",branch_dirs), ("reference",dirs @ exact)):
            metrics = ns["fd_eps_sweep"](pred, fd).values()
            best_error = min(np.median(row["rel_error"]) for row in metrics)
            best_cos = max(row["cosine"] for row in metrics)
            yield dict(nx=nx,seed=seed,probe=p,scalar_readout=observable,io=np.dtype(dtype).name,
                       candidate=name,unrolled_full_E=error,best_median=best_error,best_cos=best_cos,
                       passed=int(best_error <= .01 and best_cos >= .99))


if __name__ == "__main__":
    ns = kernel_functions()
    with (review/"mosaic_loss.csv").open("w", newline="") as fh:
        writer = None
        for nx, s in ((12,100),(12,104),(16,101),(16,102),(16,103)):
            for dtype in (np.float64,np.float32):
                for obs in ("mass","nodal"):
                    rows = list(test(ns,nx,s,dtype,obs))
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
                    fh.flush()
