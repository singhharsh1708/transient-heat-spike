"""Check the executed study gradients by complex step, and compare live/replayed CG."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import lu_factor, lu_solve

study = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(study))
import sweep
from spike.problem import assemble_dense, element_matrices, mesh

jax.config.update("jax_enable_compilation_cache", False)


@jax.jit
def live_count(a, m, u, rtol):
    b = m @ u
    target = rtol*jnp.linalg.norm(b)
    inv = 1./jnp.diag(a)
    r = b - a @ u
    z = inv*r
    def cond(s):
        return (s[0] < 5000) & (jnp.linalg.norm(s[2]) >= target)
    def step(s):
        n, x, r, d, rho = s
        q = a @ d
        alpha = rho/(d @ q)
        x, r = x+alpha*d, r-alpha*q
        z = inv*r
        new = r @ z
        return n+1, x, r, z+(new/rho)*d, new
    n, *_ = jax.lax.while_loop(cond, step, (0, u, r, z, r @ z))
    return n


def study_adjoints(nx, field, steps, frac, tol):
    # Observe return values of the actual nested adjoint, without editing study files.
    result = []
    def collect(frame, event, value):
        if event == "return" and frame.f_code.co_name == "adjoint" and frame.f_code.co_filename == sweep.__file__:
            result.append(value.copy())
    saved = sweep.PROBES
    sweep.PROBES = ()  # the separate reproduction runs all FD calls
    sys.setprofile(collect)
    try:
        sweep.run_case(nx, nx//2, field, steps, frac, tol, None)
    finally:
        sys.setprofile(None)
        sweep.PROBES = saved
    return np.stack(result).reshape(2, 2, -1)  # observable, exact/implicit, cell


def independent_complex_step(p, k, dt, M, u0, right, states):
    K, M0 = assemble_dense(p, k)
    A = M0 + dt*K
    np.testing.assert_array_equal(M0, M)
    coords, cells, free = mesh(p)
    free_index = np.full(len(coords), -1)
    free_index[free] = np.arange(len(free))
    ke, _ = element_matrices(p.Lx/p.nx, p.Ly/p.ny)
    # Two columns: true dynamics and dynamics with fixed residual forcings.
    # The latter exactly tracks the approximate base states and differentiates
    # the equations while holding those forcings fixed.
    defects = [A @ v - M0 @ u for u, v in zip(states[:-1], states[1:])]
    eta = 1.e-30
    grads = []
    for cell in cells:
        ij = free_index[cell]
        keep = ij >= 0
        B = A.astype(complex)
        B[np.ix_(ij[keep], ij[keep])] += (1j*eta*dt)*ke[np.ix_(keep, keep)]
        factors = lu_factor(B)
        u = np.column_stack([u0, u0])
        for d in defects:
            rhs = M0 @ u
            rhs[:, 1] += d
            u = lu_solve(factors, rhs)
        # Ordinary transpose is the analytic continuation of real thermal energy.
        grads.append(np.stack([np.sum(u*(M0 @ u), axis=0), right @ u]).imag/eta)
    return np.moveaxis(np.array(grads), 0, 2)


spot = {(12,100,1,1.,.001), (12,100,10,.1,.001), (16,102,1,.01,.001), (16,104,10,.1,.0001)}
max_state = max_asm = 0.
disagreements = []
n_steps = 0
for nx in (12,16):
    p, kej, mj, x, *_ = sweep.setup(nx, nx//2)
    ke, m = np.array(kej), np.array(mj)
    for seed in sweep.FIELDS:
        rng = np.random.default_rng(seed)
        k = np.exp(.5*rng.standard_normal(p.n_cells))
        mode = np.sin(np.pi*x/(2*p.Lx))
        mode /= np.sqrt(mode @ m @ mode)
        noise = rng.standard_normal(len(mode))
        u0 = mode + .3*noise/np.sqrt(noise @ m @ noise)
        right = np.isclose(x,p.Lx).astype(float)
        right /= right.sum()
        K, M0 = assemble_dense(p, k)
        np.testing.assert_array_equal(M0, m)
        K0 = np.einsum("e,eij->ij", k, ke)
        max_asm = max(max_asm,np.max(np.abs(K0-K)))
        for steps, frac in sweep.SCHEDULES:
            dt = frac*p.t_end
            a = m+dt*K0
            aj = mj+dt*jnp.einsum("e,eij->ij", k, kej)
            for tol in sweep.RTOLS:
                key = (nx,seed,steps,frac,tol)
                states, uj = [u0], jnp.asarray(u0)
                for n in range(steps):
                    u = states[-1]
                    u, count = sweep.solve_cg(a,m@u,u,tol)
                    independent = int(live_count(aj,mj,uj,tol))
                    if independent != count:
                        disagreements.append((key,n,count,independent))
                    uj = sweep.pcg(aj, mj@uj, uj, 1/jnp.diag(aj), count)
                    max_state = max(max_state, np.linalg.norm(u-np.array(uj))/np.linalg.norm(u))
                    states.append(u)
                    n_steps += 1
                if key in spot:
                    actual = study_adjoints(*key)
                    expected = independent_complex_step(p,k,dt,m,u0,right,states)
                    discrepancy = np.linalg.norm(actual-expected, axis=2)/np.linalg.norm(expected, axis=2)
                    print("complex-step", key, "(energy,right) x (reference,implicit)", discrepancy.tolist(),
                          flush=True)
print("assembly max abs",max_asm,"steps",n_steps,"max rel state replay",max_state,
      "native JAX vs SciPy count mismatches",len(disagreements),disagreements[:12],flush=True)
