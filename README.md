# transient-heat-spike

A spike for idea 4 of [mosaic#375](https://github.com/pasteurlabs/mosaic/issues/375): transient heat conduction
with a long horizon, which should expose the memory and AD-strategy axis of differentiable solvers, with a closed-form
reference; and its follow-on, temperature-dependent conductivity, which should separate implicit differentiation
from differentiating unrolled Newton iterations. The issue asks for each idea to be "checked with a quick spike
before full integration"; this is that check, on CPU, outside mosaic.

## Problem

Slab [0, 2] x [0, 1], cold left face, other faces insulated, conductivity k0 rho per cell (rho as in mosaic's
thermal-mesh), initial temperature the single eigenmode sin(pi x / 4). Then T(x, t) = sin(pi x / 4) exp(-lam t),
and J = integral of T(t_end)^2 and its derivative along uniform conductivity have closed forms (`spike/problem.py`).
Every backend uses the same bilinear (Q1) elements, consistent mass and backward Euler with a fixed number of steps
to t_end, so forward results agree (to 1e-11 or better; OrdinaryDiffEq's drifts to 3.5e-12 at 100000 steps) and
differences come from how the gradient is taken. Besides the continuous closed form, `spike/problem.py` gives the
exact gradient of the discrete scheme itself (the scheme separates into a 1D eigenproblem; checked against the full
2D one), so the differentiation error of a backend is measured apart from the discretisation error.

## Backends and strategies

| backend | strategy | what is kept for the reverse sweep |
|---|---|---|
| JAX (`backends/jax/heat.py`) | `tape` | every step (plain `jax.grad` through `lax.scan`) |
| | `checkpoint` | steps / c + c states, c the largest divisor of steps not above isqrt(steps) (650 at 100000 steps); chunks recomputed (`jax.checkpoint`) |
| Firedrake + pyadjoint (`backends/firedrake/heat.py`) | `tape` | every step (default tape) |
| | `checkpoint` | Revolve with ceil(sqrt(steps)) snapshots (317 at 100000 steps; `checkpoint_schedules`) |
| OrdinaryDiffEq + SciMLSensitivity (`backends/julia/heat.jl`) | `interp` | checkpoints at isqrt(steps) times (`InterpolatingAdjoint(checkpointing = true)`); each interval is re-solved adaptively at the solve tolerances, not at the fixed step |
| | `interp-dense`, `quadrature`, `gauss` | the dense forward solution |
| | `backsolve` | nothing: the state is integrated backwards (ill-posed for diffusion) |

JAX and pyadjoint differentiate the discrete scheme; SciMLSensitivity's adjoints are continuous adjoints solved at
the same step, so their gradient differs from the discrete one by O(dt), except where the checkpointed re-solve
adds its own error. Linear solves: JAX uses conjugate gradients per step (`jax.scipy.sparse.linalg.cg`, implicit
derivative), Firedrake one MUMPS factorisation reused for every step (`constant_jacobian`), and OrdinaryDiffEq's
ImplicitEuler rebuilds and factorises its matrix at every step here (docs/log.md), so wall times across backends are
not a like-for-like comparison of the libraries. Finite-difference checks use a random direction in each
backend's own cell order, so their values are not comparable across backends; the uniform direction is.

`backends/jax/heat_kt.py` is the follow-on: k = k0 rho (1 + T), Newton per step, `unrolled` (reverse mode through
the Newton iterations) against `implicit` (`jax.lax.custom_root`), over the number of Newton iterations, with dense
Newton solves; the converged reference runs in its own process.

## Running

```sh
uv venv && uv pip install "jax[cpu]==0.12.1" numpy scipy
.venv/bin/python backends/jax/heat.py --nx 32 --ny 16 --steps 10000 --strategy tape --fd
julia --project=backends/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=backends/julia backends/julia/heat.jl --nx 32 --ny 16 --steps 10000 --strategy gauss
docker run --rm -v "$PWD":/work -w /work firedrakeproject/firedrake-vanilla-default:2026.10.0 \
  python3 backends/firedrake/heat.py --nx 32 --ny 16 --steps 10000 --strategy checkpoint
```

`.github/workflows/spike.yml` runs the whole sweep (10 to 100000 steps; three mesh sizes for JAX and Firedrake, two
for OrdinaryDiffEq) on GitHub's CPU runners,
one process per case, killed and recorded at 13 GB, and `spike/summarise.py` turns the results into tables and
plots (the run's `summary` artifact). Saved runs go in `results/`.

## Results

From CI run 1 ([results/ci-run-1/summary.md](results/ci-run-1/summary.md); notes in docs/log.md; GB = 1024 MB):

- A full tape grows linearly with the step count: at 1e5 steps JAX needs +1.5 GB on 32 x 16 and +6.1 GB on
  64 x 32 and refuses 128 x 64 up front (it asks for 24.4 GB); pyadjoint needs +4.4 and +9.0 GB and is killed at
  13 GB on 128 x 64.
- JAX's checkpointing stays within +102 MB on every mesh (1.6-1.8 times the tape's time at 1e5 steps).
  pyadjoint's Revolve stores no per-step state but still grows (+0.64 to +1.6 GB at 1e5 steps, 2.2-3.3 times the
  time, default settings): its tape keeps three blocks per step (5-6 KB/step on both meshes measured), and garbage from
  the recomputation, which grows with the mesh, waits for the cycle collector; gc_timestep_frequency removes the
  second part at 1.3-2 times the time (repro/pyadjoint_checkpoint_memory.py, docs/log.md).
- JAX and pyadjoint match the exact discrete gradient to 2.1e-11 or better; SciMLSensitivity's continuous adjoints
  are O(dt) off (1e-5 to 2e-5 at 1e5 steps), and BacksolveAdjoint is 89-100 % off (5.3e289 at 1e5 steps on 32 x 16).
- k(T): with one Newton iteration per step, unrolled differentiation is 7-22 times closer to the converged gradient
  than implicit differentiation; from three iterations both agree with it to 2.1e-14 or better.

Separately, repro/sciml_checkpoint_resolve.jl shows that SciMLSensitivity's checkpointed InterpolatingAdjoint
re-solves each checkpoint interval adaptively, ignoring a fixed dt: its gradient error stops shrinking with the step
(1.5, 0.78 and 0.72 % at 1e2, 1e3 and 1e4 steps, against 1.0, 0.099 and 0.0099 % without checkpointing). With
checkpoints at isqrt(steps) times, as in the CI sweep, the effect is small.
