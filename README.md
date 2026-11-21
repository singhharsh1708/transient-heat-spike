# transient-heat-spike

A spike for idea 4 of [mosaic#375](https://github.com/pasteurlabs/mosaic/issues/375): transient heat conduction
with a long horizon, which should expose the memory and AD-strategy axis of differentiable solvers, with a closed-form
reference; and its follow-on, temperature-dependent conductivity, which should separate implicit differentiation
from differentiating unrolled Newton iterations. The issue asks for each idea to be "checked with a quick spike
before full integration"; this is that check, on CPU, outside mosaic.

**Answer for idea 4.** It separates solvers on memory, not on gradient accuracy: every exact-discrete adjoint (JAX,
pyadjoint, tape or checkpointed) returns the same gradient to 3e-13, while a full tape grows linearly with the step
count (+6.1 GB at 1e5 steps on 64 x 32 for JAX) and checkpointing does not, or barely (JAX within about +100 MB;
pyadjoint's Revolve +0.6 to +1.6 GB, from its tape records and recomputation garbage). Continuous adjoints
(SciMLSensitivity) are first order in the step, and BacksolveAdjoint fails on diffusion. For #375's CI budget (a few
minutes per solver call, under about 30 minutes per solver): the memory separation is already clear at 1e4 steps,
where a call takes seconds (JAX) to minutes (pyadjoint Revolve, 2 to 3 minutes at 64 x 32 in CI); at 1e5 steps a
single pyadjoint Revolve call on 128 x 64 takes 29 to 31 minutes, so the top of the sweep belongs outside per-PR CI.
The mosaic integration on a fork (singhharsh1708/mosaic#2) runs its default suite to 1e4 steps in 21 to 26 minutes
per solver.

## Problem

Slab [0, 2] x [0, 1], cold left face, other faces insulated, conductivity k0 rho per cell (linear in rho;
mosaic's thermal-mesh maps rho through SIMP, which the fork integration does), initial temperature the single eigenmode sin(pi x / 4). Then T(x, t) = sin(pi x / 4) exp(-lam t),
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
| | `checkpoint` | steps / c + c states, c the largest divisor of steps not above isqrt(steps) (650 at 100000 steps; for a prime step count c = 1 and every step is stored, as with the tape); chunks recomputed (`jax.checkpoint`) |
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

From CI run 1 ([results/ci-run-1/summary.md](results/ci-run-1/summary.md)) and a T4 run
([results/t4-run-1](results/t4-run-1/README.md)); notes in docs/log.md; GB = 1024 MB:

- A full tape grows linearly with the step count: at 1e5 steps JAX needs +1.5 GB on 32 x 16 and +6.1 GB on
  64 x 32 and refuses 128 x 64 up front (it asks for 24.4 GB); pyadjoint needs +4.4 and +9.0 GB and is killed at
  13 GB on 128 x 64.
- JAX's checkpointing stays within about +100 MB on every mesh. pyadjoint's Revolve stores no per-step state but
  still grows (+0.64 to +1.6 GB at 1e5 steps, default settings): its tape keeps three blocks per step (about 5-6 KB/step on both meshes measured), and
  garbage from the recomputation, which grows with the mesh, waits for the cycle collector; gc_timestep_frequency
  removes the second part at 1.2-2 times the time (repro/pyadjoint_checkpoint_memory.py, docs/log.md).
- The memory figures reproduce: in six full sweeps of the same code (CI run 1 and five later runs in
  [results/ci-later](results/ci-later/README.md)), the tape's growth at 1e5 steps varied by under 1 % (JAX +1587 to
  +1593 and +6266 to +6271 MB, pyadjoint +4430 to +4469 and +9244 to +9277 MB on 32 x 16 and 64 x 32). The time
  ratios do not: tape and checkpoint run as separate jobs on different runners, and at 1e5 steps checkpointing took
  0.88 to 2.12 times the tape's time for JAX and 1.19 to 3.24 times for pyadjoint across those runs, so a time
  comparison needs both strategies in one job.
- On a T4 GPU a growing JAX tape's device memory is what XLA's memory analysis of the compiled gradient predicts
  (to 1.1 MB; 6254.7 MB against 6254.4 MB at 64 x 32 and 1e5 steps), 128 x 64 at 1e5 steps is refused (it asks for
  24.4 GB, the allocator's limit is 14.6 GB), checkpointing needs at most 94 MB of temporaries by XLA's analysis,
  and gradients equal the CPU run's to 8.4e-14.
- JAX and pyadjoint match the exact gradient of the discrete scheme along uniform conductivity to 3e-13 or better at
  every step count and mesh (JAX 2.3e-13 including the T4, pyadjoint 3.0e-13), and every cell's JAX gradient
  matches its closed form to 1.1e-12 of the largest (spike/check_full_gradient.py); SciMLSensitivity's continuous adjoints
  are O(dt) off (1e-5 to 2e-5 at 1e5 steps), and BacksolveAdjoint is 89-100 % off, except at 1e5 steps on 32 x 16,
  where it reaches 5.3e289.
- k(T): with one Newton iteration per step, unrolled differentiation is 7-22 times closer to the converged gradient
  than implicit differentiation, along one random direction (the ratio depends on the direction); from three iterations both agree with it to 2.1e-14 or better.
  Firedrake (pyadjoint: hand-written Newton on the tape against a NonlinearVariationalSolver) gives the same J and
  gradients to 1e-14 ([results/kt-firedrake-local](results/kt-firedrake-local/README.md)); at two iterations which
  one is closer depends on the case.

Separately, repro/sciml_checkpoint_resolve.jl shows that SciMLSensitivity's checkpointed InterpolatingAdjoint
re-solves each checkpoint interval adaptively, ignoring a fixed dt, so its error has a floor set by the re-solve's
tolerances (the adjoint defaults, abstol = 1e-6 and reltol = 1e-3) rather than by the step. With one checkpoint
interval (saveat = [T], so the only checkpoints are the start and the end) the error stalls at the defaults (1.5,
0.78 and 0.72 % at 1e2, 1e3 and 1e4 steps, against 1.0, 0.099 and 0.0099 % without checkpointing), and with abstol =
1e-12 and reltol = 1e-10 (passed to solve) it is first order again (0.75, 0.075, 0.0077 %). Same root cause as
SciMLSensitivity#1705, where fixed-step-only methods raise an error; with ImplicitEuler nothing warns. With
checkpoints at isqrt(steps) times, as in the CI sweep, the effect is small.

## Studies

Three follow-up studies in [studies/](studies/), each with its criteria written before the run (PREREG.md), its
data, and a README that separates registered results from later observations. The two FD studies use a Jacobi-PCG
solver written for the study, not any mosaic backend; the Newton study uses Optimistix.

- [studies/fd_heldout](studies/fd_heldout/README.md): a best-step finite-difference gradient check, scored as mosaic
  scores it, with the FD forward at the solver's own tolerance. At CG rtol 1e-3 over one short step it accepted
  truncated-CG gradients 12-17 % off and rejected the exact gradient on the same 20/20 probe sets. Tightening only
  the FD forward to 1e-12 (not registered) stopped it accepting those (0/20).
- [studies/fd_configured](studies/fd_configured/README.md): the fork's configured 50-step FD workload with the same
  hypothetical CG. No truncated gradient crossed 10 % (registered endpoint 0/30; largest 9.1 %), but at 1e-3 the
  check rejected the exact gradient on 20 of 30 fresh fields (observed, not registered).
- [studies/newton_heldout](studies/newton_heldout/README.md): Optimistix Newton stopped at rtol = atol = 1e-4 gives
  gradients within 6.1e-9 by both its default implicit adjoint and by differentiating the iterations (36 cases);
  needs `optimistix==0.1.0`.
