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
to t_end, so forward results agree to machine precision and differences come from how the gradient is taken.

## Backends and strategies

| backend | strategy | what is kept for the reverse sweep |
|---|---|---|
| JAX (`backends/jax/heat.py`) | `tape` | every step (plain `jax.grad` through `lax.scan`) |
| | `checkpoint` | about sqrt(steps) states; chunks recomputed (`jax.checkpoint`) |
| Firedrake + pyadjoint (`backends/firedrake/heat.py`) | `tape` | every step (default tape) |
| | `checkpoint` | Revolve with sqrt(steps) snapshots (`checkpoint_schedules`) |
| OrdinaryDiffEq + SciMLSensitivity (`backends/julia/heat.jl`) | `interp` | checkpoints (`InterpolatingAdjoint(checkpointing = true)`) |
| | `quadrature`, `gauss` | the dense forward solution |
| | `backsolve` | nothing: the state is integrated backwards (ill-posed for diffusion) |

JAX and pyadjoint differentiate the discrete scheme; SciMLSensitivity's adjoints are continuous adjoints solved at
the same step, so their gradient differs from the discrete one by O(dt).

`backends/jax/heat_kt.py` is the follow-on: k = k0 rho (1 + T), Newton per step, `unrolled` (reverse mode through
the Newton iterations) against `implicit` (`jax.lax.custom_root`), over the number of Newton iterations.

## Running

```sh
uv venv && uv pip install "jax[cpu]" numpy scipy
.venv/bin/python backends/jax/heat.py --nx 32 --ny 16 --steps 10000 --strategy tape --fd
julia --project=backends/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=backends/julia backends/julia/heat.jl --nx 32 --ny 16 --steps 10000 --strategy gauss
docker run --rm -v "$PWD":/work -w /work firedrakeproject/firedrake-vanilla-default:2026.10.0 \
  python3 backends/firedrake/heat.py --nx 32 --ny 16 --steps 10000 --strategy checkpoint
```

`.github/workflows/spike.yml` runs the whole sweep (10 to 100000 steps, three mesh sizes) on GitHub's CPU runners,
one process per case, and `spike/summarise.py` turns the results into tables and plots. Results: `results/`.
