# Working log

## 2026-11-19

- JAX backend: the first gradient was off by about 1e-4 relative although the forward matched the dense reference
  to 1e-13. Cause: the operator handed to `jax.scipy.sparse.linalg.cg` was not symmetric (free-node rows read the
  Dirichlet nodes' values, Dirichlet rows were the identity). CG's derivative solves with the same operator,
  assuming symmetry, so the adjoint was wrong while the forward was not (the Dirichlet values are zero). Masking the
  input as well as the output fixed it (gradient against central differences 2e-11). Only an FD check shows this;
  it is the kind of failure mosaic's intervention records describe.
- Julia backend: SciMLSensitivity's adjoints are continuous adjoints, so at 100 steps the gradient differs from the
  derivative of the backend's own forward result by 1-1.5 % (O(dt); 0.1 % at 1000 steps for Quadrature and Gauss,
  0.9 % for InterpolatingAdjoint with checkpointing). JAX and pyadjoint match FD of their forward to 1e-11.
- BacksolveAdjoint returns a gradient 96 % off with no error: it integrates the heat equation backwards in time.
- Firedrake 2026.10.0 (pyadjoint 2026.10.0, checkpoint_schedules 1.0.4): forward J equal to the JAX and dense
  results to 14 digits; gradient against FD 3e-12, both strategies.
- Memory is measured as the process's peak RSS after a 10-step warm-up (`baseline_rss_mb`) and after the measured
  run (`peak_rss_mb`); Julia's warm-up peak (compilation, about 1.5 GB) hides small growth.
- k(T) follow-on (JAX, local): unrolled differentiation's memory and time grow with the Newton iteration count,
  implicit's do not; with one Newton iteration the unrolled gradient is closer to the converged gradient than the
  implicit one (6.5e-4 vs 4.6e-3 at 10 steps); from three iterations both are at machine precision.
