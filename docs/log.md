# Working log

## 2026-11-19

- JAX backend: the first gradient was wrong although the forward matched the dense reference to 1e-13 (16 x 8,
  100 steps: 1.3e-4 off along uniform conductivity, 3.5e-4 against central differences along a random direction).
  Cause: the operator handed to `jax.scipy.sparse.linalg.cg` was not symmetric (free-node rows read the
  Dirichlet nodes' values, Dirichlet rows were the identity). CG's derivative solves with the same operator,
  assuming symmetry, so the adjoint was wrong while the forward was not (the Dirichlet values are zero). Masking the
  input as well as the output fixed it (gradient against central differences 2e-11). Only an FD check shows this;
  it is the kind of failure mosaic's intervention records describe.
- Julia backend: SciMLSensitivity's adjoints are continuous adjoints, so their gradient differs from the exact
  gradient of the discrete scheme by O(dt): Quadrature and Gauss 1.0e-3, 1.0e-4, 1.0e-5 at 1e3, 1e4, 1e5 steps
  (32 x 16). JAX matches the discrete exact gradient to 1e-12 (2e-11 at 1e5 steps).
- InterpolatingAdjoint(checkpointing = true) re-solves each checkpoint interval with the forward algorithm at the
  solve tolerances, adaptively: it does not pass on dt / adaptive = false (SciMLSensitivity 7.119.14,
  concrete_solve.jl and interpolating_adjoint.jl). Saving only t_end gives one interval, the whole run, so the first
  version's "checkpointed" gradient was a coarse adaptive reconstruction: 0.65-0.9 % off at every step count and
  with flat memory for that reason (reltol 1e-12 instead makes the re-solve hit maxiters and the gradient 20 % off).
  With checkpoints at isqrt(steps) times it is 2.1e-3 off at 1000 steps on 16 x 8, close to the non-checkpointed
  InterpolatingAdjoint (2.0e-3). Worth an upstream issue.
- Checking against the continuous closed form alone would have missed the adjoint bug above (3.5e-4) on the 32 x 16
  mesh, whose spatial error is 6e-4; it can also make an O(dt) adjoint look better than an exact one when the two
  errors partly cancel. Hence the discrete exact gradient (spike/problem.py: discrete_exact, by separation of
  variables; exact on every mesh; against it JAX is within 1e-13 up to 1000 steps and 2e-11 at
  100000, where the forward J drifts by the same amount: rounding over the steps).
- BacksolveAdjoint returns a gradient 96 % off (98 % at 1000 steps) with no error: it integrates the heat equation
  backwards in time.
- Firedrake 2026.10.0 (pyadjoint 2026.10.0, checkpoint_schedules 1.0.4): forward J equal to the JAX and dense
  results to 14 digits; gradient against FD 3e-12, both strategies.
- Memory is measured as the process's peak RSS after a 10-step warm-up (`baseline_rss_mb`) and after the measured
  run (`peak_rss_mb`); Julia's warm-up peak (compilation, about 1.5 GB) hides small growth.
- k(T) follow-on (JAX, local, dense Newton solves): unrolled differentiation's memory grows with the Newton
  iteration count and implicit's does not; both take longer with more iterations, implicit more slowly. With one
  Newton iteration the unrolled gradient is closer to the converged gradient than the implicit one (6.5e-4 vs
  4.6e-3 at 10 steps, 1.2e-5 vs 2.1e-4 at 50, 5.8e-7 vs 1.3e-5 at 200); at two iterations it depends on the case;
  from three, both agree with the converged gradient to 1e-14 or better.
- spike/run_case.py first killed only /usr/bin/time on a timeout and left the solver running (and overwriting the
  failure record); it now runs each case in its own process group and kills the group on timeout or at a memory
  limit.
- OrdinaryDiffEq's cost: for 1e4 fixed steps of this linear problem, ImplicitEuler evaluates the Jacobian and
  builds and factorises W at every step (DEStats: 10001 Jacobians, 10001 W, 10003 linear solves; 1.8 s forward on
  32 x 16), although W is constant. JAX does CG per step and Firedrake factorises once (constant_jacobian), so the
  Julia times include 1e4 sparse factorisations and are not a like-for-like speed comparison. Not yet tried: a
  linear-operator formulation (SciMLOperators) or a solver option that keeps W for a constant step.


## 2026-11-20

- CI run 1 (results/ci-run-1, run 44586316772): every case ran; the summary job failed while plotting because
  BacksolveAdjoint's error reached 5.3e289 at 1e5 steps on 32 x 16 and broke matplotlib's log-axis ticks (errors
  above 1e3 are now drawn as a triangle at the cap). Full tape at 1e5 steps: JAX +1.5 GB (32 x 16) and +6.1 GB
  (64 x 32), and on 128 x 64 XLA refuses up front (it asks for 26216674456 bytes, 24.4 GB); pyadjoint +4.4 and
  +9.0 GB, killed at 13 GB on 128 x 64. JAX checkpointing stays within +102 MB on every mesh, at 1.6-1.8 times the
  tape's time at 1e5 steps on the two meshes where the tape finished. Gradients of JAX and pyadjoint agree with the
  exact discrete gradient to 2.1e-11 or better everywhere. (GB here is 1024 MB, as in the tables.)
- pyadjoint's Revolve does not give flat memory: +0.64, +0.84 and +1.6 GB at 1e5 steps on the three meshes
  (against +4.4, +9.0 and over 13 GB for its tape), at 2.2-3.3 times the tape's time, with pyadjoint's default
  settings; from 1e4 to 1e5 steps it grows 6.5, 8.4 and 15.6 KB/step. repro/pyadjoint_checkpoint_memory.py
  (Firedrake 2026.10.0, pyadjoint 2026.10.0, checkpoint_schedules 1.0.4; slopes between 1000 and 4000 steps, after
  a 10-step annotated warm-up): with Revolve 3 of the tape's variables hold a state after the derivative at any
  step count (the full tape: every step's state and adjoint value, 45.6 KB/step on 32 x 16 and 290 KB/step on
  128 x 64). Revolve grows 6.6 KB/step plus 15 MB on 32 x 16 and 15.9 KB/step plus 96 MB on 128 x 64. Two parts:
  - the tape's own records, three blocks per step: with garbage collected every 50 steps
    (gc_timestep_frequency=50) the growth is 5.2 and 5.8 KB/step and the Python heap grows 4.4 KB/step on both
    meshes (tracemalloc), about 0.5 GB at 1e5 steps that no setting here removes;
  - garbage from the recomputation, which grows with the mesh and waits for the cycle collector; collecting every
    50 steps removes it at 1.3-1.4 times the time on 128 x 64 and about 2 times on 32 x 16, with the same gradient
    to 13 digits.
  The CI sweep used the default (no periodic collection). The time-dependent benchmark should report memory against
  step count for checkpointed runs too, and state the collector setting.
