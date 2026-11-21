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
  exact discrete gradient to 3e-13 or better everywhere (see the correction below). (GB here is 1024 MB, as in
  the tables.)
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
    50 steps removes it at 1.2-1.4 times the time on 128 x 64 (timings vary under load) and about 2 times on
    32 x 16, with the same gradient to 13 digits.
  The CI sweep used the default (no periodic collection). The time-dependent benchmark should report memory against
  step count for checkpointed runs too, and state the collector setting.
- repro/sciml_checkpoint_resolve.jl now also runs at abstol = 1e-12, reltol = 1e-10: the checkpointed
  InterpolatingAdjoint is then first order (0.75, 0.075, 0.0077 % at 1e2, 1e3, 1e4 steps), so the stall at the
  adjoint's default tolerances (abstol 1e-6, reltol 1e-3; 1.46, 0.78, 0.72 %) is the re-solve's tolerance error.
  The tolerances are passed to solve, which hands them to the adjoint. Tightening one is not enough here: reltol =
  1e-10 alone gives 0.78, 0.11, 0.039 %, abstol = 1e-12 alone 1.46, 0.78, 0.71 % (probe, not in the repro). SciMLSensitivity 7.119.14, interpolating_adjoint.jl: the first
  re-solve (lines 89-92; the only one in the repro, which has one checkpoint interval) gets the algorithm and
  tolerances only; later re-solves add an initial dt taken from the previous re-solve; for ODEs none passes
  adaptive = false or the forward dt (the SDE branches do re-solve on the forward grid), although the backward
  adjoint solve itself does receive them. Same root cause as SciMLSensitivity#1705.
- Times for Revolve without periodic collection against gc_timestep_frequency = 50
  (repro/pyadjoint_checkpoint_memory.py, this laptop under load): 10 s against 14 s and 46 s against 58 s at 1000
  and 4000 steps on 128 x 64; 5 s against 10 s and 21 s against 40 s on 32 x 16.
- T4 (Kaggle, results/t4-run-1). Run 1 failed in 2 minutes: the image's jax-cuda13 0.11.1 plugin and the
  installed jax-cuda12 0.12.1 one both registered "cuda", JAX 0.12.1 discovers plugins from a set so each process
  started whichever won, and the old one rejected the new client option memory_fraction_policy (7 of 13 cases).
  Run 2 removes the image's plugin and probes with a real tiny case first. Result: device memory of the tape equals
  XLA's memory analysis of the compiled gradient to 1.1 MB; tape at 128 x 64, 1e5 steps refused (24.42 GiB against
  a 14.6 GB allocator limit); checkpointing at most 94 MB of temporaries; gradients equal to the CPU CI run's to
  8.4e-14 relative. So for the growing tapes XLA's memory analysis, which works on any platform without running
  the case, predicted the GPU memory; whether it does for checkpointed cases is not measured (they stay under the
  96 MB device floor).
- Correction to the gradient and J errors above and on 2026-11-19: the "exact discrete" reference in
  spike/problem.py was itself up to 2.1e-11 off at 1e5 steps. It raised the rounded 1 + dt lam to the power -2N,
  which amplifies that rounding by 2N, and the dense cross-check used the same formula, so the two agreed; on
  128 x 64 eigh's lowest eigenvalue was also about 1e-12 off, about 2e-12 in J at every step count. Against a
  40-digit evaluation of the same closed form, JAX's gradients (CPU and T4) are within 2.3e-13 and pyadjoint's
  within 3.0e-13, and JAX's forward J within 2.7e-13, so the "2e-11 at 1e5 steps" (and the "forward J drifts by the
  same amount" of 2026-11-19) was the reference's error, not the solvers'. JAX's gradient error is about 5e-15,
  3e-14 and 1.3e-13 on the three meshes up to 1000 steps and at most 2.3e-13 (64 x 32, tape, 1e5 steps); its J error
  depends on the mesh, not the step count (about 2e-14, 7e-14, 2.6e-13). discrete_exact now uses the single-mode
  closed form (the nodal sin(pi x / 2 Lx) is an exact eigenvector of the 1D pencil) with exp(-2N log1p(dt lam1)):
  within 4.5e-16 of the 40-digit value on every mesh and step count here. (spike/check_reference_mp.py, added
  2026-11-21, reruns that comparison on the CI meshes: worst 4.5e-16.)
- k(T) on Firedrake (backends/firedrake/heat_kt.py): the JAX scheme in UFL, with the cell average of T through
  cell_avg (on a bilinear rectangle the mean of the nodal values, as in JAX); unrolled = Newton iterations written
  out as taped linear solves, implicit = a NonlinearVariationalSolver forced to take exactly n iterations
  (snes_convergence_test skip), which pyadjoint differentiates by the implicit function theorem. All 27 cases agree
  with the JAX records to 1e-14 (J, and the gradient along uniform conductivity and along the JAX random direction
  mapped to Firedrake's cell order), so the unrolled-vs-implicit result is the scheme's, not one AD system's: 7-22
  times closer at one Newton iteration, and at two it depends on the case, as with JAX. Measured along uniform
  conductivity alone the two-iteration ratios are 0.7 at every step count, so the direction matters there. Local run
  in results/kt-firedrake-local; CI job kt-firedrake.
- Corrections and checks after an outside review (2026-11-20):
  - Every cell's gradient, not only the sum: the initial temperature is an exact eigenvector and the adjoint stays
    in the same mode, so dJ/dk_e = (s' K_e s / s' K s) dJ/deps. JAX's per-cell gradients match that to 1.1e-12 of
    the largest cell gradient on all three meshes, 10 to 1e4 steps, tape and checkpoint
    (spike/check_full_gradient.py). The mode does not depend on y, so this checks the x-dependence of the gradient:
    cells in one column have equal shares, and an error that permuted cells within a column would pass.
  - BacksolveAdjoint is silent: at 1000 and 1e5 steps on 32 x 16 the process exits 0 with nothing on stderr (no
    warning, no error; the harness's ok flag means no exception), while its gradient is 99 % off and -1.2e287 (CI
    run 1 had 5.3e289 for the same case).
  - JAX checkpointing keeps steps / c + c states with c the largest divisor of steps not above isqrt(steps); for a
    prime step count c = 1 and it keeps every step. The sweep's step counts all have a divisor near the square root.
  - The k(T) ratios are along one random direction; along uniform conductivity they differ (see above).
  - OrdinaryDiffEq timing: with adaptive = false, OrdinaryDiffEqDifferentiation's do_newJW rebuilds J and W at every
    step ("Not adaptive will always refactorize"); only a linear SciMLOperator skips it. With a MatrixOperator and
    the mass matrix the forward builds W twice instead of 10001 times but is not faster (2.2 s against 1.8 s,
    32 x 16, 1e4 steps; J differs by 1e-11, since that path solves with a Krylov method rather than a
    factorisation). A plain loop with one LU takes about 0.14 s for the same steps (three runs; an earlier
    measurement under heavy load gave 0.47 to 0.55 s), about 13 times faster than the ODE forward. So there is no
    like-for-like Julia timing within the current API, and Julia times are not compared with the others.
