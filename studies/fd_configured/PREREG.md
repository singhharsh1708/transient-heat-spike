# Does the FD verdict reversal reach the fork's configured thermal-transient FD experiment? (fixed 2026-11-21, before the run)

These criteria were written before the run but committed together with its results (commit b932232), so the
order rests on this statement, not on a timestamp.

Workload = mosaic fork config gradient/fd_check: 8x4 slab (the single-layer HEX8 mesh with z-uniform data is
equivalent to Q1 2D up to a factor Lz that leaves CG iterates unchanged), density rho = clip(0.5 + 0.3 N(0,1), 0.05,
0.95) in float32, SIMP k = 1e-3 + (1 - 1e-3) rho^3, T0 = sin(pi x / 2Lx), t_end = 1/(k(0.5) (pi/4)^2), 50 backward
Euler steps, J = sum of nodal T^2. Gradient with respect to rho.

Solver under test: Jacobi-PCG per step, warm-started, stopped at rtol (atol 0). Primary rtol 1e-3; control 1e-6.
Candidates: unrolled (AD through the executed iterations) and exact (dense solves + discrete adjoint).
FD as mosaic: 6 float32 unit normal directions (probe seed 0), eps in {1e-1, 1e-2, 1e-3, 1e-4}, h = eps * rms(rho),
pass = min-over-eps median relative error <= 1% AND max-over-eps cosine >= 0.99 (the fork's absolute checks).
Outputs scored both in float64 and rounded to float32 (the tesseract output dtype). Peer check not modelled.

Fields: 30 fresh seeds 1000-1029, plus the configured seed 0 reported separately.
Endpoint: number of fields with unrolled E > 10%, unrolled accepted and exact rejected.
Prediction: none, I do not know. Zero at 1e-3 confines the claim to the short-step workload.
