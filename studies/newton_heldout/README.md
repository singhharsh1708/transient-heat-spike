# Held-out Newton study: tolerance-stopped Newton gives accurate gradients either way (2026-11-21)

Criteria in PREREG.md (written before the run, committed with the results). From studies/newton_heldout, `python -B sweep.py > results.csv` with the `studies` extra (optimistix==0.1.0) installed
(about 90 s). 216 runs: meshes 12x6 and 20x10, three held-out lognormal rho fields, three
conductivity laws (1+T, exp T, 1+2T^2), 10 and 100 steps, rtol = atol in {1e-2, 1e-4, 1e-8}, Optimistix's default
ImplicitAdjoint and RecursiveCheckpointAdjoint (differentiating the iterations). All solves converged. The reference
agrees with central FD of its own forward to 2.0e-8 or better in every case.

| tol | Newton iterations per step | worst E, implicit | worst E, unrolled |
|---|---|---|---|
| 1e-2 | 1-4 | 8.4e-5 | 1.0e-5 |
| 1e-4 | 2-5 | 6.1e-9 | 5.4e-9 |
| 1e-8 | 3-6 | 2.0e-14 | 2.3e-14 |

- H1 holds: at 1e-4 (the Z-pinch paper's listing, [arXiv 2511.13262](https://arxiv.org/abs/2511.13262)) both routes are within 6.1e-9 in all 72 runs.
- H2 fails: even at 1e-2 no gradient is off by more than 8.4e-5. The stopping test catches Newton after its
  quadratic step, so the returned state is far more accurate than the tolerance, and so are both gradients.
- Interpretation (not tested here): this is consistent with the 7-22x gap in the spike's k(T) result coming from its
  forced single Newton iteration per step. Some steps at 1e-2 stopped after one iteration and were still accurate,
  so iteration count alone does not decide it. Memory was not measured in this study.
