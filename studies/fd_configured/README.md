# The fork's configured FD experiment: endpoint not reached; correct gradients still rejected at 1e-3 (2026-11-21)

Criteria in PREREG.md, fixed before the run. `python -B run.py` (about 55 s)
writes results.csv. 30 fresh density fields (seeds 1000-1029) plus the configured seed 0, the fork's 8x4 slab, SIMP
conductivity, eigenmode start, 50 steps to t_end, J = sum of nodal T^2, mosaic's FD grid and absolute checks.

| CG rtol | unrolled E (min..max) | unrolled passes | exact gradient passes | registered endpoint |
|---|---|---|---|---|
| 1e-3 | 0.26 % .. 9.1 % | 30/30 | **10/30** | **0/30** |
| 1e-6 (control) | 2.7e-4 % .. 2.7e-2 % | 30/30 | 30/30 | 0/30 |

Rounding J to float32 changes no verdict. Seed 0 (the configured field) at 1e-3: unrolled E = 2.7 % passes
(median FD error 6e-10), the exact gradient fails (best median FD error 1.96 %).

- Registered endpoint (E > 10 %, unrolled accepted, exact rejected): **0 of 30**. The largest truncated-gradient
  error in the configured workload is 9.1 %, so the "passes a >10 % wrong gradient" claim stays confined to the
  short-step workload in ../fd_heldout.
- Not registered, reported as observed: at rtol 1e-3 the check rejects the exact gradient on 20 of 30 fields while
  accepting the executed-solver gradient on all 30. A backend with an exact adjoint at its loose forward's returned
  states (transpose solves along the loose trajectory) was rejected on 17 of 30
  (probe_implicit.py, probe_implicit.log; it also shows mosaic's J^2 loss changes no unrolled or exact verdict). At 1e-6 the effect is gone.
- The "configured workload" is the fork's FD experiment with a hypothetical Jacobi-PCG solver; the fork's actual
  transient backends (Firedrake, FEniCS) solve with LU. mosaic differentiates J^2; using J changes no verdict.
- Not modelled: the peer-outlier check, FP32 inputs inside the solver, Tesseract transport.
