# Same-tolerance FD check through a served backend of a mosaic fork (2026-11-21)

The fd_heldout mechanism, run through the torch-fem thermal-transient backend of an open fork PR
(singhharsh1708/mosaic#4, commit 9ef61ca = 66f4ce6 after a rebase; upstream mosaic has neither the problem nor this
backend) instead of a standalone script: mosaic's own FD functions and `gradient/fd_check`
status checks, scored against an exact discrete adjoint assembled here without torch-fem. 30 lognormal density
fields (seeds 2000-2029), nx = 8, both adjoint modes, `cg_rtol` 1e-3 and 1e-6, one step of t_end/100 and the
50-step fd_check workload. Counts in summary.txt, all rows in results.csv; how to rerun is in fd_served.py.

- One loose short step (`cg_rtol` 1e-3): the check accepted unrolled gradients more than 10 % off on 20 of 30
  fields and rejected the exact gradient on all 30.
- The 50-step workload at 1e-3: no gradient more than 1.9 % off; the exact gradient rejected on 6 of 30. This is
  milder than studies/fd_configured (9.1 %, 20 of 30) because these lognormal fields have a median SIMP conductivity
  contrast of 29 against 762 for the clipped-Gaussian fields there (14 cells in 13 of the 30 fields sit at the
  0.95 clip).
- `cg_rtol` 1e-6, the backend's default: every gradient within 0.16 % and every check passed.

Not a criterion fixed before the run; it repeats fd_heldout's design through a served backend.
