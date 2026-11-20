# Held-out FD check study: criteria fixed before the run (2026-11-21)

Question: does mosaic's best-step FD check (six random unit directions, central differences at
eps * rms(input), eps in {1e-1, 1e-2, 1e-3, 1e-4}; pass when at one eps the median symmetric relative error over
directions is <= 1% and the cosine is >= 0.99) pass a gradient that is more than 10% wrong, on inputs that were not
hand-built (no eigenmode warm start)?

Inputs (none used before): meshes 12x6 and 16x8; lognormal conductivity exp(0.5 z), field seeds 100-104;
initial temperature = slab eigenmode + 0.3 * M-normalised Gaussian noise (same seed); backward Euler with
(steps, dt) in {(1, t_end), (1, t_end/100), (10, t_end/10)}; Jacobi-PCG per step, warm-started from the previous
state, stopping at rtol in {1e-3, 1e-4, 1e-5, 1e-6, 1e-8}, atol 0; observables J1 = T'MT and J2 = mean T on the
right face, at the end time. FD probe seeds 0-3 (six directions each). Every perturbed forward solve runs its own
stopping test.

Gradients scored: unrolled (AD through the CG iterations executed at the base point) and implicit (exact transpose
solve at each returned state). Reference: dense exact solves and the discrete adjoint. E = relative L2 error of the
full per-cell gradient.

Definitions: wrong = E > 0.10. False pass = the check passes a wrong gradient for a probe seed.

H1: some unrolled gradient with E > 0.10 passes at rtol 1e-3 or 1e-4 on BOTH meshes. Killed if either mesh has none.
H2 (control): the check rejects nothing with E < 0.01 (no false failures of accurate gradients). Reported either way.
H3: implicit gradients with E > 0.10 also pass. Reported either way; no prediction.
All rows are kept in results.csv; nothing is dropped after the run.
