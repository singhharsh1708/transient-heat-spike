# Held-out Newton study: criteria fixed before the run (2026-11-21)

These criteria were written before the run but committed together with its results (commit b932232), so the
order rests on this statement, not on a timestamp.

Question: at the tolerances people use (Optimistix Newton, rtol = atol in {1e-2, 1e-4, 1e-8}), are per-step Newton
gradients from Optimistix's default ImplicitAdjoint and from differentiating the iterations
(RecursiveCheckpointAdjoint) accurate on inputs not used before?

Inputs: meshes 12x6 and 20x10; heterogeneous lognormal rho = exp(0.4 z), field seeds 200-202; conductivity
k = rho * g(Tbar) with Tbar the cell mean and g in {1 + T, exp(T), 1 + 2 T^2}; initial temperature = eigenmode +
0.3 noise (same seed); backward Euler, 10 and 100 steps to t_end. Loss J = T'MT at t_end. Reference: FP64 Newton to
roundoff with hand-assembled Jacobians and a discrete adjoint, itself checked against central FD of the reference
forward along one random direction (relative error < 1e-6 or the case is reported as a reference failure).
E = relative L2 error of the full gradient.

H1: at 1e-4 both routes give E < 1e-6 in every case. One case above kills it.
H2: at 1e-2 some case has E > 1e-3 for at least one route. Reported either way.
All rows kept.
