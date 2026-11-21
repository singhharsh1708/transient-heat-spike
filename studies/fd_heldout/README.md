# Same-tolerance FD accepts five truncated-CG energy gradients and rejects their converged counterparts

Finite differences (FD) of the stopped solver accepted a gradient with over 10% error against the converged discrete heat solve and rejected the exact gradient on the same 20/20 probe sets. The five qualifying cases all measured energy after one short step at CG relative tolerance `1e-3`. [Study criteria, dated 2026-11-21](PREREG.md)

## Study design

Q1 backward-Euler heat uses lognormal cell conductivities `exp(0.5 z)` on 12×6 and 16×8 meshes. Seeds 100–104 repeat across meshes, so each 72-cell field equals the first 72 entries of its 128-cell counterpart. The initial temperature is the mass-normalized slab mode plus a mass-normalized Gaussian vector of weight 0.3. Jacobi-PCG starts from the previous temperature. Outputs are mass-weighted terminal energy and the arithmetic mean of the right-face nodes. Two schedules reach `t_end` in one or ten steps. The third stops after one step at `t_end/100`. [Case generator](sweep.py)

Each of four probe seeds gives six unit Gaussian directions. Central FD uses `h = eps × rms(k)` for `eps` in `{0.1, 0.01, 0.001, 0.0001}`. Both signs re-run the stopping test. A pass requires median symmetric directional error ≤1% and six-direction cosine ≥0.99 at one epsilon. Full-gradient error `E` is the relative L2 error against dense solves and their discrete adjoint.

Mosaic's thermal-transient configuration uses the same direction count, epsilon grid and RMS scaling. It uses FP32 directions, a squared-output loss, separate best epsilons for the two absolute checks, and a 50× peer-median check when at least three entries are valid and their median is positive. Here the draws are FP64, the observables are scored directly, and only absolute checks are evaluated. Using separate extrema changes zero archived verdicts. The configured Mosaic workload uses random density and 50 steps on an 8×4×1 mesh. [Configuration](https://github.com/singhharsh1708/mosaic/blob/749d77576b71a21b72b4af4b00d0e0f80b772f28/mosaic/benchmarks/problems/thermal_transient/config.py) · [FD kernel](https://github.com/singhharsh1708/mosaic/blob/749d77576b71a21b72b4af4b00d0e0f80b772f28/mosaic/benchmarks/problems/shared/gradient.py) · [Status aggregation](https://github.com/singhharsh1708/mosaic/blob/749d77576b71a21b72b4af4b00d0e0f80b772f28/mosaic/benchmarks/core/status.py)

## Results against the criteria in PREREG.md

The matched-tolerance run contains 300 cases, 1,200 case/probe sets and 4,800 epsilon rows. Unrolled AD replays the SciPy iteration counts in JAX. The implicit estimate uses exact transpose solves along the approximate trajectory. [Matched-tolerance results](results.csv)

| Matched-tolerance FD | Unrolled | Implicit | Converged reference |
|---|---:|---:|---:|
| Cases with E > 10% | 5/300 | 0/300 | 0/300 |
| Qualifying probe sets accepted | 20/20 | — | — |
| Rejected among probe sets with E < 1% | 0/1,092 | 112/1,200 | 115/1,200 |
| Rejected in the short, one-step energy group at rtol 1e-3, regardless of E | 0/40 | 40/40 | 40/40 |

H1 holds. Five of ten short, loose energy cases exceeded 10% error (12.465–16.774%), two on 12×6 and three on 16×8. The other 290 cases had none. H2 fails. The exact gradient was rejected 88 times at `1e-3` and 27 at `1e-4`. H3 had no qualifying implicit gradient. The maximum unrolled error at `1e-5` or tighter was 0.3145%.

On 16×8, field 102, the base solve and every signed perturbation used seven updates. For probe 0 at `eps = 1e-4`, the median directional error was `2.5854e-9` for the unrolled gradient and 11.3638% for the reference. The FD/reference probe cosine was 0.993594. The unrolled **full-gradient** error was 13.0316%. Across the five qualifying cases, the unrolled gradient passed at all 80 probe/epsilon combinations. Cosine alone would accept the exact reference in 15/20 probe sets. [Numerical audit](checks/audit.log)

Not in PREREG.md, run after the main sweep: tightening only the FD forward to `1e-12` revisited the two loosest base tolerances (120 cases, 480 probe sets, 1,920 rows). Wrong unrolled gradients passed on 0/20 probe sets. The exact gradient had 0/480 rejections. Among gradients with E < 1%, the check rejected 16/372 unrolled and 7/480 implicit estimates. [Tight-FD results](results_fd_tight.csv)

## Scope and prior work

In a separate review of ten fresh seeds at `rtol = 1e-3` and `dt = t_end/100`, the one-step energy inversion occurred on 4/10 fields at 16×8, 7/10 at 32×16 and 8/10 at 64×32. At 16×8 it occurred on 8/10 with unpreconditioned CG, on 0/10 with pure-noise initial temperatures, and on 0/10 after 100 steps to the full end time. [Review results](checks/challenge_summary.log)

FD on a stable iteration branch measures the derivative of the executed solver. The fork's analytic check covers the uniform-density direction. Its steady thermal-mesh GPU torch-fem path uses CG at `1e-6` with an implicit adjoint, and steady JAX-FEM uses BiCGSTAB with relative and absolute tolerance `1e-10`. Both are excluded from the transient experiments. [Analytic reference](https://github.com/singhharsh1708/mosaic/blob/749d77576b71a21b72b4af4b00d0e0f80b772f28/mosaic/benchmarks/problems/thermal_transient/references.py) · [Backend exclusions](https://github.com/singhharsh1708/mosaic/blob/749d77576b71a21b72b4af4b00d0e0f80b772f28/mosaic/benchmarks/problems/thermal_transient/exclusions.py) · [torch-fem](https://github.com/singhharsh1708/mosaic/blob/749d77576b71a21b72b4af4b00d0e0f80b772f28/mosaic/tesseracts/thermal-mesh/torch-fem-thermal/tesseract_api.py) · [JAX-FEM](https://github.com/singhharsh1708/mosaic/blob/749d77576b71a21b72b4af4b00d0e0f80b772f28/mosaic/tesseracts/thermal-mesh/jax-fem/tesseract_api.py)

Moré and Wild (2014), §§3–5, compare AD and FD for tolerance-stopped Krylov solves. Theorem 2.1 of their preprint gives the fixed-iteration derivative away from stopping boundaries for a varying right-hand side. Gratton et al. (2014) derive finite-iterate CG Jacobians, and Christianson (2018), footnote 5, gives a CG early-termination counterexample. [Moré and Wild](https://doi.org/10.1016/j.jcp.2014.04.056) · [Theorem 2.1](https://optimization-online.org/wp-content/uploads/2012/05/3453.pdf) · [Gratton et al.](https://doi.org/10.1137/120889848) · [Christianson](https://doi.org/10.1080/10556788.2018.1425862)

Padway and Mavriplis (2020), §I and Figs. 6–9, discuss FD of unconverged solvers. Shu et al. (2026), Table 1 and Appendix F/Table 6, quantify a one-sweep equation-adjoint discrepancy and separately validate a finite-sweep derivative by central FD. This study measures the paired verdict under the specified heat protocol. [Padway and Mavriplis](https://doi.org/10.2514/6.2020-3136) · [Shu et al.](https://arxiv.org/html/2608.08559v1)

From the repository root, run `python -B studies/fd_heldout/checks/reproduce.py` to regenerate both CSVs and compare them with the archived ones (about 100 s and 52 s). `analyse.py` reads `results.csv` from its working directory. `checks/mosaic_loss.py` runs mosaic's own FD functions and needs `MOSAIC_CHECKOUT` set to a checkout of github.com/singhharsh1708/mosaic at 749d775. [Main replay](checks/reproduce.log) · [Tight replay](checks/reproduce_tight_cold.log) · [Scoring script](analyse.py) · [Shared helpers](../common.py)
