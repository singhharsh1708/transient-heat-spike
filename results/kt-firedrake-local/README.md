# k(T) on Firedrake, local run

backends/firedrake/heat_kt.py on the 16 x 8 mesh (10, 50, 200 steps; 1, 2, 3, 5 Newton iterations; unrolled and
implicit; the 30-iteration implicit reference), run on 2026-11-20 in firedrakeproject/firedrake-vanilla-default
(Firedrake 2026.10.0, pyadjoint 2026.10.0, arm64), one process per case. The CI job `kt-firedrake` in
.github/workflows/spike.yml runs the same cases. The records compare with the JAX ones in results/ci-run-1
(`python spike/summarise.py <a folder holding both sets> <out>` prints the comparison):

- J, the gradient along uniform conductivity and the gradient along the JAX table's random direction (mapped to
  Firedrake's cell order) agree with JAX's to 8.0e-15, 5.9e-15 and 1.0e-14 relative in all 27 cases.
- Along that direction, with one Newton iteration per step the unrolled gradient is 7.1, 16.7 and 22.5 times
  closer to the converged gradient than the implicit one at 10, 50 and 200 steps, as in the JAX table; at two
  iterations it depends on the case (4.9, 0.31 and 0.08; in the last the implicit error is at round-off, so the ratio is noise, and JAX's is 0.13); from three both
  are at round-off.
- The reference agrees with central differences (along the same direction) to 4e-10, 9e-10 and 1e-9 at 10, 50 and
  200 steps.
