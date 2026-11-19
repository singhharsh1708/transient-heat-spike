# CI run 1

[Run 44586316772](https://github.com/singhharsh1708/transient-heat-spike/actions/runs/44586316772), commit ce64b6d,
2026-11-19, GitHub's `ubuntu-latest` CPU runners (4 cores, 16 GB), one process per case, killed at 13 GB.
All 23 experiment jobs passed. The run's `summary` job failed while plotting: backsolve's 5e289 error broke the
log axis (fixed in e9abd23). `summary.md` and the plots here were made from the run's artifacts by
`python spike/summarise.py results/ci-run-1 results/ci-run-1`, which is the summary that job would have printed.

- `summary.md`: memory, gradient time and gradient error per backend, strategy, mesh and step count; FD checks;
  the k(T) follow-on.
- `memory.png`, `gradient_error.png`, `time.png`: the same, plotted (a triangle marks an error above 1e3).
- `<backend>-<strategy>-<nx>x<ny>-<steps>.json`: one record per case, as each case wrote it.
- `fdcheck-*.json` (16 x 8 mesh, 100 steps; the run named them after their job's mesh, renamed here), `kt-*.json`: the
  finite-difference checks and the k(T) cases.
