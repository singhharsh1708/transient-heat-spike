# T4 run 1

The JAX sweep on Kaggle's GPU T4 x2 (one T4 used, 15 GB, driver 580.178.04), private kernel run by gpu/push.sh at
commit 4ad2e03 (the second attempt; the first failed at JAX start-up, see docs/log.md): JAX 0.12.1 with the
jax-cuda12 plugin (the same XLA as the CPU CI tables; `environment.json` lists the JAX and NVIDIA packages),
float64, one process per case through spike/run_case.py, cases one after another. `summary.md` and the plots were
made here from these records with `python spike/summarise.py results/t4-run-1 results/t4-run-1`.

- Device memory is the allocator's peak bytes in use; for a growing tape it matches XLA's memory analysis of the
  compiled gradient (`xla_temp_mb`) to within 1.1 MB (0.6 MB with its argument and output buffers added; 64 x 32
  at 1e5 steps: 6254.7 MB against 6254.4 MB). The device peak has a floor of about 96 MB, set before the measured
  run, so below it only XLA's analysis is available (in the records, not in the summary): by it, checkpointing
  needs at most 94 MB of temporaries, at 128 x 64 and 1e5 steps; that figure was not measured on the device.
- Tape at 128 x 64 and 1e5 steps is refused up front: "Out of memory while trying to allocate 24.42GiB", against an
  allocator limit of 14.6 GB.
- Gradients are within 2.3e-13 of the exact gradient of the discrete scheme, and gradients and J agree with the CPU
  CI run (results/ci-run-1) to 8.4e-14 and 1.5e-14 relative in every case both completed.
- Times are float64 on a T4: slower than the CPU runner on 32 x 16, at 10 steps on every mesh and at 100 steps on
  64 x 32; on 64 x 32 from 1000 steps checkpointing takes the same time or up to 6 % less and the tape 13-21 %
  longer; faster on 128 x 64 from 100 steps (checkpoint at 1e5 steps: 82.5 s against 118 s).
