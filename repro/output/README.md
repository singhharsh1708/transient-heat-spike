# Reproducer outputs (rerun 2026-11-21)

- `pyadjoint_checkpoint_memory.txt`: all three modes at 1000 and 4000 steps on 32 x 16 and 128 x 64. Slopes
  (peak at 4000 minus peak at 1000, over 3000 steps): tape 44.7 and 283 KB/step, Revolve 6.4 and 16.1 KB/step,
  Revolve with gc_timestep_frequency = 50 5.2 and 5.8 KB/step; periodic collection costs 1.3 to 1.6 times the time.
  docs/log.md's figures from the first run (45.6, 290, 6.6, 15.9, 5.2, 5.8) agree to about 3 %, run to run.
- `sciml_checkpoint_resolve.txt`: relative error of the checkpointed adjoint at 1e2, 1e3, 1e4 steps, 1.46, 0.78,
  0.715 % at the default tolerances and 0.746, 0.0746, 0.00766 % at abstol 1e-12 / reltol 1e-10, against 1.0, 0.099,
  0.0099 % without checkpointing, as the README quotes.
