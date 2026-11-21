# Five later full sweeps (raw records)

The result files of the five CI sweeps that ran after CI run 1, on pushes that did not change the backend compute
code: runs 44619522608 (42b3170), 44722533183, 44735036065, 44769292992 and 44785914340 (123966c). Same layout as
results/ci-run-1; `python spike/summarise.py results/ci-later/run-<id> <out>` rebuilds a summary.

What they show next to CI run 1 (32 x 16 and 64 x 32, 1e5 steps):
- memory reproduces: tape growth JAX +1587 to +1593 and +6266 to +6271 MB, pyadjoint +4430 to +4469 and +9244 to
  +9277 MB; JAX checkpoint +25 to +46 MB, Revolve +657 to +855 MB;
- time ratios do not: checkpoint / tape (warm) 0.88 to 2.12 for JAX and 1.19 to 3.24 for pyadjoint, since each
  strategy ran in its own job on its own runner;
- in run 44785914340 Firedrake checkpointing at 128 x 64 and 1e5 steps hit the 2400 s per-case limit (1710-1860 s
  in the other runs); the limit is now 3600 s.
