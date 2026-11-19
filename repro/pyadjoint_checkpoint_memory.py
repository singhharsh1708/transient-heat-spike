# Where pyadjoint's memory goes with and without Revolve checkpointing, on the spike's Firedrake backend.
#
#   for m in tape checkpoint checkpoint-gc; do for n in 1000 4000; do
#     docker run --rm -v "$PWD":/work -w /work firedrakeproject/firedrake-vanilla-default:2026.10.0 \
#       python3 repro/pyadjoint_checkpoint_memory.py $m $n [--mesh 128 64] [--trace]
#   done; done
#
# One mode and step count per process, since the peak is the process's high-water mark, measured over a 10-step
# annotated derivative in the same mode (so compilation is outside it). The growth has a part that does not depend
# on the step count, so the cost per step is the slope between two step counts: (peak at 4000 - peak at 1000) / 3000.
# It also prints how many tape variables still hold a stored state or an adjoint value after the derivative.
# --trace adds the Python heap (tracemalloc, which inflates the peak, so take peaks from runs without it).
# checkpoint-gc is Revolve with gc_timestep_frequency=50. (Measured on the :latest image of 2026-11-20, which is
# Firedrake 2026.10.0, pyadjoint 2026.10.0, checkpoint_schedules 1.0.4.)

import argparse
import gc
import math
import resource
import sys
import time
import tracemalloc

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["tape", "checkpoint", "checkpoint-gc"])
ap.add_argument("steps", type=int)
ap.add_argument("--mesh", type=int, nargs=2, default=[32, 16], metavar=("NX", "NY"))
ap.add_argument("--trace", action="store_true")
a = ap.parse_args()
sys.argv = sys.argv[:1]  # PETSc reads the command line when Firedrake starts
sys.path.insert(0, ".")
from checkpoint_schedules import MultistageCheckpointSchedule  # noqa: E402
from firedrake import Function  # noqa: E402
from firedrake.adjoint import (  # noqa: E402
    Control, ReducedFunctional, continue_annotation, get_working_tape, pause_annotation,
)

from backends.firedrake.heat import Heat  # noqa: E402
from spike.problem import Problem  # noqa: E402

mode, steps = a.mode, a.steps
h = Heat(Problem(nx=a.mesh[0], ny=a.mesh[1]))


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def start(n):
    """A clean tape for n steps in this mode; returns the control and the time stepper."""
    tape.clear_tape()
    gc.collect()
    if mode.startswith("checkpoint"):
        kw = {"gc_timestep_frequency": 50} if mode == "checkpoint-gc" else {}
        tape.enable_checkpointing(MultistageCheckpointSchedule(n, math.ceil(math.sqrt(n)), 0), **kw)
        return Function(h.D).assign(1.0), tape.timestepper
    return Function(h.D).assign(1.0), None


tape = get_working_tape()
rho, stepper = start(10)  # warm-up: compiles the forward and adjoint forms
continue_annotation()
J = h.run(rho, 10, stepper)
pause_annotation()
ReducedFunctional(J, Control(rho)).derivative()
rho, stepper = start(steps)
base = peak_mb()
trace = a.trace
if trace:
    tracemalloc.start()
t0 = time.perf_counter()
continue_annotation()
J = h.run(rho, steps, stepper)
pause_annotation()
g = ReducedFunctional(J, Control(rho)).derivative()
wall = time.perf_counter() - t0
heap = f", Python heap {tracemalloc.get_traced_memory()[0] / 2**20:.0f} MB" if trace else ""
variables = {id(bv): bv for b in tape.get_blocks() for bv in [*b.get_dependencies(), *b.get_outputs()]}
states = sum(bv.checkpoint is not None for bv in variables.values())
adjoints = sum(bv.adj_value is not None for bv in variables.values())
grow = peak_mb() - base
print(f"{mode}, {a.mesh[0]} x {a.mesh[1]}, {steps} steps: peak +{grow:.1f} MB{heap}, {wall:.0f} s, "
      f"{len(tape.get_blocks())} blocks, {len(variables)} variables of which {states} hold a state and {adjoints} an "
      f"adjoint value, dJ = {g.dat.data_ro.sum():.12e}")
