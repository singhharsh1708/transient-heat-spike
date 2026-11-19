# Where pyadjoint's memory goes with and without Revolve checkpointing, on the spike's Firedrake backend.
#
#   for m in tape checkpoint checkpoint-gc; do for n in 1000 4000; do
#     docker run --rm -v "$PWD":/work -w /work firedrakeproject/firedrake-vanilla-default:2026.10.0 \
#       python3 repro/pyadjoint_checkpoint_memory.py $m $n
#   done; done
#
# One mode and step count per process, since the peak is the process's high-water mark, measured over a 10-step
# annotated derivative in the same mode (so compilation is outside it). The growth has a part that does not depend
# on the step count, so the cost per step is the slope between two step counts. It also prints how many tape
# variables still hold a stored state or an adjoint value after the derivative; a third argument `trace` adds the
# Python heap (tracemalloc, which inflates the peak, so take peaks from runs without it).
# With Revolve (ceil(sqrt(steps)) snapshots) no per-step state is held, yet memory still grows with the step count.

import gc
import math
import resource
import sys
import time
import tracemalloc

sys.path.insert(0, ".")
from checkpoint_schedules import MultistageCheckpointSchedule  # noqa: E402
from firedrake import Function  # noqa: E402
from firedrake.adjoint import (  # noqa: E402
    Control, ReducedFunctional, continue_annotation, get_working_tape, pause_annotation,
)

from backends.firedrake.heat import Heat  # noqa: E402
from spike.problem import Problem  # noqa: E402

mode = sys.argv[1]  # tape, checkpoint, or checkpoint-gc (collect garbage every 50 steps)
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
h = Heat(Problem(nx=32, ny=16))


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


if mode not in ("tape", "checkpoint", "checkpoint-gc"):
    raise ValueError(mode)


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
trace = sys.argv[3:] == ["trace"]
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
print(f"{mode}, {steps} steps: peak +{grow:.0f} MB ({grow * 1024 / steps:.1f} KB/step){heap}, "
      f"{len(tape.get_blocks())} blocks, {len(variables)} variables of which {states} hold a state and {adjoints} an "
      f"adjoint value, {wall:.0f} s, dJ = {g.dat.data_ro.sum():.12e}")
