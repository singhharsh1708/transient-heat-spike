"""CI form of verify_gradients.py: the same checks with tolerances instead of a byte-exact log.

verify_gradients.log holds round-off quantities (about 1e-14) that differ between machines and BLAS builds; this
asserts they stay at round-off and that the JAX replay takes the same iteration counts as SciPy CG.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
out = subprocess.run([sys.executable, "-B", str(here / "verify_gradients.py")], stdout=subprocess.PIPE, text=True,
                     check=True, cwd=here).stdout
lines = out.strip().splitlines()
complex_step = [line for line in lines if line.startswith("complex-step")]
assert len(complex_step) == 4, out
worst = max(max(row) for line in complex_step for row in ast.literal_eval(line[line.index("[["):]))
summary = lines[-1]
assembly = float(re.search(r"assembly max abs (\S+)", summary).group(1))
replay = float(re.search(r"max rel state replay (\S+)", summary).group(1))
mismatches = int(re.search(r"count mismatches (\d+)", summary).group(1))
print(f"complex-step worst {worst:.2e}, assembly {assembly:.2e}, replay {replay:.2e}, count mismatches {mismatches}")
assert worst < 1e-12 and assembly < 1e-13 and replay < 1e-12 and mismatches == 0
