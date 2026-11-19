"""Run one backend case in its own process and always leave a result file, also when the process is killed.

    python spike/run_case.py <out.json> <timeout_s> -- <command ...>

The command must accept `--out <file>`; this script adds it. On Linux the command runs under `/usr/bin/time -v`,
whose "Maximum resident set size" is recorded as external_peak_rss_mb, including for a process the kernel killed
for running out of memory. If the command left no result, a stub with ok = false and the exit status is written.
"""

import json
import os
import re
import subprocess
import sys
import time

out, timeout = sys.argv[1], float(sys.argv[2])
cmd = sys.argv[sys.argv.index("--") + 1 :] + ["--out", out]
timed = os.path.exists("/usr/bin/time") and sys.platform.startswith("linux")
full = (["/usr/bin/time", "-v"] if timed else []) + cmd
if os.path.exists(out):
    os.remove(out)
t0 = time.perf_counter()
try:
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    code, stderr = r.returncode, r.stderr
except subprocess.TimeoutExpired as e:
    code, stderr = "timeout", (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
wall = time.perf_counter() - t0
m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
if os.path.exists(out):
    with open(out) as fh:
        res = json.load(fh)
else:
    killed = code in (-9, 137, "timeout") or (isinstance(code, int) and code < 0)
    res = {"ok": False, "error": f"no result (exit status {code}{', killed' if killed else ''})",
           "stderr_tail": stderr[-1500:]}
res["exit_status"] = code
res["process_wall_s"] = wall
if m:
    res["external_peak_rss_mb"] = int(m.group(1)) / 1024
with open(out, "w") as fh:
    json.dump(res, fh)
print(json.dumps({k: res.get(k) for k in ("backend", "strategy", "nx", "steps", "ok", "exit_status", "warm_s",
                                           "peak_rss_mb", "external_peak_rss_mb", "error")}))
