"""Run one backend case in its own process group and always leave a result file.

    python spike/run_case.py <out.json> <timeout_s> [--mem-limit-mb N] -- <command ...>

The command must accept `--out <file>`; this script adds it. The command runs in a new session (process group).
A watchdog samples the group's resident memory every 0.2 s (Linux: /proc) and kills the whole group when it passes
--mem-limit-mb or when the timeout expires, so a killed case leaves nothing running behind it and a memory failure
is recorded the same way every time instead of depending on swap or the kernel's OOM killer. Under /usr/bin/time -v
(Linux) the command's maximum resident set size is recorded as external_peak_rss_mb; the watchdog's largest
group sample is sampled_peak_rss_mb. If the command left no valid result, a stub with ok = false is written.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time

args = sys.argv[1:]
sep = args.index("--")
opts, cmd = args[:sep], args[sep + 1 :]
out, timeout = opts[0], float(opts[1])
mem_limit = float(opts[opts.index("--mem-limit-mb") + 1]) if "--mem-limit-mb" in opts else None
cmd = cmd + ["--out", out]
timed = os.path.exists("/usr/bin/time") and sys.platform.startswith("linux")
full = (["/usr/bin/time", "-v"] if timed else []) + cmd
if os.path.exists(out):
    os.remove(out)


def group_rss_mb(pgid: int) -> float:
    total = 0
    for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:  # field 5 of stat: pgrp
                continue
            with open(f"/proc/{pid}/status") as fh:
                m = re.search(r"VmRSS:\s+(\d+) kB", fh.read())
            total += int(m.group(1)) if m else 0
        except (OSError, ValueError, IndexError):
            continue
    return total / 1024


t0 = time.perf_counter()
proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
reason, sampled = None, 0.0
while proc.poll() is None:
    time.sleep(0.2)
    rss = group_rss_mb(proc.pid)
    sampled = max(sampled, rss)
    if mem_limit and rss > mem_limit:
        reason = f"killed: group memory {rss / 1024:.1f} GB passed the {mem_limit / 1024:.1f} GB limit"
    elif time.perf_counter() - t0 > timeout:
        reason = f"killed: timeout after {timeout:.0f} s"
    if reason:
        os.killpg(proc.pid, signal.SIGKILL)
        break
stdout, stderr = proc.communicate()
wall = time.perf_counter() - t0
m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)

res = None
if os.path.exists(out) and not reason:
    try:
        with open(out) as fh:
            res = json.load(fh)
    except (OSError, ValueError):
        res = None
if res is None:
    res = {"ok": False, "error": reason or f"no result (exit status {proc.returncode})", "stderr_tail": stderr[-1500:]}
res["exit_status"] = proc.returncode
res["process_wall_s"] = wall
res["sampled_peak_rss_mb"] = sampled
if m:
    res["external_peak_rss_mb"] = int(m.group(1)) / 1024
with open(out, "w") as fh:
    json.dump(res, fh)
print(json.dumps({k: res.get(k) for k in ("backend", "strategy", "nx", "steps", "ok", "exit_status", "warm_s",
                                           "peak_rss_mb", "sampled_peak_rss_mb", "error")}))
