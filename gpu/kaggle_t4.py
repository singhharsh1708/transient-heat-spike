"""The JAX sweep on a Kaggle T4: tape and checkpoint, three meshes, 10 to 100000 steps, plus an FD check each.

Pushed by gpu/push.sh, which sets COMMIT to the repository commit to run. Each case runs in its own process through
spike/run_case.py, as in CI, one after the other on one GPU (so the times are not shared with another case). A
strategy stops at its first failure on a mesh, since larger step counts would fail too. Results, the environment and
the summary go to /kaggle/working; the repository is cloned outside it, so it is not part of the output.

For a CPU test of this script: SPIKE_SMOKE=1 KAGGLE_WORK=<dir> SPIKE_REPO=<local clone> python gpu/kaggle_t4.py
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

COMMIT = "main"
REPO = os.environ.get("SPIKE_REPO", "https://github.com/singhharsh1708/transient-heat-spike.git")
MESHES = [(32, 16), (64, 32), (128, 64)]
STEPS = [10, 100, 1000, 10000, 100000]
SMOKE = os.environ.get("SPIKE_SMOKE") == "1"
if SMOKE:
    MESHES, STEPS = [(8, 4)], [10, 100]
CASE_TIMEOUT_S = 3600
HOST_MEM_LIMIT_MB = 13312  # per case, as in CI
STOP_AT = time.time() + 10 * 3600  # Kaggle ends GPU sessions at 12 h; a case may still run CASE_TIMEOUT_S after this

WORK = pathlib.Path(os.environ.get("KAGGLE_WORK", "/kaggle/working")).resolve()
OUT = WORK / "results"
OUT.mkdir(parents=True, exist_ok=True)
CODE = pathlib.Path(tempfile.mkdtemp()) / "spike"


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr[-2000:]).strip()


def jax_platform() -> str:
    return sh(f"{sys.executable} -c \"import jax; print(jax.__version__, jax.devices()[0].platform)\"")


subprocess.run(["git", "clone", "-q", REPO, str(CODE)], check=True)
subprocess.run(["git", "-C", str(CODE), "checkout", "-q", COMMIT], check=True)

# JAX with CUDA: the version CI pins (same XLA as the CPU tables); if that cannot be installed, the image's own,
# else the latest.
if not SMOKE:
    for pkg in ("jax[cuda12]==0.12.1", None, "jax[cuda12]"):
        if pkg:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg], check=False)
        if jax_platform().partition("\n")[0].endswith("gpu"):  # stdout first; stderr may follow
            break
    else:
        sys.exit(f"no GPU JAX: {jax_platform()}")

env = {
    "commit": sh(f"git -C {CODE} rev-parse HEAD"),
    "gpus": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "jax": jax_platform(),
    "python": sys.version.split()[0],
}
(WORK / "environment.json").write_text(json.dumps(env, indent=1))
print(env, flush=True)

case_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
if not SMOKE:
    case_env["JAX_PLATFORMS"] = "cuda"  # a silent fall back to CPU would be an error, not hours of CPU runs


def run(name: str, args: list[str]) -> dict:
    out = OUT / f"{name}.json"
    cmd = [sys.executable, str(CODE / "spike/run_case.py"), str(out), str(CASE_TIMEOUT_S),
           "--mem-limit-mb", str(HOST_MEM_LIMIT_MB), "--",
           sys.executable, str(CODE / "backends/jax/heat.py"), *args]
    subprocess.run(cmd, env=case_env, check=False)
    d = json.loads(out.read_text()) if out.exists() else {"ok": False, "error": "no result"}
    print(name, "ok" if d.get("ok") else str(d.get("error"))[:160], d.get("device_peak_mb"), d.get("warm_s"),
          flush=True)
    return d


for strategy in ("tape", "checkpoint"):
    run(f"fdcheck-jax-{strategy}-16x8", ["--nx", "16", "--ny", "8", "--steps", "100", "--strategy", strategy, "--fd"])
    for nx, ny in MESHES:
        for steps in STEPS:
            if time.time() > STOP_AT:
                print("not starting more cases: close to the session limit", flush=True)
                break
            d = run(f"jax-{strategy}-{nx}x{ny}-{steps}",
                    ["--nx", str(nx), "--ny", str(ny), "--steps", str(steps), "--strategy", strategy])
            if not d.get("ok"):
                break

subprocess.run([sys.executable, str(CODE / "spike/summarise.py"), str(OUT), str(WORK / "summary")], check=False)
summary = WORK / "summary" / "summary.md"
print(summary.read_text() if summary.exists() else "no summary")
