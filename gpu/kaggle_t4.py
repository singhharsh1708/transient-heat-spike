"""The JAX sweep on Kaggle's GPU T4 x2: tape and checkpoint, three meshes, 10 to 100000 steps.

Pushed by gpu/push.sh, which sets COMMIT to the repository commit to run. Each case runs in its own process
through spike/run_case.py, as in CI; tape runs on one GPU and checkpoint on the other, at the same time. A strategy
stops at its first failure on a mesh (larger step counts would fail too). Results, the environment and the summary
are written to /kaggle/working.
"""

import json
import os
import pathlib
import subprocess
import sys
import threading
import time

COMMIT = "main"
REPO = os.environ.get("SPIKE_REPO", "https://github.com/singhharsh1708/transient-heat-spike.git")
MESHES = [(32, 16), (64, 32), (128, 64)]
STEPS = [10, 100, 1000, 10000, 100000]
SMOKE = os.environ.get("SPIKE_SMOKE") == "1"  # a small CPU run of this script, to test it before a GPU session
if SMOKE:
    MESHES, STEPS = [(8, 4)], [10, 100]
CASE_TIMEOUT_S = 3600
HOST_MEM_LIMIT_MB = 13312  # per case, as in CI
STOP_AT = time.time() + 11 * 3600  # Kaggle ends GPU sessions at 12 h

WORK = pathlib.Path(os.environ.get("KAGGLE_WORK", "/kaggle/working"))
OUT = WORK / "results"
OUT.mkdir(parents=True, exist_ok=True)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def jax_on_gpu() -> str:
    return sh(f"{sys.executable} -c \"import jax; print(jax.__version__, jax.devices()[0].platform)\"")


os.chdir(WORK)
subprocess.run(["git", "clone", "-q", REPO, "spike"], check=True)
subprocess.run(["git", "-C", "spike", "checkout", "-q", COMMIT], check=True)

# JAX with CUDA: the image's own if it sees the GPU, else the version CI pins, else the latest.
for pkg in (None, "jax[cuda12]==0.12.1", "jax[cuda12]"):
    if SMOKE:
        break
    if pkg:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg], check=False)
    if jax_on_gpu().endswith("gpu"):
        break
else:
    sys.exit(f"no GPU JAX: {jax_on_gpu()!r}")

n_gpus = max(1, len(sh("nvidia-smi -L").splitlines()))
env = {
    "commit": sh("git -C spike rev-parse HEAD"),
    "gpus": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "jax": jax_on_gpu(),
    "python": sys.version.split()[0],
}
(WORK / "environment.json").write_text(json.dumps(env, indent=1))
print(env, flush=True)


def run(name: str, gpu: int, args: list[str]) -> dict:
    out = OUT / f"{name}.json"
    cmd = [sys.executable, "spike/spike/run_case.py", str(out), str(CASE_TIMEOUT_S),
           "--mem-limit-mb", str(HOST_MEM_LIMIT_MB), "--",
           sys.executable, "spike/backends/jax/heat.py", *args]
    case_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}
    subprocess.run(cmd, env=case_env, check=False)
    d = json.loads(out.read_text()) if out.exists() else {"ok": False, "error": "no result"}
    print(name, "ok" if d.get("ok") else d.get("error", "")[:120], d.get("device_peak_mb"), d.get("warm_s"),
          flush=True)
    return d


def sweep(strategy: str, gpu: int) -> None:
    run(f"fdcheck-jax-{strategy}-16x8", gpu, ["--nx", "16", "--ny", "8", "--steps", "100", "--strategy", strategy,
                                               "--fd"])
    for nx, ny in MESHES:
        for steps in STEPS:
            if time.time() > STOP_AT:
                print("stopping before the session limit", flush=True)
                return
            d = run(f"jax-{strategy}-{nx}x{ny}-{steps}", gpu,
                    ["--nx", str(nx), "--ny", str(ny), "--steps", str(steps), "--strategy", strategy])
            if not d.get("ok"):
                break


workers = [threading.Thread(target=sweep, args=(s, i % n_gpus)) for i, s in enumerate(("tape", "checkpoint"))]
if n_gpus > 1:
    for w in workers:
        w.start()
    for w in workers:
        w.join()
else:
    for w in workers:
        w.start()
        w.join()

subprocess.run([sys.executable, "spike/spike/summarise.py", str(OUT), str(WORK / "summary")], check=False)
print((WORK / "summary" / "summary.md").read_text() if (WORK / "summary" / "summary.md").exists() else "no summary")
