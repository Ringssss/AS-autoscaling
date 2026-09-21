"""Measured lifecycle of dedicated local SGLang engines.

There are no actions at import time. Callers must first close their ingress and
drain foreground requests and both ends of every hybrid transfer. This module
does not destroy NCCL groups or infer those prerequisites from scheduler load.
"""

from __future__ import annotations

import asyncio
import csv
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import psutil


ENVIRONMENT_KEYS = frozenset({
    "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LANG",
    "LC_ALL", "TMPDIR", "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
})
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _endpoint(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in LOCAL_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or parsed.port is None):
        raise ValueError("engine lifecycle requires an explicit local HTTP host and port")
    return parsed.hostname, parsed.port


def _option(argv, name, default=None):
    for index, arg in enumerate(argv):
        if arg == name and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return default


def _matches_launch(argv, port):
    module = any(argv[i:i + 2] == ["-m", "sglang.launch_server"] for i in range(len(argv)))
    return (module and _option(argv, "--port") == str(port)
            and _option(argv, "--host", "127.0.0.1") in LOCAL_HOSTS)


def _check_safe_arguments(argv):
    # Reject instead of storing credentials or attempting an unreproducible
    # redacted restart. Dedicated experimental servers do not require them.
    sensitive = ("api-key", "auth-token", "password", "secret", "access-token")
    if any(arg.startswith("--") and any(key in arg.lower().split("=", 1)[0]
                                       for key in sensitive) for arg in argv):
        raise ValueError("credential-bearing server command is outside lifecycle scope")


def _process_identity(process):
    return {"pid": process.pid, "create_time": process.create_time()}


def _same_process(identity):
    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() != identity["create_time"]:
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except psutil.NoSuchProcess:
        return None


def _validate_root(identity):
    _, port = _endpoint(identity["url"])
    process = _same_process(identity)
    if (process is None or process.cmdline() != identity["cmdline"]
            or not _matches_launch(process.cmdline(), port)
            or process.uids().real != os.getuid() or process.pid == os.getpid()):
        raise RuntimeError("owned engine identity changed or PID is no longer alive")
    _check_safe_arguments(process.cmdline())
    return process


def _numeric(value):
    try:
        return float(value)
    except ValueError:
        return None


def _gpu_snapshot():
    def query(arguments):
        result = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=5)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    processes = [{"pid": int(row[0]), "gpu_uuid": row[1], "used_memory_mib": _numeric(row[2])}
                 for row in query(["--query-compute-apps=pid,gpu_uuid,used_memory"])
                 if len(row) == 3 and row[0].isdigit()]
    gpus = [{"uuid": row[0], "pci_bus_id": row[1], "memory_used_mib": _numeric(row[2])}
            for row in query(["--query-gpu=uuid,pci.bus_id,memory.used"]) if len(row) == 3]
    return {"compute_processes": processes, "gpus": gpus}


def capture_owned_engine(url):
    """Capture an exact local launch process and only reproducible nonsecret env."""
    _, port = _endpoint(url)
    matches = []
    for process in psutil.process_iter():
        try:
            if (_matches_launch(process.cmdline(), port)
                    and process.uids().real == os.getuid()
                    and process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE):
                matches.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one owned SGLang launch process on port {port}; found {len(matches)}")
    root = matches[0]
    if root.pid == os.getpid():
        raise RuntimeError("refusing to capture the experiment process itself")
    argv = root.cmdline()
    _check_safe_arguments(argv)
    owned = [_process_identity(root)]
    for child in root.children(recursive=True):
        try:
            owned.append(_process_identity(child))
        except psutil.NoSuchProcess:
            pass
    snapshot = _gpu_snapshot()
    pids = {p["pid"] for p in owned}
    gpu_processes = [p for p in snapshot["compute_processes"] if p["pid"] in pids]
    uuids = sorted({p["gpu_uuid"] for p in gpu_processes})
    environment = {k: v for k, v in root.environ().items() if k in ENVIRONMENT_KEYS}
    return {
        **_process_identity(root), "url": url.rstrip("/"), "cmdline": argv,
        "cwd": root.cwd(), "environment": environment, "owned_processes": owned,
        "gpu_uuids": uuids, "captured_monotonic": time.perf_counter(),
        "gpu_evidence": {"compute_processes": gpu_processes,
                         "gpus": [g for g in snapshot["gpus"] if g["uuid"] in uuids]},
    }


def _endpoint_responds(url):
    host, port = _endpoint(url)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _http_ready(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _release_observation(identity, owned):
    snapshot = _gpu_snapshot()
    pids = {p["pid"] for p in owned}
    return {
        "monotonic": time.perf_counter(),
        "live_owned_pids": [p["pid"] for p in owned if _same_process(p) is not None],
        "owned_gpu_processes": [p for p in snapshot["compute_processes"] if p["pid"] in pids],
        "gpus": [g for g in snapshot["gpus"] if g["uuid"] in identity["gpu_uuids"]],
        "endpoint_responds": _endpoint_responds(identity["url"]),
    }


async def release_engine(identity, *, no_foreground, no_transfers,
                         graceful_timeout=15.0, confirmation_timeout=15.0,
                         poll_seconds=0.5):
    """Stop a drained owned engine; report measured release, never logical release.

    Guards are assertions supplied by the ingress/transfer owner. This helper
    cannot independently infer them from the runtime's incomplete idle metric.
    A forced fallback is recorded separately and is not a graceful-drain pass.
    """
    if no_foreground is not True or no_transfers is not True:
        raise ValueError("foreground and hybrid-transfer drain must both be confirmed")
    if graceful_timeout < 0 or confirmation_timeout <= 0 or poll_seconds <= 0:
        raise ValueError("lifecycle timeout and polling values must be positive")
    root = _validate_root(identity)
    owned = {(p["pid"], p["create_time"]): dict(p) for p in identity["owned_processes"]}
    for process in [root, *root.children(recursive=True)]:
        try:
            p = _process_identity(process)
            owned[(p["pid"], p["create_time"])] = p
        except psutil.NoSuchProcess:
            pass
    owned = list(owned.values())
    before = await asyncio.to_thread(_release_observation, identity, owned)
    uuids = set(identity["gpu_uuids"]) | {p["gpu_uuid"] for p in before["owned_gpu_processes"]}
    if not uuids:
        raise RuntimeError("cannot measure GPU release without an observed owned GPU worker")
    identity = {**identity, "gpu_uuids": sorted(uuids)}
    root = _validate_root(identity)  # revalidate after observations, before signal
    result = {"pid": identity["pid"], "create_time": identity["create_time"],
              "url": identity["url"], "gpu_uuids": identity["gpu_uuids"],
              "released": False, "graceful": False, "forced_kill": False,
              "confirmed_monotonic": None, "observations": [before],
              "prerequisites": {"no_foreground": True, "no_transfers": True},
              "signal_monotonic": time.perf_counter(), "killed_pids": []}
    root.terminate()
    deadline = result["signal_monotonic"] + graceful_timeout
    hard_deadline = deadline + confirmation_timeout
    empty_observations = 0
    while time.perf_counter() <= hard_deadline:
        try:
            observation = await asyncio.to_thread(_release_observation, identity, owned)
        except Exception as exc:
            # Once SIGTERM has been sent, retain the partial lifecycle result
            # even if telemetry fails. Missing evidence cannot confirm release.
            observation = {"monotonic": time.perf_counter(), "observation_error": str(exc)}
        result["observations"].append(observation)
        disappeared = ("observation_error" not in observation
                       and not observation["live_owned_pids"]
                       and not observation["owned_gpu_processes"]
                       and not observation["endpoint_responds"]
                       and uuids <= {g["uuid"] for g in observation["gpus"]})
        empty_observations = empty_observations + 1 if disappeared else 0
        if empty_observations >= 2:
            result.update(released=True, graceful=not result["forced_kill"],
                          confirmed_monotonic=observation["monotonic"])
            result["shutdown_seconds"] = result["confirmed_monotonic"] - result["signal_monotonic"]
            return result
        if time.perf_counter() >= deadline and not result["forced_kill"]:
            live = [(p, _same_process(p)) for p in owned]
            live = [(p, process) for p, process in live if process is not None]
            if live:
                result["forced_kill"] = True
                # Children first; check every process incarnation immediately
                # before signaling, including potentially orphaned descendants.
                for p, _ in sorted(live, key=lambda item: item[0]["pid"] == identity["pid"]):
                    process = _same_process(p)
                    if process is not None:
                        try:
                            process.kill()
                            result["killed_pids"].append(p["pid"])
                        except psutil.NoSuchProcess:
                            pass
        await asyncio.sleep(poll_seconds)
    result["error"] = "physical GPU/process/endpoint disappearance was not confirmed before timeout"
    return result


async def start_engine(identity, log_path, *, readiness_timeout=180.0):
    """Restart captured dedicated launch arguments and return its new identity."""
    _, port = _endpoint(identity["url"])
    argv = list(identity["cmdline"])
    _check_safe_arguments(argv)
    if not _matches_launch(argv, port):
        raise ValueError("captured command does not match local engine endpoint")
    if _same_process(identity) is not None or await asyncio.to_thread(_endpoint_responds, identity["url"]):
        raise RuntimeError("refusing to start an engine whose prior process or port is still live")
    if readiness_timeout <= 0:
        raise ValueError("readiness timeout must be positive")
    environment = {key: value for key, value in identity["environment"].items() if key in ENVIRONMENT_KEYS}
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(argv, cwd=identity["cwd"], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    try:
        launch_create_time = psutil.Process(process.pid).create_time()
    except psutil.NoSuchProcess:
        launch_create_time = None

    def startup_error(message):
        error = RuntimeError(message)
        error.launch_pid = process.pid
        error.launch_create_time = launch_create_time
        error.started_monotonic = started
        error.log_path = str(path.resolve())
        return error

    while time.perf_counter() - started < readiness_timeout:
        code = process.poll()
        if code is not None:
            raise startup_error(f"engine launch PID {process.pid} exited with {code}; inspect {path}")
        if await asyncio.to_thread(_http_ready, identity["url"]):
            captured = await asyncio.to_thread(capture_owned_engine, identity["url"])
            if captured["pid"] != process.pid:
                raise startup_error("ready endpoint does not belong to the launched PID")
            return {**captured, "started_monotonic": started,
                    "ready_monotonic": time.perf_counter(), "log_path": str(path.resolve())}
        await asyncio.sleep(0.5)
    # Preserve a possibly still-loading owned engine for caller-directed
    # recovery; never hide its continued GPU residency from cost accounting.
    raise startup_error(f"engine launch PID {process.pid} not ready within {readiness_timeout}s; inspect {path}")
