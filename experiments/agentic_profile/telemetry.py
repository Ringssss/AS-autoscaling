"""External, sampled process/GPU accounting; no imports into the serving runtime.

CPU accounting sums *only* each process's own user+system counters, never the
children_user/children_system counters (which would count reaped children twice).
Sampling cannot recover a process that starts and exits between observations.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import psutil


def child_pids(process):
    """Linux direct descendants, including children created by non-main threads.

    psutil.children() scans every process on a shared host, even when our tree
    is tiny. Reading per-thread children visits only the selected workload.
    """
    children = set()
    with os.scandir(f"/proc/{process.pid}/task") as threads:
        for thread in threads:
            try:
                children.update(map(int, Path(thread.path, "children").read_text().split()))
            except FileNotFoundError:
                pass
    return children


def gpu_sample(indices=None):
    """Driver telemetry, scoped to GPU indices, not attribution to these PIDs."""
    fields = ["index", "uuid", "utilization.gpu", "memory.used", "memory.total", "power.draw"]
    command = ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    if indices is not None:
        command += ["--id=" + ",".join(map(str, indices))]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout
        rows = []
        for values in csv.reader(io.StringIO(output)):
            if len(values) != len(fields):
                raise ValueError("unexpected nvidia-smi field count")
            raw = dict(zip(fields, (v.strip() for v in values)))
            row = {"index": int(raw["index"]), "uuid": raw["uuid"]}
            for source, target in [("utilization.gpu", "utilization_percent"),
                                   ("memory.used", "memory_used_mib"),
                                   ("memory.total", "memory_total_mib"), ("power.draw", "power_watts")]:
                try:
                    value = float(raw[source])
                    row[target] = value if math.isfinite(value) else None
                except ValueError:
                    row[target] = None
            rows.append(row)
        return {"devices": rows}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"devices": [], "error": str(exc)}


class Sampler:
    """Sample roots and recursively discovered children, with PID reuse protection.

    The first observation is a baseline. Newly discovered processes born after
    sampler creation contribute their full own CPU counter; older processes get
    a baseline. RSS is a sum of process RSS, so shared pages may be counted twice.
    start()/stop() optionally run the same sample() function in a background thread.
    """

    def __init__(self, pids, output, interval=1.0, include_gpu=False, gpu_indices=None):
        if not pids or interval <= 0 or not math.isfinite(interval):
            raise ValueError("nonempty pids and finite positive interval required")
        self.pids = set(map(int, pids))
        self.output = Path(output)
        self.interval = interval
        self.include_gpu = include_gpu
        self.gpu_indices = gpu_indices
        self.started_epoch = time.time()
        self.started_monotonic = time.monotonic()
        self.previous = {}
        self.known = {}
        self.cpu_seconds = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output.open("x")
        self._write({"event": "metadata", "epoch_seconds": self.started_epoch,
                     "monotonic_seconds": self.started_monotonic, "root_pids": sorted(self.pids),
                     "interval_seconds": interval, "gpu_indices": gpu_indices,
                     "process_discovery": "linux_per_thread_children_v2_no_host_scan",
                     "cpu_accounting": "sum own user+system deltas; child counters excluded",
                     "limitations": ["short-lived unsampled processes and final exit CPU tails can be missed",
                                     "RSS sum may double-count shared pages",
                                     "GPU metrics are device-wide, not PID-attributed"]})

    def _write(self, row):
        self._file.write(json.dumps(row, allow_nan=False) + "\n")
        self._file.flush()

    def _discover(self):
        candidates = {}
        # children(recursive=True) builds a global /proc parent map. Calling it
        # for every known descendant repeated that expensive scan per process.
        # Traverse only selected roots using Linux per-thread children files.
        for pid in sorted(self.pids):
            if any(key[0] == pid for key in candidates):
                continue
            try:
                process = psutil.Process(pid)
                key = (pid, process.create_time())
                # Do not reinterpret a replaced root PID as our original workload.
                if any(old[0] == pid and old != key for old in self.known):
                    continue
                candidates[key] = process
                queue = [process]
                visited = {pid}
                while queue:
                    current = queue.pop()
                    try:
                        children = child_pids(current)
                    except (FileNotFoundError, PermissionError, psutil.NoSuchProcess):
                        continue
                    for child_pid in children - visited:
                        visited.add(child_pid)
                        try:
                            child = psutil.Process(child_pid)
                            candidates[(child.pid, child.create_time())] = child
                            queue.append(child)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        for key, process in self.known.items():
            try:
                if process.is_running():
                    candidates[key] = process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.known.update(candidates)
        return candidates

    def sample(self):
        began = time.monotonic()
        epoch = time.time()
        rows, errors = [], []
        for key, process in sorted(self._discover().items()):
            try:
                with process.oneshot():
                    cpu = process.cpu_times()
                    switches = process.num_ctx_switches()
                    row = {"pid": key[0], "create_time": key[1], "ppid": process.ppid(),
                           "cpu_seconds": cpu.user + cpu.system,
                           "rss_bytes": process.memory_info().rss,
                           "threads": process.num_threads(), "affinity": process.cpu_affinity(),
                           "voluntary_context_switches": switches.voluntary,
                           "involuntary_context_switches": switches.involuntary}
                previous = self.previous.get(key)
                count_initial = self.samples > 0 and key[1] >= self.started_epoch
                for field in ["cpu_seconds", "voluntary_context_switches", "involuntary_context_switches"]:
                    baseline = previous[field] if previous else (0 if count_initial else row[field])
                    row[field + "_delta"] = max(0, row[field] - baseline)
                self.previous[key] = row
                rows.append(row)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError) as exc:
                errors.append({"pid": key[0], "error": type(exc).__name__})
        delta = sum(row["cpu_seconds_delta"] for row in rows)
        self.cpu_seconds += delta
        result = {"event": "sample", "epoch_seconds": epoch, "monotonic_seconds": began,
                  "elapsed_seconds": began - self.started_monotonic, "processes": rows,
                  "errors": errors, "cpu_seconds_delta": delta,
                  "sampled_cpu_seconds": self.cpu_seconds,
                  "rss_sum_bytes": sum(row["rss_bytes"] for row in rows)}
        if self.include_gpu:
            result["gpu"] = gpu_sample(self.gpu_indices)
        result["sampling_wall_seconds"] = time.monotonic() - began
        self.samples += 1
        self._write(result)
        return result

    def _run(self):
        try:
            while not self._stop.wait(self.interval):
                self.sample()
        except Exception as exc:
            self.error = str(exc)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self.sample()
        self._thread = threading.Thread(target=self._run, name="process-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._file.closed:
            return
        self.sample()
        summary = {"event": "summary", "epoch_seconds": time.time(),
                   "monotonic_seconds": time.monotonic(), "samples": self.samples,
                   "sampled_cpu_seconds": self.cpu_seconds, "error": self.error}
        self._write(summary)
        self._file.close()
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--gpu-indices", type=int, nargs="+")
    args = parser.parse_args()
    if args.duration is not None and (args.duration <= 0 or not math.isfinite(args.duration)):
        parser.error("duration must be finite and positive")
    sampler = Sampler(args.pids, args.output, args.interval,
                      include_gpu=args.gpu or args.gpu_indices is not None,
                      gpu_indices=args.gpu_indices).start()
    try:
        sampler._stop.wait(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        print(json.dumps(sampler.stop()))


if __name__ == "__main__":
    main()
