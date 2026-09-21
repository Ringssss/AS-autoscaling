import contextlib
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import psutil
import pytest

from experiments.agentic_profile.telemetry import Sampler, gpu_sample, child_pids


class FakeProcess:
    def __init__(self, user=1, system=2):
        self.user, self.system = user, system

    def oneshot(self):
        return contextlib.nullcontext()

    def cpu_times(self):
        return SimpleNamespace(user=self.user, system=self.system, children_user=1000, children_system=2000)

    def num_ctx_switches(self):
        return SimpleNamespace(voluntary=int(self.user * 10), involuntary=2)

    def ppid(self):
        return 1

    def memory_info(self):
        return SimpleNamespace(rss=4096)

    def num_threads(self):
        return 2

    def cpu_affinity(self):
        return [0, 1]


def test_own_cpu_deltas_exclude_reaped_child_counters(tmp_path, monkeypatch):
    sampler = Sampler([123], tmp_path / "sample.jsonl")
    parent, child = FakeProcess(), FakeProcess()
    monkeypatch.setattr(sampler, "_discover", lambda: {(123, 0): parent, (124, 0): child})
    assert sampler.sample()["cpu_seconds_delta"] == 0
    parent.user += 0.5
    child.system += 0.25
    row = sampler.sample()
    assert row["cpu_seconds_delta"] == 0.75
    assert row["rss_sum_bytes"] == 8192
    assert row["processes"][0]["voluntary_context_switches_delta"] == 5
    assert sampler.stop()["sampled_cpu_seconds"] == 0.75
    records = [json.loads(line) for line in sampler.output.read_text().splitlines()]
    assert records[0]["event"] == "metadata"
    assert records[-1]["event"] == "summary"


def test_new_process_count_and_disappearing_process(tmp_path, monkeypatch):
    sampler = Sampler([123], tmp_path / "sample.jsonl")
    monkeypatch.setattr(sampler, "_discover", lambda: {})
    sampler.sample()
    process = FakeProcess(user=0.5, system=0.25)
    key = (123, sampler.started_epoch + 0.01)
    monkeypatch.setattr(sampler, "_discover", lambda: {key: process})
    assert sampler.sample()["cpu_seconds_delta"] == 0.75
    def vanished():
        raise psutil.NoSuchProcess(123)
    process.cpu_times = vanished
    result = sampler.sample()
    assert result["cpu_seconds_delta"] == 0
    assert result["errors"][0]["error"] == "NoSuchProcess"
    sampler.stop()


def test_gpu_na_values_and_failure(monkeypatch):
    import experiments.agentic_profile.telemetry as telemetry
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="0, GPU-uuid, 42, 1024, 81920, [N/A]\n"))
    row = gpu_sample([0])["devices"][0]
    assert row["utilization_percent"] == 42
    assert row["power_watts"] is None
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(telemetry.subprocess, "run", missing)
    assert "error" in gpu_sample()


def test_output_does_not_overwrite_and_interval_validation(tmp_path):
    path = tmp_path / "existing.jsonl"
    path.write_text("preserved")
    with pytest.raises(FileExistsError):
        Sampler([123], path)
    with pytest.raises(ValueError):
        Sampler([123], tmp_path / "new", interval=0)


def test_procfs_children_includes_nonmain_thread_children():
    children = []
    ready = threading.Event()
    stop = threading.Event()
    def launch():
        children.append(subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(10)']))
        ready.set()
        stop.wait(10)
    thread = threading.Thread(target=launch)
    thread.start()
    try:
        assert ready.wait(5)
        assert children[0].pid in child_pids(psutil.Process(os.getpid()))
    finally:
        for child in children:
            child.terminate()
            child.wait()
        stop.set()
        thread.join()
