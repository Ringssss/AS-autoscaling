import asyncio
import pytest

from experiments.fluidkv import engine_lifecycle as lifecycle


class FakeProcess:
    def __init__(self, pid, argv, born=1.0, children=()):
        self.pid, self.argv, self.born = pid, argv, born
        self.descendants = list(children)
        self.alive = True
        self.signals = []

    def cmdline(self):
        return self.argv

    def create_time(self):
        return self.born

    def uids(self):
        return type("Uids", (), {"real": lifecycle.os.getuid()})()

    def children(self, recursive=False):
        return self.descendants

    def cwd(self):
        return "/tmp"

    def environ(self):
        return {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/runtime", "API_SECRET": "must-not-leak"}

    def is_running(self):
        return self.alive

    def status(self):
        return "running" if self.alive else lifecycle.psutil.STATUS_ZOMBIE

    def terminate(self):
        self.signals.append("TERM")
        self.alive = False
        for child in self.descendants:
            child.alive = False

    def kill(self):
        self.signals.append("KILL")
        self.alive = False


@pytest.fixture
def owned(monkeypatch):
    worker = FakeProcess(100002, ["scheduler"])
    root = FakeProcess(100001, ["/python", "-m", "sglang.launch_server", "--host", "127.0.0.1", "--port", "31480", "--model-path", "/model"], children=[worker])
    processes = {root.pid: root, worker.pid: worker}
    monkeypatch.setattr(lifecycle.psutil, "process_iter", lambda: list(processes.values()))
    monkeypatch.setattr(lifecycle.psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(lifecycle, "_gpu_snapshot", lambda: {
        "compute_processes": [{"pid": worker.pid, "gpu_uuid": "GPU-test", "used_memory_mib": 70000.0}] if worker.alive else [],
        "gpus": [{"uuid": "GPU-test", "memory_used_mib": 70000.0 if worker.alive else 0.0}],
    })
    monkeypatch.setattr(lifecycle, "_endpoint_responds", lambda url: root.alive)
    return root, worker


def test_capture_exact_local_port_and_no_secret_environment(owned):
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")
    assert identity["pid"] == 100001
    assert identity["gpu_uuids"] == ["GPU-test"]
    assert {p["pid"] for p in identity["owned_processes"]} == {100001, 100002}
    assert identity["environment"] == {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/runtime"}
    with pytest.raises(ValueError):
        lifecycle.capture_owned_engine("http://remote.example:31480")
    with pytest.raises(ValueError):
        lifecycle.capture_owned_engine("http://user:secret@localhost:31480")
    with pytest.raises(RuntimeError, match="exactly one"):
        lifecycle.capture_owned_engine("http://127.0.0.1:3148")


@pytest.mark.parametrize("guards", [{"no_foreground": False, "no_transfers": True}, {"no_foreground": True, "no_transfers": False}])
def test_active_work_cannot_be_signaled(owned, guards):
    root, worker = owned
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")
    with pytest.raises(ValueError, match="drain"):
        asyncio.run(lifecycle.release_engine(identity, **guards))
    assert not root.signals and not worker.signals


def test_reused_pid_is_rejected_before_signal(owned):
    root, worker = owned
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")
    root.born = 2.0
    with pytest.raises(RuntimeError, match="identity"):
        asyncio.run(lifecycle.release_engine(identity, no_foreground=True, no_transfers=True))
    assert not root.signals and not worker.signals


def test_graceful_release_records_gpu_and_process_confirmation(owned):
    root, worker = owned
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")
    result = asyncio.run(lifecycle.release_engine(identity, no_foreground=True, no_transfers=True, poll_seconds=0.001))
    assert result["released"] and result["graceful"]
    assert result["confirmed_monotonic"] >= result["signal_monotonic"]
    assert root.signals == ["TERM"] and not worker.signals
    assert result["observations"][-1]["owned_gpu_processes"] == []
    assert result["observations"][-1]["live_owned_pids"] == []


def test_force_fallback_is_explicit_and_does_not_kill_reused_child(owned, monkeypatch):
    root, worker = owned
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")

    def stuck_term():
        root.signals.append("TERM")
        # Original worker exits and its numeric PID is reused by another process.
        worker.born = 9.0
        worker.alive = False

    monkeypatch.setattr(root, "terminate", stuck_term)
    result = asyncio.run(lifecycle.release_engine(identity, no_foreground=True, no_transfers=True, graceful_timeout=0, poll_seconds=0.001))
    assert result["released"] and result["forced_kill"] and not result["graceful"]
    assert root.signals == ["TERM", "KILL"]
    assert not worker.signals


def test_gpu_observation_failure_prevents_signaling(owned, monkeypatch):
    root, _ = owned
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")
    monkeypatch.setattr(lifecycle, "_gpu_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("nvidia query failed")))
    with pytest.raises(RuntimeError, match="nvidia"):
        asyncio.run(lifecycle.release_engine(identity, no_foreground=True, no_transfers=True))
    assert not root.signals


def test_missing_post_signal_gpu_evidence_cannot_confirm_release(owned, monkeypatch):
    root, _ = owned
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")
    original = lifecycle._gpu_snapshot

    def observation():
        if root.alive:
            return original()
        raise RuntimeError("GPU observation unavailable")

    monkeypatch.setattr(lifecycle, "_gpu_snapshot", observation)
    result = asyncio.run(lifecycle.release_engine(identity, no_foreground=True, no_transfers=True,
        graceful_timeout=0, confirmation_timeout=0.01, poll_seconds=0.001))
    assert not result["released"] and result["confirmed_monotonic"] is None
    assert "observation_error" in result["observations"][-1]


def test_capture_rejects_credential_bearing_launch(owned):
    root, _ = owned
    root.argv.extend(["--api-key", "must-not-store"])
    with pytest.raises(ValueError, match="credential"):
        lifecycle.capture_owned_engine("http://127.0.0.1:31480")


def test_restart_rejects_a_still_live_prior_engine(owned, monkeypatch, tmp_path):
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(RuntimeError, match="still live"):
        asyncio.run(lifecycle.start_engine(identity, tmp_path / "engine.log"))


def test_restart_uses_captured_arguments_environment_and_new_identity(owned, monkeypatch, tmp_path):
    root, _ = owned
    identity = lifecycle.capture_owned_engine("http://127.0.0.1:31480")
    root.alive = False
    called = {}

    def spawn(argv, **kwargs):
        called.update(argv=argv, **kwargs)
        root.born = 5.0
        root.alive = True
        return type("Spawned", (), {"pid": root.pid, "poll": lambda self: None})()

    monkeypatch.setattr(lifecycle.subprocess, "Popen", spawn)
    monkeypatch.setattr(lifecycle, "_http_ready", lambda url: True)
    restarted = asyncio.run(lifecycle.start_engine(identity, tmp_path / "engine.log"))
    assert restarted["create_time"] == 5.0
    assert called["argv"] == identity["cmdline"]
    assert called["env"] == identity["environment"]
    assert called["start_new_session"] is True
    assert restarted["ready_monotonic"] >= restarted["started_monotonic"]
