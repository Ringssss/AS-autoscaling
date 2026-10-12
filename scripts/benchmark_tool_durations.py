#!/usr/bin/env python3
"""Measure real coding-agent tool durations on this machine.

Independent of scripts/benchmark_real_tools.py, which ran in a conda env that
no longer exists. This harness shells out to tools that are actually present
(git, ripgrep, gcc, make, python3) against this repository, so every number is
reproducible by re-running this file.

Each command is run `--reps` times after `--warmup` discarded warmups.
Warm page cache is the default because the question the paper asks is about
the *steady-state* tool a serving system would overlap against, not first-touch
I/O. `--cold` drops the page cache between reps instead, which needs root.
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_ripgrep() -> str | None:
    """Locate a real ripgrep binary.

    `rg` on PATH is a shell *function* installed by the editor integration that
    re-execs the editor CLI; it is not ripgrep, and under /bin/sh it does not
    exist at all. Shell out to a concrete binary or not at all.
    """
    import shutil

    for candidate in (
        shutil.which("rg"),
        "/home/zhujianian/.vscode-server/bin/a5b500951314efd502d07465bd138dfbd714a960"
        "/node_modules/@vscode/ripgrep-universal/bin/linux-x64/rg",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


RG = find_ripgrep()

# The `sglang` conda env is the only interpreter on this box with pytest, numpy
# and torch, so it is what a real test-run tool call would use.
_CONDA = ("/home/zhujianian/spack/opt/spack/linux-x86_64_v4/"
          "miniconda3-25.5.1-iszegyxvurvtr57qc54q7txr3g2oaxng")
SP = f"{_CONDA}/envs/sglang/bin/python"
BP = f"{_CONDA}/bin"

# (label, class, command). Commands are chosen so each class has at least one
# representative that is genuinely present on the box.
TOOLS: list[tuple[str, str, str]] = [
    ("git status", "vcs metadata", "git status --short"),
    ("git diff stat", "vcs metadata", "git diff --stat HEAD"),
    ("git log", "vcs metadata", "git log --oneline -500"),
    *([("rg test names", "search", f"{RG} -c 'def test_' tests/"),
       ("rg TODO", "search", f"{RG} -n --no-heading TODO agentshift/"),
       ("rg defs wide", "search",
        f"{RG} -n --no-heading 'def ' agentshift/ experiments/")] if RG else []),
    ("read scheduler.py", "file read", "cat experiments/weka_baselines/environment/runtime/managers/scheduler.py"),
    ("python syntax check", "python", "python3 -m py_compile experiments/weka_baselines/environment/runtime/managers/scheduler.py"),
    ("python compileall", "python", "python3 -m compileall -q agentshift"),
    ("gcc one unit", "build", "gcc -c -O2 -o /tmp/fkv_bench.o /tmp/fkv_bench.c"),
    ("gcc -O0 link", "build", "gcc -O0 -o /tmp/fkv_bench_O0 /tmp/fkv_bench.c"),
    # Heavy tools: these are the operations that actually exceed a migration.
    # Exit code 1 is meaningful for ripgrep (no match) and pytest (tests failed),
    # so validity is decided per-tool rather than by "rc == 0".
    ("pytest collect", "test", f"{SP} -m pytest --collect-only -q tests/"),
    ("pytest unit subset", "test", f"{SP} -m pytest -q tests/test_state_store.py"),
    ("pytest mixed subset", "test",
     f"{SP} -m pytest -q tests/test_migration_coordinator.py "
     f"tests/test_baseline_policies.py tests/test_traces.py"),
    ("pytest full suite", "test", f"{SP} -m pytest -q tests/"),
    ("pip download", "network", f"{BP}/pip download --no-deps -q --dest /tmp/fkv_dl pytest"),
]

# rc values that still mean "the tool ran and its measurement is meaningful".
ACCEPTED_RC = {"pytest collect": {0, 1, 5}, "pytest unit subset": {0, 1},
               "pytest mixed subset": {0, 1}, "pytest full suite": {0, 1},
               "rg TODO": {0, 1}}


def run_once(cmd: str) -> tuple[float, int, int]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, shell=True, cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - start
    return elapsed, proc.returncode, len(proc.stdout)


def drop_caches() -> bool:
    try:
        with open("/proc/sys/vm/drop_caches", "w") as handle:
            handle.write("3\n")
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cold", action="store_true",
                    help="drop page cache between reps (needs root)")
    ap.add_argument("--out", default="results/tool-durations.json")
    args = ap.parse_args()

    # A deterministic C workload so the build rows are honest work, not a stub.
    csrc = Path("/tmp/fkv_bench.c")
    body = "\n".join(
        f"static double f{i}(double x){{ return x*{i + 1}.0 + {i}; }}" for i in range(400)
    )
    csrc.write_text(
        '#include <stdio.h>\n#include <math.h>\n'
        + body
        + "\nint main(void){double s=0;"
        + "".join(f"s+=f{i}({i}.5);" for i in range(400))
        + 'printf("%f\\n",s);return 0;}\n'
    )

    cold_ok = None
    rows = []
    for label, cls, cmd in TOOLS:
        samples: list[float] = []
        rc_last = 0
        nbytes = 0
        for i in range(args.warmup + args.reps):
            if args.cold and cold_ok is not False:
                cold_ok = drop_caches() if cold_ok is None else cold_ok
                if not cold_ok:
                    print("  ! cold mode unavailable (drop_caches denied); "
                          "falling back to warm", flush=True)
                    args.cold = False
            elapsed, rc, nbytes = run_once(cmd)
            rc_last = rc
            if i >= args.warmup:
                samples.append(elapsed)
        samples.sort()
        valid = rc_last in ACCEPTED_RC.get(label, {0})
        med = statistics.median(samples)
        p95 = samples[max(0, int(round(0.95 * len(samples))) - 1)]
        rows.append({
            "label": label, "class": cls, "command": cmd,
            "reps": len(samples), "returncode": rc_last, "output_bytes": nbytes,
            "valid": valid,
            "median_s": med if valid else None, "p95_s": p95 if valid else None,
            "min_s": samples[0], "max_s": samples[-1],
            "mean_s": statistics.fmean(samples),
            "samples_s": samples,
        })
        flag = "" if valid else "  <-- INVALID (nonzero exit)"
        print(f"{label:22s} {cls:14s} med={med * 1000:9.3f} ms  "
              f"p95={p95 * 1000:9.3f} ms  rc={rc_last}{flag}", flush=True)

    payload = {
        "host": platform.node(), "platform": platform.platform(),
        "reps": args.reps, "warmup": args.warmup, "cold": bool(args.cold),
        "repo_root": str(ROOT), "rows": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
