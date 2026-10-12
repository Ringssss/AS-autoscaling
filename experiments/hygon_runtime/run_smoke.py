"""Wait for the dedicated engines and retain stdout, including failed checks."""
import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--base-port", type=int, default=29820)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    urls = [f"http://127.0.0.1:{port}" for port in (31400, 31401)]
    for url in urls:
        deadline = time.monotonic() + 600
        while True:
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{url} did not become healthy")
            time.sleep(5)
        print(f"Ready: {url}", flush=True)
    for name, script, port_arg, port in (
        ("direct", "validate_direct_two_engine.py", "--rendezvous-port", args.base_port),
        ("coordinator", "validate_hybrid_coordinator.py", "--base-port", args.base_port + 8),
    ):
        command = [sys.executable, str(root / "experiments/hybrid_state_mobility" / script),
                   "--source", urls[0], "--destination", urls[1],
                   "--prefix-tokens", str(args.prefix_tokens), "--continuation-tokens", "16",
                   port_arg, str(port), "--output-dir", str(output / name)]
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root)),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=900)
        print((output / f"{name}.log").read_text(), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
