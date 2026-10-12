"""Four-node connectivity/tool-window campaign, NOT concurrent fleet scaling.

Requires one healthy TP2 engine on each of four physical hosts. Evaluates
source→three destinations sequentially, using distinct rendezvous port ranges.
Does not launch engines, share occupied cards, or fall back to fewer nodes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=30500)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = "http://12.11.3.27:31600"
    servers = {}
    for node in (27, 28, 29, 30):
        url = f"http://12.11.3.{node}:31600"
        for endpoint in ("health", "get_server_info"):
            try:
                with urllib.request.urlopen(url + "/" + endpoint, timeout=15) as response:
                    if response.status != 200:
                        raise RuntimeError("server unhealthy")
                    if endpoint == "get_server_info":
                        servers[str(node)] = json.load(response)
            except Exception as exc:
                raise SystemExit(f"Four-node gate failed at node{node}/{endpoint}: {exc}. No experiments started.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "server_info.json").write_text(json.dumps(servers, indent=2) + "\n")
    for index, destination in enumerate((28, 29, 30)):
        name = f"node27-node{destination}"
        command = [sys.executable, str(root / "experiments/multinode/benchmark_tool_window.py"),
                   "--source", source, "--destination", f"http://12.11.3.{destination}:31600",
                   "--master-address", "12.11.3.27", "--base-port", str(args.base_port + 8 * index),
                   "--repetitions", str(args.repetitions), "--output", str(output / name)]
        with (output / (name + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root)),
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{name} failed with {result.returncode}; inspect retained log")
        subprocess.run([sys.executable, str(root / "experiments/multinode/summarize_tool_window.py"),
                        str(output / name)], cwd=root, check=True, stdout=subprocess.DEVNULL)
        print(f"Completed {name}", flush=True)


if __name__ == "__main__":
    main()
