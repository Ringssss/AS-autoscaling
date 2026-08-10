#!/usr/bin/env python3
"""Measure labeled blocking operations after completed Qwen3 turns."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_e2e import flush, timed_generate

from agentshift.engine.sglang import SGLangAgentShiftClient


@dataclass(frozen=True, slots=True)
class ToolSpec:
    tool_class: str
    operation: str
    arguments: tuple[str, ...]


def tool_specs(root: Path) -> list[ToolSpec]:
    pdf_root = root / "paper_rewriting_output/reference_materials"
    return [
        *[
            ToolSpec("Web search", query, ("search", "--query", query))
            for query in (
                "LLM serving systems",
                "distributed systems scheduling",
                "KV cache inference",
                "LLM agent systems",
            )
        ],
        *[
            ToolSpec("Page fetch", name, ("page", "--url", url))
            for name, url in (
                ("Python asyncio", "https://docs.python.org/3/library/asyncio.html"),
                ("Python sqlite3", "https://docs.python.org/3/library/sqlite3.html"),
                (
                    "Python concurrent futures",
                    "https://docs.python.org/3/library/concurrent.futures.html",
                ),
                ("RFC 9110", "https://www.rfc-editor.org/rfc/rfc9110.html"),
            )
        ],
        *[
            ToolSpec(
                "External API",
                city,
                (
                    "api",
                    "--latitude",
                    str(latitude),
                    "--longitude",
                    str(longitude),
                ),
            )
            for city, latitude, longitude in (
                ("Berlin", 52.52, 13.41),
                ("Shanghai", 31.23, 121.47),
                ("San Francisco", 37.77, -122.42),
                ("Sydney", -33.87, 151.21),
            )
        ],
        *[
            ToolSpec(
                "PDF parsing",
                path.stem,
                ("pdf", "--path", str(path), "--max-pages", "12"),
            )
            for path in (
                pdf_root / "fastserve_atc23.pdf",
                pdf_root / "agentix_nsdi26.pdf",
                pdf_root / "blitzscale_osdi25.pdf",
                pdf_root / "continuum.pdf",
            )
        ],
        *[
            ToolSpec("Python execution", name, arguments)
            for name, arguments in (
                ("AST analysis", ("python", "--mode", "ast", "--root", str(root))),
                (
                    "JSON trace analysis",
                    (
                        "python",
                        "--mode",
                        "json",
                        "--json-path",
                        str(
                            root
                            / "results/autoscaling/manifests/"
                            "autoscaling-manifest-qwen8b-30m-v1.json"
                        ),
                    ),
                ),
                ("PBKDF2", ("python", "--mode", "hash", "--rounds", "250000")),
                ("Numeric sort", ("python", "--mode", "sort", "--items", "250000")),
            )
        ],
    ]


async def run_tool(
    command: list[str], cwd: Path, timeout_seconds: float
) -> dict[str, Any]:
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timed_out = False
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout, stderr = await process.communicate()
        stderr += f"\ntimed out after {timeout_seconds:.1f}s".encode()
    return {
        "tool_seconds": time.perf_counter() - started,
        "returncode": process.returncode,
        "success": process.returncode == 0 and not timed_out,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


class AgentToolGapBenchmark:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(__file__).resolve().parents[1]
        self.engine = SGLangAgentShiftClient(
            "engine-a", args.engine, timeout=args.request_timeout
        )
        self.runner = Path(__file__).resolve().with_name("agent_tool_operations.py")
        self.serial = 0

    def cases(self) -> list[tuple[int, int, ToolSpec]]:
        cases = [
            (prefix, repeat, spec)
            for prefix in self.args.prefix_lengths
            for repeat in range(self.args.repeats)
            for spec in tool_specs(self.root)
        ]
        random.Random(self.args.seed).shuffle(cases)
        return cases

    async def run_one(
        self, prefix_length: int, repeat: int, spec: ToolSpec
    ) -> dict[str, Any]:
        await flush(self.engine)
        self.serial += 1
        agent_id = f"tool-gap-{self.serial}"
        prompt = [32000 + self.serial] + [100] * (prefix_length - 1)
        _, first = await timed_generate(
            self.engine,
            prompt,
            max_new_tokens=self.args.first_output_tokens,
            rid=f"{agent_id}-turn-1",
        )
        completed = tuple(prompt + first["output_ids"])

        arguments = list(spec.arguments)
        if arguments[0] == "page":
            arguments.extend(("--nonce", f"{self.args.seed}-{self.serial}"))
        command = [self.args.python, str(self.runner), *arguments]
        outcome = await run_tool(
            command, self.root, timeout_seconds=self.args.tool_timeout
        )
        tool_result_bytes = outcome["stdout"] + outcome["stderr"]
        tool_result_tokens = max(
            16,
            min(
                self.args.max_tool_result_tokens,
                (len(tool_result_bytes) + 3) // 4,
            ),
        )
        next_seconds, second = await timed_generate(
            self.engine,
            list(completed) + [200] * tool_result_tokens,
            max_new_tokens=self.args.next_output_tokens,
            rid=f"{agent_id}-turn-2",
        )
        cached = int(second["meta_info"]["cached_tokens"])
        return {
            "agent_id": agent_id,
            "tool_class": spec.tool_class,
            "operation": spec.operation,
            "command": command,
            "repeat": repeat,
            "requested_prefix_tokens": prefix_length,
            "suspension_prefix_tokens": len(completed),
            "tool_seconds": outcome["tool_seconds"],
            "success": outcome["success"],
            "timed_out": outcome["timed_out"],
            "returncode": outcome["returncode"],
            "tool_stdout_bytes": len(outcome["stdout"]),
            "tool_stderr_bytes": len(outcome["stderr"]),
            "tool_result_tokens": tool_result_tokens,
            "next_turn_seconds": next_seconds,
            "cached_tokens": cached,
            "full_prefix_hit": cached >= len(completed) - 1,
            "stderr_tail": outcome["stderr"].decode(errors="replace")[-400:],
        }

    async def run(self) -> dict[str, Any]:
        started = time.time()
        records = []
        cases = self.cases()
        for index, (prefix, repeat, spec) in enumerate(cases, 1):
            record = await self.run_one(prefix, repeat, spec)
            records.append(record)
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(cases)}",
                        **{
                            key: record[key]
                            for key in (
                                "tool_class",
                                "operation",
                                "requested_prefix_tokens",
                                "tool_seconds",
                                "success",
                                "full_prefix_hit",
                            )
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        await flush(self.engine)
        return {
            "workload": "controlled labeled agent tool-gap characterization",
            "methodology": (
                "Each tool starts after a completed model turn. The next turn consumes "
                "the tool result on the same engine. Cases are deterministically shuffled; "
                "network errors and timeouts are retained."
            ),
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "config": {
                **vars(self.args),
                "output_dir": str(self.args.output_dir),
            },
            "records": records,
        }


def model_info(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            f"{url.rstrip('/')}/get_model_info", timeout=5
        ) as response:
            return json.loads(response.read())
    except Exception as exc:
        return {"error": str(exc)}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="http://127.0.0.1:32200")
    parser.add_argument("--model-label", default="Qwen3-8B")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument(
        "--prefix-lengths", type=int, nargs="+", default=[4096, 16384, 32768]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--next-output-tokens", type=int, default=1)
    parser.add_argument("--max-tool-result-tokens", type=int, default=256)
    parser.add_argument("--tool-timeout", type=float, default=20.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--python",
        default="/home/zhujianian/miniconda3/envs/crossstage/bin/python",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/tool-gap-workloads"
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = AgentToolGapBenchmark(args)
    payload = await benchmark.run()
    payload["model_info"] = model_info(args.engine)
    output = args.output_dir / f"agent-tool-gaps-{time.time_ns()}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
