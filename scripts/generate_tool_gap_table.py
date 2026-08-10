#!/usr/bin/env python3
"""Generate the measured tool-gap characterization table.

The table keeps two controlled workloads separate because they use different
tensor-parallel configurations.  Each observed interval is compared with a
Qwen3-8B preparation calibration measured under the matching configuration.
Synthetic Azure/Kimi gaps and unlabeled FlowPrefill inter-turn deltas are
intentionally excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

TOOL_LABELS = {
    "git-status": ("Shell", r"\texttt{git status --short}", "git status --short"),
    "state-store-tests": (
        "Targeted test",
        r"\texttt{pytest} state store",
        "pytest state store",
    ),
    "migration-tests": (
        "Targeted test",
        r"\texttt{pytest} migration protocol",
        "pytest migration protocol",
    ),
    "control-plane-tests": (
        "Build/test",
        r"\texttt{pytest} full suite",
        "pytest full suite",
    ),
}

REPRESENTATIVE_TOOL_LABELS = {
    "Web search": ("Web search", "OpenAlex query", "OpenAlex query"),
    "Page fetch": (
        "Page fetch",
        "HTTP fetch + HTML parse",
        "HTTP fetch + HTML parse",
    ),
    "External API": (
        "External API",
        "Open-Meteo request",
        "Open-Meteo request",
    ),
    "PDF parsing": (
        "PDF parsing",
        "Parse up to 12 pages",
        "Parse up to 12 pages",
    ),
    "Python execution": (
        "Python execution",
        "AST/JSON/hash/sort",
        "AST/JSON/hash/sort",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile over an empty sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def suspension_prefix(config: dict[str, Any], turn: int) -> int:
    """Reconstruct the immutable prefix present when the tool starts."""

    nominal = min(
        int(config["max_prefix"]),
        int(config["initial_prefix"]) + (turn - 1) * int(config["turn_growth"]),
    )
    return nominal + int(config["output_tokens"])


def load_preparation_p95(path: Path) -> dict[int, float]:
    payload = read_json(path)
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in payload["records"]:
        if row.get("scenario") != "agentshift":
            continue
        seconds = float(row.get("migration_wall_seconds", 0.0))
        if seconds > 0:
            grouped[int(row["prefix_length"])].append(seconds)
    if not grouped:
        raise ValueError(f"no AgentShift preparation measurements in {path}")
    return {
        prefix: percentile(samples, 0.95) for prefix, samples in sorted(grouped.items())
    }


def nearest_preparation_p95(prefix: int, thresholds: dict[int, float]) -> float:
    bucket = min(thresholds, key=lambda candidate: (abs(candidate - prefix), candidate))
    return thresholds[bucket]


def load_blocked_turns(
    workload_path: Path, preparation_p95: dict[int, float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(workload_path)
    config = payload["config"]
    rows: list[dict[str, Any]] = []
    agent_shift_runs = [
        run for run in payload["records"] if run.get("scenario") == "agentshift"
    ]
    for run in agent_shift_runs:
        for record in run["turn_records"]:
            tool = record["tool"]
            if tool not in TOOL_LABELS:
                raise ValueError(f"unclassified real tool: {tool}")
            prefix = suspension_prefix(config, int(record["turn"]))
            threshold = nearest_preparation_p95(prefix, preparation_p95)
            rows.append(
                {
                    "agent_id": record["agent_id"],
                    "repeat": int(run["repeat"]),
                    "turn": int(record["turn"]),
                    "tool": tool,
                    "wait_seconds": float(record["tool_seconds"]),
                    "suspension_prefix_tokens": prefix,
                    "preparation_p95_seconds": threshold,
                    "covered": float(record["tool_seconds"]) >= threshold,
                }
            )
    if not rows:
        raise ValueError(f"no AgentShift blocked turns in {workload_path}")
    return rows, config


def load_representative_turns(
    workload_path: Path, preparation_p95: dict[int, float]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    payload = read_json(workload_path)
    config = payload["config"]
    rows: list[dict[str, Any]] = []
    outcomes = {"successful": 0, "timed_out": 0, "full_prefix_hits": 0}
    for record in payload["records"]:
        tool_class = record["tool_class"]
        if tool_class not in REPRESENTATIVE_TOOL_LABELS:
            raise ValueError(f"unclassified representative tool: {tool_class}")
        prefix = int(record["suspension_prefix_tokens"])
        threshold = nearest_preparation_p95(prefix, preparation_p95)
        successful = bool(record["success"])
        timed_out = bool(record["timed_out"])
        full_prefix_hit = bool(record["full_prefix_hit"])
        outcomes["successful"] += int(successful)
        outcomes["timed_out"] += int(timed_out)
        outcomes["full_prefix_hits"] += int(full_prefix_hit)
        rows.append(
            {
                "agent_id": record["agent_id"],
                "repeat": int(record["repeat"]),
                "tool": tool_class,
                "operation": record["operation"],
                "wait_seconds": float(record["tool_seconds"]),
                "suspension_prefix_tokens": prefix,
                "preparation_p95_seconds": threshold,
                "covered": float(record["tool_seconds"]) >= threshold,
                "success": successful,
                "timed_out": timed_out,
                "full_prefix_hit": full_prefix_hit,
            }
        )
    if not rows:
        raise ValueError(f"no representative blocked turns in {workload_path}")
    return rows, config, outcomes


def summarize(
    rows: list[dict[str, Any]],
    labels: dict[str, tuple[str, str, str]],
    order: list[str],
) -> list[dict[str, Any]]:
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tool[row["tool"]].append(row)

    summaries = []
    for tool in order:
        samples = by_tool.get(tool, [])
        if not samples:
            continue
        tool_class, operation_tex, operation_text = labels[tool]
        waits = [row["wait_seconds"] for row in samples]
        prefixes = [float(row["suspension_prefix_tokens"]) for row in samples]
        summaries.append(
            {
                "tool": tool,
                "class": tool_class,
                "operation_tex": operation_tex,
                "operation": operation_text,
                "blocked_turns": len(samples),
                "wait_p50_ms": percentile(waits, 0.50) * 1000.0,
                "wait_p90_ms": percentile(waits, 0.90) * 1000.0,
                "prefix_p50_k": percentile(prefixes, 0.50) / 1024.0,
                "prefix_p90_k": percentile(prefixes, 0.90) / 1024.0,
                "covered_fraction": sum(row["covered"] for row in samples)
                / len(samples),
            }
        )

    waits = [row["wait_seconds"] for row in rows]
    prefixes = [float(row["suspension_prefix_tokens"]) for row in rows]
    summaries.append(
        {
            "tool": "overall",
            "class": "Overall",
            "operation_tex": "All operations",
            "operation": "All operations",
            "blocked_turns": len(rows),
            "wait_p50_ms": percentile(waits, 0.50) * 1000.0,
            "wait_p90_ms": percentile(waits, 0.90) * 1000.0,
            "prefix_p50_k": percentile(prefixes, 0.50) / 1024.0,
            "prefix_p90_k": percentile(prefixes, 0.90) / 1024.0,
            "covered_fraction": sum(row["covered"] for row in rows) / len(rows),
        }
    )
    return summaries


def format_pair(left: float, right: float, decimals: int = 1) -> str:
    return f"{left:.{decimals}f} / {right:.{decimals}f}"


def render_markdown(groups: list[dict[str, Any]]) -> str:
    lines = [
        "# Tool-Wait Characteristics",
        "",
        "| Configuration | Tool class | Representative operation | Blocked turns | Wait p50 / p90 (ms) | Prefix p50 / p90 (K tokens) | Covered by p95 preparation |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        for index, row in enumerate(group["summaries"]):
            lines.append(
                "| {configuration} | {class_} | {operation} | {n} | {wait} | {prefix} | {covered:.0%} |".format(
                    configuration=group["markdown_label"] if index == 0 else "",
                    class_=(
                        f"**{row['class']}**"
                        if row["tool"] == "overall"
                        else row["class"]
                    ),
                    operation=row["operation"],
                    n=row["blocked_turns"],
                    wait=format_pair(row["wait_p50_ms"], row["wait_p90_ms"]),
                    prefix=format_pair(row["prefix_p50_k"], row["prefix_p90_k"]),
                    covered=row["covered_fraction"],
                )
            )
    lines.extend(
        [
            "",
            "Coverage compares each observed interval with the p95 measured Qwen3-8B completed-prefix preparation time for the nearest 4K, 16K, or 32K prefix bucket under the matching TP configuration.",
            "",
            "Scope: controlled labeled operations, not a production-frequency distribution. Azure/Kimi synthetic gaps and unlabeled FlowPrefill inter-turn proxies are excluded.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_latex(
    groups: list[dict[str, Any]], representative_outcomes: dict[str, int]
) -> str:
    body = []
    for group_index, group in enumerate(groups):
        if group_index:
            body.append(r"\addlinespace[2pt]")
        body.append(r"\multicolumn{6}{l}{\textit{" + group["latex_label"] + r"}} \\")
        for row in group["summaries"]:
            values = (
                row["class"],
                row["operation_tex"],
                str(row["blocked_turns"]),
                format_pair(row["wait_p50_ms"], row["wait_p90_ms"]),
                format_pair(row["prefix_p50_k"], row["prefix_p90_k"]),
                f"{row['covered_fraction'] * 100:.0f}\\%",
            )
            line = " & ".join(values) + r" \\"
            if row["tool"] == "overall":
                line = (
                    r"\textbf{" + values[0] + "} & " + " & ".join(values[1:]) + r" \\"
                )
            body.append(line)
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Blocking intervals observed after a completed Qwen3-8B turn. Coding rows come from an eight-agent, three-turn replay repeated three times. Representative rows aggregate four operations over 4K, 16K, and 32K prefixes, each repeated twice. `Covered' compares each interval with p95 prefix-preparation time under the matching TP configuration. All "
            + str(representative_outcomes["successful"])
            + r" representative calls completed successfully; the replay retains any failures and timeouts.}",
            r"\label{tab:tool-gaps}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            "Tool class & Representative operation & Blocked & Wait p50/p90 & Prefix p50/p90 & Covered \\\\",
            " & & turns & (ms) & (K tokens) & by wait \\\\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coding-workload",
        type=Path,
        default=root / "results/coding-agent-replay-1784829317183895026.json",
    )
    parser.add_argument(
        "--coding-preparation",
        type=Path,
        default=root / "results/blocked-window-1784565792789187724.json",
    )
    parser.add_argument(
        "--representative-workload",
        type=Path,
        default=root
        / "results/tool-gap-workloads/formal/agent-tool-gaps-1786347666986609309.json",
    )
    parser.add_argument(
        "--representative-preparation",
        type=Path,
        default=root
        / "results/tool-gap-workloads/preparation/blocked-window-1786347451876687273.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_rewriting_output/tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coding_preparation_p95 = load_preparation_p95(args.coding_preparation)
    coding_turns, coding_config = load_blocked_turns(
        args.coding_workload, coding_preparation_p95
    )
    representative_preparation_p95 = load_preparation_p95(
        args.representative_preparation
    )
    representative_turns, representative_config, representative_outcomes = (
        load_representative_turns(
            args.representative_workload, representative_preparation_p95
        )
    )
    groups = [
        {
            "id": "coding",
            "markdown_label": "Coding replay (TP=1)",
            "latex_label": "Controlled coding replay (TP=1)",
            "summaries": summarize(
                coding_turns,
                TOOL_LABELS,
                [
                    "git-status",
                    "state-store-tests",
                    "migration-tests",
                    "control-plane-tests",
                ],
            ),
        },
        {
            "id": "representative",
            "markdown_label": "Representative operations (TP=2)",
            "latex_label": "Representative blocking operations (TP=2)",
            "summaries": summarize(
                representative_turns,
                REPRESENTATIVE_TOOL_LABELS,
                list(REPRESENTATIVE_TOOL_LABELS),
            ),
        },
    ]

    markdown_path = args.output_dir / "table_tool_gaps.md"
    latex_path = args.output_dir / "table_tool_gaps.tex"
    data_path = args.output_dir / "table_tool_gaps_data.json"
    markdown_path.write_text(render_markdown(groups))
    latex_path.write_text(render_latex(groups, representative_outcomes))
    data_path.write_text(
        json.dumps(
            {
                "evidence_type": "measured controlled labeled agent-tool workloads",
                "scope_exclusions": [
                    "Azure and Kimi gaps synthesized from FlowPrefill",
                    "unlabeled FlowPrefill inter-turn deltas",
                    "20-case tool-operation smoke run",
                ],
                "coverage_definition": (
                    "observed tool_seconds >= p95 AgentShift migration_wall_seconds "
                    "for the nearest measured prefix bucket under the matching TP config"
                ),
                "groups": groups,
                "coding": {
                    "workload_path": str(args.coding_workload.resolve()),
                    "workload_sha256": sha256(args.coding_workload),
                    "workload_config": coding_config,
                    "preparation_path": str(args.coding_preparation.resolve()),
                    "preparation_sha256": sha256(args.coding_preparation),
                    "preparation_p95_seconds": coding_preparation_p95,
                    "blocked_turns": coding_turns,
                },
                "representative": {
                    "workload_path": str(args.representative_workload.resolve()),
                    "workload_sha256": sha256(args.representative_workload),
                    "workload_config": representative_config,
                    "preparation_path": str(args.representative_preparation.resolve()),
                    "preparation_sha256": sha256(args.representative_preparation),
                    "preparation_p95_seconds": representative_preparation_p95,
                    "outcomes": representative_outcomes,
                    "blocked_turns": representative_turns,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for path in (markdown_path, latex_path, data_path):
        print(path)


if __name__ == "__main__":
    main()
