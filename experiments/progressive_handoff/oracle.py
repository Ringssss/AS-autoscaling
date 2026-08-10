#!/usr/bin/env python3
"""Replay a progressive AgentShift KV handoff as a two-stage flow shop.

The script uses measured AgentShift results as inputs. It does not modify or
launch SGLang. Copy work releases immutable layer groups; destination compute
may consume a group only after that group is fully present.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Measurement:
    layers: int
    sticky_ms: float
    on_return_ms: float
    agentshift_ms: float
    remaining_copy_ms: float
    tool_gap_ms: float

    @property
    def transfer_ms(self) -> float:
        # AgentShift starts migration during the tool. Its exposed time is the
        # transfer tail still visible after the tool returns.
        return self.remaining_copy_ms + self.tool_gap_ms

    @property
    def precopied_layers(self) -> float:
        return self.layers * self.tool_gap_ms / self.transfer_ms


@dataclass(frozen=True)
class Profile:
    name: str
    copy_slowdown: float
    compute_slowdown: float
    tp_straggler: float


@dataclass(frozen=True)
class ReplayResult:
    profile: str
    group_layers: int
    post_tool_ms: float
    speedup_vs_agentshift: float
    reduction_vs_agentshift_pct: float
    overhead_vs_sticky_pct: float
    overlap_ms: float
    groups_ready_at_return: int


def find_row(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["prefix_length"] == 32768
        and row["tool"] == "git-status"
        and row["scenario"] == scenario
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {scenario!r} row, found {len(matches)}")
    return matches[0]


def load_measurement(path: Path, layers: int) -> Measurement:
    data = json.loads(path.read_text())
    rows = data["rows"]
    sticky = find_row(rows, "sticky")
    on_return = find_row(rows, "on-return")
    agentshift = find_row(rows, "agentshift")
    return Measurement(
        layers=layers,
        sticky_ms=sticky["post_tool_mean_ms"],
        on_return_ms=on_return["post_tool_mean_ms"],
        agentshift_ms=agentshift["post_tool_mean_ms"],
        remaining_copy_ms=agentshift["exposed_mean_ms"],
        tool_gap_ms=agentshift["tool_mean_ms"],
    )


def layer_groups(layers: int, group_layers: int) -> list[int]:
    groups = []
    left = layers
    while left:
        size = min(left, group_layers)
        groups.append(size)
        left -= size
    return groups


def replay(
    measurement: Measurement,
    profile: Profile,
    group_layers: int,
    publish_overhead_us: float,
    precopied_layers: float | None = None,
) -> ReplayResult:
    groups = layer_groups(measurement.layers, group_layers)
    copy_per_layer = measurement.transfer_ms / measurement.layers
    compute_per_layer = measurement.sticky_ms / measurement.layers
    publish_ms = publish_overhead_us / 1000.0

    # TP publication waits for the slowest rank. Model that as additional copy
    # work per group. Publication overhead is deliberately charged serially to
    # the copy stage, which is conservative for an event-based implementation.
    copy_work = [
        size * copy_per_layer * (1.0 + profile.tp_straggler) + publish_ms
        for size in groups
    ]
    compute_work = [size * compute_per_layer for size in groups]

    if precopied_layers is None:
        progress = measurement.tool_gap_ms
    else:
        progress = max(0.0, precopied_layers) * copy_per_layer

    ready: set[int] = set()
    copy_idx = 0
    copy_remaining = 0.0
    while copy_idx < len(groups):
        if progress + 1e-12 >= copy_work[copy_idx]:
            progress -= copy_work[copy_idx]
            ready.add(copy_idx)
            copy_idx += 1
        else:
            copy_remaining = copy_work[copy_idx] - progress
            progress = 0.0
            break
    ready_at_return = len(ready)

    compute_idx = 0
    compute_remaining: float | None = None
    now = 0.0
    overlap_ms = 0.0

    while compute_idx < len(groups):
        if compute_remaining is None and compute_idx in ready:
            compute_remaining = compute_work[compute_idx]

        copy_active = copy_idx < len(groups)
        compute_active = compute_remaining is not None
        if not copy_active and not compute_active:
            raise RuntimeError("flow shop stalled without active work")

        if copy_active and compute_active:
            copy_rate = 1.0 / (1.0 + profile.copy_slowdown)
            compute_rate = 1.0 / (1.0 + profile.compute_slowdown)
        else:
            copy_rate = 1.0
            compute_rate = 1.0

        copy_done_in = copy_remaining / copy_rate if copy_active else float("inf")
        compute_done_in = (
            compute_remaining / compute_rate if compute_active else float("inf")
        )
        step = min(copy_done_in, compute_done_in)
        if copy_active and compute_active:
            overlap_ms += step
        now += step

        if copy_active:
            copy_remaining -= step * copy_rate
            if copy_remaining <= 1e-10:
                ready.add(copy_idx)
                copy_idx += 1
                if copy_idx < len(groups):
                    copy_remaining = copy_work[copy_idx]

        if compute_active:
            assert compute_remaining is not None
            compute_remaining -= step * compute_rate
            if compute_remaining <= 1e-10:
                compute_idx += 1
                compute_remaining = None

    return ReplayResult(
        profile=profile.name,
        group_layers=group_layers,
        post_tool_ms=now,
        speedup_vs_agentshift=measurement.agentshift_ms / now,
        reduction_vs_agentshift_pct=(1.0 - now / measurement.agentshift_ms) * 100.0,
        overhead_vs_sticky_pct=(now / measurement.sticky_ms - 1.0) * 100.0,
        overlap_ms=overlap_ms,
        groups_ready_at_return=ready_at_return,
    )


def print_measurement(measurement: Measurement) -> None:
    print("Measured 32K git-status case")
    print(f"  layers:                       {measurement.layers}")
    print(f"  sticky continuation:          {measurement.sticky_ms:8.3f} ms")
    print(f"  on-return migration:          {measurement.on_return_ms:8.3f} ms")
    print(f"  current AgentShift:            {measurement.agentshift_ms:8.3f} ms")
    print(f"  copy tail after tool return:   {measurement.remaining_copy_ms:8.3f} ms")
    print(f"  inferred full transfer:        {measurement.transfer_ms:8.3f} ms")
    print(f"  inferred pre-copy progress:    {measurement.precopied_layers:8.2f} layers")
    print()


def print_results(results: list[ReplayResult]) -> None:
    print(
        "profile       group  ready  post-tool(ms)  speedup  reduction  sticky-gap"
    )
    for result in results:
        print(
            f"{result.profile:<13}"
            f"{result.group_layers:>5}"
            f"{result.groups_ready_at_return:>7}"
            f"{result.post_tool_ms:>15.3f}"
            f"{result.speedup_vs_agentshift:>9.3f}x"
            f"{result.reduction_vs_agentshift_pct:>10.1f}%"
            f"{result.overhead_vs_sticky_pct:>11.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/real-tools-all-1784548799599095961-summary.json"),
    )
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--publish-overhead-us", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    measurement = load_measurement(args.summary, args.layers)
    profiles = [
        Profile("ideal", 0.00, 0.00, 0.00),
        Profile("mild", 0.10, 0.10, 0.05),
        Profile("conservative", 0.25, 0.25, 0.10),
        Profile("pessimistic", 0.50, 0.50, 0.20),
    ]
    results = []
    for profile in profiles:
        for group_layers in (1, 2, 4, 8):
            result = replay(
                measurement,
                profile,
                group_layers,
                args.publish_overhead_us,
            )
            results.append(result)

    print_measurement(measurement)
    print_results(results)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "measurement": asdict(measurement),
                    "publish_overhead_us": args.publish_overhead_us,
                    "results": [asdict(result) for result in results],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
