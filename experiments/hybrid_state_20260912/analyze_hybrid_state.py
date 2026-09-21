#!/usr/bin/env python3
"""Account for Qwen3.5's two state families in agentic trace replay."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DTYPE_SIZES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
}


@dataclass(frozen=True)
class HybridStateBudget:
    full_layers: int
    linear_layers: int
    full_kv_bytes_per_token: int
    linear_state_bytes: int

    @classmethod
    def from_config(cls, config: dict) -> "HybridStateBudget":
        text = config.get("text_config", config)
        num_layers = int(text["num_hidden_layers"])
        interval = int(text["full_attention_interval"])
        full_layers = math.ceil(num_layers / interval)
        linear_layers = num_layers - full_layers

        kv_dtype = text.get("dtype", "bfloat16")
        conv_dtype = text.get("dtype", "bfloat16")
        ssm_dtype = text.get("mamba_ssm_dtype", "float32")
        kv_bytes = DTYPE_SIZES[kv_dtype.lower()]
        conv_bytes = DTYPE_SIZES[conv_dtype.lower()]
        ssm_bytes = DTYPE_SIZES[ssm_dtype.lower()]

        num_kv_heads = int(text["num_key_value_heads"])
        head_dim = int(text["head_dim"])
        full_kv_per_token = full_layers * 2 * num_kv_heads * head_dim * kv_bytes

        kernel = int(text["linear_conv_kernel_dim"])
        num_key_heads = int(text["linear_num_key_heads"])
        key_dim = int(text["linear_key_head_dim"])
        num_value_heads = int(text["linear_num_value_heads"])
        value_dim = int(text["linear_value_head_dim"])

        # Matches SGLang KimiLinearStateShape: a conv window over projected V
        # plus two projected K lanes, followed by an HVxK recurrent matrix.
        conv_elements = (kernel - 1) * (
            num_value_heads * value_dim + 2 * num_key_heads * key_dim
        )
        # SGLang stores the recurrent matrix as (value heads, value dim, state dim).
        temporal_elements = num_value_heads * value_dim * value_dim
        linear_state = linear_layers * (
            conv_elements * conv_bytes + temporal_elements * ssm_bytes
        )
        return cls(
            full_layers=full_layers,
            linear_layers=linear_layers,
            full_kv_bytes_per_token=full_kv_per_token,
            linear_state_bytes=linear_state,
        )

    @property
    def linear_state_mib(self) -> float:
        return self.linear_state_bytes / 1024**2

    @property
    def crossover_tokens(self) -> int:
        return math.ceil(self.linear_state_bytes / self.full_kv_bytes_per_token)


@dataclass(frozen=True)
class LinkCost:
    prefix_tokens: int
    gap_seconds: float
    full_kv_bytes: int
    linear_state_bytes: int
    total_bytes: int
    transfer_times_seconds: tuple[float, ...]
    full_only_fits_gap: bool
    full_plus_linear_fits_gap: bool

    @classmethod
    def from_link(
        cls,
        *,
        prefix_tokens: int,
        gap_seconds: float,
        budget: HybridStateBudget,
        bandwidths_bytes_per_second: tuple[float, ...],
    ) -> "LinkCost":
        full = prefix_tokens * budget.full_kv_bytes_per_token
        total = full + budget.linear_state_bytes
        times = tuple(total / bandwidth for bandwidth in bandwidths_bytes_per_second)
        return cls(
            prefix_tokens=int(prefix_tokens),
            gap_seconds=float(gap_seconds),
            full_kv_bytes=int(full),
            linear_state_bytes=int(budget.linear_state_bytes),
            total_bytes=int(total),
            transfer_times_seconds=times,
            full_only_fits_gap=any(
                full / bandwidth <= gap_seconds
                for bandwidth in bandwidths_bytes_per_second
            ),
            full_plus_linear_fits_gap=any(time <= gap_seconds for time in times),
        )


@dataclass(frozen=True)
class ComponentCost:
    root_index: int
    request_count: int
    max_input_tokens: int
    full_kv_bytes: int
    linear_snapshots: int
    linear_state_bytes: int
    total_bytes: int
    linear_state_share: float


@dataclass(frozen=True)
class TraceSummary:
    trace_path: str
    request_count: int
    usable_link_count: int
    link_component_count: int
    link_prefix_tokens_mean: float
    link_prefix_tokens_p50: float
    link_prefix_tokens_p90: float
    link_prefix_tokens_p99: float
    aggregate_full_kv_gib: float
    aggregate_hybrid_gib: float
    aggregate_linear_overhead_pct: float
    mha_link_bytes_p50: float
    mha_link_bytes_p90: float
    hybrid_vs_mha_bytes_p50: float
    hybrid_vs_mha_bytes_p90: float
    fit_rates_by_bandwidth: dict[str, dict[str, float]]
    fit_switch_count_by_bandwidth: dict[str, int]
    drain_component_count: int
    max_component_requests: int
    component_requests_p50: float
    component_requests_p90: float
    component_hybrid_gib_p50: float
    component_hybrid_gib_p90: float
    component_linear_share_p50: float
    component_linear_share_p90: float
    all_metrics_finite: bool


def _percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, q))


def _fit_rates(
    links: list[LinkCost],
    budget: HybridStateBudget,
    bandwidths_bytes_per_second: tuple[float, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    rates: dict[str, dict[str, float]] = {}
    switches: dict[str, int] = {}
    for bandwidth in bandwidths_bytes_per_second:
        label = f"{bandwidth / 1024**3:g}_gib_s"
        full_fit = 0
        hybrid_fit = 0
        switch = 0
        for link in links:
            full_time = link.full_kv_bytes / bandwidth
            hybrid_time = link.total_bytes / bandwidth
            full_ok = full_time <= link.gap_seconds
            hybrid_ok = hybrid_time <= link.gap_seconds
            full_fit += int(full_ok)
            hybrid_fit += int(hybrid_ok)
            switch += int(full_ok and not hybrid_ok)
        denom = max(1, len(links))
        rates[label] = {
            "full_kv_only": full_fit / denom,
            "full_plus_linear": hybrid_fit / denom,
        }
        switches[label] = switch
    return rates, switches


def summarize_trace(
    path: str | Path,
    budget: HybridStateBudget,
    *,
    bandwidths_bytes_per_second: tuple[float, ...],
    baseline_mha_bytes_per_token: int,
) -> TraceSummary:
    data = np.load(path)
    usable = np.asarray(data["usable_parent_link"], dtype=bool)
    parents = np.asarray(data["parent_index"], dtype=np.int64)
    gaps = np.asarray(data["arrival_gap_s"], dtype=np.float64)
    prefixes = np.asarray(data["parent_prefix_tokens"], dtype=np.int64)
    roots = np.asarray(data["root_index"], dtype=np.int64)
    inputs = np.asarray(data["input_length"], dtype=np.int64)

    link_mask = usable & (parents >= 0) & np.isfinite(gaps) & (prefixes >= 0)
    links = [
        LinkCost.from_link(
            prefix_tokens=int(prefixes[index]),
            gap_seconds=float(gaps[index]),
            budget=budget,
            bandwidths_bytes_per_second=bandwidths_bytes_per_second,
        )
        for index in np.flatnonzero(link_mask)
    ]

    components: list[ComponentCost] = []
    for root in np.unique(roots):
        indices = np.flatnonzero(roots == root)
        if indices.size == 0:
            continue
        max_tokens = int(np.max(inputs[indices]))
        full = max_tokens * budget.full_kv_bytes_per_token
        snapshots = int(indices.size)
        linear = snapshots * budget.linear_state_bytes
        total = full + linear
        components.append(
            ComponentCost(
                root_index=int(root),
                request_count=snapshots,
                max_input_tokens=max_tokens,
                full_kv_bytes=full,
                linear_snapshots=snapshots,
                linear_state_bytes=linear,
                total_bytes=total,
                linear_state_share=linear / total,
            )
        )

    link_tokens = [link.prefix_tokens for link in links]
    aggregate_full = sum(link.full_kv_bytes for link in links)
    aggregate_total = sum(link.total_bytes for link in links)
    mha_bytes = [
        link.prefix_tokens * baseline_mha_bytes_per_token for link in links
    ]
    hybrid_vs_mha = [
        link.total_bytes / baseline
        for link, baseline in zip(links, mha_bytes, strict=True)
        if baseline > 0
    ]
    rates, switches = _fit_rates(links, budget, bandwidths_bytes_per_second)
    component_hybrid_gib = [
        component.total_bytes / 1024**3 for component in components
    ]
    component_linear_share = [component.linear_state_share for component in components]

    summary = TraceSummary(
        trace_path=str(path),
        request_count=int(inputs.size),
        usable_link_count=len(links),
        link_component_count=len(set(roots[link_mask].tolist())),
        link_prefix_tokens_mean=float(np.mean(link_tokens))
        if link_tokens
        else float("nan"),
        link_prefix_tokens_p50=_percentile(link_tokens, 50),
        link_prefix_tokens_p90=_percentile(link_tokens, 90),
        link_prefix_tokens_p99=_percentile(link_tokens, 99),
        aggregate_full_kv_gib=aggregate_full / 1024**3,
        aggregate_hybrid_gib=aggregate_total / 1024**3,
        aggregate_linear_overhead_pct=(
            (aggregate_total / aggregate_full - 1.0) * 100.0
            if aggregate_full
            else float("nan")
        ),
        mha_link_bytes_p50=_percentile(mha_bytes, 50),
        mha_link_bytes_p90=_percentile(mha_bytes, 90),
        hybrid_vs_mha_bytes_p50=_percentile(hybrid_vs_mha, 50),
        hybrid_vs_mha_bytes_p90=_percentile(hybrid_vs_mha, 90),
        fit_rates_by_bandwidth=rates,
        fit_switch_count_by_bandwidth=switches,
        drain_component_count=len(components),
        max_component_requests=max(
            (component.request_count for component in components), default=0
        ),
        component_requests_p50=_percentile(
            [component.request_count for component in components], 50
        ),
        component_requests_p90=_percentile(
            [component.request_count for component in components], 90
        ),
        component_hybrid_gib_p50=_percentile(component_hybrid_gib, 50),
        component_hybrid_gib_p90=_percentile(component_hybrid_gib, 90),
        component_linear_share_p50=_percentile(component_linear_share, 50),
        component_linear_share_p90=_percentile(component_linear_share, 90),
        all_metrics_finite=False,
    )
    raw_summary = asdict(summary)
    numeric_values: list[float] = []
    for field_name, field_value in raw_summary.items():
        if field_name == "trace_path":
            continue
        queue = [field_value]
        while queue:
            value = queue.pop()
            if isinstance(value, dict):
                queue.extend(value.values())
            elif not isinstance(value, str):
                numeric_values.append(value)
    return TraceSummary(
        **{
            **raw_summary,
            "all_metrics_finite": all(
                math.isfinite(number) for number in numeric_values
            ),
        }
    )


def make_report(budget: HybridStateBudget, summaries: list[TraceSummary]) -> str:
    lines = [
        "# Qwen3.5 Hybrid State Accounting",
        "",
        f"- Full-attention layers: {budget.full_layers}",
        f"- Linear-attention layers: {budget.linear_layers}",
        f"- Full-attention KV: {budget.full_kv_bytes_per_token:,} bytes/token",
        f"- One linear-attention state: {budget.linear_state_bytes:,} bytes "
        f"({budget.linear_state_mib:.3f} MiB)",
        f"- Equal-size crossover: {budget.crossover_tokens:,} tokens",
        "",
        "| Trace | Requests | Usable links | Prefix p50/p90/p99 | Linear overhead | Hybrid vs 8B MHA p50/p90 | Fit switch @50GiB/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        name = Path(item.trace_path).name
        switch = item.fit_switch_count_by_bandwidth.get("50_gib_s", 0)
        lines.append(
            f"| {name} | {item.request_count:,} | {item.usable_link_count:,} | "
            f"{item.link_prefix_tokens_p50:.0f}/{item.link_prefix_tokens_p90:.0f}/"
            f"{item.link_prefix_tokens_p99:.0f} | "
            f"{item.aggregate_linear_overhead_pct:.1f}% | "
            f"{item.hybrid_vs_mha_bytes_p50:.2f}x/{item.hybrid_vs_mha_bytes_p90:.2f}x | "
            f"{switch:,} |"
        )
    lines.extend(
        [
            "",
            "Aggregate link bytes are not resident-cache bytes because shared prefixes are counted per link. Component rows use max context length for deduplicated full KV and one linear snapshot per completed request as the drain worst case.",
            "Bandwidth-normalized fit rates are sensitivity analysis, not measured migration performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
    )
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    parser.add_argument("--bandwidth-gib-s", type=float, default=[25.0, 50.0, 100.0])
    parser.add_argument(
        "--baseline-config",
        type=Path,
        required=True,
        help="Path to a homogeneous-MHA baseline model config, typically Qwen3-8B.",
    )
    args = parser.parse_args()

    config = json.loads(args.model_config.read_text())
    budget = HybridStateBudget.from_config(config)
    baseline_config = json.loads(args.baseline_config.read_text())
    baseline_text = baseline_config.get("text_config", baseline_config)
    baseline_mha_bytes_per_token = (
        int(baseline_text["num_hidden_layers"])
        * 2
        * int(baseline_text["num_key_value_heads"])
        * int(baseline_text["head_dim"])
        * DTYPE_SIZES[baseline_text.get("dtype", "bfloat16").lower()]
    )
    bandwidths = tuple(value * 1024**3 for value in args.bandwidth_gib_s)
    summaries = [
        summarize_trace(
            path,
            budget,
            bandwidths_bytes_per_second=bandwidths,
            baseline_mha_bytes_per_token=baseline_mha_bytes_per_token,
        )
        for path in args.trace
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model_config": str(args.model_config),
        "budget": asdict(budget) | {
            "linear_state_mib": budget.linear_state_mib,
            "crossover_tokens": budget.crossover_tokens,
        },
        "baseline_mha_bytes_per_token": baseline_mha_bytes_per_token,
        "bandwidths_gib_s": args.bandwidth_gib_s,
        "traces": [asdict(item) for item in summaries],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(make_report(budget, summaries))
    print(make_report(budget, summaries), end="")


if __name__ == "__main__":
    main()
