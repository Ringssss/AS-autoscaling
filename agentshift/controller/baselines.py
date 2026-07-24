from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TTLCalibration:
    ttl_seconds: float
    expected_cost_seconds: float
    recompute_seconds: float
    kv_gib: float
    hbm_cost_per_gib_second: float


def calibrate_ttl(
    gap_samples_seconds: Iterable[float],
    *,
    recompute_seconds: float,
    kv_gib: float,
    hbm_cost_per_gib_second: float,
) -> TTLCalibration:
    """Choose a retention TTL from an empirical tool-gap distribution.

    The mechanism-equivalent Continuum cost is GPU residency until eviction,
    plus historical-prefix recomputation when the tool outlives the TTL.
    """

    samples = tuple(max(0.0, float(value)) for value in gap_samples_seconds)
    if not samples:
        raise ValueError("TTL calibration requires at least one gap sample")
    if recompute_seconds < 0 or kv_gib < 0 or hbm_cost_per_gib_second < 0:
        raise ValueError("TTL calibration costs must be non-negative")

    candidates = sorted({0.0, *samples})
    best_ttl = 0.0
    best_cost = float("inf")
    for ttl_seconds in candidates:
        total = 0.0
        for gap_seconds in samples:
            total += (
                min(gap_seconds, ttl_seconds)
                * kv_gib
                * hbm_cost_per_gib_second
            )
            if gap_seconds > ttl_seconds:
                total += recompute_seconds
        expected = total / len(samples)
        if expected < best_cost:
            best_ttl = ttl_seconds
            best_cost = expected

    return TTLCalibration(
        ttl_seconds=best_ttl,
        expected_cost_seconds=best_cost,
        recompute_seconds=recompute_seconds,
        kv_gib=kv_gib,
        hbm_cost_per_gib_second=hbm_cost_per_gib_second,
    )


@dataclass(frozen=True, slots=True)
class AgentixRoutingDecision:
    engine: str
    source_cost_seconds: float
    destination_cost_seconds: float


def agentix_style_route(
    *,
    source_engine: str,
    destination_engine: str,
    source_queue_seconds: float,
    destination_queue_seconds: float,
    next_call_service_seconds: float,
    destination_reprefill_seconds: float,
) -> AgentixRoutingDecision:
    """Locality-aware expected-completion routing without state movement."""

    values = (
        source_queue_seconds,
        destination_queue_seconds,
        next_call_service_seconds,
        destination_reprefill_seconds,
    )
    if any(value < 0 for value in values):
        raise ValueError("routing costs must be non-negative")
    source_cost = source_queue_seconds + next_call_service_seconds
    destination_cost = (
        destination_queue_seconds
        + destination_reprefill_seconds
        + next_call_service_seconds
    )
    engine = source_engine if source_cost <= destination_cost else destination_engine
    return AgentixRoutingDecision(engine, source_cost, destination_cost)
