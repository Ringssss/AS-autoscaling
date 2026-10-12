"""Paper Algorithm 1 of arXiv:2608.19677, not the authors' private harness.

Rates must come from an earlier training interval, not the evaluated future.
The plan reserves no physical cache and transfers no state. Actual engine
cache hits and request latency must be measured by the common serving harness.
"""
from dataclasses import dataclass
import math
import random
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Plan:
    assignments: dict[str, tuple[int, ...]]
    expected_load: tuple[float, ...]
    slots_used: int
    construction_seconds: float


def build_plan(rates: Mapping[str, float], replicas: int, qcap: float, slots: int) -> Plan:
    started = time.perf_counter()
    if replicas < 1 or slots < 0 or not math.isfinite(qcap) or qcap <= 0:
        raise ValueError("positive replica count/qcap and nonnegative slot budget required")
    if any(not math.isfinite(rate) or rate < 0 for rate in rates.values()):
        raise ValueError("rates must be finite and nonnegative")
    ordered = sorted(((key, rate) for key, rate in rates.items() if rate > 0), key=lambda x: (-x[1], x[0]))
    admitted, used = [], 0
    for key, rate in ordered:
        count = max(1, math.ceil(rate / qcap))
        if count > replicas:
            raise ValueError(f"key {key!r} exceeds total calibrated destination capacity")
        if used + count > slots:
            break
        admitted.append((key, rate, count))
        used += count
    cold_rate = sum(rate for _, rate in ordered) - sum(rate for _, rate, _ in admitted)
    loads = [cold_rate / replicas] * replicas
    table = {}
    for key, rate, count in admitted:
        targets = tuple(sorted(range(replicas), key=lambda i: (loads[i], i))[:count])
        table[key] = targets
        for target in targets:
            loads[target] += rate / count
    return Plan(table, tuple(loads), used, time.perf_counter() - started)


def dispatch(plan: Plan, key: str, live_load: Sequence[float], rng: random.Random) -> int:
    if len(live_load) != len(plan.expected_load) or any(not math.isfinite(x) or x < 0 for x in live_load):
        raise ValueError("finite, nonnegative load required for every destination")
    targets = plan.assignments.get(key)
    if targets is None:
        targets = rng.sample(range(len(live_load)), min(2, len(live_load)))
    # Stable deterministic tie breaking inside planned sets; random sampled order
    # breaks ties for the cold-tail power-of-two choices.
    return min(targets, key=lambda i: live_load[i])
