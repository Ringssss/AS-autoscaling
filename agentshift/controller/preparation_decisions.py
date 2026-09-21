"""Small, explicit cost comparisons for execution-state preparation.

Estimates describe available actions, not future requests. Callers must record
the estimates and actual outcomes; these functions do not assert engine capacity.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CostCurve:
    points: tuple[tuple[int, float], ...]

    def __post_init__(self):
        if (not self.points or any(n <= 0 or t < 0 or not isfinite(t) for n, t in self.points)
                or any(a[0] >= b[0] for a, b in zip(self.points, self.points[1:]))):
            raise ValueError('cost points must have increasing positive lengths and finite nonnegative times')

    def seconds(self, tokens: int) -> float:
        if tokens <= 0:
            return 0.0
        points = ((0, 0.0),) + self.points
        for (n0, t0), (n1, t1) in zip(points, points[1:]):
            if tokens <= n1:
                return max(0.0, t0 + (t1 - t0) * (tokens - n0) / (n1 - n0))
        if len(self.points) == 1:
            return self.points[0][1] * tokens / self.points[0][0]
        (n0, t0), (n1, t1) = self.points[-2:]
        return t1 + max(0.0, (t1 - t0) / (n1 - n0)) * (tokens - n1)

    def saved_seconds(self, input_tokens: int, reusable_tokens: int) -> float:
        reusable_tokens = max(0, min(input_tokens, reusable_tokens))
        return max(0.0, self.seconds(input_tokens) - self.seconds(input_tokens - reusable_tokens))


@dataclass(frozen=True)
class PreparationEstimate:
    method: str
    ready_seconds: float
    service_seconds: float
    queue_seconds: float
    capacity_feasible: bool = True
    interference_seconds: float = 0.0

    def __post_init__(self):
        values = (self.ready_seconds, self.service_seconds, self.queue_seconds, self.interference_seconds)
        if any(t < 0 or not isfinite(t) for t in values):
            raise ValueError('cost estimates must be finite and nonnegative')

    @property
    def finish_seconds(self):
        # Queue service and state preparation may overlap. This assumes the
        # preparation duration already includes its own resource queueing.
        return max(self.ready_seconds, self.queue_seconds) + self.service_seconds


def choose_preparation(options, *, deadline_seconds, minimum_gain_seconds=0.02):
    """Return a feasible faster action, retaining cold execution for small gains.

    Missing the deadline is recorded by the caller; it never makes an infeasible
    action valid or silently discards a request. An action with faster TTFT but
    excessive predicted foreground interference is rejected.
    """
    feasible = [option for option in options if option.capacity_feasible]
    cold = next(option for option in feasible if option.method == 'cold')
    best = min(feasible, key=lambda option: (
        option.finish_seconds + option.interference_seconds, option.method != 'cold'))
    if best.method == 'cold':
        return cold
    improvement = cold.finish_seconds - best.finish_seconds - best.interference_seconds
    if improvement < minimum_gain_seconds:
        return cold
    return best
