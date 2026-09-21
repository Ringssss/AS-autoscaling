"""Shared observations of engine-confirmed hybrid snapshot boundaries.

The directory contains bounded routing estimates, never reservations or a
guarantee of present cache residency. Every policy must receive the same current
inspection metadata; source/destination preparation still validates state and
capacity. No boundary is inferred from block size, prompt length or future input.
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _tokens(token_ids):
    if isinstance(token_ids, (str, bytes, bytearray)):
        raise ValueError("token_ids must contain nonnegative integer token IDs")
    try:
        tokens = tuple(token_ids)
    except TypeError as exc:
        raise ValueError("token_ids must be iterable") from exc
    for token in tokens:
        _nonnegative_int(token, "token ID")
    return tokens


@dataclass(frozen=True)
class CapacityObservation:
    """Raw free and evictable capacity at inspection time, kept separate."""

    available_token_slots: int
    evictable_token_slots: int
    available_snapshot_slots: int
    evictable_snapshot_slots: int


def _parse(token_ids, response):
    tokens = _tokens(token_ids)
    if not isinstance(response, Mapping):
        raise ValueError("inspection response must be an object")
    try:
        query_tokens = _nonnegative_int(response["query_tokens"], "query_tokens")
        matched_tokens = _nonnegative_int(response["matched_tokens"], "matched_tokens")
        boundaries = response["snapshot_boundaries"]
        capacity = CapacityObservation(**{
            name: _nonnegative_int(response[name], name)
            for name in CapacityObservation.__dataclass_fields__
        })
    except KeyError as exc:
        raise ValueError(f"inspection response is missing {exc.args[0]}") from exc
    if query_tokens != len(tokens):
        raise ValueError("query_tokens does not match inspected token_ids")
    if matched_tokens > query_tokens:
        raise ValueError("matched_tokens exceeds inspected input length")
    if not isinstance(boundaries, (list, tuple)):
        raise ValueError("snapshot_boundaries must be an ordered list")
    previous = 0
    prefixes = []
    for boundary in boundaries:
        _nonnegative_int(boundary, "snapshot boundary")
        if boundary <= previous:
            raise ValueError("snapshot boundaries must be positive, unique and increasing")
        if boundary > matched_tokens:
            raise ValueError("snapshot boundary exceeds matching inspected prefix")
        prefixes.append(tokens[:boundary])
        previous = boundary
    return tokens, tuple(prefixes), capacity


def validated_snapshots(token_ids, response) -> tuple[tuple[int, ...], ...]:
    """Return only exact prefix tuples explicitly confirmed by inspection.

    Required response fields are ``query_tokens``, ``matched_tokens``,
    ``snapshot_boundaries``, and all four ``CapacityObservation`` fields. A raw KV
    match does not itself establish a snapshot. The prompt end is included only
    if it appears in ``snapshot_boundaries``. Malformed/overclaimed responses
    raise ``ValueError``; callers may record the inspection failure and fall back
    to stale estimates or cold execution, but must not fabricate a successful hit.
    """
    return _parse(token_ids, response)[1]


def best_reusable_prefix(token_ids, observedprefixes: Iterable, require_suffix=True) -> int:
    """Longest confirmed matching boundary; require a nonempty suffix by default."""
    tokens = _tokens(token_ids)
    maximum = len(tokens) - int(require_suffix)
    longest = 0
    for observed in observedprefixes:
        prefix = _tokens(observed)
        if longest < len(prefix) <= maximum and tokens[:len(prefix)] == prefix:
            longest = len(prefix)
    return longest


@dataclass
class _EngineObservations:
    prefixes: OrderedDict
    capacity: CapacityObservation
    token_count: int = 0


class SnapshotDirectory:
    """Bounded LRU observations keyed by stable physical engine identity.

    Limits apply per engine to both snapshot count and total retained token IDs;
    shared prefixes are conservatively counted more than once. Engines have a
    separate LRU bound. Forgetting entries loses estimated reuse; it never claims
    state that was not observed. Call ``discard_engine`` after an engine restart
    or retirement rather than reinterpreting a physical ID as a compacted slot.

    Calls are synchronous and designed for a single controller event loop.
    ``update`` validates atomically before changing anything. HTTP/inspection
    failures should be handled by the caller; existing entries are still stale
    estimates if a refresh fails. No future arrival or child identity is needed.
    """

    def __init__(self, *, max_snapshots_per_engine=64, max_tokens_per_engine=131072,
                 max_engines=128):
        for name, value in (("max_snapshots_per_engine", max_snapshots_per_engine),
                            ("max_tokens_per_engine", max_tokens_per_engine),
                            ("max_engines", max_engines)):
            if _nonnegative_int(value, name) == 0:
                raise ValueError(f"{name} must be positive")
        self.max_snapshots_per_engine = max_snapshots_per_engine
        self.max_tokens_per_engine = max_tokens_per_engine
        self.max_engines = max_engines
        self._engines = OrderedDict()

    @staticmethod
    def _engine_key(physical_engine):
        if ((type(physical_engine) is int and physical_engine >= 0)
                or (isinstance(physical_engine, str) and physical_engine)):
            return physical_engine
        raise ValueError("physical_engine must be a nonnegative integer or nonempty string")

    def update(self, physical_engine, token_ids, response) -> tuple[tuple[int, ...], ...]:
        engine = self._engine_key(physical_engine)
        tokens, confirmed, capacity = _parse(token_ids, response)
        state = self._engines.get(engine)
        if state is None:
            state = _EngineObservations(OrderedDict(), capacity)
            self._engines[engine] = state
        confirmed_set = set(confirmed)
        # A response is authoritative only for boundaries along this inspected
        # path. Longer snapshots and unrelated branches were not inspected.
        stale = [prefix for prefix in state.prefixes
                 if len(prefix) <= len(tokens) and tokens[:len(prefix)] == prefix
                 and prefix not in confirmed_set]
        for prefix in stale:
            state.token_count -= len(prefix)
            del state.prefixes[prefix]
        for prefix in confirmed:
            if len(prefix) > self.max_tokens_per_engine:
                continue  # Return confirmed observation without retaining it.
            if prefix not in state.prefixes:
                state.prefixes[prefix] = None
                state.token_count += len(prefix)
            state.prefixes.move_to_end(prefix)
            while (len(state.prefixes) > self.max_snapshots_per_engine
                   or state.token_count > self.max_tokens_per_engine):
                discarded, _ = state.prefixes.popitem(last=False)
                state.token_count -= len(discarded)
        state.capacity = capacity
        self._engines.move_to_end(engine)
        while len(self._engines) > self.max_engines:
            self._engines.popitem(last=False)
        return confirmed

    def prefixes(self, physical_engine) -> tuple[tuple[int, ...], ...]:
        state = self._engines.get(self._engine_key(physical_engine))
        return tuple(state.prefixes) if state is not None else ()

    def resources(self, physical_engine) -> CapacityObservation | None:
        state = self._engines.get(self._engine_key(physical_engine))
        return state.capacity if state is not None else None

    def best_reusable_prefix(self, physical_engine, token_ids, require_suffix=True) -> int:
        engine = self._engine_key(physical_engine)
        tokens = _tokens(token_ids)
        state = self._engines.get(engine)
        if state is None:
            return 0
        longest = best_reusable_prefix(tokens, state.prefixes, require_suffix)
        if longest:
            state.prefixes.move_to_end(tokens[:longest])
        self._engines.move_to_end(engine)
        return longest

    def discard_engine(self, physical_engine):
        self._engines.pop(self._engine_key(physical_engine), None)

    def clear(self):
        self._engines.clear()
