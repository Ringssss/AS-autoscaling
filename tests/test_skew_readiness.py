"""Regressions for measurement failures found in the October p03/p04 runs."""
import importlib.util
import json
from pathlib import Path
import time
import unittest

spec = importlib.util.spec_from_file_location("skew", Path(__file__).parents[1] /
    "experiments/fluidkv/p06_skew_readiness.py")
skew = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skew)


class RouterTests(unittest.TestCase):
    def test_first_burst_reserves_inflight_and_completion_releases_it(self):
        router = skew.Router("balanced", {}, 8, [True, False])
        targets = [router.reserve() for _ in range(16)]
        self.assertEqual(targets.count(0), 8)
        self.assertEqual(targets.count(1), 8)
        for target in targets:
            router.release(target, True)
        self.assertEqual(router.inflight, [0, 0])

    def test_failed_requests_are_slo_violations(self):
        good = dict(error=None, ttft=.1, e2e=.2, tpot=.01,
                    token_intervals=[.01], coalesced_tokens=0,
                    cached_tokens=10, target=0)
        bad = dict(good, error="timeout", ttft=None, e2e=None, tpot=None)
        report = skew.summarize([good, bad], 1, 1, .1)
        self.assertEqual(report["slo_attainment"], .5)
        self.assertEqual(report["errors"], 1)

    def test_cache_cost_preserves_affinity_until_batch_queue_makes_cold_cheaper(self):
        router = skew.Router("cache_cost", {"warm_ttft": .1, "cold_ttft": 2.,
                            "warm_service": 3.}, 8, [True, False])
        self.assertEqual([router.reserve() for _ in range(8)], [0] * 8)
        self.assertEqual(router.reserve(), 1)


class TimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_ttft_and_completion_include_before_dispatch_wait(self):
        class Content:
            def __aiter__(self):
                async def iterate():
                    for count in [1, 2]:
                        yield b"data:" + json.dumps({"meta_info": {
                            "completion_tokens": count, "cached_tokens": 10},
                            "output_ids": [42] * count}).encode() + b"\n"
                return iterate()

        class Response:
            content = Content()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def raise_for_status(self):
                pass

        class Session:
            def post(self, *args, **kwargs):
                return Response()

        row = await skew.generate(Session(), "http://test", [1, 2], 2,
                                  arrival=time.perf_counter() - 1)
        self.assertGreaterEqual(row["ttft"], 1)
        self.assertGreaterEqual(row["e2e"], row["ttft"])
        self.assertGreaterEqual(row["router_wait"], 1)
        self.assertLess(row["serve_ttft"], .1)


if __name__ == "__main__":
    unittest.main()
