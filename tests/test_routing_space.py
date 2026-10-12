"""Accounting controls for the same-router preparation pilot."""
import importlib.util
from pathlib import Path
import unittest
import asyncio
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

spec = importlib.util.spec_from_file_location('routing_space', Path(__file__).parents[1]/
    'experiments/fluidkv/p07_routing_space.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Accounting(unittest.TestCase):
    def router(self, policy='lmetric'):
        return m.Router(policy, dict(decode_coefficients=[.01,.001,.002],
            prefill_points=[[32768,4],[65536,8]]),
            [[np.arange(32768,dtype=np.int32)],[]],None)

    def test_admission_step_and_hit_correction_are_distinct(self):
        r=self.router()
        ids=np.arange(33792,dtype=np.int32)
        target,_=r.reserve('x',ids)
        self.assertEqual(target,0)
        self.assertEqual(r.pending,[1024,0])
        r.step(0,{'scheduled_requests':{'sum_prefill_tokens':256}})
        self.assertEqual(r.pending[0],768)
        r.progress(0,'x',{'meta_info':{'cached_tokens':33024}},1,0)
        self.assertEqual(r.pending[0],512)
        r.progress(0,'x',{'meta_info':{'cached_tokens':33024}},2,1)
        self.assertEqual(r.pending[0],512)
        r.release(0,'x',ids,True)
        self.assertEqual(r.pending,[0,0])
        self.assertFalse(r.active[0])

    def test_future_state_is_not_published_on_admission_or_failure(self):
        r=self.router()
        ids=np.arange(34816,dtype=np.int32)
        target,_=r.reserve('x',ids)
        self.assertEqual(r.hit(target,ids),32768)
        r.release(target,'x',ids,False)
        self.assertEqual(r.hit(target,ids),32768)

    def test_real_added_warm_candidate_changes_same_score(self):
        ids=np.arange(33280,dtype=np.int32)
        r=self.router()
        self.assertEqual([r.reserve(str(i),ids)[0] for i in range(3)],[0,0,0])
        warm=self.router()
        warm.cache[1].append(np.arange(32768,dtype=np.int32))
        self.assertEqual([warm.reserve(str(i),ids)[0] for i in range(4)],[0,1,0,1])

    def test_two_stage_hotspot_fallback_offloads_after_threshold(self):
        r=self.router('lmetric_guard')
        ids=np.arange(33280,dtype=np.int32)
        self.assertEqual([r.reserve(str(i),ids)[0] for i in range(3)],[0,0,1])
        self.assertEqual(r.decisions[-1]['filtered'],[0])
        r.cache[1].append(np.arange(32768,dtype=np.int32))
        r.reserve('next',ids)
        self.assertEqual(r.decisions[-1]['filtered'],[])


class PreparationTiming(unittest.IsolatedAsyncioTestCase):
    async def test_recompute_does_not_block_source_consumers(self):
        foreground_started=asyncio.Event()
        order=[]
        async def fake_generate(session,url,ids,output,arrival=None,callback=None):
            is_prepare=url=='target' and len(ids)==32768
            if is_prepare:
                order.append('prepare-start')
                await foreground_started.wait()
                order.append('prepare-done')
            elif len(ids)>32768:
                order.append('foreground')
                foreground_started.set()
            if callback:
                callback({'meta_info':{'cached_tokens':0 if is_prepare else 32768}},output,0)
            return dict(error=None,ttft=.001,e2e=.002,tpot=.001,
                service_seconds=.002,cached_tokens=32768,token_intervals=[.001],coalesced_tokens=0)
        async def fake_inspect(session,url,ids):
            return {'matched_tokens':len(ids)}
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(urls=['source','target'],placement='skew',gap=.02,
                lookahead=128,background_requests=0,output_cap=None,out=Path(directory),
                ttft_slo=10,tpot_slo=.2)
            parent=dict(token_offset=0,input_tokens=32768)
            child=dict(token_offset=0,input_tokens=32832,output_tokens=2,
                       native_output_tokens=2,token_sha256='fixed')
            case=dict(name='fake',common_parent_tokens=32768,streams=[dict(key='stream-0',
                requests=[parent,child,child,child],first_offset=0,gaps=[0,0])])
            telemetry=SimpleNamespace(router=None,events=[1,1],peaks=[0,0])
            with patch.object(m,'generate',fake_generate),patch.object(m,'inspect',fake_inspect),\
                 patch.object(m.skew,'reset',AsyncMock()),patch.object(m.skew,'drain',AsyncMock()):
                result=await asyncio.wait_for(m.trial(None,args,case,'lmetric','recompute',0,
                    np.arange(32832,dtype=np.int32),{},telemetry,None),timeout=1)
            self.assertLess(order.index('foreground'),order.index('prepare-done'))
            self.assertEqual(result['preparation']['mode'],'asynchronous')
            self.assertEqual(result['summary']['requests'],3)


if __name__=='__main__':
    unittest.main()
