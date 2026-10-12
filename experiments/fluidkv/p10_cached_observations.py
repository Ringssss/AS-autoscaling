"""Memoize immutable request/cache-version lookups; leave upstream policy intact."""
import numpy as np

from experiments.fluidkv.p08_upstream_routers import UpstreamRouter


class CachedObservationRouter(UpstreamRouter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observed_arrays={}
        self.observed_hits={}
        if self.policy=='dualmap':
            for worker in range(self.workers):
                self.dualmap.replica_for_engine(worker).get_num_recompute_token_ids=(
                    lambda ids,w=worker:len(ids)-self.observed_hit(w,ids))

    def observed_hit(self,worker,ids):
        # Request inputs are immutable, and this campaign's directory only
        # appends confirmed boundaries. Strong references prevent ID reuse.
        key=id(ids)
        if key not in self.observed_arrays:
            self.observed_arrays[key]=(ids,np.asarray(ids,dtype=np.int32))
        version=len(self.cache[worker])
        cached=self.observed_hits.get((worker,key))
        if cached is None or cached[0]!=version:
            cached=(version,self.hit(worker,self.observed_arrays[key][1]))
            self.observed_hits[worker,key]=cached
        return cached[1]
