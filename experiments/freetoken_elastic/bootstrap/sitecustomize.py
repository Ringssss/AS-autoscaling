"""Opt-in research wrapper, installed only in explicitly launched experiment processes."""
import os
if os.environ.get('FLUIDKV_ELASTIC_POLICY') and os.environ.get('FLUIDKV_ELASTIC_LOG'):
    import importlib.abc
    import importlib.machinery
    import sys
    class ElasticLoader(importlib.abc.Loader):
        def __init__(self, original): self.original=original
        def create_module(self, spec):
            fn=getattr(self.original,'create_module',None)
            return fn(spec) if fn else None
        def exec_module(self, module):
            self.original.exec_module(module)
            from elastic_runtime_hook import install
            install(module.Scheduler)
    class ElasticFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname!='freetoken.scheduler.scheduler':return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec and spec.loader:spec.loader=ElasticLoader(spec.loader)
            return spec
    sys.meta_path.insert(0,ElasticFinder())
