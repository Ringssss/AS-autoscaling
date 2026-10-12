"""Build an opt-in idle-only importer that retains unrelated radix state.

The vendor insert routine owns duplicate KV freeing. Full temporary destination
allocation remains necessary; there is no eviction or under-load concurrency.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).parent
source=ROOT.parent/'weka_baselines/environment/runtime'
target=ROOT/'environment/nonempty-runtime'


def modify(text):
    old='''                if self.s.tree_cache.root_node.children:
                    raise ValueError("destination cache must be empty; flush before migration")
'''
    new='''                # Opt-in nonempty import. Reserve a complete private copy;
                # vendor insert() frees duplicate prefix indices atomically
                # while the scheduler is idle. No other cache is flushed.
                if self.s.tree_cache.root_node.children and os.environ.get("AGENTSHIFT_NONEMPTY_IMPORT") != "1":
                    raise ValueError("destination cache must be empty; flush before migration")
'''
    assert text.count(old)==1;text=text.replace(old,new)
    old='''                installed = True
                if inserted.prefix_len != 0 or inserted.mamba_exist:
                    raise RuntimeError("unexpected destination prefix collision")
'''
    new='''                installed = True
                if os.environ.get("AGENTSHIFT_NONEMPTY_IMPORT") == "1":
                    # MambaRadixCache._insert_helper frees duplicate KV slots
                    # but deliberately leaves redundant recurrent state to
                    # its caller. An existing executable snapshot wins.
                    if inserted.mamba_exist:
                        self.pool.mamba_pool.free(mamba)
                    self.s.tree_cache.sanity_check()
                elif inserted.prefix_len != 0 or inserted.mamba_exist:
                    raise RuntimeError("unexpected destination prefix collision")
'''
    assert text.count(old)==1;text=text.replace(old,new)
    ast.parse(text);return text


if __name__=='__main__':
    shutil.copytree(source,target,dirs_exist_ok=True)
    p=target/'managers/hygon_hybrid_control.py';before=p.read_text();after=modify(before);p.write_text(after)
    (target/'nonempty-manifest.json').write_text(json.dumps(dict(
        before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),
        limitation='Opt-in; same model/TP, both idle, full temporary capacity, no background migration, source retained. '
                   'Nonempty insertion delegates shared-prefix KV freeing to vendor MambaRadixCache; free redundant recurrent slot explicitly.'),indent=2)+'\n')
    print(target)
