"""Isolate control collectives from normal scheduler CPU collectives."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

root=Path(__file__).parent
source=Path('/home/zhujianian/hygon-runtime-overlay')
target=root/'environment'/'runtime'
shutil.copytree(source,target,dirs_exist_ok=True)
path=target/'managers/hygon_hybrid_control.py'
original=path.read_text()
old='        self.groups = {}\n'
new='''        self.groups = {}
        # Scheduler metrics/control handling must not interleave differently
        # typed object collectives on the shared TP CPU communicator.
        self.control_cpu_group = dist.new_group(
            ranks=dist.get_process_group_ranks(scheduler.tp_cpu_group),
            backend="gloo",
        )
'''
assert original.count(old)==1
modified=original.replace(old,new).replace('group=self.s.tp_cpu_group','group=self.control_cpu_group')
# Exceptions before an inner TP gather must not send one rank directly to the
# final result gather while another rank still gathers preparation status.
begin=modified.index('        store = dist.TCPStore(',modified.index('    def init_group('))
end=modified.index('        pg = init_custom_process_group(',begin)
stage=modified[begin:end]
body=stage[:stage.index('        errors = self.gather(')]
guarded='        init_error = None\n        try:\n'+''.join('    '+line+'\n' for line in body.splitlines())
guarded+='''        except Exception as exc:
            init_error = repr(exc)
            logger.exception("Hybrid group store/identity initialization failed")
        errors = self.gather(init_error)
        if any(errors):
            raise ValueError(str(errors))
        errors = self.gather(None if identity == peer else "peer model/layout mismatch")
        if any(errors):
            raise ValueError(str(errors))
'''
modified=modified[:begin]+guarded+modified[end:]
ast.parse(modified);path.write_text(modified)
(target/'control-isolation.json').write_text(json.dumps(dict(
    original_sha256=hashlib.sha256(original.encode()).hexdigest(),
    patched_sha256=hashlib.sha256(modified.encode()).hexdigest(),
    reason='Dedicated control group plus TP-synchronized TCPStore/identity initialization failures; no placement policy change.'),indent=2)+'\n')
print(target)
