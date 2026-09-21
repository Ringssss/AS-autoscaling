"""Capacity-only ballast: reserve unused VRAM without doing ongoing GPU compute.

A separate PID makes this unavailable-memory emulation explicit in process
accounting. It does not emulate a consumer GPU's compute, HBM or PCIe bandwidth.
"""
import argparse,json,os,signal,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--leave-free-gib',type=float,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    if a.output.exists():raise ValueError('output exists')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,{80})
    import torch
    torch.cuda.set_device(a.device);torch.cuda.init()
    free,total=torch.cuda.mem_get_info(a.device);target=int(a.leave_free_gib*(1<<30));margin=32<<20
    amount=free-target-margin
    if amount<=0:raise ValueError('not enough free VRAM for requested capacity envelope')
    tensor=torch.empty(amount,dtype=torch.uint8,device=f'cuda:{a.device}')
    torch.cuda.synchronize(a.device)
    after,_=torch.cuda.mem_get_info(a.device)
    row={'pid':os.getpid(),'started_epoch':time.time(),'device':a.device,'name':torch.cuda.get_device_name(a.device),
         'requested_free_gib':a.leave_free_gib,'before_free_bytes':free,'device_total_bytes':total,'ballast_tensor_bytes':amount,
         'after_free_bytes':after,'safety_margin_bytes':margin,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
         'scope':'Capacity-only separate-process VRAM reservation; not consumer hardware performance emulation.','alive':True}
    a.output.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not a.stop_file.exists():time.sleep(1)
    except KeyboardInterrupt:pass
    finally:
        del tensor;torch.cuda.empty_cache();row.update(alive=False,stopped_epoch=time.time())
        a.output.write_text(json.dumps(row,indent=2)+'\n')
if __name__=='__main__':main()
