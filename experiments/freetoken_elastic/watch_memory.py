"""Persistent NVML memory observer; read failures are records, not silent exit."""
import argparse,json,os,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True);p.add_argument('--interval',type=float,default=.5);a=p.parse_args();os.sched_setaffinity(0,{80})
 import pynvml as nv
 nv.nvmlInit();h=nv.nvmlDeviceGetHandleByIndex(0)
 with a.output.open('x',buffering=1) as f:
  while not a.stop_file.exists():
   r={'epoch':time.time(),'monotonic':time.perf_counter()}
   try:
    m=nv.nvmlDeviceGetMemoryInfo(h);r.update(total_bytes=m.total,used_bytes=m.used,free_bytes=m.free,
       processes=[{'pid':p.pid,'used_gpu_bytes':p.usedGpuMemory} for p in nv.nvmlDeviceGetComputeRunningProcesses(h)])
   except nv.NVMLError as e:r['error']=str(e)
   f.write(json.dumps(r)+'\n');time.sleep(a.interval)
 nv.nvmlShutdown()
if __name__=='__main__':main()
