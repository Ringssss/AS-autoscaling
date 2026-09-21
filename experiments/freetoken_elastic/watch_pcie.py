"""Read-only sampled PCIe rates; samples are not exact DMA byte accounting."""
import argparse,json,os,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True);p.add_argument('--interval',type=float,default=.1);a=p.parse_args();os.sched_setaffinity(0,{81})
 import pynvml as nv
 nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByIndex(0)
 with a.output.open('x',buffering=1) as f:
  while not a.stop_file.exists():
   row={'epoch':time.time(),'monotonic':time.perf_counter()}
   try:
    row.update(rx_kib_s=nv.nvmlDeviceGetPcieThroughput(handle,nv.NVML_PCIE_UTIL_RX_BYTES),tx_kib_s=nv.nvmlDeviceGetPcieThroughput(handle,nv.NVML_PCIE_UTIL_TX_BYTES))
   except nv.NVMLError as e:row['error']=str(e)
   f.write(json.dumps(row)+'\n');time.sleep(a.interval)
 nv.nvmlShutdown()
if __name__=='__main__':main()
