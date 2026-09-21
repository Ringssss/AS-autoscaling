"""Stop an owned idle experimental FreeToken process tree, preserving evidence."""
import argparse,json,os,time,subprocess,urllib.request
from pathlib import Path
import psutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=31580);a=p.parse_args()
    launch=json.loads(a.launch.read_text());root=psutil.Process(launch['pid']);cmd=root.cmdline()
    if root.uids().real!=os.getuid() or 'serve' not in cmd or str(a.port) not in cmd:raise RuntimeError('owned engine identity mismatch')
    with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/v1/stats',timeout=5) as r:stats=json.load(r)
    if stats['requests']['active']!=0:raise RuntimeError('foreground still active')
    owned=[root,*root.children(recursive=True)];record={'started_epoch':time.time(),'before':stats,'identity':[{'pid':q.pid,'create_time':q.create_time(),'cmdline':q.cmdline()} for q in owned],'scope':'Experimental lifecycle, excluded from algorithm benefits.'}
    root.terminate();_,alive=psutil.wait_procs(owned,timeout=15);record['forced_pids']=[]
    for q in alive:
        if q.is_running() and q.status()!=psutil.STATUS_ZOMBIE:record['forced_pids'].append(q.pid);q.kill()
    psutil.wait_procs(alive,timeout=5);record['finished_epoch']=time.time()
    record['gpu_processes_after']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
    record['released']=not any(q.is_running() and q.status()!=psutil.STATUS_ZOMBIE for q in owned)
    a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'released':record['released'],'forced_pids':record['forced_pids']}))
if __name__=='__main__':main()
