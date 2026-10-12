"""Verify every downloaded repository file against pinned LFS/blob metadata."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    metadata=json.loads(args.metadata.read_text())
    began=time.perf_counter()
    def check(item):
        path=args.model/item['rfilename']
        if path.stat().st_size!=item['size']:raise ValueError('size mismatch '+str(path))
        lfs=item.get('lfs')
        digest=hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:digest.update(b'blob '+str(path.stat().st_size).encode()+b'\0')
        with path.open('rb') as f:
            for block in iter(lambda:f.read(16*1024*1024),b''):digest.update(block)
        expected=lfs['sha256'] if lfs else item['blobId']
        if digest.hexdigest()!=expected:raise ValueError('content hash mismatch '+str(path))
        return {'file':item['rfilename'],'size':item['size'],'hash':digest.hexdigest(),'algorithm':'sha256-lfs' if lfs else 'sha1-git-blob'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        rows=list(executor.map(check,metadata['siblings']))
    report={'revision':metadata['sha'],'verified_files':len(rows),'total_bytes':sum(r['size'] for r in rows),
            'seconds':time.perf_counter()-began,'files':rows}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Verified',len(rows),'files against pinned repository hashes',flush=True)


if __name__=='__main__':main()
