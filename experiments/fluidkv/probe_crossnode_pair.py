"""Can a clean CROSS-NODE pair establish a migration group?

Same-node pairs on both nodes failed group init four times running with
`DistStoreError: Timed out after 91 seconds waiting for clients. 1/2 clients
joined.` The historically successful topology (sweep line D) was cross-node, so
this probes that shape with the only two engines currently clean:

    27:31782  (CLEAN, 262144/256)  ->  29:31762  (CLEAN, 262144/256)

Only group init is exercised here, with generous timeouts and several ports,
so the answer is actionable either way and no prefix state is consumed.
"""
import json
import time
import urllib.error
import urllib.request

SRC = 'http://12.11.3.27:31782'
DST = 'http://12.11.3.29:31762'
SRC_HOST = '12.11.3.27'
PORTS = (44160, 44176, 44192, 44208, 44224)


def post(url, path, payload, timeout=150):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return time.perf_counter() - t0, json.loads(fh.read())
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - t0, {
            'http': exc.code, 'b': exc.read().decode('utf-8', 'replace')[:220]}
    except Exception as exc:
        return time.perf_counter() - t0, {'exc': repr(exc)[:130]}


def main():
    print(f'SRC {SRC}   DST {DST}   master {SRC_HOST}', flush=True)
    for port in PORTS:
        name = f'xn-{port}-{int(time.time()) % 100000}'
        r_src = post(SRC, '/agentshift/hybrid_state/direct_group/init',
                     {'group_rank': 0, 'group_name': name,
                      'master_address': SRC_HOST, 'port': port,
                      'world_size': 2, 'backend': 'nccl'})
        r_dst = post(DST, '/agentshift/hybrid_state/direct_group/init',
                     {'group_rank': 1, 'group_name': name,
                      'master_address': SRC_HOST, 'port': port,
                      'world_size': 2, 'backend': 'nccl'})
        ok = all(isinstance(v[1], dict) and v[1].get('success') is True
                 for v in (r_src, r_dst))
        print(f'port{port}: src {r_src[0]:6.1f}s dst {r_dst[0]:6.1f}s ok={ok}',
              flush=True)
        if ok:
            print('  => 27:31782 -> 29:31762 USABLE for cross-node migration',
                  flush=True)
            return
        print(f'   src: {str(r_src[1])[:180]}', flush=True)
        print(f'   dst: {str(r_dst[1])[:180]}', flush=True)
    print('  => no port worked; cross-node init also fails on this pair',
          flush=True)


if __name__ == '__main__':
    main()
