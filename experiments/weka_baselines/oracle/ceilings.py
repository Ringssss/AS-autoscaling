"""Decompose the reuse ceiling into routing-reachable vs mobility-reachable.

Three nested upper bounds over the same workload:

  B0  static-perfect   every stream keeps one home engine, infinite capacity,
                       no eviction ever. The best any single-home
                       (DualMap-class, hash-ring, sticky) router can do.
  B1  mobile-perfect   state may be moved between engines at request time for
                       free. Best any *online-routing-only* system can do if
                       it could relocate state, i.e. the FluidKV ideal.
  B2  structural       every block that repeats anywhere in the workload,
                       ignoring which stream/engine holds it.

The gap B1-B0 is the headroom that mobility buys; B2-B1 is headroom that
requires cross-stream sharing or unbounded capacity.
"""
import json, hashlib, random
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.simulate import build_workload, BLOCK, ROOT


def b0_static_perfect(rows, engines):
    """One home engine per stream, unbounded cache: full intra-stream reuse."""
    home = {}
    hit = 0
    for r in rows:
        home.setdefault(r['key'], int(hashlib.sha256(('i:'+r['key']).encode()).hexdigest(),16) % engines)
        prior = [q for q in rows if q['key']==r['key'] and q['ordinal'] < r['ordinal']]
        # with unbounded state the best prior is the longest shared prefix
        best = 0
        for q in prior:
            n = 0
            for a, b in zip(q['hash_ids'], r['hash_ids']):
                if a != b: break
                n += 1
            best = max(best, n)
        hit += best*BLOCK
    return hit


def b1_mobile_perfect(rows, engines, cap_blocks_per_engine=None):
    """Free state mobility: a request's blocks are placed on whichever engine
    served its longest available prefix, and blocks migrate with the request.
    Effectively one shared global cache, i.e. ideal FluidKV."""
    resident = {}          # hash_id -> owner engine
    free = [0]*engines
    hit = 0
    for r in rows:
        n = 0
        for h in r['hash_ids']:
            if h in resident:
                n += 1
            else:
                break
        hit += n*BLOCK
        # install this request's blocks; assign new blocks to the emptiest engine
        t = min(range(engines), key=lambda i: free[i])
        for h in r['hash_ids']:
            if h not in resident:
                resident[h] = t; free[t] += 1
    return hit


def b2_structural(rows):
    """Any block reused anywhere counts; no placement, no capacity."""
    seen = set(); hit = 0
    for r in rows:
        for h in r['hash_ids']:
            if h in seen: hit += BLOCK
            else: seen.add(h)
    return hit


def main():
    rows = build_workload(60.0, 128, 'balanced')
    total = sum(r['input_tokens'] for r in rows)
    out = {}
    for engines in (3, 4, 8):
        b0 = b0_static_perfect(rows, engines)
        b1 = b1_mobile_perfect(rows, engines)
        b2 = b2_structural(rows)
        out[engines] = dict(B0_static_perfect=b0, B1_mobile_perfect=b1, B2_structural=b2)
        print(f"engines={engines}")
        print(f"  B0 static-perfect   {b0:>12,}  {b0/total:6.1%}   (best single-home router)")
        print(f"  B1 mobile-perfect   {b1:>12,}  {b1/total:6.1%}   (ideal FluidKV)")
        print(f"  B2 structural       {b2:>12,}  {b2/total:6.1%}   (any repeat anywhere)")
        print(f"  mobility headroom B1-B0 = {(b1-b0)/total:6.1%} of input")
        print(f"  sharing headroom B2-B1  = {(b2-b1)/total:6.1%} of input")
        print()
    (ROOT/'oracle/results/ceilings.json').write_text(json.dumps(
        dict(total_input_tokens=total, by_engines=out), indent=2)+'\n')
    print(f"total input tokens {total:,}")


if __name__ == '__main__':
    main()
