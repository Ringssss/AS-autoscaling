"""Structural reuse census of the materialized weka workload.

Answers, without any routing policy: how much reuse exists, and how long a
prefix survives on the engine that last served it. Pure trace arithmetic.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = [json.loads(l) for l in (ROOT/'data/requests.jsonl').open()]
BLOCK = 16

by_key = defaultdict(list)
for r in ROWS:
    by_key[r['key']].append(r)

def match_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def main():
    turns = Counter()
    lcp_blocks = []
    lcp_tokens = []
    cold = 0
    reuse_rows = []
    for key, rs in sorted(by_key.items()):
        rs.sort(key=lambda r: r['turn'])
        turns[len(rs)] += 1
        for prev, cur in zip(rs, rs[1:]):
            m = match_len(prev['hash_ids'], cur['hash_ids'])
            lcp_blocks.append(m)
            lcp_tokens.append(BLOCK*m)
            if m == 0:
                cold += 1
            reuse_rows.append(dict(key=key, turn=cur['turn'],
                prev_in=prev['input_tokens'], cur_in=cur['input_tokens'],
                lcp_blocks=m, lcp_tokens=BLOCK*m,
                cur_blocks=len(cur['hash_ids']),
                native_gap=cur['native_t']-prev['native_t'],
                api_time=prev['native_api_time']))
    n = len(lcp_tokens)
    tot_in = sum(r['input_tokens'] for r in ROWS)
    import statistics
    def pct(v, p):
        v = sorted(v); return v[min(len(v)-1, int(p/100*len(v)))]
    print(f"streams                {len(by_key)}")
    print(f"requests               {len(ROWS)}")
    print(f"turns-per-stream       {dict(sorted(turns.items()))}")
    print(f"consecutive pairs      {n}")
    print(f"total input tokens     {tot_in:,}")
    print()
    print("--- consecutive-turn prefix reuse ---")
    print(f"pairs with zero LCP    {cold} ({cold/n:.1%})")
    print(f"reusable tokens total  {sum(lcp_tokens):,} ({sum(lcp_tokens)/tot_in:.1%} of all input)")
    print(f"LCP tokens  p10/p50/p90/p99/max  "
          f"{pct(lcp_tokens,10)}/{pct(lcp_tokens,50)}/{pct(lcp_tokens,90)}/{pct(lcp_tokens,99)}/{max(lcp_tokens)}")
    print(f"cur input   p10/p50/p90/p99/max  "
          f"{pct([r['cur_in'] for r in reuse_rows],10)}/{pct([r['cur_in'] for r in reuse_rows],50)}"
          f"/{pct([r['cur_in'] for r in reuse_rows],90)}/{pct([r['cur_in'] for r in reuse_rows],99)}"
          f"/{max(r['cur_in'] for r in reuse_rows)}")
    hit_ratio = [r['lcp_tokens']/r['cur_in'] for r in reuse_rows]
    print(f"prefix-fraction of cur input   p50={statistics.median(hit_ratio):.3f} "
          f"mean={statistics.mean(hit_ratio):.3f}")
    nz = [h for h in hit_ratio if h > 0]
    print(f"when nonzero (n={len(nz)})     p50={statistics.median(nz):.3f} mean={statistics.mean(nz):.3f}")
    print()
    print("--- native inter-turn gap (seconds, from trace) ---")
    gaps = [r['native_gap'] for r in reuse_rows if r['native_gap'] >= 0]
    print(f"p10/p50/p90/p99   {pct(gaps,10):.1f}/{pct(gaps,50):.1f}/{pct(gaps,90):.1f}/{pct(gaps,99):.1f}")
    print(f"gap > api_time of prev (state still resident if never evicted): "
          f"{sum(1 for r in reuse_rows if r['native_gap'] > r['api_time'])}/{n}")
    print()
    print("--- longest prefix vs ANY earlier turn in same stream (not just prev) ---")
    allbest = []
    for key, rs in by_key.items():
        rs.sort(key=lambda r: r['turn'])
        for i, cur in enumerate(rs):
            best = max([match_len(rs[j]['hash_ids'], cur['hash_ids']) for j in range(i)], default=0)
            allbest.append(BLOCK*best)
    print(f"best-over-all-prior tokens  p10/p50/p90/max  "
          f"{pct(allbest,10)}/{pct(allbest,50)}/{pct(allbest,90)}/{max(allbest)}")
    print(f"total recoverable         {sum(allbest):,} ({sum(allbest)/tot_in:.1%})")

    out = ROOT/'oracle/results'
    out.mkdir(parents=True, exist_ok=True)
    (out/'structure.json').write_text(json.dumps(dict(
        streams=len(by_key), requests=len(ROWS), total_input_tokens=tot_in,
        consecutive_pairs=n, zero_lcp_pairs=cold,
        reusable_tokens_consecutive=sum(lcp_tokens),
        reusable_tokens_best_prior=sum(allbest),
        lcp_tokens_percentiles={p: pct(lcp_tokens,p) for p in (10,50,90,99)},
    ), indent=2)+'\n')
    print(f"\nwrote {out/'structure.json'}")

if __name__ == '__main__':
    main()
