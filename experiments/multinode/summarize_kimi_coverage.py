"""Coverage and explicitly hypothetical latency sensitivity, not policy replay."""
import argparse
import json
from pathlib import Path

import numpy as np


def main(root):
    report = json.loads((root / 'analysis.json').read_text())
    data = np.load(root / 'native_numeric_fields.npz')
    x, c, t, a = (data[k] for k in ('input_tokens', 'cached_tokens', 'ttft', 'arrival'))
    valid = (c >= 0) & (c <= x)
    latency_valid = np.isfinite(t) & np.isfinite(data['e2e']) & (t >= 0) & (data['e2e'] >= 0) & (t <= data['e2e'] + .001)
    hour = ((a - a.min()) // 3600).astype(int)
    full_hours = np.arange(int((a.max() - a.min()) // 3600))
    rows = []
    for cohort in report['cohorts']:
        if cohort['kind'] != 'observed_long_cache':
            continue
        threshold = cohort['threshold']
        mask = valid & (c >= threshold)
        counts = np.array([(mask & (hour == h)).sum() for h in full_hours])
        totals = np.array([(hour == h).sum() for h in full_hours])
        b = cohort['oracle_budget']['ttft_sum_share']
        e = report['ttft_sum_fraction_of_e2e_sum']
        rows.append({
            'threshold': threshold,
            'requests': int(mask.sum()),
            'full_hour_count_min_median_max': np.percentile(counts, [0, 50, 100]).tolist(),
            'full_hour_fraction_min_median_max': np.percentile(counts / totals, [0, 50, 100]).tolist(),
            'ttft_ge_10s_requests': int((mask & latency_valid & (t >= 10)).sum()),
            'fraction_of_ttft_ge_10s_requests': float((mask & latency_valid & (t >= 10)).sum() / (latency_valid & (t >= 10)).sum()),
            'long_input_median_observed_cache_fraction': float(np.median(c[valid & (x >= threshold)] / x[valid & (x >= threshold)])),
            'hypothetical_sensitivity': [
                {'affected_cohort_ttft_mass_fraction': q, 'local_ttft_speedup': s,
                 'global_mean_ttft_speedup': 1 / (1 - b * q * (1 - 1 / s)),
                 'global_mean_e2e_speedup': 1 / (1 - e * b * q * (1 - 1 / s))}
                for q in (.1, .25, .5, 1.) for s in (2., 7., 10.)
            ],
        })
    result = {
        'source_archive_sha256': report['archive_sha256'],
        'full_hours': len(full_hours),
        'semantics': 'Observed serving-path cache counts do not identify transferable history or migration opportunities. Hours are relative to first arrival; finish-log cohort has censoring.',
        'sensitivity_assumptions': 'Hypothetical only: q is affected fraction of cohort summed TTFT, not observed event probability. Only affected TTFT changes by factor s; all other latency and decode remain fixed. No queue/network feedback. Not measured FluidKV performance, a policy comparison, or a global upper bound.',
        'cohorts': rows,
    }
    (root / 'coverage-sensitivity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    main(parser.parse_args().input)
