"""Calibration (Figure 3, Section 3.2) over the three B0 test runs of the six decision models. Read-only, no inference.

Runs as thresholds_runs123_eval.py and sources/baseline.json: Run 1 runs/quality (PPLX v1.2 prompt12_run01_20261019),
Runs 2/3 replications/run0{2,3}/runs/quality (PPLX v1.2 pplx12_b0_run0{2,3}_20261021); B0, all test pairs of the eight
benchmarks. Method of paper scripts/import_calibration.py: ten equal-width bins of p(match) (bin min(int(10 p), 9)),
Brier score, Murphy partition on the same ten bins (REL, RES, UNC; exact up to the within-bin variance of p), the
share of pairs with p < 0.1 (share_low) and with 0.1 <= p < 0.9 (share_mid).
Per model and run: the bins pooled over the eight benchmarks (n, positives, mean_p, match_rate), pooled Brier, REL,
RES, UNC and shares, their macro means over the benchmarks, and the same quantities per benchmark.
Over the runs: mean and sample SD (ddof=1) of every quantity, and the bins pooled over all three runs (reliability
diagram). Run 1 must reproduce paper sources/calibration.json (asserted, 1e-12).

    .venv/bin/python src/calibration_runs123_eval.py --paper-sources <dir with calibration.json> \
        --out artifacts/results/calibration_runs123_20261104.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from thresholds_runs123_eval import MODELS, RUNS, test_path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ['abt-buy', 'walmart-amazon', 'wdc', 'dblp-acm', 'dblp-scholar', 'dn7-walmart-amazon', 'billiger-de', 'semi-heter']
EDGES = [i/10 for i in range(11)]
KEYS = ('brier', 'reliability', 'resolution', 'uncertainty', 'share_low', 'share_mid')


def empty():
    return [{'n': 0, 'positives': 0, 'sum_p': 0.} for _ in range(10)]


def stats(bins, sq):
    """Brier, Murphy partition and shares of one set of bins (sq: sum of squared errors)."""
    n = sum(x['n'] for x in bins)
    rate = sum(x['positives'] for x in bins)/n
    return {'n': n, 'brier': sq/n,
            'reliability': sum(x['n']*(x['sum_p']/x['n']-x['positives']/x['n'])**2 for x in bins if x['n'])/n,
            'resolution': sum(x['n']*(x['positives']/x['n']-rate)**2 for x in bins if x['n'])/n,
            'uncertainty': rate*(1-rate), 'share_low': bins[0]['n']/n, 'share_mid': sum(x['n'] for x in bins[1:9])/n}


def table(bins):
    return [{'lower': EDGES[i], 'upper': EDGES[i+1], 'n': x['n'], 'positives': x['positives'],
             'mean_p': x['sum_p']/x['n'] if x['n'] else None, 'match_rate': x['positives']/x['n'] if x['n'] else None}
            for i, x in enumerate(bins)]


def mean_sd(xs):
    return {'mean': statistics.mean(xs), 'sd': statistics.stdev(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper-sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    ref = json.loads((args.paper_sources/'calibration.json').read_text())['models']
    labels = {b: {c['pair_id']: c['label'] for c in json.loads((ROOT/'data'/b/'test.json').read_text())} for b in BENCHMARKS}
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'bin_edges': EDGES,
           'units': 'fractions (Brier, REL, RES, UNC, shares, mean_p, match_rate)', 'models': {}, 'checks': {}}
    for m in MODELS:
        res, all_bins = {'runs': {}}, empty()
        for r in RUNS:
            bins, sq_total, per_benchmark = empty(), 0., {}
            for b in BENCHMARKS:
                local, sq, seen = empty(), 0., set()
                for line in test_path(m, b, r).read_text().splitlines():
                    x = json.loads(line)
                    p, y = x['probability_match'], labels[b][x['pair_id']]
                    assert x['pair_id'] not in seen and p is not None and 0 <= p <= 1, (m, b, r)
                    seen.add(x['pair_id'])
                    k = min(int(p*10), 9)
                    for target in (bins[k], local[k], all_bins[k]):
                        target['n'] += 1; target['positives'] += y; target['sum_p'] += p
                    sq += (p-y)**2
                assert seen == labels[b].keys(), (m, b, r)
                per_benchmark[b] = {**stats(local, sq), 'bins': table(local)}
                sq_total += sq
            pooled = stats(bins, sq_total)
            macro = {k: statistics.mean(per_benchmark[b][k] for b in BENCHMARKS) for k in KEYS}
            res['runs'][str(r)] = {'pooled': pooled, 'macro': macro, 'bins': table(bins), 'per_benchmark': per_benchmark}
            if r == 1:  # sources/calibration.json
                old = ref[m]
                assert pooled['n'] == old['n'] and abs(pooled['brier']-old['brier_pooled']) < 1e-12 \
                    and abs(macro['brier']-old['brier_macro']) < 1e-12, m
                for k in ('reliability', 'resolution', 'uncertainty', 'share_mid'):
                    assert abs(pooled[k]-old['pooled'][k]) < 1e-12 and abs(macro[k]-old[k+'_macro']) < 1e-12, (m, k)
                for new, o in zip(table(bins), old['bins']):
                    assert new['n'] == o['n'] and new['positives'] == o['positives'] and \
                        (o['mean_p'] is None or abs(new['mean_p']-o['mean_p']) < 1e-12), m
                for b in BENCHMARKS:
                    assert abs(per_benchmark[b]['brier']-old['per_benchmark'][b]['brier']) < 1e-12, (m, b)
                    for k in ('reliability', 'resolution', 'uncertainty', 'share_mid'):
                        assert abs(per_benchmark[b][k]-old['per_benchmark'][b][k]) < 1e-12, (m, b, k)
        runs = [res['runs'][str(r)] for r in RUNS]
        res['over_runs'] = {
            'pooled': {k: mean_sd([x['pooled'][k] for x in runs]) for k in KEYS},
            'macro': {k: mean_sd([x['macro'][k] for x in runs]) for k in KEYS},
            'per_benchmark': {b: {k: mean_sd([x['per_benchmark'][b][k] for x in runs]) for k in KEYS} for b in BENCHMARKS},
            'bins_pooled_all_runs': table(all_bins)}
        out['models'][m] = res
        out['checks'][m] = 'Run 1 reproduces sources/calibration.json (bins, pooled and macro Brier, REL, RES, UNC, share_mid, per benchmark).'
        print(m, {k: (round(v['mean'], 4), round(v['sd'], 4)) for k, v in res['over_runs']['pooled'].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
