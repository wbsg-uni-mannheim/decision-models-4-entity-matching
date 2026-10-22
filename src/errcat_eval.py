"""Error categorization over three runs: stability, group frequencies, link to p(match) and the cascade.

Read-only, no inference. Group set per case and run from runs/error_categorization/sol_flex_full498_run{1,2,3}.
Stability per group: share of cases where all three runs agree on presence/absence, and Fleiss' kappa.
Final assignment: majority (at least two of three runs).

    PYTHONPATH=src python3 src/errcat_eval.py [sample prefix tag] > errcat_summary.json
    F0 sample: PYTHONPATH=src python3 src/errcat_eval.py f0_v1 sol_flex_f0 f0_20261021
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8']
RUNS = [ROOT/f'runs/error_categorization/sol_flex_full498_run{r}' for r in (1, 2, 3)]  # results.jsonl plus fill passes


def fleiss(votes):
    """votes: list of counts of 'present' among 3 raters per case."""
    n, k = len(votes), 3
    p = sum(votes)/(n*k)
    pe = p*p+(1-p)*(1-p)
    po = sum((v*(v-1)+(k-v)*(k-v-1))/(k*(k-1)) for v in votes)/n
    return (po-pe)/(1-pe) if pe < 1 else 1.


def main(sample='preparation_v1', prefix='sol_flex_full498', tag='20261020'):
    cases = {c['case_id']: c for c in json.loads((ROOT/f'artifacts/astra_error_analysis/{sample}/cases_private.json').read_text())['evaluation']}
    runs = []
    for directory in [ROOT/f'runs/error_categorization/{prefix}_run{r}' for r in (1, 2, 3)]:
        rows = {}
        for path in sorted(directory.glob('results*.jsonl')):
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if 'result' in r:
                    assert r['case_id'] not in rows, ('two valid results', r['case_id'])
                    rows[r['case_id']] = {a['group'] for a in r['result']['assignments']}
        runs.append(rows)
    complete = [cid for cid in cases if all(cid in r for r in runs)]
    out = {'cases': len(cases), 'complete_in_all_runs': len(complete), 'stability': {}, 'majority': {}}
    for g in GROUPS:
        votes = [sum(g in r[cid] for r in runs) for cid in complete]
        out['stability'][g] = {'all_agree': sum(v in (0, 3) for v in votes)/len(votes), 'fleiss_kappa': fleiss(votes),
                               'present_majority': sum(v >= 2 for v in votes)}
    final = {cid: {g for g in GROUPS if sum(g in r[cid] for r in runs) >= 2} for cid in complete}
    out['exact_set_agreement_all_runs'] = sum(runs[0][c] == runs[1][c] == runs[2][c] for c in complete)/len(complete)
    by = {}
    for cid, gs in final.items():
        c = cases[cid]
        for key in [('all',), (c['error_type'],), (c['benchmark'], c['error_type'])]:
            d = by.setdefault('|'.join(key), {'n': 0, **{g: 0 for g in GROUPS}})
            d['n'] += 1
            for g in gs:
                d[g] += 1
    out['majority'] = by
    out['groups_per_case'] = sum(len(v) for v in final.values())/len(final)
    out['g7_candidates'] = sorted(cid for cid, gs in final.items() if 'G7' in gs)
    json.dump(final and {k: sorted(v) for k, v in final.items()}, open(ROOT/f'artifacts/results/errcat_final_assignments_{tag}.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:4]) if len(sys.argv) > 1 else main()
