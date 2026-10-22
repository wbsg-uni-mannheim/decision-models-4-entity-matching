"""Offline family-mean B0 planning proxy for the audited missing variant pairs."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from fresh_run_cost_plan import inventory, sha

PROOF = 'artifacts/verification/baseline_only_source_proof_20261014.json'
DIAGNOSIS = 'artifacts/operations/variants_terminal_diagnosis_20261015T0003Z.json'
FAMILIES = {'billiger-de-en-100un': 'billiger-de', 'billiger-en-de-100un': 'billiger-de',
            'billiger-en-en-100un': 'billiger-de', 'wdc-000un': 'wdc', 'wdc-050un': 'wdc'}


def project(cells, sources):
    index = {(s['model'], s['benchmark']): s for s in sources
             if s['namespace'] == 'quality' and s['variant'] == 'B0' and s['split'] == 'test'}
    seen, result = set(), []
    for c in cells:
        key = c['model'], c['benchmark']
        if key in seen or c['benchmark'] not in FAMILIES:
            raise ValueError('Duplicate or unknown target cell')
        seen.add(key)
        if (type(c['missing']) is not int or not 0 <= c['missing'] <= c['expected']
                or c['responses'] + c['missing'] != c['expected']):
            raise ValueError('Invalid remaining pair count')
        if not c['missing']:
            continue
        source = index[c['model'], FAMILIES[c['benchmark']]]
        if source['responses'] <= 0:
            raise ValueError('Empty source')
        factor = c['missing']/source['responses']
        result.append({'model': c['model'], 'benchmark': c['benchmark'], 'missing': c['missing'],
                       'missing_pair_ids_sha256': c['missing_pair_ids_sha256'],
                       'source': source, 'known_proxy_usd': factor*source['known_usd'],
                       'historical_unknown_scenario_usd': factor*source['unknown_attempts']*.05})
    return result


def build(root):
    diagnosis = json.loads((root/DIAGNOSIS).read_text())
    if diagnosis['job'] != '365279' or diagnosis['status'] != 'terminal_scoped_audit_no_recovery':
        raise ValueError('Wrong diagnosis')
    for path, expected in diagnosis['files_sha256'].items():
        if sha(root/path) != expected:
            raise ValueError('Audited variant journal changed')
    sources, proofs = inventory(root, [PROOF])
    cells = project(diagnosis['cells'], sources)
    missing = sum(c['missing'] for c in cells)
    if missing != diagnosis['totals']['missing']:
        raise ValueError('Remaining count mismatch')
    known = math.fsum(c['known_proxy_usd'] for c in cells)
    unknown = math.fsum(c['historical_unknown_scenario_usd'] for c in cells)
    budget = json.loads((root/'artifacts/budget.json').read_text())
    return {'utc': datetime.now(timezone.utc).isoformat(), 'diagnosis_sha256': sha(root/DIAGNOSIS),
            'missing_pairs': missing, 'known_proxy_usd': known, 'historical_unknown_scenario_usd': unknown,
            'source_proofs': proofs, 'source_cells_verified': len(sources), 'cells': cells,
            'budget_snapshot': budget, 'budget_cap_usd': 100,
            'scenario_fits_snapshot': budget['spent']+budget['reserved']+known+unknown <= 100,
            'limitations': ['Same-model Run1 B0 family/test mean including observed failure attempts; not exact same-pair cost matching.',
                           'Changed language, unseen split and token lengths are not cost-adjusted; WDC pair sets differ.',
                           'Historical unknown allowance scenario is not a provider charge or forecast of future errors.',
                           'No provider quote, hard bound, complete-study funding claim or admission.',
                           'Historical unjournaled variant outcomes remain unpriced and may be requested again after an admitted continuation.'],
            'new_paid_calls': 0, 'execution_admitted': False, 'script_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build(args.root.resolve())
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('missing_pairs', 'known_proxy_usd', 'historical_unknown_scenario_usd', 'scenario_fits_snapshot')}))
