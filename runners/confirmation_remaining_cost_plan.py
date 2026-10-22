"""Read-only remaining confirmation scenarios from matching complete development cells.

Frozen selection alone defines scope. Development means are historical proxies,
not test-token-adjusted quotes, representative future error rates or hard bounds.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

from confirmation_drained import selected_tests
from diagnose_confirmation_terminal import audit_confirmation_rows
from resume_confirmation_drained import inventory

DIAG = 'artifacts/operations/confirmation_terminal_diagnosis_20261015T0233Z.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def scenario(source, missing):
    n = source['responses']
    if type(n) is not int or n <= 0 or type(missing) is not int or missing < 0:
        raise ValueError('Invalid source or missing count')
    known, unknown = source['known_usd'], source['unknown_attempts']
    if (type(known) not in (int, float) or not math.isfinite(known) or known < 0
            or type(unknown) is not int or unknown < 0):
        raise ValueError('Invalid source costs')
    return {'known_proxy_usd': known*missing/n,
            'historical_unknown_scenario_usd': unknown*.05*missing/n}


def build(root, output):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root/'src'))
    import expanded_confirmation as ec
    import analyze
    import protocol
    import billiger50
    def denied(*args, **kwargs):
        raise AssertionError('Network prohibited in cost planning')
    socket.socket = denied
    diagnosis = read(root/DIAG)
    if diagnosis['job'] != '365636' or diagnosis['status'] != 'terminal_partial_confirmation_audited_no_recovery':
        raise ValueError('Wrong diagnosis')
    frozen = {**diagnosis['source_sha256'], **diagnosis['frozen_artifact_sha256']}
    protected = {p: sha(root/p) for p in ('artifacts/budget.json', 'artifacts/progress.json')}
    if any(sha(root/p) != h for p, h in frozen.items()) or inventory(root, diagnosis) != diagnosis['files_sha256']:
        raise ValueError('Terminal source/cache inventory drift')
    cells, selection_hashes = selected_tests(root, ec, analyze)
    billiger50.verify()
    diagnosed = {(c['model'], c['variant'], c['benchmark']): c for c in diagnosis['cells'] if c['split'] == 'test'}
    if set(cells) != set(diagnosed) or len(cells) != len(diagnosed):
        raise ValueError('Incomplete selected scope')
    sources, targets, input_hashes = {}, [], {}
    for m, v, b in cells:
        c = diagnosed[m, v, b]
        ip = f'data/{b}/test.json'
        cases = read(root/ip)
        input_hashes[ip] = sha(root/ip)
        path = root/'runs/quality'/m/v/b/'test.jsonl'
        current = rows(path)
        ids = [r['pair_id'] for r in current]
        expected = {x['pair_id'] for x in cases}
        missing = sorted(expected-set(ids))
        if (len(expected) != len(cases) or len(ids) != len(set(ids)) or not set(ids) <= expected
                or len(current) != c['responses'] or len(missing) != c['missing']
                or protocol.digest(missing) != c['missing_pair_ids_sha256']):
            raise ValueError('Current missing-pair inventory differs from diagnosis')
        sb = 'billiger-de' if b == 'billiger-de-50un' else b
        key = (m, v, sb)
        if key not in sources:
            rel = f'runs/quality/{m}/{v}/{sb}/prompt.jsonl'
            fp = (root/rel).with_name('prompt.failures.jsonl')
            source_cases = read(root/'data'/sb/'prompt.json')
            input_hashes[f'data/{sb}/prompt.json'] = sha(root/'data'/sb/'prompt.json')
            audit = audit_confirmation_rows(source_cases, rows(root/rel), rows(fp), ec, protocol, m, v, sb)
            if not audit['complete']:
                raise ValueError('Incomplete development cost source')
            sources[key] = {'model': m, 'variant': v, 'benchmark': sb, 'path': rel, **audit,
                'cache_sha256': sha(root/rel), 'failure_sha256': sha(fp) if fp.exists() else None}
        source = sources[key]
        targets.append({'model': m, 'variant': v, 'benchmark': b, 'missing': len(missing),
            'missing_pair_ids_sha256': protocol.digest(missing), 'source_path': source['path'],
            'source_benchmark': sb, 'transfer': sb != b, **scenario(source, len(missing))})
    if (inventory(root, diagnosis) != diagnosis['files_sha256']
            or any(sha(root/p) != h for p, h in {**frozen, **protected, **input_hashes}.items())):
        raise ValueError('Source changed during planning')
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'historical_remaining_scenario_not_admission',
        'diagnosis': DIAG, 'diagnosis_sha256': sha(root/DIAG), 'script_sha256': sha(Path(__file__)),
        'source_sha256': frozen, 'input_sha256': input_hashes, 'protected_sha256': protected,
        'selection_sha256': selection_hashes, 'sources': list(sources.values()), 'cells': targets,
        'missing_pairs': sum(c['missing'] for c in targets), 'test_cells': len(targets),
        'incomplete_test_cells': sum(c['missing'] > 0 for c in targets),
        **{k: math.fsum(c[k] for c in targets) for k in ('known_proxy_usd', 'historical_unknown_scenario_usd')},
        'quote_or_hard_bound': False, 'admitted': False, 'new_paid_calls': 0,
        'historical_unjournaled_outcomes_resolved': False, 'budget_snapshot': read(root/'artifacts/budget.json'),
        'method': 'Matching model/variant/family complete D_prompt cost including preserved retries/failures, divided by valid source responses, multiplied by exact missing selected test pairs. Billiger50 transfers billiger-de source.',
        'limitations': ['Not same-pair costs or test-token-length adjustment; development and test lengths/output costs can differ.',
            'Unknown scenario is historical unknown attempts per source response times .05 USD; not fees or future retry forecast.',
            'Partially observed test costs do not select the source, quality does not select a cost scenario.',
            'Historical unjournaled outcomes remain unresolved; a missing pair may already have incurred a call.',
            'Remaining confirmation only; no other stages, incremental past total or full bill. No admission.']}
    with output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(args.root.resolve(), args.output)
    print(json.dumps({k: result[k] for k in ('test_cells', 'incomplete_test_cells', 'missing_pairs', 'known_proxy_usd', 'historical_unknown_scenario_usd')}))
