"""Offline planning from completed, verified Run1 cells; never admission or a quote."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

PROOFS = ['artifacts/results/three_runs/run01_verified_base_ap2_screen_v2/verification.json',
          'artifacts/results/three_runs/run01_verified_stability_v2/verification.json']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def inventory(root, proof_paths):
    sources, proofs = {}, []
    for rel in proof_paths:
        path = root/rel
        proof = read(path)
        if proof['run_id'] != 1 or proof['verified_source_cells'] != len(proof['sources']):
            raise ValueError('Complete Run1 source proof required')
        proofs.append({'path': str(rel), 'sha256': sha(path), 'stages': proof['stages']})
        for source in proof['sources']:
            key = tuple(source[k] for k in ['namespace', 'model', 'variant', 'benchmark', 'split'])
            if any(not isinstance(k, str) or Path(k).name != k or k in ('.', '..') for k in key):
                raise ValueError('Invalid source path component')
            if key in sources and source != sources[key]:
                raise ValueError('Conflicting duplicate source proof')
            sources[key] = source
    cells = []
    for key, source in sorted(sources.items()):
        ns, model, variant, benchmark, split = key
        path = root/'runs'/ns/model/variant/benchmark/(split+'.jsonl')
        failures = path.with_name(split+'.failures.jsonl')
        if not source['request_hashes_verified'] or sha(path) != source['cache_sha256']:
            raise ValueError('Unverified or changed cache')
        if (sha(failures) if failures.exists() else None) != source['failure_sha256']:
            raise ValueError('Changed failure journal')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        by_id = {row['pair_id']: row for row in rows}
        if len(by_id) != len(rows) or len(rows) != source['response_count']:
            raise ValueError('Duplicate or incomplete response cells')
        errors = [json.loads(line) for line in failures.read_text().splitlines()] if failures.exists() else []
        for row in errors:
            if row['pair_id'] not in by_id or row['request_hash'] != by_id[row['pair_id']]['request_hash']:
                raise ValueError('Failure outside verified request set')
        attempts = [a for row in rows+errors for a in row['attempts']]
        known = [a['cost'] for a in attempts if a['cost'] is not None]
        if any(type(c) not in (int, float) or not math.isfinite(c) or c < 0 for c in known):
            raise ValueError('Invalid recorded cost')
        cells.append({'namespace': ns, 'model': model, 'variant': variant, 'benchmark': benchmark,
                      'split': split, 'responses': len(rows), 'attempts': len(attempts),
                      'known_usd': math.fsum(known), 'unknown_attempts': sum(a['cost'] is None for a in attempts),
                      'cache_sha256': source['cache_sha256'], 'failure_sha256': source['failure_sha256']})
    return cells, proofs


def plan(root, proof_paths=PROOFS):
    cells, proofs = inventory(root, proof_paths)
    budget_path = root/'artifacts/budget.json'
    # One byte snapshot keeps the live budget numbers and their digest consistent.
    budget_bytes = budget_path.read_bytes()
    budget = json.loads(budget_bytes)
    known = math.fsum(c['known_usd'] for c in cells)
    unknown = sum(c['unknown_attempts'] for c in cells)
    remaining = 100-budget['spent']-budget['reserved']
    return {'utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Completed Run1 source-cell subset only',
            'source_proofs': proofs, 'deduplicated_source_cells': len(cells),
            'responses_per_run_proxy': sum(c['responses'] for c in cells),
            'observed_known_usd': known, 'observed_unknown_attempts': unknown,
            'two_fresh_runs_known_cost_proxy_usd': 2*known,
            'two_fresh_runs_unknown_allowance_scenario_usd': 2*unknown*.05,
            'unknown_allowance_per_attempt_usd': .05,
            'budget_snapshot': budget, 'budget_snapshot_sha256': hashlib.sha256(budget_bytes).hexdigest(),
            'uncommitted_headroom_usd': remaining,
            'known_proxy_excess_over_headroom_usd': max(0, 2*known-remaining),
            'proxy_plus_unknown_allowance_excess_usd': max(0, 2*(known+unknown*.05)-remaining),
            'not_in_this_proxy': ['remaining Run1 work', 'calibration and extra development cells',
                                  'expanded confirmation', 'variants', 'schema', 'rules',
                                  'fresh core timing and live latency calls',
                                  'strong selective ensemble', 'Astra codebook and categorization'],
            'limitations': ['Run1 costs and selected AP2 test cells reused only as planning observations, never as fresh responses',
                            'Fresh runs refit selections; selected test variants and token usage can change',
                            'Historical errors are not a forecast of future error frequency',
                            'Unknown allowances are not actual provider charges',
                            'Not a full-study estimate, lower bound, provider hard cap or spending authorization'],
            'new_paid_calls': 0, 'execution_admitted': False, 'cells': cells}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = plan(args.root.resolve())
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cells'}))
