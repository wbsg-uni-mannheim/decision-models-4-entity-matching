"""Offline, cell-matched historical B0 cost proxy for the two Peeters arms.

This deliberately does not convert a changed prompt into a provider quote.
No inference, admission, reservation, or source-manifest mutation occurs here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from fresh_run_cost_plan import inventory, sha
from peeters_control import ARMS
from peeters_run import EXECUTION, MANIFEST, load_modules, verify_matrix

PROOF = 'artifacts/verification/baseline_only_source_proof_20261014.json'


def project_cells(source_cells, payload_cells):
    sources = {}
    for c in source_cells:
        key = c['model'], c['benchmark'], c['split']
        if key in sources or c['namespace'] != 'quality' or c['variant'] != 'B0':
            raise ValueError('Duplicate or non-B0 source')
        sources[key] = c
    expected = {(*key, arm) for key in sources for arm in ARMS}
    keys = [(c['model'], c['benchmark'], c['split'], c['arm']) for c in payload_cells]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Incomplete, extra or duplicate Peeters cost matrix')
    result = []
    for c, key in zip(payload_cells, keys, strict=True):
        source = sources[key[:3]]
        if c['pairs'] != source['responses']:
            raise ValueError('Source pair count differs from prospective cell')
        result.append({k: c[k] for k in ['model', 'benchmark', 'split', 'arm', 'pairs']} | {
            'ordered_payloads_sha256': c['ordered_payloads_sha256'],
            'b0_cache_sha256': source['cache_sha256'],
            'b0_failure_sha256': source['failure_sha256'],
            'known_cost_proxy_usd_per_run': source['known_usd'],
            'historical_unknown_attempts_per_run': source['unknown_attempts'],
            'historical_attempts_per_run': source['attempts']})
    return result


def totals(cells, runs):
    return {'runs': runs, 'prospective_calls_before_retries': runs*sum(c['pairs'] for c in cells),
            'historical_b0_known_cost_proxy_usd': runs*math.fsum(c['known_cost_proxy_usd_per_run'] for c in cells),
            'separate_historical_unknown_allowance_scenario_usd': runs*.05*sum(c['historical_unknown_attempts_per_run'] for c in cells)}


def build(root):
    execution = json.loads((root/EXECUTION).read_text())
    if execution['payload_manifest_sha256'] != sha(root/MANIFEST):
        raise ValueError('Payload manifest drift')
    for rel, expected in execution['source_sha256'].items():
        if sha(root/rel) != expected:
            raise ValueError('Execution source drift: '+rel)
    _, protocol, orchestrate, billiger50 = load_modules(root)
    orchestrate.verify()
    billiger50.verify()
    payload_cells = verify_matrix(root, root, json.loads((root/MANIFEST).read_text()), protocol)
    sources, proofs = inventory(root, [PROOF])
    cells = project_cells(sources, payload_cells)
    budget_bytes = (root/'artifacts/budget.json').read_bytes()
    budget = json.loads(budget_bytes)
    headroom = 100-budget['spent']-budget['reserved']
    all_runs = totals(cells, 3)
    return {'utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Both fixed Peeters arms, all development and test cells, three fresh study runs',
            'source_proofs': proofs, 'verified_b0_source_cells': len(sources),
            'prospective_cells_per_run': len(cells),
            'execution_manifest_sha256': sha(root/EXECUTION), 'payload_manifest_sha256': sha(root/MANIFEST),
            'per_run': totals(cells, 1), 'all_three_runs': all_runs,
            'by_arm_three_runs': {a: totals([c for c in cells if c['arm'] == a], 3) for a in ARMS},
            'by_split_three_runs': {s: totals([c for c in cells if c['split'] == s], 3) for s in ['prompt', 'route', 'test']},
            'budget_snapshot': budget, 'budget_snapshot_sha256': hashlib.sha256(budget_bytes).hexdigest(),
            'uncommitted_headroom_usd': headroom,
            'historical_known_proxy_excess_over_current_headroom_usd': max(0, all_runs['historical_b0_known_cost_proxy_usd']-headroom),
            'limitations': [
                'Each Peeters cell is mapped to the same model, dataset and split of completed Run1 B0; observed costs include its failure journal.',
                'The two arms receive identical historical cost proxies, not a prediction that their actual costs are equal.',
                'Question and serialization change input length, output usage and potentially error frequency; no token-length or output-cost adjustment is claimed.',
                'Historical unknown attempts times USD0.05 are shown separately, not actual charges or a future error forecast.',
                'This is an incremental planning scenario, not a provider quote, hard bound, full-study funding estimate or admission.',
                'Remaining original and fresh-run work, selective ensemble and Astra follow-ups are additional.'],
            'new_paid_calls': 0, 'execution_admitted': False, 'hard_bound': False,
            'script_sha256': sha(Path(__file__)), 'cells': cells}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build(args.root.resolve())
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ['utc', 'verified_b0_source_cells', 'prospective_cells_per_run', 'per_run', 'all_three_runs', 'uncommitted_headroom_usd']}))
