"""Read-only Run2/3 fixed rules/manual historical cost scenarios.

Actual own-run manual cache only; no hypothetical confirmation reuse, no
cross-run answers, no admission, account binding, inference or credit mutation.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

from fixed_extension_cost_plan import source, key
from fresh_rules_drained import audit, hashes, untouched
from repeated_runs import replica_path, verify
from rules_drained import prepare, sha, unchanged
from rules_remaining_cost_plan import matrix, validate_matrix

FIELDS = ('known_proxy_usd', 'historical_unknown_scenario_usd')
POLICY = 'configs/parallel_grouped_core3_second_20261015.json'
POLICY_SHA = '6248b6dbc3b832448994a005136dc2e60795b20110b60cb4e50fa460246965f4'


def read(p):
    return json.loads(p.read_text())


def inventory(root, target, run_id, rules, protocol):
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Exact own replica identity required')
    untouched(target, protocol)
    frozen, _, cells, protected = prepare(target, rules, protocol)
    journals = hashes(target, cells)
    checked = audit(target, run_id, frozen, protocol, cells)
    if hashes(target, cells) != journals:
        raise ValueError('Manual cache changed during cost planning')
    unchanged(target, protected)
    untouched(target, protocol)
    return dict(run_id=run_id, cells=checked, protected_sha256=protected,
                journals_sha256_or_absent=journals, b0_controls_replayed=False,
                admission=False)


def price(sources, cells, models, benchmarks):
    validate_matrix(cells, matrix(models, benchmarks))
    indexed = {key(s): s for s in sources}
    expected = {('quality', m, v, b, 'prompt') for m in models
                for v in ('B0', 'X_P_domain_0') for b in benchmarks}
    if len(indexed) != len(sources) or set(indexed) != expected:
        raise ValueError('Complete unique development cost sources required')
    for s in sources:
        if type(s['responses']) is not int or s['responses'] <= 0 or any(
                type(s[k]) not in (int, float) or not math.isfinite(s[k]) or s[k] < 0 for k in FIELDS):
            raise ValueError('Invalid historical source count/cost')
    result = []
    for c in cells:
        if any(type(c[k]) is not int or c[k] < 0 for k in ('expected', 'responses', 'missing')) or c['expected'] != c['responses']+c['missing']:
            raise ValueError('Invalid exact cache inventory')
        if c['variant'] == 'R_rules' and c['responses']:
            raise ValueError('Existing rule answers require separate recovery review')
        b = 'billiger-de' if c['benchmark'] == 'billiger-de-50un' else c['benchmark']
        v = 'B0' if c['variant'] == 'R_rules' else c['variant']
        s = indexed['quality', c['model'], v, b, 'prompt']
        result.append({**c, 'source_key': list(key(s)), 'source_responses': s['responses'],
            'rule_prompt_cost_not_observed': c['variant'] == 'R_rules',
            'billiger50_transfer': b != c['benchmark'],
            **{k: s[k]*c['missing']/s['responses'] for k in FIELDS}})
    def total(rows):
        return {**{k: sum(c[k] for c in rows) for k in ('expected', 'responses', 'missing')},
                **{k: math.fsum(c[k] for c in rows) for k in FIELDS}}
    return dict(cells=result, totals=total(result),
        by_variant={v: total([c for c in result if c['variant'] == v]) for v in ('R_rules', 'X_P_domain_0')},
        by_phase={s: total([c for c in result if c['split'] == s]) for s in ('prompt', 'test')})


def replica_inventory(root, rid):
    target = replica_path(root, rid)
    spec = verify(root, rid)
    sys.path.insert(0, str(target/'src'))
    import ssl
    import protocol
    import expanded_rules as rules
    assert ssl.SSLContext
    with patch.object(socket, 'socket', side_effect=AssertionError('No network')):
        result = inventory(root, target, rid, rules, protocol)
    result['protected_sha256'].update(spec['immutable_sha256'])
    result['protected_sha256']['replica.json'] = sha(target/'replica.json')
    result['frozen_files_verified'] = len(spec['immutable_sha256'])
    return result


def build(root, output):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root/'src'))
    import ssl
    import protocol
    import expanded_rules as rules
    assert ssl.SSLContext
    if sha(root/POLICY) != POLICY_SHA:
        raise ValueError('Active policy changed; review required')
    protected = {POLICY: POLICY_SHA, **read(root/POLICY)['files_sha256']}
    with patch.object(socket, 'socket', side_effect=AssertionError('No network')):
        frozen, _, _, inputs = prepare(root, rules, protocol)
        protected.update(inputs)
        specs = [dict(namespace='quality', model=m, variant=v, benchmark=b, split='prompt')
                 for m in protocol.MODELS for v in ('B0', 'X_P_domain_0') for b in protocol.BENCHMARKS]
        journal_hashes = hashes(root, specs)
        sources = [source(root, c, frozen, protocol) for c in specs]
    replicas = {}
    for rid in (2, 3):
        inv = json.loads(subprocess.check_output([sys.executable, __file__, '--root', str(root), '--inventory-run', str(rid)], text=True))
        inv.update(price(sources, inv['cells'], list(protocol.MODELS), list(protocol.BENCHMARKS)))
        replicas[str(rid)] = inv
        protected.update({f'replications/run{rid:02d}/{p}': h for p, h in inv['protected_sha256'].items()})
        journal_hashes.update({f'replications/run{rid:02d}/{p}': h for p, h in inv['journals_sha256_or_absent'].items()})
    unchanged(root, protected)
    for p, h in journal_hashes.items():
        if (sha(root/p) if (root/p).exists() else None) != h:
            raise ValueError('Cost source or target journal changed: '+p)
    totals = {k: math.fsum(r['totals'][k] for r in replicas.values()) for k in FIELDS}
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
        status='fresh_rules_historical_costs_not_admission', admitted=False, new_paid_calls=0,
        new_credits=0, quote_or_hard_bound=False, sources=sources, replicas=replicas,
        two_runs_total=totals, protected_sha256=protected, journal_sha256_or_absent=journal_hashes,
        script_sha256=sha(Path(__file__)), budget_snapshot=read(root/'artifacts/budget.json'),
        method='Full fixed 238-cell matrix per run, minus only exact own-run raw-validated manual answers. Rules use same-model/family complete Run1 B0 development mean; manual uses matching manual development mean. Retained retries/failures included; billiger50 transfers billiger-de.',
        limitations=[
            'Rule responses do not exist. Changed rule prompt/output lengths are unpriced: B0 is only a historical proxy.',
            'Known costs and separate historical unknown allowance scenarios are not a quote, hard bound or representative future error forecast.',
            'No hypothetical future confirmation cache credit and no cross-run response reuse.',
            'Own complete 119 B0 controls and core/supplement/confirmation/schema prerequisites still required; controls are not purchased or certified complete by this plan.',
            'Current source/cache/cost/budget/scheduler/entire history/all joins and both locks remain required before execution.',
            'Private grouped 8 USD credit/checkpoint512 binding and exact durable reconciliation required by future wrapper.',
            'Active core journals/progress/budget may advance; only named source and manual/rule scopes protected.',
            'Historical unjournaled outcomes remain unresolved. No full-study financing, invoice or zero-duplicate claim.'])
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path); ap.add_argument('--inventory-run', type=int, choices=(2, 3))
    args = ap.parse_args()
    if args.inventory_run:
        print(json.dumps(replica_inventory(args.root.resolve(), args.inventory_run)))
    else:
        if args.output is None: ap.error('--output required')
        result = build(args.root.resolve(), args.output)
        print(json.dumps({k: r['totals'] for k, r in result['replicas'].items()}))
