"""Offline per-stage historical costs for four untouched fresh-run extensions.

These stages have fixed request matrices. Reused core/screen controls are not
charged again; separate stability repeat namespaces remain distinct requests.
No execution, selection, admission, quote or full-study financing claim.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys

from core_phase_cost_plan import aggregate, costs
from diagnose_confirmation_terminal import audit_confirmation_rows
from operational_journal import inspect_operations
from repeated_runs import sha, verify

STAGES = ('supplement', 'screen', 'calibration', 'stability')
SUFFIXES = ('.jsonl', '.failures.jsonl', '.executions.jsonl', '.operations.jsonl')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def key(c):
    return tuple(c[k] for k in ('namespace', 'model', 'variant', 'benchmark', 'split'))


def matrix(models, screen, calibration, stability):
    result = []
    def add(stage, ns, m, v, b, s):
        result.append(dict(stage=stage, namespace=ns, model=m, variant=v, benchmark=b, split=s))
    for m in models:
        add('supplement', 'quality', m, 'B0', 'billiger-de-50un', 'test')
    for m in screen['models']:
        for b in screen['benchmarks']:
            for v in screen['variants']:
                add('screen', 'quality', m, v, b, 'expanded_screen')
    for m in calibration['models']:
        for b, info in calibration['benchmarks'].items():
            if info['extra_n']:
                add('calibration', 'quality', m, 'B0', b, 'calibration_extra')
    for m in stability['models']:
        for b in stability['benchmarks']:
            for v in ('I3_identity_json', 'I3_domain_json'):
                add('stability', 'quality', m, v, b, 'expanded_screen')
            for ns in stability['repeats']['fresh_namespaces']:
                add('stability', ns, m, 'B0', b, 'expanded_repeat')
    if len({key(c) for c in result}) != len(result):
        raise ValueError('Overlapping incremental stage requests')
    return result


def source(root, c, payload_module, protocol):
    ns, m, v, b, split = key(c)
    cases = read(root/f'data/{b}/{split}.json')
    base = root/'runs'/ns/m/v/b
    successes, failures = rows(base/(split+'.jsonl')), rows(base/(split+'.failures.jsonl'))
    audit = audit_confirmation_rows(cases, successes, failures, payload_module, protocol, m, v, b)
    if not audit['complete'] or not cases:
        raise ValueError('Complete nonempty historical source required')
    from types import SimpleNamespace
    bound = SimpleNamespace(**{**vars(protocol), 'payload': payload_module.payload})
    operations = inspect_operations(base/(split+'.operations.jsonl'), cases, failures, bound, m, v, 1)
    if operations['unknown_outcome_events']:
        raise ValueError('Unresolved source operational outcome')
    amounts = costs([a for r in successes+failures for a in r['attempts']])
    return {**c, 'responses': len(successes), 'attempts': audit['attempts'], **amounts,
            'pair_ids_sha256': protocol.digest(sorted(x['pair_id'] for x in cases)),
            'audit': audit, 'operations': operations}


def build(root, output):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root/'src'))
    import protocol
    import expanded_protocol
    import expanded_screen
    import expanded_calibration
    import expanded_stability
    import billiger50
    import orchestrate
    def denied(*a, **kw):
        raise AssertionError('Network prohibited in cost planning')
    socket.socket = denied
    socket.create_connection = denied
    for module in (expanded_screen, expanded_calibration, expanded_stability, billiger50, orchestrate):
        if module.PROJECT != root:
            raise ValueError('Wrong source workspace')
        module.verify()
    specs = {n: verify(root, n) for n in (2, 3)}
    manifests = {s: read(root/p) for s, p in {
        'screen': 'configs/expanded_screen_manifest.json', 'calibration': 'configs/calibration_manifest.json',
        'stability': 'configs/expanded_stability_manifest.json'}.items()}
    cells = matrix(protocol.MODELS, **manifests)
    expected_counts = {'supplement': 31059, 'screen': manifests['screen']['requests'],
                       'calibration': manifests['calibration']['new_decisions'],
                       'stability': manifests['stability']['new_decisions_before_cache']}
    protected = {'artifacts/budget.json', 'artifacts/progress.json',
                 'artifacts/operations/approval_runs23_aaron_20261014.json',
                 'configs/expanded_screen_manifest.json', 'configs/calibration_manifest.json',
                 'configs/expanded_stability_manifest.json'}
    for n, spec in specs.items():
        protected |= {f'replications/run{n:02d}/{p}' for p in spec['immutable_sha256']}
        protected |= {p for p in spec['immutable_sha256'] if p.startswith(('src/', 'configs/'))}
        protected |= {f'replications/run{n:02d}/replica.json',
                      f'replications/run{n:02d}/artifacts/stage_execution.jsonl',
                      f'replications/run{n:02d}/artifacts/progress.json'}
    protected |= {f'data/{c["benchmark"]}/{c["split"]}.json' for c in cells}
    snapshot = {p: sha(root/p) for p in protected}
    source_journals = {}
    for c in cells:
        ns, m, v, b, split = key(c)
        for suffix in SUFFIXES:
            rel = f'runs/{ns}/{m}/{v}/{b}/{split}{suffix}'
            source_journals[rel] = sha(root/rel) if (root/rel).exists() else None
    replicas = {}
    for n in (2, 3):
        target = root/f'replications/run{n:02d}'
        history = rows(target/'artifacts/stage_execution.jsonl')
        if any(r.get('stage') in STAGES for r in history):
            raise ValueError('Fixed extension already attempted; scoped diagnosis required')
        # All existing fresh journals, not only prospective extension paths,
        # stay hash-bound to preserve the incomplete core and its provenance.
        existing = {str(p.relative_to(root)): sha(p) for p in (target/'runs').rglob('*.jsonl')}
        absent = []
        for c in cells:
            ns, m, v, b, split = key(c)
            for suffix in SUFFIXES:
                p = target/f'runs/{ns}/{m}/{v}/{b}/{split}{suffix}'
                if p.exists():
                    raise ValueError('Prospective extension has cache; need missing-pair plan')
                absent.append(str(p.relative_to(root)))
        replicas[str(n)] = {'run_id': n, 'stage_history': history, 'existing_journal_sha256': existing,
                            'expected_absent_journals': absent, 'frozen_files_verified': len(specs[n]['immutable_sha256'])}
    sources = [source(root, c, expanded_stability if c['stage'] == 'stability' else expanded_protocol, protocol)
               for c in cells]
    stage_totals = {}
    for stage in STAGES:
        selected = [c for c in sources if c['stage'] == stage]
        count = sum(c['responses'] for c in selected)
        if count != expected_counts[stage]:
            raise ValueError('Manifest request count differs from fixed stage matrix')
        stage_totals[stage] = {'cells': len(selected), 'prospective_responses_per_run': count,
            'source_attempts': sum(c['attempts'] for c in selected), **aggregate(selected)}
    total = aggregate(sources)
    for rel, expected in {**snapshot, **source_journals,
                          **{p:h for r in replicas.values() for p,h in r['existing_journal_sha256'].items()}}.items():
        if (sha(root/rel) if (root/rel).exists() else None) != expected:
            raise ValueError('Protected input/source/replica changed during planning')
    for r in replicas.values():
        if any((root/p).exists() for p in r['expected_absent_journals']):
            raise ValueError('Fresh extension journal appeared during planning')
        current = {str(p.relative_to(root)): sha(p)
                   for p in (root/f"replications/run{r['run_id']:02d}/runs").rglob('*.jsonl')}
        if current != r['existing_journal_sha256']:
            raise ValueError('Replica journal inventory changed during planning')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'fixed_extension_historical_scenarios_not_admission',
        'stages': stage_totals, 'sources': sources, 'per_run_total': total,
        'two_runs_total': {k: 2*v for k, v in total.items()}, 'replicas': replicas,
        'protected_sha256': snapshot, 'source_journal_sha256_or_absent': source_journals,
        'script_sha256': sha(Path(__file__)), 'quote_or_hard_bound': False,
        'new_paid_calls': 0, 'execution_admitted': False, 'budget_snapshot': read(root/'artifacts/budget.json'),
        'limitations': ['Four fixed incremental stages only, not all remaining work or full-study financing.',
            'Historical Run1 costs including preserved retries, not a quote, token bound or future error forecast.',
            'Unknown allowances stay separate from known costs and are not provider fees.',
            'Same fixed cases/requests are source observations, never reused responses across fresh runs.',
            'Core/route/test and screen controls reused within each fresh run are excluded from incremental costs.',
            'Stability repeat namespaces are distinct paid requests; no collapsing by pair ID.',
            'Excluded: remaining core and Run1, confirmation, schema, rules, variants, live, Peeters, ensemble, Astra.',
            'Historical unjournaled outcomes remain unresolved; no full-billing or zero-duplicate guarantee.',
            'Actual prerequisites, drained stage integration, scheduler/policy/scopes/joins and budget admission still required.']}
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    value = build(args.root.resolve(), args.output)
    print(json.dumps({'stages': value['stages'], 'two_runs_total': value['two_runs_total']}))
