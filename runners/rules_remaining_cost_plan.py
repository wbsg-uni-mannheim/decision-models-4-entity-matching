"""Read-only Run1 rules/manual-control inventory and historical cost scenarios.

No inference, selection, admission or result export. Rule costs use B0 development
means because no rule responses exist; changed prompt/output lengths are unpriced.
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
from confirmation_remaining_cost_plan import scenario
from diagnose_confirmation_terminal import audit_confirmation_rows

SUFFIXES = ('.jsonl', '.failures.jsonl', '.executions.jsonl', '.operations.jsonl')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def matrix(models, benchmarks):
    return [{'model': m, 'variant': v, 'namespace': ns, 'benchmark': b, 'split': s}
            for m in models for v, ns in (('R_rules', 'rules'), ('X_P_domain_0', 'quality'))
            for b in [*benchmarks, 'billiger-de-50un']
            for s in (['test'] if b == 'billiger-de-50un' else ['prompt', 'test'])]


def identity(c):
    return tuple(c[k] for k in ('namespace', 'model', 'variant', 'benchmark', 'split'))


def validate_matrix(cells, expected):
    keys = [identity(c) for c in cells]
    if len(keys) != len(set(keys)) or set(keys) != {identity(c) for c in expected}:
        raise ValueError('Incomplete or duplicate fixed rule/control matrix')


def partition_missing(cases, cached, confirmation_ids):
    expected = {c['pair_id'] for c in cases}
    present = [r['pair_id'] for r in cached]
    if len(expected) != len(cases) or len(present) != len(set(present)) or not set(present) <= expected:
        raise ValueError('Duplicate or foreign cache pair')
    if not confirmation_ids <= expected:
        raise ValueError('Foreign confirmation overlap')
    missing = expected-set(present)
    return sorted(missing), sorted(missing & confirmation_ids), sorted(missing-confirmation_ids)


def build(root, output):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root/'src'))
    import expanded_rules as er
    import expanded_confirmation as ec
    import analyze
    import protocol
    import orchestrate
    import billiger50
    socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network prohibited'))
    if er.PROJECT != root:
        raise ValueError('Wrong workspace')
    spec = er.verify()
    orchestrate.verify()
    billiger50.verify()
    expected = matrix(protocol.MODELS, protocol.BENCHMARKS)
    validate_matrix(spec['cells'], expected)
    frozen_paths = ['src/expanded_rules.py', 'src/protocol.py', 'src/expanded_protocol.py',
        'src/run.py', 'configs/expanded_rules_manifest.json', 'configs/learned_rules.json',
        'configs/rule_agent_training_manifest.json', 'artifacts/astra_agent/acceptance.json',
        'configs/expanded_confirmation_manifest.json', 'configs/expanded_confirmation_cells.json',
        'configs/expanded_selected_variants.json', 'artifacts/budget.json', 'artifacts/progress.json']
    protected = {p: sha(root/p) for p in frozen_paths}
    # Confirmation uses the same manual request namespace. Its entire historical
    # inventory is retained so overlap is not inferred merely from a variant name.
    diag_path = 'artifacts/operations/confirmation_terminal_diagnosis_20261015T0233Z.json'
    diag = read(root/diag_path)
    protected[diag_path] = sha(root/diag_path)
    if any(sha(root/p) != h for p, h in diag['files_sha256'].items()):
        raise ValueError('Confirmation cache drift requires a new diagnosis')
    selected, selection_hashes = selected_tests(root, ec, analyze)
    selected = set(map(tuple, selected))
    journals, inputs, audits = {}, {}, {}
    def audit(c):
        key = identity(c)
        if key in audits:
            return audits[key]
        ns, m, v, b, split = key
        ip = f'data/{b}/{split}.json'
        cases = read(root/ip)
        inputs[ip] = sha(root/ip)
        path = root/'runs'/ns/m/v/b/(split+'.jsonl')
        for suffix in SUFFIXES:
            p = path.with_name(split+suffix)
            journals[str(p.relative_to(root))] = sha(p) if p.exists() else None
        if path.with_name(split+'.operations.jsonl').exists():
            raise ValueError('New operational history requires updated scoped cost review')
        success, failed = rows(path), rows(path.with_name(split+'.failures.jsonl'))
        info = audit_confirmation_rows(cases, success, failed, er, protocol, m, v, b)
        value = {**c, 'path': str(path.relative_to(root)), **info}
        audits[key] = (value, cases, success)
        return audits[key]
    sources, targets, controls = {}, [], []
    for c in expected:
        record, cases, success = audit(c)
        m, v, b, split = (c[k] for k in ('model', 'variant', 'benchmark', 'split'))
        # All existing B0 controls must be valid and complete; do not purchase them.
        bc = {**c, 'namespace': 'quality', 'variant': 'B0'}
        base, _, _ = audit(bc)
        if not base['complete']:
            raise ValueError('B0 report control incomplete')
        if identity(bc) not in {identity(x) for x in controls}:
            controls.append(base)
        sb = 'billiger-de' if b == 'billiger-de-50un' else b
        sv = 'B0' if v == 'R_rules' else v
        source_key = (m, sv, sb)
        if source_key not in sources:
            source, _, _ = audit({'model': m, 'variant': sv, 'benchmark': sb,
                                  'split': 'prompt', 'namespace': 'quality'})
            if not source['complete']:
                raise ValueError('Development source incomplete')
            sources[source_key] = source
        source = sources[source_key]
        overlap_ids = set()
        if split == 'test' and (m, v, b) in selected:
            if v != er.MANUAL or c['namespace'] != 'quality':
                raise ValueError('Unexpected shared scope')
            for case in cases:
                if protocol.canonical(er.payload(case, m, v)) != protocol.canonical(ec.payload(case, m, v)):
                    raise ValueError('Shared cache requests differ')
            overlap_ids = {case['pair_id'] for case in cases}
        missing, shared, additional = partition_missing(cases, success, overlap_ids)
        targets.append({**record, 'source_path': source['path'],
            'source_variant': sv, 'source_benchmark': sb, 'billiger50_transfer': sb != b,
            'rule_prompt_cost_not_observed': v == 'R_rules',
            'confirmation_missing_overlap': len(shared), 'confirmation_overlap_ids_sha256': protocol.digest(shared),
            'additional_after_confirmation': len(additional), 'additional_ids_sha256': protocol.digest(additional),
            'standalone_scenario': scenario(source, len(missing)),
            'additional_after_confirmation_scenario': scenario(source, len(additional))})
    def current():
        return {p: sha(root/p) if (root/p).exists() else None for p in journals}
    if (current() != journals or any(sha(root/p) != h for p, h in {**protected, **inputs, **selection_hashes,
            **diag['files_sha256']}.items())):
        raise ValueError('Source changed during planning')
    totals = {k: sum(c[k] for c in targets) for k in
              ('expected', 'responses', 'missing', 'confirmation_missing_overlap', 'additional_after_confirmation')}
    totals['complete_cells'] = sum(c['complete'] for c in targets)
    for scope in ('standalone_scenario', 'additional_after_confirmation_scenario'):
        totals[scope] = {k: math.fsum(c[scope][k] for c in targets)
            for k in ('known_proxy_usd', 'historical_unknown_scenario_usd')}
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'rules_remaining_cost_preparation_only',
        'admitted': False, 'new_paid_calls': 0, 'quote_or_hard_bound': False,
        'script_sha256': sha(Path(__file__)), 'protected_sha256': protected, 'input_sha256': inputs,
        'journal_sha256_or_absent': journals, 'confirmation_diagnosis': diag_path,
        'selection_sha256': selection_hashes, 'sources': list(sources.values()), 'cells': targets,
        'b0_controls': controls, 'totals': totals, 'budget_snapshot': read(root/'artifacts/budget.json'),
        'method': 'Fixed full rule/manual matrix. Rule costs use same-model/family B0 D_prompt mean; manual costs use matching full manual D_prompt mean. Includes preserved retries/failures. Multiply by exact missing pairs; billiger50 transfers billiger-de.',
        'deduplication': 'Existing valid cache pairs excluded. B0 report controls read-only, no new calls. Standalone includes manual pairs also missing in selected confirmation; additional-after-confirmation removes only exact shared requests. These two scenarios are alternatives, never additive. No future cache completion is claimed.',
        'limitations': ['Rule prompt/output lengths differ from B0; no observed rule cost, quote, token-length correction or hard bound.',
            'Development costs are historical proxies, not representative future test/error costs.',
            'Unknown allowance scenario is separate from known cost; not actual fees.',
            'Rule induction CLI usage is separate; not inferred as dollars or zero cost.',
            'Historical unjournaled outcomes remain unresolved; missing pairs may already have incurred calls.',
            'Run1 rules/manual only; does not finance the entire study; no test-quality or partial-test-cost selection.']}
    with output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.output)['totals']))

