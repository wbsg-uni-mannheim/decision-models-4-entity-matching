"""Offline historical cost plan for untouched Run2/3 variant stages.

No admission, provider call, budget mutation, selection, or execution wrapper.
The completed Run1 variant observations price the same fixed request matrix;
their predictions are never installed in another run's cache.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

from core_phase_cost_plan import aggregate
from fixed_extension_cost_plan import source, SUFFIXES
from repeated_runs import sha, verify

AUDIT = 'artifacts/operations/variants_complete_audit_366563_20261015.json'
AUDIT_SHA = 'd7b6b2224599daead5b660b31646d7a08ddafd2346d57a0ed2522c26279f991f'
PROOF = 'artifacts/verification/variants_result_verification_366563_20261015.json'
PROOF_SHA = 'f4e79f38d9aa722722d160774b9bf34f1bc855e6008a7fad7f24220e6727ad0e'
POLICY = 'configs/parallel_grouped_core3_second_20261015.json'
POLICY_SHA = '6248b6dbc3b832448994a005136dc2e60795b20110b60cb4e50fa460246965f4'


def read(p):
    return json.loads(p.read_text())


def matrix(models, conditions):
    cells = [dict(stage='variants', namespace='quality', model=m, variant='B0',
                  benchmark=b, split='test') for m in models for b in conditions]
    if not cells or len({(c['model'], c['benchmark']) for c in cells}) != len(cells):
        raise ValueError('Nonempty unique variant matrix required')
    return cells


def untouched(target, cells):
    """Absence is a current planning fact, never a future execution permit."""
    history = target/'artifacts/stage_execution.jsonl'
    if history.exists() and any(json.loads(s).get('stage') == 'variants'
                                for s in history.read_text().splitlines()):
        raise ValueError('Variants already attempted; separate terminal review required')
    if ((target/'artifacts/expanded_variants_complete.json').exists()
            or (target/'artifacts/results/expanded_variants').exists()):
        raise ValueError('Existing variant output requires separate review')
    absent = []
    for c in cells:
        directory = target/f"runs/quality/{c['model']}/B0/{c['benchmark']}"
        if list(directory.glob('*.jsonl')):
            raise ValueError('Fresh variant journals must be absent')
        absent.extend(str(directory/('test'+s)) for s in SUFFIXES)
    return absent


def build(root, output):
    root = root.resolve()
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root/'src'))
    import protocol
    import expanded_variants as variants
    if variants.PROJECT != root:
        raise ValueError('Wrong source workspace')
    protected = {AUDIT: AUDIT_SHA, PROOF: PROOF_SHA, POLICY: POLICY_SHA}
    if any(sha(root/p) != h for p, h in protected.items()):
        raise ValueError('Reviewed source proof or policy changed')
    audit, proof = read(root/AUDIT), read(root/PROOF)
    if (proof['diagnosis_sha256'] != AUDIT_SHA or audit['job'] != '366563'
            or audit['totals']['missing'] or audit['complete_cells'] != 35):
        raise ValueError('Completed Run1 variant proof required')
    protected.update(read(root/POLICY)['files_sha256'])
    protected.update(audit['files_sha256'])
    protected.update(audit['protected_sha256'])
    # Protect only frozen and completed scopes. Active core journals/progress
    # and the shared ledger are intentionally allowed to advance concurrently.
    manifest = variants.verify()
    specs = {n: verify(root, n) for n in (2, 3)}
    cells = matrix(protocol.MODELS, variants.VARIANTS)
    if set(manifest['conditions']) != set(variants.VARIANTS) or len(cells) != 35:
        raise ValueError('Frozen variant matrix changed')
    for n, spec in specs.items():
        protected.update({f'replications/run{n:02d}/{p}': h
                          for p, h in spec['immutable_sha256'].items()})
        rel = f'replications/run{n:02d}/replica.json'
        protected[rel] = sha(root/rel)
    for b in variants.VARIANTS:
        p = f'data/{b}/test.json'
        protected[p] = sha(root/p)
    absent_sources = list(audit['absent_journals'])
    absent_targets = {str(n): untouched(root/f'replications/run{n:02d}', cells) for n in (2, 3)}
    def stable():
        for p, h in protected.items():
            if sha(root/p) != h:
                raise ValueError('Protected source drift: '+p)
        if any((root/p).exists() for p in absent_sources):
            raise ValueError('Historical source journal appeared')
        for n in (2, 3):
            if untouched(root/f'replications/run{n:02d}', cells) != absent_targets[str(n)]:
                raise ValueError('Prospective scope changed')
    stable()
    with patch.object(socket, 'socket', side_effect=AssertionError('No network in cost planner')):
        sources = [source(root, c, protocol, protocol) for c in cells]
    historical = {(c['model'], c['benchmark']): c for c in audit['cells']}
    for c in sources:
        old = historical[c['model'], c['benchmark']]
        if any(c['audit'][k] != old[k] for k in ('responses', 'attempts', 'failure_rows', 'unknown_attempts')):
            raise ValueError('Current raw source differs from completed audit')
        if abs(c['known_proxy_usd']-old['known_usd']) > 1e-10:
            raise ValueError('Historical source costs differ')
    responses = sum(c['responses'] for c in sources)
    if responses != manifest['new_decisions_before_cache']:
        raise ValueError('Manifest decision count differs')
    stable()
    total = aggregate(sources)
    result = {'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'historical_variant_scenario_not_admission', 'sources': sources,
        'cells_per_run': len(cells), 'responses_per_run': responses,
        'per_run_total': total, 'two_runs_total': {k: 2*v for k, v in total.items()},
        'protected_sha256': protected, 'absent_source_journals': absent_sources,
        'absent_target_journals': {n: [str(Path(p).relative_to(root)) for p in paths]
                                   for n, paths in absent_targets.items()},
        'frozen_files_verified': {str(n): len(s['immutable_sha256']) for n, s in specs.items()},
        'budget_snapshot': read(root/'artifacts/budget.json'), 'script_sha256': sha(Path(__file__)),
        'new_paid_calls': 0, 'execution_admitted': False, 'quote_or_hard_bound': False,
        'limitations': [
            'Same fixed Run1 variant requests, including retained retries/failures; no cross-run answer reuse.',
            'Unknown allowance scenario is separate from known costs, not a fee or future error forecast.',
            'Core controls are required in each own run; this plan prices only 35 additional variant cells.',
            'Core completion, full history adapter, grouped credit binding and bounded wrapper remain required.',
            'Actual scheduler, scopes, locks, sources, missing pairs, costs, budget and all joins need fresh admission.',
            'Historical unjournaled outcomes remain unresolved; no full-billing or zero-duplicate claim.',
            'No full-study financing guarantee or additional budget authorization.']}
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    value = build(args.root, args.output)
    print(json.dumps({k: value[k] for k in ('cells_per_run', 'responses_per_run', 'per_run_total', 'two_runs_total')}))
