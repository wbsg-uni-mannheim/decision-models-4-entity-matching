"""One separately admitted Run1 variant continuation, with both drain layers.

No submission or admission generation. Requires actual Slurm identity, terminal
predecessor, exact audited cache inventory, reviewed jobs/joins and source hashes.
Existing stages, selection and frozen source files are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from drained_inference import build, drain_model_tasks
from operational_journal import inspect_operations
from parallel_admission import review
from resume_schema_once import terminal_steps

REQUIRED = {'src/run.py', 'src/protocol.py', 'src/expanded_variants.py', 'src/analyze.py',
            'src/orchestrate.py', 'src/prepare.py', 'configs/expanded_variant_manifest.json',
            'artifacts/operations/resume_variants_drained.py', 'artifacts/operations/drained_inference.py',
            'artifacts/operations/operational_journal.py', 'artifacts/operations/parallel_admission.py',
            'artifacts/operations/resume_schema_once.py'}
JOURNAL = 'artifacts/operations/variants_drained_recovery_v1.jsonl'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(spec, own, budget):
    expected = {'admitted': True, 'run_id': 1, 'stage': 'variants', 'predecessor': '365279',
                'slurm_job_id': own, 'budget_cap_usd': 100, 'max_invocations': 1,
                'historical_unknown_outcomes_disclosed': True,
                'unobserved_worker_causes': ['kev', 'luna', 'pplx']}
    if not own or any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual reviewed variant recovery admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Incomplete executable source provenance')
    if any(type(budget.get(k)) not in (int, float) or not math.isfinite(budget[k]) or budget[k] < -1e-8
           for k in ('spent', 'reserved')) or budget['spent']+budget['reserved']+.05 > 100:
        raise ValueError('Shared budget unavailable')


def inventory(root, models, benchmarks):
    return {str(p.relative_to(root)): sha(p) for model in models for b in benchmarks
            for p in sorted((root/'runs/quality'/model/'B0'/b).glob('*.jsonl'))}


def verify_ready(root, spec, own, models, benchmarks):
    budget = json.loads((root/'artifacts/budget.json').read_text())
    validate(spec, own, budget)
    for rel, expected in spec['source_sha256'].items():
        if sha(root/rel) != expected:
            raise ValueError('Source drift: '+rel)
    diagnosis_path = root/spec['diagnosis']
    if sha(diagnosis_path) != spec['diagnosis_sha256']:
        raise ValueError('Diagnosis changed')
    diagnosis = json.loads(diagnosis_path.read_text())
    if diagnosis['job'] != spec['predecessor'] or diagnosis['status'] != 'terminal_scoped_audit_no_recovery':
        raise ValueError('Wrong terminal diagnosis')
    if inventory(root, models, benchmarks) != diagnosis['files_sha256']:
        raise ValueError('Audited response inventory changed')
    cost_path = root/spec['cost_plan']
    if sha(cost_path) != spec['cost_plan_sha256']:
        raise ValueError('Cost plan changed')
    cost = json.loads(cost_path.read_text())
    if (cost.get('diagnosis_sha256') != spec['diagnosis_sha256']
            or cost.get('missing_pairs') != diagnosis['totals']['missing']
            or any(type(cost.get(k)) not in (int, float) or not math.isfinite(cost[k]) or cost[k] < 0
                   for k in ('known_proxy_usd', 'historical_unknown_scenario_usd'))):
        raise ValueError('Cost plan scope or values invalid')
    if budget['spent']+budget['reserved']+cost['known_proxy_usd']+cost['historical_unknown_scenario_usd'] > 100:
        raise ValueError('Remaining-cost planning scenario exceeds current headroom')
    accounting = subprocess.check_output(['sacct', '-n', '-j', spec['predecessor'],
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(accounting, spec['predecessor'])
    policy = review(root, root/spec['parallel_policy'], own)
    if (policy['policy_sha256'] != spec['parallel_policy_sha256']
            or not any(r.get('self') for r in policy['reviewed_jobs'])):
        raise ValueError('Current scheduler policy does not admit this job')
    config = json.loads((root/'configs/expanded_execution.json').read_text())
    if str(config['e6_variants_job']) != own:
        raise ValueError('Not the configured variant successor')
    stat = str(config['statistics_full_job'])
    meta = subprocess.check_output(['scontrol', 'show', 'job', stat, '-o'], text=True)
    dependency = next((s.split('=', 1)[1] for s in meta.split() if s.startswith('Dependency=')), '')
    ids = re.findall(r'\d+', dependency)
    if own not in ids or spec['predecessor'] in ids or 'afterok:' not in dependency:
        raise ValueError('Statistics dependency not updated to successor')
    # Known independent jobs remain allowed. Reject another variant entry point,
    # even pending, after the common policy has verified every project job.
    for item in policy['reviewed_jobs']:
        if item.get('self') or item.get('scope') == 'different project':
            continue
        script = subprocess.check_output(['scontrol', 'write', 'batch_script', item['id'], '-'], text=True)
        if any(s in script for s in ('expanded_variants.py', 'resume_variants', '--stage variants')):
            raise ValueError('Potential overlapping variant writer')
    return {'budget': budget, 'policy': policy, 'accounting_utc': accounting,
            'cost_plan_sha256': spec['cost_plan_sha256'], 'cost_plan_is_quote_or_hard_bound': False}


def run_once(root, variants, protocol, inference, models_journal):
    _, cell, adapter = build(inference, 1)
    def task(model):
        for benchmark in variants.VARIANTS:
            cell(model, 'B0', benchmark, 'test', workers=8)
    outcomes = drain_model_tasks(task, list(protocol.MODELS), models_journal, protocol.canonical)
    operations = []
    for model in protocol.MODELS:
        for benchmark in variants.VARIANTS:
            directory = root/'runs/quality'/model/'B0'/benchmark
            failure = directory/'test.failures.jsonl'
            rows = [json.loads(x) for x in failure.read_text().splitlines()] if failure.exists() else []
            cases = json.loads((root/'data'/benchmark/'test.json').read_text())
            operations.append({'model': model, 'benchmark': benchmark,
                **inspect_operations(directory/'test.operations.jsonl', cases, rows, protocol, model, 'B0', 1)})
    return outcomes, operations, adapter


def execute(root, admission_path):
    root = root.resolve()
    if any(m in sys.modules for m in ('run', 'protocol', 'expanded_variants', 'analyze')):
        raise ValueError('Fresh Python process required')
    spec = json.loads(admission_path.read_text())
    own = os.environ.get('SLURM_JOB_ID')
    with (root/'artifacts/operations/variants_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/JOURNAL).exists() or (root/'artifacts/expanded_variants_complete.json').exists():
            raise ValueError('Already attempted/completed; fresh terminal review required')
        sys.path.insert(0, str(root/'src'))
        import expanded_variants as variants
        import orchestrate
        import protocol
        import run
        if any(m.PROJECT != root for m in (run, variants, orchestrate)):
            raise ValueError('Wrong workspace')
        variants.verify()
        orchestrate.verify()
        checked = verify_ready(root, spec, own, protocol.MODELS, variants.VARIANTS)
        with (root/JOURNAL).open('x') as journal:
            def log(event, **fields):
                journal.write(protocol.canonical({'utc': datetime.now(timezone.utc).isoformat(),
                    'event': event, 'job_id': own, 'admission_sha256': sha(admission_path), **fields})+'\n')
                journal.flush()
                os.fsync(journal.fileno())
            log('start', invocations_limit=1, verification=checked,
                historical_unknown_outcomes=True, historical_worker_causes_unobserved=['kev', 'luna', 'pplx'])
            try:
                outcomes, operations, adapter = run_once(root, variants, protocol, run,
                    root/'artifacts/operations/variants_drained_models_v1.jsonl')
                log('inference_end', models=outcomes, operations=operations, adapter=adapter)
                if any(r['status'] != 'complete' for r in outcomes):
                    log('end', status='failed', report_created=False)
                    return 1
                if any(o['unknown_outcome_events'] for o in operations):
                    raise ValueError('Unknown operational outcome requires diagnosis before report')
                variants.report()
                log('end', status='complete', report_created=True,
                    cost_scope='Journaled costs only; historical unknown outcomes still unresolved')
                return 0
            except BaseException as exc:
                log('end', status='failed', exception_type=type(exc).__name__)
                raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--admission', type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(execute(args.root, args.admission.resolve()))
