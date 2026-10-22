"""One separately admitted confirmation continuation; no submission or retries.

The old supervisor journal is appended to, never reset. Frozen report functions
are bound privately to the original expanded payload. A complete raw-response
audit precedes the report and completion marker.
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
import types

from confirmation_drained import run_selected_tests, selected_tests
from diagnose_confirmation_terminal import audit_confirmation_rows
from parallel_admission import review
from resume_schema_once import terminal_steps

ATTEMPT = 'artifacts/operations/confirmation_drained_recovery_v1'
PRIOR = 'artifacts/operations/confirmation_recovery_365636.jsonl'
REQUIRED = {'src/run.py', 'src/protocol.py', 'src/analyze.py', 'src/expanded_confirmation.py',
    'src/expanded_protocol.py', 'src/orchestrate.py', 'src/billiger50.py',
    'configs/expanded_confirmation_manifest.json', 'configs/expanded_selected_variants.json',
    'configs/expanded_confirmation_cells.json', 'artifacts/operations/resume_confirmation_drained.py',
    'artifacts/operations/confirmation_drained.py', 'artifacts/operations/drained_inference.py',
    'artifacts/operations/operational_journal.py', 'artifacts/operations/diagnose_confirmation_terminal.py',
    'artifacts/operations/diagnose_fresh_baseline.py', 'artifacts/operations/parallel_admission.py',
    'artifacts/operations/resume_schema_once.py'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate(spec, own, budget):
    expected = {'admitted': True, 'run_id': 1, 'stage': 'confirmation', 'predecessor': '365636',
        'slurm_job_id': own, 'budget_cap_usd': 100, 'max_invocations': 1,
        'historical_unknown_outcomes_disclosed': True, 'unobserved_worker_causes_disclosed': True}
    if not own or any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual reviewed confirmation admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Incomplete executable source provenance')
    if any(type(budget.get(k)) not in (int, float) or not math.isfinite(budget[k]) or budget[k] < -1e-8
           for k in ('spent', 'reserved')) or budget['spent']+budget['reserved']+.05 > 100:
        raise ValueError('Shared budget unavailable')


def inventory(root, diagnosis):
    files = {}
    for c in diagnosis['cells']:
        folder = root/'runs/quality'/c['model']/c['variant']/c['benchmark']
        for p in folder.glob(c['split']+'.*.jsonl'):
            files[str(p.relative_to(root))] = sha(p)
        p = folder/(c['split']+'.jsonl')
        if p.exists():
            files[str(p.relative_to(root))] = sha(p)
    return files


def verify_ready(root, spec, own, cells):
    budget = read(root/'artifacts/budget.json')
    validate(spec, own, budget)
    for rel, expected in spec['source_sha256'].items():
        if sha(root/rel) != expected:
            raise ValueError('Source drift: '+rel)
    dp = root/spec['diagnosis']
    if sha(dp) != spec['diagnosis_sha256']:
        raise ValueError('Diagnosis changed')
    diagnosis = read(dp)
    if diagnosis['job'] != '365636' or diagnosis['status'] != 'terminal_partial_confirmation_audited_no_recovery':
        raise ValueError('Wrong terminal diagnosis')
    for rel, expected in {**diagnosis['source_sha256'], **diagnosis['frozen_artifact_sha256']}.items():
        if sha(root/rel) != expected:
            raise ValueError('Terminal source or supervisor history changed')
    if inventory(root, diagnosis) != diagnosis['files_sha256']:
        raise ValueError('Audited response inventory changed')
    diagnosed_tests = {(c['model'], c['variant'], c['benchmark']) for c in diagnosis['cells'] if c['split'] == 'test'}
    if diagnosed_tests != set(cells):
        raise ValueError('Diagnosed test scope differs from frozen selection')
    controls = {str(p.relative_to(root)): sha(p) for m, b in {(m, b) for m, _, b in cells}
                for p in (root/'runs/quality'/m/'B0'/b).glob('test*.jsonl')}
    if controls != spec['control_files_sha256']:
        raise ValueError('B0 report controls changed')
    cost_path = root/spec['cost_plan']
    if sha(cost_path) != spec['cost_plan_sha256']:
        raise ValueError('Cost plan changed')
    cost = read(cost_path)
    missing = sum(c['missing'] for c in diagnosis['cells'] if c['split'] == 'test')
    if (cost.get('diagnosis_sha256') != spec['diagnosis_sha256'] or cost.get('missing_pairs') != missing
            or cost.get('quote_or_hard_bound') is not False
            or sha(root/spec['cost_source_verification']) != cost.get('source_verification_sha256')
            or any(type(cost.get(k)) not in (int, float) or not math.isfinite(cost[k]) or cost[k] < 0
                   for k in ('known_proxy_usd', 'historical_unknown_scenario_usd'))):
        raise ValueError('Incomplete remaining-cost evidence')
    if budget['spent']+budget['reserved']+cost['known_proxy_usd']+cost['historical_unknown_scenario_usd'] > 100:
        raise ValueError('Remaining-cost scenario exceeds current headroom')
    accounting = subprocess.check_output(['sacct', '-n', '-j', '365636', '--format=JobID,State,ExitCode,End', '-P'],
        text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(accounting, '365636')
    policy = review(root, root/spec['parallel_policy'], own)
    if policy['policy_sha256'] != spec['parallel_policy_sha256'] or not any(j.get('self') for j in policy['reviewed_jobs']):
        raise ValueError('Current policy does not admit this job')
    config = read(root/'configs/expanded_execution.json')
    if str(config['e2_confirmation_job']) != own:
        raise ValueError('Not configured confirmation successor')
    joins = {str(config['e2_rules_job']), str(config['statistics_full_job'])}
    protected_cells = {('quality', c['model'], c['variant'], c['benchmark'], c['split']) for c in diagnosis['cells']}
    protected_cells |= {('quality', m, 'B0', b, 'test') for m, _, b in cells}
    dependencies = {}
    for job in policy['reviewed_jobs']:
        if job.get('scope') == 'different project':
            continue
        jid = job['id']
        meta = subprocess.check_output(['scontrol', 'show', 'job', jid, '-o'], text=True)
        dep = next((x.split('=', 1)[1] for x in meta.split() if x.startswith('Dependency=')), '')
        dependencies[jid] = dep
        ids = re.findall(r'\d+', dep)
        if '365636' in ids or (jid in joins and (own not in ids or 'afterok:' not in dep)):
            raise ValueError('Dependency joins not updated')
        if not job.get('self') and jid not in joins:
            # Scope records are part of the reviewed admission. Policy review
            # binds their job IDs to exact batch scripts and source hashes.
            scope = spec['reviewed_job_write_cells'].get(jid)
            if scope is None or any(tuple(c) in protected_cells for c in scope):
                raise ValueError('Missing or overlapping writer scope')
    if not joins <= dependencies.keys() or dependencies != spec['reviewed_dependencies']:
        raise ValueError('Incomplete current dependency review')
    return {'budget': budget, 'policy': policy, 'accounting_utc': accounting,
        'dependencies': dependencies, 'diagnosis_sha256': spec['diagnosis_sha256'],
        'prior_journal_sha256': sha(root/PRIOR), 'cost_plan_sha256': spec['cost_plan_sha256']}


def finish_report(root, confirmation, analysis, protocol, cells):
    selected = read(root/'configs/expanded_selected_variants.json')
    all_cells = set(cells) | {(m, 'B0', b) for m, _, b in cells}
    for m, v, b in sorted(all_cells):
        folder = root/'runs/quality'/m/v/b
        successes = [json.loads(x) for x in (folder/'test.jsonl').read_text().splitlines()]
        fail = folder/'test.failures.jsonl'
        failures = [json.loads(x) for x in fail.read_text().splitlines()] if fail.exists() else []
        audit = audit_confirmation_rows(read(root/'data'/b/'test.json'), successes, failures,
                                        confirmation, protocol, m, v, b)
        if not audit['complete']:
            raise ValueError('Report requires complete selected and B0 cells')
    bound = types.SimpleNamespace(**vars(analysis))
    bound.load = types.FunctionType(analysis.load.__code__,
        {**analysis.load.__globals__, 'PROJECT': root, 'payload': confirmation.payload},
        argdefs=analysis.load.__defaults__)
    report = types.FunctionType(confirmation.report.__code__,
        {**confirmation.report.__globals__, 'PROJECT': root, 'analyze': bound},
        argdefs=confirmation.report.__defaults__)
    out = root/'artifacts/results/expanded_confirmation'
    if out.exists() and any(out.iterdir()):
        raise ValueError('Existing report requires separate review')
    out.mkdir(parents=True, exist_ok=True)
    report(selected)
    marker = root/'artifacts/expanded_confirmation_complete.json'
    expected = {'status': 'complete', 'selection_sha256': protocol.digest(selected),
        'unique_test_cells': len(cells), 'selected_regime_rows': len(selected), 'full_expanded_study_complete': False}
    if read(marker) != expected:
        raise ValueError('Completion marker mismatch')
    return {str(p.relative_to(root)): sha(p) for p in [marker, *sorted(out.iterdir())] if p.is_file()}


def execute(root, admission_path):
    root = root.resolve()
    spec = read(admission_path)
    own = os.environ.get('SLURM_JOB_ID')
    validate(spec, own, read(root/'artifacts/budget.json'))
    if any(m in sys.modules for m in ('run', 'protocol', 'expanded_confirmation', 'analyze')):
        raise ValueError('Fresh Python process required')
    sys.path.insert(0, str(root/'src'))
    import expanded_confirmation as ec
    import analyze
    import protocol
    import run
    return _execute_bound(root, admission_path, spec, own, ec, analyze, protocol, run)


def _execute_bound(root, admission_path, spec, own, ec, analyze, protocol, run):
    validate(spec, own, read(root/'artifacts/budget.json'))
    with (root/'artifacts/operations/confirmation_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = root/ATTEMPT
        if attempt.exists() or (root/'artifacts/expanded_confirmation_complete.json').exists():
            raise ValueError('Already attempted/completed; fresh diagnosis required')
        report_dir = root/'artifacts/results/expanded_confirmation'
        if report_dir.exists() and any(report_dir.iterdir()):
            raise ValueError('Existing report requires separate review')
        if run.PROJECT != root:
            raise ValueError('Wrong workspace')
        cells, selection_hashes = selected_tests(root, ec, analyze)
        checked = verify_ready(root, spec, own, cells)
        attempt.mkdir(exist_ok=False)
        def log(event, **fields):
            row = protocol.canonical({'utc': datetime.now(timezone.utc).isoformat(),
                'event': 'drained_recovery_'+event, 'job_id': own, 'predecessor': '365636',
                'admission_sha256': sha(admission_path), 'prior_journal_sha256': checked['prior_journal_sha256'],
                'parallel_policy_sha256': checked['policy']['policy_sha256'], **fields})+'\n'
            for path in (attempt/'execution.jsonl', root/PRIOR):
                with path.open('a') as handle:
                    handle.write(row); handle.flush(); os.fsync(handle.fileno())
        log('start', verification=checked, max_invocations=1, historical_unjournaled_outcomes_resolved=False)
        try:
            result = run_selected_tests(root, ec, analyze, run, protocol, attempt/'models')
            log('inference_end', result=result)
            if any(m['status'] != 'complete' for m in result['models']):
                log('end', status='failed', completion_marker_written=False)
                return 1
            if any(o['unknown_outcome_events'] for o in result['operations']):
                raise ValueError('Unknown operational outcomes require diagnosis')
            if any(sha(root/p) != h for p, h in selection_hashes.items()):
                raise ValueError('Selection changed')
            artifacts = finish_report(root, ec, analyze, protocol, cells)
            log('end', status='complete', report_sha256=artifacts,
                cost_scope='Journaled attempts, reused B0 included; not incremental or complete historical costs')
            return 0
        except BaseException as exc:
            log('end', status='failed', exception_type=type(exc).__name__)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.root, args.admission.resolve()))
