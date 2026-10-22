"""Bounded replacement for the never-started old rules job, not a submitter.

No automatic recovery or admission. The old pending job must first be separately
retired and its prerequisites actually completed. Shared manual caches remain
locked through the privately bound frozen report. No frozen module is edited.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import rules_drained as component
from parallel_admission import review
from resume_schema_once import terminal_steps
from rules_remaining_cost_plan import identity, validate_matrix

ATTEMPT = 'artifacts/operations/rules_drained_attempt_v1'
STAGE = 'artifacts/operations/rules_drained_stage.jsonl'
LOCKS = ('artifacts/operations/confirmation_recovery.lock', 'artifacts/operations/rules_recovery.lock')
REQUIRED = {'src/run.py', 'src/protocol.py', 'src/expanded_protocol.py', 'src/analyze.py',
    'src/expanded_rules.py', 'src/orchestrate.py', 'src/billiger50.py', 'src/prepare.py',
    'configs/expanded_rules_manifest.json', 'configs/learned_rules.json',
    'configs/rule_agent_training_manifest.json', 'artifacts/astra_agent/acceptance.json',
    'artifacts/operations/run_rules_drained.py', 'artifacts/operations/rules_drained.py',
    'artifacts/operations/drained_inference.py', 'artifacts/operations/operational_journal.py',
    'artifacts/operations/diagnose_confirmation_terminal.py', 'artifacts/operations/diagnose_fresh_baseline.py',
    'artifacts/operations/rules_remaining_cost_plan.py', 'artifacts/operations/confirmation_remaining_cost_plan.py',
    'artifacts/operations/confirmation_drained.py', 'artifacts/operations/parallel_admission.py',
    'artifacts/operations/resume_schema_once.py', 'artifacts/operations/repeated_runs.py',
    'artifacts/operations/resume_confirmation_drained.py',
    'artifacts/verification/check_rules_remaining_cost_20261015.py'}
sha = component.sha


def read(path):
    return json.loads(path.read_text())


def validate(spec, own, budget):
    expected = {'admitted': True, 'run_id': 1, 'stage': 'rules', 'predecessor': '365300',
        'predecessor_never_started': True, 'slurm_job_id': own, 'budget_cap_usd': 100,
        'max_invocations_per_model_per_phase': 1, 'historical_unknown_outcomes_disclosed': True}
    if not own or spec.get('admitted') is not True or any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual reviewed rules admission required')
    if not REQUIRED <= spec.get('source_sha256', {}).keys():
        raise ValueError('Incomplete executable source provenance')
    if any(type(budget.get(k)) not in (float, int) or not math.isfinite(budget[k]) or budget[k] < -1e-8
           for k in ('spent', 'reserved')) or budget['spent']+budget['reserved']+.05 > 100:
        raise ValueError('Shared budget unavailable')


def inventory(root, cells, reviewed_other=None):
    reviewed_other = reviewed_other or {}
    found = {}
    for c in cells:
        folder = root/'runs'/c['namespace']/c['model']/c['variant']/c['benchmark']
        for p in folder.glob(c['split']+'*.jsonl'):
            found[str(p.relative_to(root))] = sha(p)
    # The shared manual namespace also contains completed screening development
    # cells. They are outside this inference matrix and require separate exact
    # historical hashes; this is not permission to write or price those cells.
    manual_dev = {(c['model'], c['benchmark']) for c in cells
                  if c['variant'] == 'X_P_domain_0' and c['split'] == 'prompt'}
    for rel, expected in reviewed_other.items():
        parts = Path(rel).parts
        if (len(parts) != 6 or parts[:2] != ('runs', 'quality') or parts[3] != 'X_P_domain_0'
                or (parts[2], parts[4]) not in manual_dev or parts[5] not in
                ('expanded_screen.jsonl', 'expanded_screen.executions.jsonl',
                 'expanded_screen.failures.jsonl', 'expanded_screen.operations.jsonl')):
            raise ValueError('Other historical journal is not manual screening')
        if sha(root/rel) != expected:
            raise ValueError('Other historical screening journal changed')
    actual = {str(p.relative_to(root)) for p in (root/'runs/rules').rglob('*.jsonl')}
    actual |= {str(p.relative_to(root)) for p in (root/'runs/quality').glob('*/X_P_domain_0/**/*.jsonl')}
    target = {p for p in found if p.startswith('runs/rules/') or '/X_P_domain_0/' in p}
    if actual != target | reviewed_other.keys():
        raise ValueError('Unreviewed rule/manual journal outside fixed scope')
    return found


def completed_prerequisites(root, spec, config):
    recorded = {}
    for stage, key in (('confirmation', 'e2_confirmation_job'), ('schema', 'e2_schema_job')):
        jid = str(config[key])
        entry = spec['prerequisites'][stage]
        if entry['job_id'] != jid:
            raise ValueError('Prerequisite identity changed')
        accounting = subprocess.check_output(['sacct', '-n', '-j', jid,
            '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
        terminal_steps(accounting, jid)
        rows = [line.split('|') for line in accounting.splitlines() if line.strip()]
        if any(row[1] != 'COMPLETED' or row[2] != '0:0' for row in rows):
            raise ValueError('Prerequisite did not complete successfully')
        marker = root/f'artifacts/expanded_{stage}_complete.json'
        if sha(marker) != entry['completion_sha256'] or read(marker).get('status') != 'complete':
            raise ValueError('Prerequisite completion marker changed')
        recorded[stage] = accounting
    return recorded


def verify_ready(root, spec, own, cells, audited):
    budget = read(root/'artifacts/budget.json')
    validate(spec, own, budget)
    for rel, expected in spec['source_sha256'].items():
        if sha(root/rel) != expected:
            raise ValueError('Source drift: '+rel)
    cp, vp, dp = (root/spec[k] for k in ('cost_scenario', 'cost_verification', 'cache_review'))
    for p, key in ((cp, 'cost_scenario_sha256'), (vp, 'cost_verification_sha256'), (dp, 'cache_review_sha256')):
        if sha(p) != spec[key]:
            raise ValueError('Cost or cache review changed')
    cost, proof, diagnosis = read(cp), read(vp), read(dp)
    if (proof.get('status') != 'independent_raw_cost_missing_matrix_deduplication_passed'
            or proof.get('scenario_sha256') != sha(cp)
            or proof.get('script_sha256') != sha(root/'artifacts/verification/check_rules_remaining_cost_20261015.py')
            or cost.get('script_sha256') != sha(root/'artifacts/operations/rules_remaining_cost_plan.py')
            or cost.get('quote_or_hard_bound') is not False
            or diagnosis.get('status') != 'real_read_only_binding_and_raw_cache_verified'
            or diagnosis.get('execution_component_invoked') is not False):
        raise ValueError('Incomplete independent cost/cache evidence')
    validate_matrix(cost['cells'], cells)
    validate_matrix(diagnosis['cells'], cells)
    current = {identity(c): (c['missing'], c['missing_pair_ids_sha256']) for c in audited}
    if any({identity(c): (c['missing'], c['missing_pair_ids_sha256']) for c in source} != current
           for source in (cost['cells'], diagnosis['cells'])):
        raise ValueError('Cost/cache missing pairs differ from current raw audit')
    controls = [dict(c, variant='B0', namespace='quality') for c in cells if c['variant'] == 'R_rules']
    observed = inventory(root, cells+controls, spec.get('reviewed_other_journal_sha256'))
    for rel, expected in spec.get('shared_history_source_sha256', {}).items():
        if sha(root/rel) != expected:
            raise ValueError('Shared screening source changed')
    expected = {p: h for p, h in cost['journal_sha256_or_absent'].items() if h is not None}
    if observed != expected:
        raise ValueError('Response/control inventory changed since cost review')
    protected = {**cost['protected_sha256'], **cost['input_sha256'], **cost['selection_sha256'],
                 **diagnosis['inventory_sha256_or_absent']}
    for rel, expected in protected.items():
        # Other independent paid work may change the ledger; use its actual
        # current values, never demand the old planning-budget snapshot.
        if rel == 'artifacts/budget.json':
            continue
        p = root/rel
        if (sha(p) if p.exists() else None) != expected:
            raise ValueError('Reviewed source/cache/absence drift: '+rel)
    # Always use the ACTUAL current standalone remainder. A hypothetical future
    # confirmation overlap is never deducted at admission time.
    scenario = cost['totals']['standalone_scenario']
    if proof.get('totals') != cost['totals'] or cost['totals']['missing'] != sum(x['missing'] for x in audited):
        raise ValueError('Cost totals differ from verified current remainder')
    amounts = [scenario.get(k) for k in ('known_proxy_usd', 'historical_unknown_scenario_usd')]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in amounts):
        raise ValueError('Invalid cost scenario')
    if budget['spent']+budget['reserved']+sum(amounts) > 100:
        raise ValueError('Remaining-cost scenario exceeds current headroom')
    old = subprocess.check_output(['sacct', '-n', '-j', '365300',
        '--format=JobID,State,ExitCode,Start,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    old_rows = [x.split('|') for x in old.splitlines() if x.strip()]
    if (len(old_rows) != 1 or old_rows[0][0] != '365300' or
            old_rows[0][1].split()[0] != 'CANCELLED' or old_rows[0][3] not in ('Unknown', 'None')
            or old_rows[0][4] in ('Unknown', 'None', '')
            or (root/'logs/rules-365300.log').exists()):
        raise ValueError('Old pending rules job not retired without execution')
    config = read(root/'configs/expanded_execution.json')
    if str(config['e2_rules_job']) != own:
        raise ValueError('Not configured rules successor')
    prereqs = completed_prerequisites(root, spec, config)
    policy = review(root, root/spec['parallel_policy'], own)
    if policy['policy_sha256'] != spec['parallel_policy_sha256'] or not any(j.get('self') for j in policy['reviewed_jobs']):
        raise ValueError('Current policy does not admit this job')
    stats = str(config['statistics_full_job'])
    protected_cells = {identity(c) for c in cells+controls}
    dependencies = {}
    for job in policy['reviewed_jobs']:
        if job.get('scope') == 'different project':
            continue
        jid = job['id']
        meta = subprocess.check_output(['scontrol', 'show', 'job', jid, '-o'], text=True)
        dep = next((x.split('=', 1)[1] for x in meta.split() if x.startswith('Dependency=')), '')
        dependencies[jid] = dep
        clean = re.sub(r'\([^)]*\)', '', dep)
        afterok = {i for group in re.findall(r'(?:^|,)afterok:([^,?]+)', clean) for i in group.split(':')}
        if job.get('self') and 'JobState=RUNNING' not in meta.split():
            raise ValueError('Admitted rules job is not running')
        if jid == '365300' or '365300' in re.findall(r'\d+', dep) or (jid == stats and
                (own not in afterok or '?' in dep or 'JobState=PENDING' not in meta.split())):
            raise ValueError('Dependency joins not updated')
        if jid == stats:
            # Confirmation and schema are already proven complete above. The
            # independent variants writer may still run alongside rules, but
            # statistics must wait for it and cannot retain a failed old join.
            variant_job = str(config['e6_variants_job'])
            accounting = subprocess.check_output(['sacct', '-n', '-j', variant_job,
                '--format=JobID,State,ExitCode,End', '-P'], text=True)
            rows = [x.split('|') for x in accounting.splitlines() if x.strip()]
            if not rows or variant_job not in {r[0] for r in rows}:
                raise ValueError('Missing variants join accounting')
            if all(r[1] == 'COMPLETED' and r[2] == '0:0' for r in rows):
                terminal_steps(accounting, variant_job)
            elif (variant_job not in afterok or any(r[1] == 'COMPLETED' and r[2] != '0:0' for r in rows)
                    or any(r[1].split()[0] not in
                    ('PENDING', 'RUNNING', 'COMPLETING', 'COMPLETED') for r in rows)):
                raise ValueError('Variants statistics join not satisfied or updated')
        if not job.get('self') and jid != stats:
            scope = spec['reviewed_job_write_cells'].get(jid)
            if scope is None or any(tuple(c) in protected_cells for c in scope):
                raise ValueError('Missing or overlapping writer scope')
    if stats not in dependencies or dependencies != spec['reviewed_dependencies']:
        raise ValueError('Incomplete current dependency review')
    return {'budget': budget, 'policy': policy, 'retired_predecessor_accounting_utc': old,
            'prerequisite_accounting_utc': prereqs, 'dependencies': dependencies,
            'cache_review_sha256': sha(dp), 'cost_scenario_sha256': sha(cp),
            'cost_verification_sha256': sha(vp)}


def finish_report(root, rules, analysis, protocol):
    frozen, spec, cells, protected = component.prepare(root, rules, protocol)
    controls = [dict(c, variant='B0', namespace='quality') for c in cells if c['variant'] == 'R_rules']
    component.audit(root, frozen, protocol, cells+controls, require_complete=True)
    load = component.bind(analysis.load, PROJECT=root, payload=frozen.payload)
    def load_cell(m, v, b, split):
        return load(m, v, b, split, namespace=frozen.CONDITIONS.get(v, 'quality'))
    report = component.bind(rules.report, PROJECT=root, load_cell=load_cell)
    out = root/'artifacts/results/expanded_rules'
    if out.exists() and any(out.iterdir()):
        raise ValueError('Existing report requires separate review')
    component.unchanged(root, protected)
    report(spec)
    marker = root/'artifacts/expanded_rules_complete.json'
    expected = {'status': 'complete', 'test_cells_per_condition': 63, 'development_cells_per_condition': 56,
        'evaluated_conditions': ['R_rules', 'X_P_domain_0'], 'controls': ['B0', 'X_P_domain_0'],
        'human_rule_quality_review': 'not performed', 'full_expanded_study_complete': False}
    if read(marker) != expected:
        raise ValueError('Completion marker mismatch')
    component.unchanged(root, protected)
    return {str(p.relative_to(root)): sha(p) for p in [marker, *sorted(out.iterdir())] if p.is_file()}


def verify_data(root):
    import orchestrate
    import billiger50
    if orchestrate.PROJECT != root or billiger50.PROJECT != root:
        raise ValueError('Wrong frozen data workspace')
    orchestrate.verify()
    billiger50.verify()


def _execute_bound(root, admission_path, spec, own, rules, analysis, protocol, inference):
    validate(spec, own, read(root/'artifacts/budget.json'))
    if inference.PROJECT != root:
        raise ValueError('Wrong inference workspace')
    with ExitStack() as stack:
        for rel in LOCKS:
            lock = stack.enter_context((root/rel).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt, stage = root/ATTEMPT, root/STAGE
        if attempt.exists() or stage.exists() or (root/'artifacts/expanded_rules_complete.json').exists():
            raise ValueError('Already attempted/completed; new terminal review required')
        out = root/'artifacts/results/expanded_rules'
        if out.exists() and any(out.iterdir()):
            raise ValueError('Existing report requires separate review')
        verify_data(root)
        frozen, _, cells, protected = component.prepare(root, rules, protocol)
        audited = component.audit(root, frozen, protocol, cells)
        checked = verify_ready(root, spec, own, cells, audited)
        component.unchanged(root, protected)
        admission_sha = sha(admission_path)
        attempt.mkdir(exist_ok=False)
        stage_handle = stack.enter_context(stage.open('x'))
        execution = stack.enter_context((attempt/'execution.jsonl').open('x'))
        def log(event, **fields):
            row = protocol.canonical({'utc': datetime.now(timezone.utc).isoformat(), 'event': event,
                'job_id': own, 'predecessor': '365300', 'predecessor_never_started': True,
                'admission_sha256': admission_sha, 'parallel_policy_sha256': checked['policy']['policy_sha256'],
                **fields})+'\n'
            for handle in (stage_handle, execution):
                handle.write(row); handle.flush(); os.fsync(handle.fileno())
        log('start', verification=checked, max_invocations_per_model_per_phase=1,
            historical_unjournaled_outcomes_resolved=False)
        try:
            result = component.run_fixed_matrix(root, rules, inference, protocol, attempt/'models')
            log('inference_end', result=result)
            if result['status'] != 'inference_complete_report_pending':
                log('end', status='failed', completion_marker_written=False)
                return 1
            component.unchanged(root, protected)
            if sha(admission_path) != admission_sha:
                raise ValueError('Admission changed during execution')
            artifacts = finish_report(root, rules, analysis, protocol)
            log('end', status='complete', report_sha256=artifacts,
                cost_scope='Journaled attempts including reused controls, not incremental or full historical billing')
            return 0
        except BaseException as exc:
            log('end', status='failed', exception_type=type(exc).__name__)
            raise


def execute(root, admission_path):
    root = root.resolve()
    spec = read(admission_path)
    own = os.environ.get('SLURM_JOB_ID')
    validate(spec, own, read(root/'artifacts/budget.json'))
    if any(m in sys.modules for m in ('run', 'protocol', 'expanded_rules', 'analyze')):
        raise ValueError('Fresh Python process required')
    sys.path.insert(0, str(root/'src'))
    import expanded_rules
    import analyze
    import protocol
    import run
    return _execute_bound(root, admission_path, spec, own, expanded_rules, analyze, protocol, run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.root, args.admission.resolve()))
