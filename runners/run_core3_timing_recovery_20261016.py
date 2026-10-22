"""Bounded timing-only successor; no quality credit, retries or Core completion.

One separate admission is required. The completed plan is never regenerated.
The original per-call account is used under the authorized USD200 ceiling.
Reports remain private staging until an independent final Core/report proof.
No CLI is exposed while that final publication/completion integration is open.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import types

import authorized_recovery_200 as authority
import core3_timing_recovery_plan_20261016 as plan
import core3_timing_analysis_20261016 as analysis
from audit_grouped_core3_terminal_366980_20261016 import history
from diagnose_fresh_baseline import audit_rows
from drained_inference import build, durable_append
from fast_transition_audit import locks
from grouped_transition import inspector
from repeated_runs import verify, shared_account
from resume_schema_once import terminal_steps
from resume_variants_second_drained import afterok_ids

SELF = 'artifacts/operations/run_core3_timing_recovery_20261016.py'
DIAGNOSIS = 'artifacts/operations/grouped_core3_terminal_366980_20261016.json'
HISTORY = ('365894', '365895', '366118', '366520', '366644', '366681',
           '366697', '366745', '366955', '366980')
REQUIRED = {SELF, analysis.PLAN, 'artifacts/operations/core3_timing_analysis_20261016.py',
    'artifacts/operations/core3_timing_recovery_plan_20261016.py',
    'artifacts/operations/audit_grouped_core3_terminal_366980_20261016.py',
    'artifacts/operations/diagnose_fresh_baseline.py',
    'artifacts/operations/drained_inference.py', 'artifacts/operations/repeated_runs.py',
    'artifacts/operations/fast_transition_audit.py', 'artifacts/operations/grouped_transition.py',
    'artifacts/operations/operational_journal.py',
    'artifacts/operations/authorized_recovery_200.py', 'artifacts/operations/parallel_admission.py',
    'artifacts/operations/resume_schema_once.py', 'artifacts/operations/resume_variants_second_drained.py',
    'artifacts/verification/grouped_stop_reconciled_20261015.json',
    'configs/grouped_core3_fourth_admission_20261016.json',
    'configs/parallel_grouped_core3_fourth_20261016.json', authority.APPROVAL, *plan.PINS}


def utc():
    return datetime.now(timezone.utc).isoformat()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_exclusive(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    fsync_dir(path.parent)
    plan.require(plan.read(path) == value, 'Durable prior/output readback failed')


def assert_states(target, states):
    for rel, digest in states.items():
        path = target/rel
        plan.require((plan.sha(path) if path.exists() else None) == digest, 'Protected state changed: '+rel)


def budget_check(root, spec):
    b = plan.read(root/'artifacts/budget.json')
    plan.require(not any(x['active'] for x in b.get('maintenance_barriers', {}).values()), 'Maintenance barrier')
    plan.require(all(math.isfinite(b[k]) and b[k] >= 0 for k in ('spent', 'reserved')), 'Invalid current budget')
    # Eight original concurrent .05 reservations; proxy is not a hard bound.
    plan.require(b['spent']+b['reserved']+max(.4, spec['known_proxy_usd']) <= 200, 'Insufficient current headroom')
    return b


def verify_new_scope(target, saved, d, attempt, prior_sha, protocol):
    expected = [(i,c) for i,c in enumerate(saved['cells']) if c['new_responses']]
    events = plan.rows(attempt/'cells.jsonl')
    plan.require(len(events) == 2*len(expected), 'Every invocation must have one start and end')
    allowed = set(d['protected_sha256_or_absent'])
    for index, (i,c) in enumerate(expected):
        start, end = events[2*index:2*index+2]
        plan.require(start['event'] == 'start' and end['event'] == 'end' and end['status'] == 'complete'
                     and all(e['cell_index'] == i and e['cell'] == c and e['prior_sha256'] == prior_sha
                             and type(e['run_id']) is int and e['run_id'] == 3 for e in (start,end)),
                     'Invocation identity/order/end mismatch')
        accepted = plan.validate_measurement(target, c, protocol)
        plan.require(end['accepted'] == accepted, 'Cell acceptance provenance changed')
        directory = plan.base(target, c['analysis_namespace'], c['model'], c['benchmark'])
        allowed.update(str((directory/('prompt'+s)).relative_to(target)) for s in plan.SUFFIXES)
    observed = {str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')}
    plan.require(observed <= allowed, 'Out-of-scope raw journal')


def quality_replay(root, target, d, protocol, orchestration):
    """Full current quality raw replay and own selection, without any requests."""
    plan.require(orchestration.PROJECT == target, 'Wrong selection root')
    inspect = inspector(root)
    result = []
    cells = [c for c in d['cells'] if c['namespace'] == 'quality']
    plan.require(len(cells) == 328, 'Complete quality matrix required')
    for c in cells:
        m, v, b, s = (c[k] for k in ('model', 'variant', 'benchmark', 'split'))
        directory = target/'runs/quality'/m/v/b
        cases = plan.read(target/'data'/b/(s+'.json'))
        success, failed = [plan.rows(directory/(s+suffix)) for suffix in ('.jsonl', '.failures.jsonl')]
        plan.require(all(r['variant'] == v and type(r.get('study_run_id')) is int
                         and r['study_run_id'] == 3 for r in success+failed), 'Strict quality RunID/variant')
        adapter = types.SimpleNamespace(payload=lambda case, alias, _, v=v: protocol.payload(case, alias, v),
            digest=protocol.digest, parse_response=protocol.parse_response)
        checked = audit_rows(cases, *[[{**r, 'variant': 'B0'} for r in rr] for rr in (success, failed)],
                             adapter, m, b, 3)
        ops = inspect(directory/(s+'.operations.jsonl'), cases, failed, protocol, m, v, 3)
        plan.require(checked['missing'] == 0 and not ops['unknown_outcome_events'], 'Quality incomplete/unknown')
        for key in ('expected', 'responses', 'attempts', 'unknown_attempts', 'failure_rows'):
            plan.require(checked[key] == c[key], 'Quality audit changed')
        plan.require(math.isclose(checked['known_usd'], c['known_usd'], abs_tol=1e-10), 'Quality cost drift')
        result.append(checked)
    selection = target/'configs/selected_variants.json'
    before = selection.read_bytes()  # absence is an error: never select anew.
    plan.require(orchestration.select() == json.loads(before) and selection.read_bytes() == before,
                 'Own development selection changed')
    plan.require(sum(c['responses'] for c in result) == 380475, 'Quality scope mismatch')
    return {'cells': len(result), 'responses': 380475, 'selection_sha256': plan.sha(selection)}


def readiness(root, protocol, orchestration):
    """Current evidence, history, quality and costs; not a scheduler admission."""
    target = root/'replications/run03'
    verify(root, 3)
    plan.require(plan.sha(root/analysis.PLAN) == analysis.PLAN_SHA, 'Finished measurement plan changed')
    for rel, digest in plan.PINS.items():
        plan.require(plan.sha(root/rel) == digest, 'Terminal proof changed')
    saved = plan.read(root/analysis.PLAN)
    analysis.validate_map(saved, protocol)
    # Read-only reconstruction is an equality gate, never republishes/replaces it.
    plan.require(plan.prepare(root, protocol) == saved, 'Current timing state differs from frozen plan')
    d = plan.read(root/DIAGNOSIS)
    assert_states(target, d['protected_sha256_or_absent'])
    observed = {str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')}
    plan.require(observed <= d['protected_sha256_or_absent'].keys(), 'Unexpected replica journal')
    events, protected, ends, provenance = history(root, target)
    plan.require(events == d['stage_history'] and ends == d['all_historical_model_journals']
                 and provenance == d['historical_admission_policy_sha256'], 'Whole FOURTH history drift')
    assert_states(target, protected)
    for rel, digest in {**provenance, **d['admission_sources_verified']}.items():
        plan.require(plan.sha(root/rel) == digest, 'Historical source/admission changed')
    source = plan.read(root/'artifacts/verification/core_phase_cost_scenarios_20261015.json')
    for rel, digest in {**source['source_files_sha256'], **source['frozen_source_sha256']}.items():
        plan.require(plan.sha(root/rel) == digest, 'Historical cost source changed')
    progress = plan.read(target/'artifacts/progress.json')
    expected = {'smoke':'complete', 'AP1_baseline':'complete', 'AP2_development':'complete',
                'AP2_confirmation':'complete', 'AP1_timing':'needs_attention'}
    plan.require({k:v['status'] for k,v in progress.items()} == expected, 'Timing-only recovery state required')
    b = budget_check(root, saved)
    credit = b['inference_credits']['fast-core3-366980']
    plan.require(credit == d['credit_reconciliation_read_only']['ledger'] and credit['status'] == 'closed',
                 'Exactly reconciled predecessor credit required')
    quality = quality_replay(root, target, d, protocol, orchestration)
    retained = [plan.validate_measurement(target, c, protocol) for c in saved['cells'] if not c['new_responses']]
    assert_states(target, d['protected_sha256_or_absent'])
    return target, saved, d, {'quality':quality, 'retained_measurements':len(retained),
                             'historical_jobs':list(HISTORY), 'budget_read_only':b}


def preflight(root, admission, own, protocol, orchestration):
    plan.require(admission.get('admitted') is True and admission.get('slurm_job_id') == own and bool(own)
                 and admission.get('predecessor') == '366980' and admission.get('stage') == 'core3_timing'
                 and type(admission.get('run_id')) is int and admission['run_id'] == 3
                 and admission.get('cap_usd') == 200 and admission.get('max_invocations') == 1,
                 'Separate exact timing admission required')
    authority.authority(root)
    plan.require(REQUIRED <= admission['sources'].keys(), 'Missing timing source bindings')
    for rel, digest in admission['sources'].items():
        plan.require(plan.sha(root/rel) == digest, 'Admitted source drift: '+rel)
    target, saved, d, proof = readiness(root, protocol, orchestration)
    accounting = {}
    for job in HISTORY:
        raw = subprocess.check_output(['sacct','-n','-j',job,'--format=JobID,State,ExitCode,End','-P'],
                                      text=True, env={**os.environ,'TZ':'UTC'})
        terminal_steps(raw, job); accounting[job] = raw
    policy = authority.private_module(root, 'parallel_admission', [('!= 100', '!= 200', 1)])
    checked = policy.review(root, root/admission['policy'], own)
    plan.require(checked['policy_sha256'] == admission['policy_sha256']
                 and any(j.get('self') and j['id'] == own for j in checked['reviewed_jobs']),
                 'Actual scheduler identity required')
    allowed = {'366694', '366695', '366696', '365300', '365355', own}
    plan.require(all(j['scope'] == 'different project' or j['id'] in allowed for j in checked['reviewed_jobs']),
                 'Unexpected project writer')
    config = plan.read(root/'configs/expanded_execution.json')
    plan.require(str(config['e2_confirmation_job']) == '366694' and str(config['e2_schema_job']) == '366695',
                 'Root successor changed')
    joins = {}
    for job in ('365300', '365355'):
        meta = subprocess.check_output(['scontrol','show','job',job,'-o'], text=True)
        fields = dict(x.split('=',1) for x in meta.split() if '=' in x)
        expected = {'366694','366695'} | ({'365300'} if job == '365355' else set())
        plan.require(afterok_ids(fields['Dependency']) == expected, 'All dependency joins required')
        plan.require(fields['JobState'] == 'PENDING' and fields['StartTime'] == 'Unknown', 'Downstream job started')
        plan.require(job != '365300' or fields['Priority'] == '0', 'Legacy rules must remain held')
        joins[job] = meta
    proof.update(policy=checked, historical_steps=accounting, joins=joins)
    budget_check(root, saved)  # reread after expensive full replay and scheduler checks.
    return target, saved, d, proof


class PerCallObserver:
    """In-memory diagnostics around the original account; no credit/ledger edits.

The original account still performs every reserve/settle transaction. Only
thread-safe counters are added; no batching or extra I/O enters measurements.
"""
    def __init__(self, account):
        self.account = account
        self.lock = threading.Lock()
        self.reserved = self.settled = self.unknown = 0
        self.costs = []
        self.io_failed = False

    def __call__(self, action, amount=0, cap=200):
        plan.require(cap == 200, 'Account cap changed')
        try:
            result = self.account(action, amount, cap)
        except OSError:
            self.io_failed = True
            raise
        with self.lock:
            if action == 'reserve':
                self.reserved += 1
            elif action == 'settle':
                self.settled += 1
                self.unknown += amount is None
                if amount is not None:
                    self.costs.append(amount)
        return result

    def snapshot(self):
        return {'attempts':self.reserved, 'settled':self.settled, 'unknown':self.unknown,
                'known_usd':math.fsum(self.costs), 'outstanding':self.reserved-self.settled,
                'account_io_failed':self.io_failed}


def execute(root, admission_path, inference, orchestration, protocol, frozen_analysis, preflight_output=None):
    """Run one bounded recovery to private reports, never publish Core complete."""
    root = Path(root).resolve()
    admission = plan.read(admission_path)
    own = admission.get('slurm_job_id') if preflight_output else os.environ.get('SLURM_JOB_ID')
    # Fail unadmitted entries before creating locks/attempts/account objects.
    plan.require(admission.get('admitted') is True and bool(own), 'Unadmitted timing preparation')
    with ExitStack() as stack:
        target = locks(root, 'core3', stack)
        target, saved, d, checked = preflight(root, admission, own, protocol, orchestration)
        if preflight_output:
            save_exclusive(preflight_output, checked)
            return checked
        plan.require(inference.PROJECT == target, 'Fresh replica inference module required')
        attempt = target/plan.ATTEMPT
        attempt.mkdir(exist_ok=False); fsync_dir(attempt.parent)
        immutable = dict(d['protected_sha256_or_absent'])
        # Only the originally empty 20 timing cells may change. The old failed
        # cell stays immutable; replacement is in its separate absent namespace.
        for c in saved['cells']:
            if c['action'] == 'measure_unstarted_cell':
                directory = plan.base(target, c['analysis_namespace'], c['model'], c['benchmark'])
                for suffix in plan.SUFFIXES:
                    immutable.pop(str((directory/('prompt'+suffix)).relative_to(target)), None)
        history_path = target/'artifacts/stage_execution.jsonl'
        prior_stage = history_path.read_bytes()
        immutable.pop('artifacts/stage_execution.jsonl', None)
        prior = {'admission_sha256':plan.sha(admission_path), 'plan_sha256':analysis.PLAN_SHA,
                 'protected_sha256_or_absent':immutable, 'prior_stage_sha256':plan.sha(history_path),
                 'checked':checked, 'attempt':plan.ATTEMPT, 'run_id':3}
        save_exclusive(attempt/'prior.json', prior)
        fsync_dir(attempt.parent)
        prior_sha = plan.sha(attempt/'prior.json')
        def log(status, **extra):
            row = {'utc':utc(), 'stage':'core', 'status':status, 'event':'timing_recovery_'+status,
                'run_id':3, 'slurm_job_id':own, 'predecessor':'366980', 'recovery':'timing_remeasurement_v1',
                'admission_sha256':plan.sha(admission_path), 'parallel_policy_sha256':admission['policy_sha256'],
                'prior_stage_journal_sha256':prior['prior_stage_sha256'], 'prior_sha256':prior_sha, **extra}
            durable_append(attempt/'execution.jsonl', row, protocol.canonical)
            durable_append(history_path, row, protocol.canonical)
            fsync_dir(attempt); fsync_dir(history_path.parent)
        log('started')
        original = shared_account(inference.account, root)
        plan.require(original.__defaults__ == (0,100), 'Original per-call account changed')
        account = types.FunctionType(original.__code__, original.__globals__, original.__name__, (0,200))
        observer = PerCallObserver(account)
        isolated = types.SimpleNamespace(**{**vars(inference), 'account':observer})
        _, cell, adapter = build(isolated, 3)
        accepted_new = []
        try:
            budget_check(root, saved)
            assert_states(target, immutable)
            for i, c in enumerate(saved['cells']):
                if not c['new_responses']:
                    continue
                plan.require(plan.sha(attempt/'prior.json') == prior_sha, 'Durable prior drift')
                directory = plan.base(target, c['analysis_namespace'], c['model'], c['benchmark'])
                plan.require(not any((directory/('prompt'+s)).exists() for s in plan.SUFFIXES), 'Never resume a new measurement')
                budget_check(root, saved)
                event = {'cell_index':i, 'cell':c, 'prior_sha256':prior_sha, 'run_id':3, 'utc':utc()}
                journal = attempt/'cells.jsonl'
                durable_append(journal, {**event,'event':'start'}, protocol.canonical); fsync_dir(attempt)
                try:
                    cell(c['model'], 'B0', c['benchmark'], 'prompt', workers=c['workers'], limit=20,
                         namespace=c['analysis_namespace'])
                    accepted = plan.validate_measurement(target, c, protocol)
                    # Confirm directory entries after the original cell's file fsyncs.
                    for p in (directory, directory.parent, directory.parent.parent,
                              directory.parent.parent.parent, directory.parent.parent.parent.parent, target/'runs'):
                        fsync_dir(p)
                except BaseException as exc:
                    durable_append(journal, {**event,'event':'end','status':'failed',
                        'exception_type':type(exc).__name__, 'ended_at':utc()}, protocol.canonical)
                    raise
                durable_append(journal, {**event,'event':'end','status':'complete','accepted':accepted,
                    'ended_at':utc()}, protocol.canonical)
                accepted_new.append(accepted)
            actual = observer.snapshot()
            plan.require(len(accepted_new) == 21 and sum(x['responses'] for x in accepted_new) == 420,
                         'Complete bounded new measurement scope required')
            plan.require(not actual['account_io_failed'] and actual['outstanding'] == 0
                         and actual['attempts'] == actual['settled'] == sum(c['attempts'] for c in accepted_new)
                         and actual['unknown'] == sum(c['unknown_attempts'] for c in accepted_new)
                         and math.isclose(actual['known_usd'], math.fsum(c['known_usd'] for c in accepted_new), abs_tol=1e-10),
                         'Durable timing attempts do not match original per-call account')
            assert_states(target, immutable)
            verify_new_scope(target, saved, d, attempt, prior_sha, protocol)
            workspace = attempt/'analysis_staging'
            workspace.mkdir(exist_ok=False)
            (workspace/'configs').mkdir(); (workspace/'manuscript').mkdir()
            (workspace/'artifacts').mkdir()
            output = workspace/'artifacts/results'
            output.mkdir()
            for name in ('manifest.json', 'selected_variants.json'):
                data = (target/'configs'/name).read_bytes()
                with (workspace/'configs'/name).open('xb') as f:
                    f.write(data); f.flush(); os.fsync(f.fileno())
            for directory in (workspace/'configs', workspace/'manuscript', workspace/'artifacts', workspace, attempt):
                fsync_dir(directory)
            mapped, accepted = analysis.bind(frozen_analysis, target, saved, protocol, output)
            save_exclusive(attempt/'timing_cost_proof.json', analysis.cost_proof(target, saved, accepted))
            save_exclusive(attempt/'timing_measurements.json', {'new':accepted_new,'account':actual,'adapter':adapter})
            mapped.main()
            reports = {}
            for p in workspace.rglob('*'):
                if p.is_file():
                    with p.open('rb') as f: os.fsync(f.fileno())
                    reports[str(p.relative_to(workspace))] = plan.sha(p)
            for directory in sorted((p for p in workspace.rglob('*') if p.is_dir()), key=lambda p:len(p.parts), reverse=True):
                fsync_dir(directory)
            fsync_dir(workspace)
            plan.require('artifacts/results/timing.csv' in reports, 'Frozen timing report missing')
            assert_states(target, immutable)
            plan.require(history_path.read_bytes().startswith(prior_stage), 'Historical stage bytes changed')
            plan.require(plan.sha(attempt/'prior.json') == prior_sha, 'Prior changed after measurement')
            result = {'status':'measured_analysis_staged_pending_independent_final_audit',
                'core_complete':False, 'reports_published':False, 'reports_sha256':reports,
                'new_timing_responses':420, 'new_quality_responses':0, 'account':actual,
                'independent_full_raw_execution_cost_report_proof_required':True}
            save_exclusive(attempt/'staged_result.json', result)
            log('measured_pending_audit', result=result)
            return result
        except BaseException as exc:
            log('failed', exception_type=type(exc).__name__, per_call_account=observer.snapshot(),
                core_complete=False, automatic_retry=False)
            raise
