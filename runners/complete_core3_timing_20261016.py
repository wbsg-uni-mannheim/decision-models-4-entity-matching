"""Bounded final integration around the unchanged timing executor.

Separate admission required by the bounded CLI. Original execution, independent proof, exclusive
local report publication and completion remain inside one pair of held locks.
A partial publication or incomplete commit blocks all automatic repetition.
The legacy generated draft is deliberately never published to any manuscript.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import types

import core3_final_validation_20261016 as final
import run_core3_timing_recovery_20261016 as executor

SELF = 'artifacts/operations/complete_core3_timing_20261016.py'
VALIDATOR = 'artifacts/operations/core3_final_validation_20261016.py'
MARKER = 'artifacts/core3_timing_verified_complete.json'
RESULTS = 'artifacts/results'
REQUIRED = executor.REQUIRED | {SELF, VALIDATOR}


def utc():
    return datetime.now(timezone.utc).isoformat()


def durable_bytes(path, content):
    with path.open('xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    executor.fsync_dir(path.parent)
    final.require(path.read_bytes() == content, 'Publication readback')


def durable_json(path, value):
    durable_bytes(path,(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode())


def current_gate(root, target, admission, own):
    """Real scheduler/source gate after measurement; no stale readiness reuse."""
    executor.authority.authority(root)
    executor.verify(root,3)
    final.require(REQUIRED <= admission['sources'].keys() and admission['slurm_job_id'] == own, 'Final sources/own job')
    for rel,h in admission['sources'].items():
        final.require(final.sha(root/rel) == h, 'Final source drift: '+rel)
    accounting = {}
    for job in executor.HISTORY:
        raw = subprocess.check_output(['sacct','-n','-j',job,'--format=JobID,State,ExitCode,End','-P'],text=True,env={**os.environ,'TZ':'UTC'})
        executor.terminal_steps(raw,job);accounting[job] = raw
    policy = executor.authority.private_module(root,'parallel_admission',[('!= 100','!= 200',1)])
    review = policy.review(root,root/admission['policy'],own)
    final.require(review['policy_sha256'] == admission['policy_sha256'] and any(j.get('self') and j['id'] == own for j in review['reviewed_jobs']), 'Current actual policy identity')
    allowed = {'366694','366695','366696','365300','365355',own}
    final.require(all(j['scope'] == 'different project' or j['id'] in allowed for j in review['reviewed_jobs']), 'Current overlapping writer')
    config = final.read(root/'configs/expanded_execution.json')
    final.require(str(config['e2_confirmation_job']) == '366694' and str(config['e2_schema_job']) == '366695','Root successor drift')
    joins = {}
    for job in ('365300','365355'):
        raw = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        fields = dict(x.split('=',1) for x in raw.split() if '=' in x)
        expected = {'366694','366695'} | ({'365300'} if job == '365355' else set())
        final.require(executor.afterok_ids(fields['Dependency']) == expected and fields['JobState'] == 'PENDING'
                      and fields['StartTime'] == 'Unknown' and (job != '365300' or fields['Priority'] == '0'), 'Current all joins/held rules')
        joins[job] = raw
    budget = final.read(root/'artifacts/budget.json')
    final.require(not any(b['active'] for b in budget.get('maintenance_barriers',{}).values()), 'Current maintenance barrier')
    # No new reservation is needed to publish already measured results.
    final.require(all(type(budget[k]) in (int,float) and final.math.isfinite(budget[k]) and budget[k] >= 0 for k in ('spent','reserved'))
                  and budget['spent']+budget['reserved'] <= 200+1e-8, 'Current budget ceiling')
    return dict(utc=utc(),historical_steps=accounting,policy=review,joins=joins,budget_read_only=budget)


def recheck_bound_bytes(target, proof):
    workspace = target/final.ATTEMPT/'analysis_staging'
    for rel,h in proof['protected_sha256_or_absent'].items():
        p = target/rel
        final.require((final.sha(p) if p.exists() else None) == h,'Protected evidence changed after proof')
    final.require({str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')} == proof['raw_sha256'].keys(),'Raw scope changed after proof')
    for rel,h in proof['raw_sha256'].items():
        final.require(final.sha(target/rel) == h, 'Raw changed after independent proof')
    observed = {str(p.relative_to(workspace)):final.sha(p) for p in workspace.rglob('*') if p.is_file()}
    final.require(observed == proof['staged_files_sha256'], 'Staging changed after proof')


def publish(target, proof, admission, admission_sha, final_gate):
    """Held-lock caller only. Exclusive files, durable progress, marker last.

Failure after progress replacement is explicitly not a committed Core outcome:
consumers must require marker AND final stage event and verify their hashes.
No rollback removes diagnostic partial files; retry remains prohibited.
"""
    attempt = target/final.ATTEMPT
    final.require(proof['status'] == 'independent_raw_execution_cost_numeric_report_verified'
        and proof['run_id'] == 3 and proof['core_complete'] is False
        and proof['admission_sha256'] == admission_sha, 'Independent final proof required')
    final.require(not (target/RESULTS).exists() and not (target/MARKER).exists(), 'Existing or partial publication')
    recheck_bound_bytes(target,proof)
    progress_path = target/'artifacts/progress.json'
    prior_progress = progress_path.read_bytes(); progress = json.loads(prior_progress)
    expected = dict(smoke='complete',AP1_baseline='complete',AP2_development='complete',AP2_confirmation='complete',AP1_timing='needs_attention')
    final.require({k:v['status'] for k,v in progress.items()} == expected, 'Exact pre-completion progress')
    durable_json(attempt/'final_audit.json',proof)
    durable_json(attempt/'final_publication_gate.json',final_gate)
    durable_bytes(attempt/'progress_before_publication.json',prior_progress)
    out = target/RESULTS; out.mkdir(exist_ok=False);executor.fsync_dir(out.parent)
    workspace = attempt/'analysis_staging'
    published = {}
    for rel,h in sorted(proof['staged_files_sha256'].items()):
        path = Path(rel)
        final.require(not path.is_absolute() and '..' not in path.parts, 'Unsafe staged path')
        if path.parts[:2] != ('artifacts','results'):
            continue
        dest = target/path
        dest.parent.mkdir(parents=True,exist_ok=True);executor.fsync_dir(dest.parent.parent)
        durable_bytes(dest,(workspace/path).read_bytes())
        final.require(final.sha(dest) == h, 'Published report SHA')
        published[rel] = h
    for name,value in [('timing_source_map.json',final.read(target.parents[1]/final.PLAN)),
                       ('timing_cost_scope.json',proof['all_timing_costs'])]:
        durable_json(out/name,value);published[RESULTS+'/'+name] = final.sha(out/name)
    executor.fsync_dir(out)
    # Recheck everything before the first completion-related status change.
    recheck_bound_bytes(target,proof)
    for rel,h in published.items():
        final.require(final.sha(target/rel) == h, 'Publication drift')
    final.require(progress_path.read_bytes() == prior_progress,'Concurrent progress change')
    when = utc()
    progress['AP1_timing'] = dict(status='complete',utc=when,detail='112 independently verified source-mapped full measurements; interrupted cell retained separately')
    progress['AP3_AP4_analysis'] = dict(status='complete_automated',utc=when,detail='Independent raw/numerical report proof; human audit and author review pending')
    progress_tmp = attempt/'progress_committed.json'
    durable_json(progress_tmp,progress)
    os.replace(progress_tmp,progress_path);executor.fsync_dir(progress_path.parent)
    final.require(final.read(progress_path) == progress,'Progress readback')
    marker = dict(status='complete',run_id=3,slurm_job_id=admission['slurm_job_id'],predecessor='366980',
        admission_sha256=admission_sha,independent_final_audit_sha256=final.sha(attempt/'final_audit.json'),
        final_gate_sha256=final.sha(attempt/'final_publication_gate.json'),progress_sha256=final.sha(progress_path),
        report_sha256=published,prior_sha256=proof['prior_sha256'],plan_sha256=proof['plan_sha256'],
        full_expanded_study_complete=False,human_audit='pending',author_review='pending',
        requires_matching_final_stage_event=True,private_legacy_draft_published=False)
    durable_json(target/MARKER,marker)
    return marker


def execute(root, admission_path, inference, orchestration, protocol, frozen_analysis):
    root = Path(root).resolve();admission_path = Path(admission_path).resolve()
    admission = final.read(admission_path);own = os.environ.get('SLURM_JOB_ID')
    final.require(admission.get('admitted') is True and own and admission.get('slurm_job_id') == own
                  and REQUIRED <= admission.get('sources',{}).keys(), 'Unadmitted final timing integration')
    admission_sha = final.sha(admission_path)
    with ExitStack() as stack:
        target = executor.locks(root,'core3',stack)
        final.require(not any((target/rel).exists() for rel in (final.ATTEMPT,RESULTS,MARKER)), 'Never repeat an attempt or partial publication')
        current_gate(root,target,admission,own)
        # Private function binding: outer locks stay held through independent
        # audit and commit. Never edit or monkeypatch the frozen executor module.
        scope = dict(vars(executor))
        def held_locks(given_root, stage, ignored_stack):
            final.require(given_root == root and stage == 'core3','Wrong held lock scope')
            return target
        scope['locks'] = held_locks
        bound = types.FunctionType(executor.execute.__code__,scope,executor.execute.__name__,executor.execute.__defaults__)
        bound(root,admission_path,inference,orchestration,protocol,frozen_analysis)
        attempt = target/final.ATTEMPT
        def log(status,**extra):
            row = dict(utc=utc(),stage='core',status=status,event='timing_final_'+status,run_id=3,slurm_job_id=own,
                predecessor='366980',recovery='timing_remeasurement_v1',admission_sha256=admission_sha,
                parallel_policy_sha256=admission['policy_sha256'],prior_sha256=final.sha(attempt/'prior.json'),**extra)
            executor.durable_append(attempt/'final_execution.jsonl',row,protocol.canonical)
            executor.durable_append(target/'artifacts/stage_execution.jsonl',row,protocol.canonical)
            executor.fsync_dir(attempt);executor.fsync_dir(target/'artifacts')
        try:
            final.require(final.sha(admission_path) == admission_sha,'Admission changed during execution')
            proof = final.verify(root,admission_path,protocol)
            checked = current_gate(root,target,admission,own)
            marker = publish(target,proof,admission,admission_sha,checked)
            log('complete',completion_marker_sha256=final.sha(target/MARKER),report_sha256=marker['report_sha256'],
                independent_final_audit_sha256=marker['independent_final_audit_sha256'],progress_sha256=marker['progress_sha256'],
                full_expanded_study_complete=False)
            # Completion is the conjunction, never merely analyzer exit/marker.
            require_committed(target)
            return marker
        except BaseException as exc:
            log('failed',exception_type=type(exc).__name__,core_complete=False,automatic_retry=False)
            raise


def require_committed(target):
    marker = final.read(target/MARKER);attempt = target/final.ATTEMPT
    events = final.rows(attempt/'final_execution.jsonl')
    final.require(len(events) == 1 and events[0]['status'] == 'complete'
                  and final.rows(target/'artifacts/stage_execution.jsonl')[-1] == events[0], 'Incomplete final stage commit')
    end = events[0]
    final.require(end['completion_marker_sha256'] == final.sha(target/MARKER)
                  and end['report_sha256'] == marker['report_sha256']
                  and end['progress_sha256'] == marker['progress_sha256'] == final.sha(target/'artifacts/progress.json')
                  and end['independent_final_audit_sha256'] == marker['independent_final_audit_sha256'] == final.sha(attempt/'final_audit.json'), 'Final commit provenance')
    for rel,h in marker['report_sha256'].items():
        final.require(final.sha(target/rel) == h,'Final report readback')
    return marker


def runtime(root):
    """Private globals only; every loaded source must match verified replica."""
    import importlib.util
    import sys
    target = root/'replications/run03'
    executor.verify(root,3)
    sys.path.insert(0,str(target/'src'))
    modules = {}
    for name in ('protocol','run','orchestrate','analyze'):
        path = target/'src'/(name+'.py')
        spec = importlib.util.spec_from_file_location('_core3_runtime_'+name,path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        final.require(final.sha(Path(module.__file__)) == final.sha(target/'src'/(name+'.py')), 'Runtime source mismatch')
        private = types.ModuleType('private_core3_'+name)
        private.__dict__.update(vars(module))
        scope = private.__dict__;scope['PROJECT'] = target
        for key,value in vars(module).items():
            if isinstance(value,types.FunctionType) and value.__globals__ is vars(module):
                scope[key] = types.FunctionType(value.__code__,scope,value.__name__,value.__defaults__,value.__closure__)
        modules[name] = private
    return modules['run'],modules['orchestrate'],modules['protocol'],modules['analyze']


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='One admitted Run3 timing recovery; never resume an existing attempt')
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--admission',required=True,type=Path)
    parser.add_argument('--preflight-output',type=Path)
    args = parser.parse_args(argv);root = args.root.resolve();admission_path = args.admission.resolve()
    admission = final.read(admission_path)
    final.require(admission.get('admitted') is True and REQUIRED <= admission.get('sources',{}).keys(), 'Unadmitted final timing CLI')
    modules = runtime(root)
    if not args.preflight_output:
        return execute(root,admission_path,*modules)
    own = admission['slurm_job_id']
    with ExitStack() as stack:
        target = executor.locks(root,'core3',stack)
        final.require(not any((target/rel).exists() for rel in (final.ATTEMPT,RESULTS,MARKER)), 'Attempt or publication already exists')
        checked = current_gate(root,target,admission,own)
        _,_,_,ready = executor.preflight(root,admission,own,modules[2],modules[1])
        final.require(final.read(admission_path) == admission,'Admission changed during preflight')
        result = dict(utc=utc(),admission_sha256=final.sha(admission_path),current_gate=checked,full_readiness=ready,
                      admitted=True,preflight_only=True,attempt_created=False,inference_executed=False)
        durable_json(args.preflight_output.resolve(),result)
        return result


if __name__ == '__main__':
    main()
