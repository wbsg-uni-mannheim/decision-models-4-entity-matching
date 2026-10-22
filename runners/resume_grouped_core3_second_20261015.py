"""One separately admitted Run3 resume after fully audited grouped job 366697.

Private reuse of the tested grouped/credit executor, with fresh exclusive
attempts. No active source, retry, prompt, selection, timing or policy changes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import types

import authorized_recovery_200 as binding
import fast_transition_audit as raw_audit
from grouped_transition import inspector
from resume_schema_once import terminal_steps
from resume_variants_second_drained import afterok_ids

SELF = 'artifacts/operations/resume_grouped_core3_second_20261015.py'
BASE = 'artifacts/operations/grouped_resume_20261015.py'
AUDITOR = 'artifacts/operations/audit_grouped_core3_terminal_20261015.py'
VERIFIER = 'artifacts/verification/verify_grouped_core3_terminal_20261015.py'
ATTEMPT = 'artifacts/operations/grouped_second_resume_core3_20261015'
PREDECESSOR = '366697'
SOURCE_PLAN = 'artifacts/verification/core_phase_cost_scenarios_20261015.json'


def read(p):
    return json.loads(p.read_text())


def validate_evidence(root, spec):
    target = root/'replications/run03'
    d = read(root/spec['diagnosis']); proof = read(root/spec['independent_proof'])
    if (binding.sha(root/spec['diagnosis']) != spec['diagnosis_sha256']
            or binding.sha(root/spec['independent_proof']) != spec['independent_proof_sha256']
            or d['job'] != PREDECESSOR or d['run_id'] != 3 or not d['all_steps_terminal']
            or d['script_sha256'] != binding.sha(root/AUDITOR)
            or proof['script_sha256'] != binding.sha(root/VERIFIER)
            or proof['diagnosis_sha256'] != spec['diagnosis_sha256']
            or proof['totals'] != d['totals'] or proof['independent_cells'] != 408
            or not proof['history_and_admission_verified'] or not proof['credit_exactly_reconciled']
            or proof['remaining_scenario'] != spec['remaining_cost_scenario']):
        raise ValueError('Fresh complete independent terminal evidence required')
    for rel, h in d['protected_sha256_or_absent'].items():
        p = target/rel
        if (binding.sha(p) if p.exists() else None) != h:
            raise ValueError('Audited source/cache/history drift '+rel)
    observed = {str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')}
    if not observed <= d['protected_sha256_or_absent'].keys():
        raise ValueError('Extra replica journal')
    for rel, h in d['historical_admission_policy_sha256'].items():
        if binding.sha(root/rel) != h:
            raise ValueError('Historical admission changed')
    source = read(root/SOURCE_PLAN)
    for rel, h in {**source['source_files_sha256'], **source['frozen_source_sha256']}.items():
        if binding.sha(root/rel) != h:
            raise ValueError('Cost source changed')
    old = read(root/'artifacts/budget.json')['inference_credits']['fast-core3-'+PREDECESSOR]
    if old != d['credit_reconciliation_read_only']['ledger'] or old['status'] != 'closed':
        raise ValueError('Predecessor credit differs from full reconciliation')
    if (target/ATTEMPT).exists():
        raise ValueError('Never reuse a successor attempt')
    if (target/'configs/selected_variants.json').exists():
        raise ValueError('Unexpected selection before resumed development')
    return target, d


def preflight(root, spec, own):
    if (spec.get('admitted') is not True or spec.get('slurm_job_id') != own or not own
            or spec.get('stage') != 'core3' or spec.get('predecessor') != PREDECESSOR
            or spec.get('cap_usd') != 200 or spec.get('max_invocations') != 1):
        raise ValueError('Exact single Run3 admission required')
    binding.authority(root)
    required = {SELF, BASE, AUDITOR, VERIFIER, SOURCE_PLAN,
                'artifacts/operations/grouped_inference.py', 'artifacts/operations/leased_budget.py',
                'artifacts/operations/grouped_transition.py',
                'artifacts/verification/grouped_stop_reconciled_20261015.json',
                'configs/parallel_grouped_20261015.json', 'configs/grouped_core3_admission_20261015.json'}
    if not required <= spec['sources'].keys():
        raise ValueError('Missing exact source bindings')
    for rel, h in spec['sources'].items():
        if binding.sha(root/rel) != h:
            raise ValueError('Source drift '+rel)
    # Other grouped jobs continue unchanged; only this scope must be terminal.
    accounting = subprocess.check_output(['sacct','-n','-j',PREDECESSOR,
        '--format=JobID,State,ExitCode,End','-P'], text=True, env={**os.environ,'TZ':'UTC'})
    terminal_steps(accounting, PREDECESSOR)
    target, diagnosis = validate_evidence(root, spec)
    from repeated_runs import verify
    verify(root, 3)
    policy = binding.private_module(root,'parallel_admission',[("!= 100","!= 200",1)])
    checked = policy.review(root,root/spec['policy'],own)
    if checked['policy_sha256'] != spec['policy_sha256'] or not any(j.get('self') for j in checked['reviewed_jobs']):
        raise ValueError('Actual scheduler admission required')
    allowed = {'366694','366695','366696',own,'365300','365355'}
    if any(j['scope'] != 'different project' and j['id'] not in allowed for j in checked['reviewed_jobs']):
        raise ValueError('Unreviewed project writer')
    config = read(root/'configs/expanded_execution.json')
    if str(config['e2_confirmation_job']) != '366694' or str(config['e2_schema_job']) != '366695':
        raise ValueError('Root successors changed; re-admit')
    joins = {}
    for job in ('365300','365355'):
        meta = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        fields = dict(x.split('=',1) for x in meta.split() if '=' in x)
        expected = {'366694','366695'} | ({'365300'} if job == '365355' else set())
        if afterok_ids(fields['Dependency']) != expected:
            raise ValueError('Full dependency join changed')
        if job == '365300' and (fields['Priority'] != '0' or fields['StartTime'] != 'Unknown' or fields['JobState'] != 'PENDING'):
            raise ValueError('Legacy rules must remain held and unstarted')
        joins[job] = meta
    variant = subprocess.check_output(['sacct','-n','-j','366563','--format=JobID,State,ExitCode,End','-P'],text=True)
    terminal_steps(variant,'366563')
    if any(x.split('|')[1:3] != ['COMPLETED','0:0'] for x in variant.splitlines() if x.strip()):
        raise ValueError('Variants prerequisite lost')
    budget = read(root/'artifacts/budget.json')
    if any(v['active'] for v in budget.get('maintenance_barriers',{}).values()):
        raise ValueError('Active maintenance barrier')
    if budget['spent']+budget['reserved']+.05 > 200:
        raise ValueError('No current budget headroom')
    return target, diagnosis, {'policy':checked,'budget':budget,'joins':joins,'predecessor_accounting_utc':accounting,
                              'diagnosis_sha256':spec['diagnosis_sha256'],
                              'historical_unjournaled_outcomes_resolved':False}


def final_audit(root, stage, require_terminal=False):
    target = root/'replications/run03'
    expected = {'AP1_baseline':7, 'AP2_development':4, 'AP2_confirmation':4}
    for phase, count in expected.items():
        events = raw_audit.rows(target/ATTEMPT/'phases'/(phase+'.jsonl'))
        ends = [e for e in events if e['event'] == 'model_end']
        if (not events or events[0]['event'] != 'start' or events[-1]['event'] != 'end'
                or events[-1]['status'] != 'complete' or events[-1]['models_drained'] != count
                or len(ends) != count or any(e['status'] != 'complete' for e in ends)
                or sorted(e['model'] for e in ends) != sorted(events[0]['models'])):
            raise ValueError('Every current phase model ending must be complete')
    source = Path(raw_audit.__file__).read_text()
    old = "attempt = (target/'artifacts/core_drained_recovery_v3') if stage.startswith('core') else root/f'artifacts/operations/{stage}_drained_recovery_v1'"
    if source.count(old) != 1:
        raise ValueError('Raw auditor source changed')
    module = types.ModuleType('second_grouped_raw_audit'); module.__file__ = __file__
    exec(compile(source.replace(old, 'attempt = target/'+repr(ATTEMPT)), '<second grouped raw audit>', 'exec'), module.__dict__)
    module.JOBS = {'core3':os.environ.get('SLURM_JOB_ID')}
    module.inspect_operations = inspector(root)
    return module.audit(root,stage,require_terminal)


def make_wrapper(root):
    source = (root/BASE).read_text()
    replacements = [("target/f'artifacts/operations/grouped_resume_{stage}_20261015'", 'target/'+repr(ATTEMPT)),
                    ('from grouped_transition import audit', 'audit = final_audit'),
                    ("'grouped_resume_'+status", "'grouped_second_resume_'+status")]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Executor binding drift: '+old)
        source = source.replace(old,new)
    module = types.ModuleType('second_grouped_executor'); module.__file__ = str(root/SELF)
    exec(compile(source,'<second grouped executor>','exec'),module.__dict__)
    module.preflight = preflight; module.final_audit = final_audit
    module.JOBS = {'core3':PREDECESSOR}
    return module


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--admission',type=Path,required=True); p.add_argument('--preflight-output',type=Path)
    a = p.parse_args(); root = a.root.resolve()
    raise SystemExit(make_wrapper(root).execute(root,a.admission.resolve(),a.preflight_output))
