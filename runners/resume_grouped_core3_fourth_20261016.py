"""One separately admitted Run3 resume after fully audited grouped_third job 366955.

Private reuse of the tested grouped/credit executor, with fresh exclusive
attempts. No active source, retry, prompt, selection, timing or policy changes.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import types

import authorized_recovery_200 as binding
import fast_transition_audit as raw_audit
from grouped_transition import inspector
from resume_schema_once import terminal_steps
from resume_variants_second_drained import afterok_ids

SELF = 'artifacts/operations/resume_grouped_core3_fourth_20261016.py'
BASE = 'artifacts/operations/grouped_resume_20261015.py'
AUDITOR = 'artifacts/operations/audit_grouped_core3_terminal_366955_20261016.py'
VERIFIER = 'artifacts/verification/verify_grouped_core3_terminal_366955_20261016.py'
ATTEMPT = 'artifacts/operations/grouped_fourth_resume_core3_20261016'
PREDECESSOR = '366955'
SOURCE_PLAN = 'artifacts/verification/core_phase_cost_scenarios_20261015.json'


HISTORY_JOBS = ('365894','365895','366118','366520','366644','366681','366697','366745','366955')
REMAINING_VERIFIER = 'artifacts/verification/verify_grouped_core3_terminal_366955_20261016.py'


def validate_remaining(root, target, spec, diagnosis, proof):
    remaining = read(root/spec['remaining_proof'])
    if (binding.sha(root/spec['remaining_proof']) != spec['remaining_proof_sha256']
            or remaining['script_sha256'] != binding.sha(root/REMAINING_VERIFIER)
            or any(remaining[k] != proof[k] for k in ('job','run_id','diagnosis_sha256','totals',
                'independent_cells','history_and_admission_verified','credit_exactly_reconciled',
                'new_durable_attempts','own_selection_reconstructed'))
            or remaining['remaining_cost_plan'] != spec['remaining_cost_scenario']):
        raise ValueError('Current independently verified remaining matrix required')
    selected = read(target/'configs/selected_variants.json')
    if {m:s['variant'] for m,s in selected.items()} != proof['own_selection_reconstructed']:
        raise ValueError('Own frozen selection changed')
    status = read(target/'artifacts/progress.json')
    expected = {'smoke':'complete','AP1_baseline':'complete','AP2_development':'complete',
                'AP2_confirmation':'needs_attention'}
    if set(status) != set(expected) or any(status[k]['status'] != v for k,v in expected.items()):
        raise ValueError('Only the audited selected-test failure is supported')
    keys = ('namespace','model','variant','benchmark','split')
    missing = {tuple(c[k] for k in keys):c['missing'] for c in diagnosis['cells'] if c['missing']}
    plan = remaining['remaining_cost_plan']
    if (len(plan['cells']) != len(missing)
            or {tuple(c['key']):c['missing'] for c in plan['cells']} != missing
            or sum(missing.values()) != plan['missing']
            or not math.isclose(math.fsum(c['known_proxy_usd'] for c in plan['cells']), plan['known_proxy_usd'],abs_tol=1e-12)
            or not math.isclose(math.fsum(c['historical_unknown_scenario_usd'] for c in plan['cells']), plan['historical_unknown_scenario_usd'],abs_tol=1e-12)
            or plan['historical_source_sha256'] != binding.sha(root/SOURCE_PLAN)):
        raise ValueError('Remaining costs/missing cells changed')
    history = diagnosis['stage_history']
    actual = [(e['slurm_job_id'],e['status']) for e in history]
    expected_history = [(j,s) for j in HISTORY_JOBS for s in (
        ('started','complete') if j=='365894' else
        ('started','credit_retained','failed') if j=='366681' else ('started','failed'))]
    if actual != expected_history or raw_audit.rows(target/'artifacts/stage_execution.jsonl') != history:
        raise ValueError('Entire initial/v1/v2/v3/fast/grouped/grouped_second/grouped_third history required')


def persist_prior(path, before):
    # No credit constructor is reached until both file and directory are durable.
    with path.open('x') as f:
        json.dump(before,f,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    fd=os.open(path.parent.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    if read(path) != before:
        raise ValueError('Prior readback failed')


def verify_drained_ends(folder):
    phases = {'AP1_baseline':7,'AP2_development':4,'AP2_confirmation':4}
    found=[]
    for name,count in phases.items():
        path=folder/(name+'.jsonl')
        if not path.exists(): break
        events=raw_audit.rows(path); ends=[e for e in events if e['event']=='model_end']
        if (not events or events[0]['event']!='start' or events[-1]['event']!='end'
                or len(ends)!=count or events[-1]['models_drained']!=count
                or sorted(e['model'] for e in ends)!=sorted(events[0]['models'])):
            raise ValueError('Missing durable drained model endings; retain credit')
        found.append(name)
    if not found or {p.stem for p in folder.glob('*.jsonl')} != set(found):
        raise ValueError('Missing or nonsequential phase journals; retain credit')


def core_component(root):
    source=(root/'artifacts/operations/fresh_core_drained.py').read_text()
    old="set(status) != {'smoke', 'AP1_baseline'} or status['smoke']['status'] != 'complete'\n            or status['AP1_baseline']['status'] != 'needs_attention'"
    new="set(status) != {'smoke','AP1_baseline','AP2_development','AP2_confirmation'} or status['smoke']['status'] != 'complete'\n            or status['AP1_baseline']['status'] != 'complete'\n            or status['AP2_development']['status'] != 'complete'\n            or status['AP2_confirmation']['status'] != 'needs_attention'"
    if source.count(old)!=1: raise ValueError('Core selected-phase guard drift')
    component=binding.private_module(root,'fresh_core_drained')
    exec(compile(source.replace(old,new),'<selected test recovery>','exec'),component.__dict__)
    original=component.run_phases
    def run_phases(root,target,run_id,inference,orchestration,protocol,journal_dir):
        selection=target/'configs/selected_variants.json'; prior=selection.read_bytes()
        # Complete baseline/development caches are replayed by frozen loaders.
        # Frozen select independently recomputes and compares without rewriting.
        result=original(root,target,run_id,inference,orchestration,protocol,journal_dir)
        if selection.read_bytes()!=prior: raise ValueError('Selection bytes changed')
        return result
    component.run_phases=run_phases
    return component


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
            or proof['totals'] != d['totals'] or proof['independent_cells'] != 440
            or not proof['history_and_admission_verified'] or not proof['credit_exactly_reconciled']
            or proof['job'] != PREDECESSOR or proof['run_id'] != 3):
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
    validate_remaining(root, target, spec, d, proof)
    return target, d


def preflight(root, spec, own):
    if (spec.get('admitted') is not True or spec.get('slurm_job_id') != own or not own
            or spec.get('stage') != 'core3' or spec.get('predecessor') != PREDECESSOR
            or spec.get('cap_usd') != 200 or spec.get('max_invocations') != 1):
        raise ValueError('Exact single Run3 admission required')
    binding.authority(root)
    required = {SELF, BASE, AUDITOR, VERIFIER, SOURCE_PLAN, REMAINING_VERIFIER,
                'artifacts/operations/fresh_core_drained.py',
                'artifacts/operations/resume_grouped_core3_third_20261016.py',
                'configs/parallel_grouped_core3_third_20261016.json',
                'configs/grouped_core3_third_admission_20261016.json',
                'artifacts/operations/resume_grouped_core3_second_20261015.py',
                'configs/parallel_grouped_core3_second_20261015.json',
                'configs/grouped_core3_second_admission_20261015.json',
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
    accounting = {}
    for job in HISTORY_JOBS:
        accounting[job] = subprocess.check_output(['sacct','-n','-j',job,
            '--format=JobID,State,ExitCode,End','-P'], text=True, env={**os.environ,'TZ':'UTC'})
        terminal_steps(accounting[job], job)
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
    if budget['spent']+budget['reserved']+max(8., spec['remaining_cost_scenario']['known_proxy_usd']) > 200:
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
    module = types.ModuleType('fourth_grouped_raw_audit'); module.__file__ = __file__
    exec(compile(source.replace(old, 'attempt = target/'+repr(ATTEMPT)), '<fourth grouped raw audit>', 'exec'), module.__dict__)
    module.JOBS = {'core3':os.environ.get('SLURM_JOB_ID')}
    module.inspect_operations = inspector(root)
    return module.audit(root,stage,require_terminal)


def make_wrapper(root):
    source = (root/BASE).read_text()
    replacements = [("target/f'artifacts/operations/grouped_resume_{stage}_20261015'", 'target/'+repr(ATTEMPT)),
                    ('from grouped_transition import audit', 'audit = final_audit'),
                    ("'grouped_resume_'+status", "'grouped_fourth_resume_'+status"),
                    ("(attempt/'prior_journals.json').write_text(json.dumps(before,sort_keys=True)+'\\n')", "persist_prior(attempt/'prior_journals.json', before)"),
                    ("{'admission_sha256':sha(admission),'run_id':run_id}", "{'admission_sha256':sha(admission),'run_id':run_id,'prior_sha256':sha(attempt/'prior_journals.json'),'journal_dir':str(attempt)}")]
    replacements.append(("        except BaseException as exc:\n", "        except BaseException as exc:\n            if isinstance(exc, OSError): credit.journal_io_failed = True\n"))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Executor binding drift: '+old)
        source = source.replace(old,new)
    module = types.ModuleType('fourth_grouped_executor'); module.__file__ = str(root/SELF)
    exec(compile(source,'<fourth grouped executor>','exec'),module.__dict__)
    module.preflight = preflight; module.final_audit = final_audit
    module.JOBS = {'core3':PREDECESSOR}
    module.core_component = core_component
    module.persist_prior = persist_prior
    original_accounting = module.verify_accounting
    def verify_accounting(credit, target, before):
        verify_drained_ends(target/ATTEMPT/'phases')
        if read(target/ATTEMPT/'prior_journals.json') != before:
            raise ValueError('Durable prior changed')
        # Timing keeps its original account, but an unknown new operation
        # anywhere in this core must still prevent this wrapper releasing credit.
        for p in (target/'runs').rglob('*.operations.jsonl'):
            rel=str(p.relative_to(target)); previous=before.get(rel,{'bytes':0})
            with p.open('rb') as f:
                prefix=f.read(previous['bytes']); tail=f.read()
            if 'sha256' in previous:
                import hashlib
                if hashlib.sha256(prefix).hexdigest()!=previous['sha256']:
                    raise ValueError('Old operations changed; retain credit')
            if any(json.loads(line)['operational_stop']['outcome_unknown'] for line in tail.splitlines()):
                raise ValueError('New unknown operations; retain credit')
        original_accounting(credit,target,before)
    module.verify_accounting = verify_accounting
    return module


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--admission',type=Path,required=True); p.add_argument('--preflight-output',type=Path)
    a = p.parse_args(); root = a.root.resolve()
    raise SystemExit(make_wrapper(root).execute(root,a.admission.resolve(),a.preflight_output))
