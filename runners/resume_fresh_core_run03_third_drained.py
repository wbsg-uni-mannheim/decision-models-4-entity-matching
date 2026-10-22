"""One new Run3 core attempt after its initial and two drained B0 failures.

Reuse the hash-pinned USD200 core component and wrapper privately. The former
attempt remains immutable, while the existing stage journal is append-only.
No submission, automatic retry, scientific changes or budget increase.
"""
import argparse
import ast
import json
import os
from pathlib import Path

import authorized_recovery_200 as budget
from resume_variants_second_drained import afterok_ids

SELF = 'artifacts/operations/resume_fresh_core_run03_third_drained.py'
ATTEMPT = 'artifacts/core_drained_recovery_v3'
PREDECESSOR = '366520'
INITIAL = '365895'
PRIOR_ADMISSION = 'configs/core_run03_second_admission_20261015.json'
PRIOR_SHA = 'e5098b64358c6b320ab9cba0602c9f5db814231d0c9533f0e17318a86912f824'
MODELS = {'jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx', 'kev', 'luna'}
SOURCE_PLAN = 'artifacts/verification/core_phase_cost_scenarios_20261015.json'
SOURCE_PROOF = 'artifacts/verification/core_phase_cost_verification_20261015.json'
AUDITOR = 'artifacts/verification/audit_run03_drained_terminal_366520_20261015.py'
VERIFIER = 'artifacts/verification/verify_run03_drained_terminal_366520_20261015.py'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def validate_history(history, run_id, predecessor):
    if (run_id != 3 or type(run_id) is not int or predecessor != PREDECESSOR or len(history) != 8
            or [r.get('stage') for r in history] != ['smoke']*2+['core']*6
            or [r.get('status') for r in history] != ['started','complete']+['started','failed']*3
            or any(type(r.get('run_id')) is not int or r['run_id'] != 3 for r in history)
            or any(r.get('slurm_job_id') != INITIAL for r in history[2:4])
            or any(r.get('slurm_job_id') != '366118' or r.get('recovery') != 'drained_v1'
                   or r.get('predecessor') != INITIAL
                   or r.get('admission_sha256') != '9129d10dfabad82cbecae61330ef1d1450dd09d49d2ef36e01e747ac35bddd85'
                   for r in history[4:6])
            or any(r.get('slurm_job_id') != PREDECESSOR or r.get('recovery') != 'drained_v2'
                   or r.get('predecessor') != '366118' or r.get('admission_sha256') != PRIOR_SHA
                   for r in history[-2:])):
        raise ValueError('Exact initial and two drained failed Run3 histories required')


def guard(root, target, run_id, spec):
    if (spec.get('admitted') is not True or run_id != 3 or spec.get('predecessor') != PREDECESSOR
            or spec.get('budget_binding_sha256') != budget.sha(root/SELF)
            or spec.get('budget_authority_sha256') != budget.APPROVAL_SHA):
        raise ValueError('Exact reviewed third Run3 binding required')
    if budget.sha(root/PRIOR_ADMISSION) != PRIOR_SHA:
        raise ValueError('Prior admission changed')
    d = read(root/spec['diagnosis'])
    proof = read(root/spec['cost_source_verification'])
    if (budget.sha(root/spec['diagnosis']) != spec['diagnosis_sha256']
            or d['job'] != PREDECESSOR or d['run_id'] != 3
            or d['script_sha256'] != budget.sha(root/AUDITOR)
            or proof['script_sha256'] != budget.sha(root/VERIFIER)
            or proof['job'] != PREDECESSOR or proof['run_id'] != 3
            or proof['diagnosis_sha256'] != spec['diagnosis_sha256']
            or proof['independent_counts'] != d['totals'] or proof['independent_cells'] != 168
            or d['admission_sha256'] != PRIOR_SHA or d['operations_events'] != 0
            or proof['operations_events'] != 0):
        raise ValueError('Fresh complete independent terminal audit required')
    history = rows(target/'artifacts/stage_execution.jsonl')
    validate_history(history, run_id, PREDECESSOR)
    if history != d['stage_history']:
        raise ValueError('Stage history differs from audit')
    for rel, expected in d['protected_sha256_or_absent'].items():
        path = target/rel
        if (budget.sha(path) if path.exists() else None) != expected:
            raise ValueError('Audited protected source or journal changed: '+rel)
    ends = rows(target/'artifacts/core_drained_recovery_v2/phases/AP1_baseline.jsonl')
    if (ends != d['model_journal'] or len(ends) != 9 or ends[0]['event'] != 'start'
            or set(ends[0]['models']) != MODELS or ends[0]['invocations_per_model'] != 1
            or {r.get('model') for r in ends[1:-1]} != MODELS
            or any(r.get('event') != 'model_end' or r.get('status') not in ('complete','failed') for r in ends[1:-1])
            or ends[-1]['event'] != 'end' or ends[-1]['status'] != 'failed' or ends[-1]['models_drained'] != 7
            or proof['model_ends'] != ends[1:-1]):
        raise ValueError('All seven actual predecessor model ends required')
    execution = rows(target/'artifacts/core_drained_recovery_v2/execution.jsonl')
    if execution != history[-2:]:
        raise ValueError('Prior attempt and append-only stage provenance disagree')
    old_files = {str(p.relative_to(target)) for p in (target/'artifacts/core_drained_recovery_v2').rglob('*') if p.is_file()}
    if old_files != {p for p,h in d['protected_sha256_or_absent'].items() if h is not None and p.startswith('artifacts/core_drained_recovery_v2/')}:
        raise ValueError('Unreviewed prior attempt journal')
    first = target/'artifacts/core_drained_recovery_v1'
    first_files = {str(p.relative_to(target)) for p in first.rglob('*') if p.is_file()}
    if (first_files != {p for p,h in d['protected_sha256_or_absent'].items()
                       if h is not None and p.startswith('artifacts/core_drained_recovery_v1/')}
            or rows(first/'execution.jsonl') != history[4:6]):
        raise ValueError('First drained attempt provenance differs')
    cost = read(root/spec['cost_plan'])
    if cost['phases'] != proof['fresh_remaining_core_scenario']['phases']:
        raise ValueError('Complete current remaining phase costs differ from independent proof')
    scenario = proof['fresh_remaining_core_scenario']
    if (budget.sha(root/SOURCE_PLAN) != scenario['historical_source_scenario_sha256']
            or budget.sha(root/SOURCE_PROOF) != scenario['historical_source_proof_sha256']):
        raise ValueError('Historical full phase source proof changed')
    sources = read(root/SOURCE_PLAN)
    for rel, digest in {**sources['source_files_sha256'], **sources['frozen_source_sha256']}.items():
        if budget.sha(root/rel) != digest:
            raise ValueError('Historical raw cost source changed: '+rel)
    if set(read(target/'artifacts/progress.json')) != {'smoke','AP1_baseline'}:
        raise ValueError('Later phase history requires separate review')
    # No pre-existing output can be overwritten by the reused selector/analyser.
    if ((target/'configs/selected_variants.json').exists()
            or any((target/'artifacts/results').rglob('*'))):
        raise ValueError('Unexpected pre-existing core outputs')
    return {'all_model_ends': 7, 'protected_entries': len(d['protected_sha256_or_absent']),
            'prior_admission_sha256': PRIOR_SHA, 'fresh_phase_cost_proof_sha256': budget.sha(root/spec['cost_source_verification'])}


def completed_variants(root, command):
    audit_path = 'artifacts/operations/variants_complete_audit_366563_20261015.json'
    proof_path = 'artifacts/verification/variants_result_verification_366563_20261015.json'
    if (budget.sha(root/audit_path) != 'd7b6b2224599daead5b660b31646d7a08ddafd2346d57a0ed2522c26279f991f'
            or budget.sha(root/proof_path) != 'f4e79f38d9aa722722d160774b9bf34f1bc855e6008a7fad7f24220e6727ad0e'):
        raise ValueError('Verified completed variants evidence changed')
    accounting = command(['sacct','-n','-j','366563','--format=JobID,State,ExitCode,End','-P'],
                         text=True, env={**os.environ,'TZ':'UTC'})
    records = [line.split('|') for line in accounting.splitlines() if line.strip()]
    if ({r[0] for r in records} != {'366563','366563.batch','366563.extern'}
            or any(r[1:3] != ['COMPLETED','0:0'] for r in records)):
        raise ValueError('All completed variants steps required')
    proof = read(root/proof_path)
    for rel, digest in {**proof['protected_sha256'], **proof['export_sha256']}.items():
        if budget.sha(root/rel) != digest:
            raise ValueError('Verified variants source or result changed: '+rel)
    return accounting


def joins(root, target, spec, own, checked, command):
    completed_variants(root, command)
    active = read(root/'configs/core_run02_third_admission_20261015.json')
    if active['slurm_job_id'] != '366627' or active['run_id'] != 2 or active['replica'] != str(root/'replications/run02'):
        raise ValueError('Independent Run2 writer binding changed')
    config = read(root/'configs/expanded_execution.json')
    rules, stats = str(config['e2_rules_job']), str(config['statistics_full_job'])
    confirmation, schema, variants = (str(config[k]) for k in ('e2_confirmation_job','e2_schema_job','e6_variants_job'))
    if (rules, stats, variants) != ('365300','365355','366563'):
        raise ValueError('Downstream identities need new integration review')
    seen = set()
    for job in checked['policy']['reviewed_jobs']:
        if job.get('scope') == 'different project':
            continue
        jid = job['id']; seen.add(jid)
        meta = command(['scontrol','show','job',jid,'-o'], text=True)
        fields = dict(x.split('=',1) for x in meta.split() if '=' in x)
        dep = fields.get('Dependency','')
        if jid == rules:
            if (afterok_ids(dep) != {confirmation,schema} or fields.get('Priority') != '0'
                    or fields.get('JobState') != 'PENDING' or fields.get('StartTime') != 'Unknown'):
                raise ValueError('Legacy rules must remain held, never started, with full join')
        elif jid == stats:
            if afterok_ids(dep) != {confirmation,schema,rules}:
                raise ValueError('Full statistics conjunction changed')
        elif dep != '(null)':
            raise ValueError('Unexpected independent-job dependency')
        if jid not in {rules,stats,own}:
            roots = spec.get('reviewed_job_write_roots',{}).get(jid)
            expected_roots = ([str(root/'replications/run02')] if jid == '366627' else [str(root/'runs')])
            if (jid not in {confirmation,schema,'366627'} or roots != expected_roots
                    or any(target == Path(p) or target in Path(p).parents or Path(p) in target.parents for p in roots)):
                raise ValueError('Unreviewed or overlapping writer root')
    if not {rules,stats,own} <= seen:
        raise ValueError('Incomplete current all-join review')


def make_wrapper(root):
    w = budget.make_wrapper(root, 'core')
    source = (root/'artifacts/operations/resume_fresh_core_drained.py').read_text()
    functions = {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    ready = functions['verify_ready']
    replacements = {
        "'budget_cap_usd': 100": "'budget_cap_usd': 200",
        "diagnosis['predecessor']": "diagnosis['job']",
        "'terminal_partial_baseline_audited_not_recovered'": "'terminal_second_drained_core_recovery_partial_baseline_audited_not_recovered'",
        "diagnosis['stage_journal_sha256']": "diagnosis['protected_sha256_or_absent']['artifacts/stage_execution.jsonl']",
        "diagnosis['response_files_sha256']": "{p:h for p,h in diagnosis['protected_sha256_or_absent'].items() if p.startswith('runs/') and h is not None}",
    }
    for old,new in replacements.items():
        if ready.count(old) != 1:
            raise ValueError('Core history binding anchor drift')
        ready = ready.replace(old,new)
    execute = functions['execute']
    if execute.count("'recovery': 'drained_v1'") != 1:
        raise ValueError('Core attempt binding anchor drift')
    execute = execute.replace("'recovery': 'drained_v1'", "'recovery': 'drained_v3'")
    w.ATTEMPT = ATTEMPT
    w.__file__ = str(root/SELF)
    w.validate_history = validate_history
    w.REQUIRED |= {SELF, AUDITOR, VERIFIER, PRIOR_ADMISSION, 'configs/core_run02_third_admission_20261015.json', 'artifacts/operations/resume_variants_second_drained.py',
                   'artifacts/operations/variants_complete_audit_366563_20261015.json',
                   'artifacts/verification/variants_result_verification_366563_20261015.json'}
    exec(compile(ready+'\n\n'+execute, '<third Run3 core attempt>', 'exec'), w.__dict__)
    prior = w.verify_ready
    def verify_ready(project, target, run_id, spec, own, replica):
        lineage = guard(project, target, run_id, spec)
        checked = prior(project, target, run_id, spec, own, replica)
        initial_accounting = w.subprocess.check_output(['sacct','-n','-j',INITIAL,
            '--format=JobID,State,ExitCode,End','-P'], text=True, env={**os.environ,'TZ':'UTC'})
        w.terminal_steps(initial_accounting, INITIAL)
        first_accounting = w.subprocess.check_output(['sacct','-n','-j','366118',
            '--format=JobID,State,ExitCode,End','-P'], text=True, env={**os.environ,'TZ':'UTC'})
        w.terminal_steps(first_accounting, '366118')
        joins(project, target, spec, own, checked, w.subprocess.check_output)
        # Read again after all expensive hashes and scheduler checks.
        w.validate_cost(read(project/spec['cost_plan']), spec['diagnosis_sha256'], read(project/'artifacts/budget.json'))
        return {**checked, 'third_attempt_lineage': lineage, 'initial_accounting_utc':initial_accounting}
    w.verify_ready = verify_ready
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); spec = read(args.admission)
    w = make_wrapper(root)
    if args.preflight_output:
        checked = budget.preflight(root, 'core', 3, spec, w)
        with args.preflight_output.open('x') as f:
            json.dump({'read_only':True,'stage_execution':False,'admission_sha256':budget.sha(args.admission),'checks':checked},f,indent=2)
            f.write('\n')
        return 0
    if os.environ.get('SLURM_JOB_ID') != spec.get('slurm_job_id'):
        raise ValueError('Actual Slurm identity required')
    return w.execute(root, 3, args.admission.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
