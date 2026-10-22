"""One newly admitted continuation after full drained 366251 end review.

Private binding of the preserved v1 wrapper: new exclusive attempt journals,
current predecessor and additional exact provenance/scope/all-join guards.
No change to inference, parsers, retries, scientific sources or active policies.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re

import authorized_recovery_200 as budget
from diagnose_variants_second_terminal_366251_20261015 import validate_history, STAGE, MODELS, ADMISSION

SELF = 'artifacts/operations/resume_variants_third_drained.py'
JOURNAL = 'artifacts/operations/variants_drained_recovery_v3.jsonl'
MODEL_JOURNAL = 'artifacts/operations/variants_drained_models_v3.jsonl'
AUDITOR = 'artifacts/operations/diagnose_variants_second_terminal_366251_20261015.py'



VERIFIER = 'artifacts/verification/check_variants_second_terminal_366251_20261015.py'
PROOF = 'artifacts/verification/variants_terminal_verification_366251_20261015.json'
HISTORY_FILES = (
    'artifacts/operations/variants_drained_recovery_v1.jsonl',
    'artifacts/operations/variants_drained_models_v1.jsonl',
    'configs/variants_budget200_admission_20261015.json',
    'artifacts/operations/diagnose_variants_drained_terminal.py',
    'artifacts/operations/resume_variants_second_drained.py',
    STAGE, MODELS, ADMISSION,
)
WRITER_ADMISSIONS = {
    '366115': 'configs/confirmation_budget200_reviewed_admission_20261015.json',
    '366116': 'configs/schema_budget200_reviewed_admission_20261015.json',
    '366520': 'configs/core_run03_second_admission_20261015.json',
    '366545': 'configs/core_run02_second_reviewed_admission_20261015.json',
}


def history(root, spec, diagnosis, aliases):
    from diagnose_variants_drained_terminal import validate_history as prior_history
    def rows(rel):
        return [json.loads(x) for x in (root/rel).read_text().splitlines()]
    first = rows(HISTORY_FILES[0]); first_models = rows(HISTORY_FILES[1])
    prior_history(first, first_models, aliases, budget.sha(root/HISTORY_FILES[2]))
    second = rows(STAGE); second_models = rows(MODELS)
    validate_history(second, second_models, aliases, budget.sha(root/ADMISSION))
    provenance = second[0]['verification']['predecessor_provenance']
    first_spec = json.loads((root/HISTORY_FILES[2]).read_text())
    second_spec = json.loads((root/ADMISSION).read_text())
    proof = json.loads((root/PROOF).read_text())
    if (first_spec['slurm_job_id'] != '366105' or first_spec['predecessor'] != '365279'
            or second_spec['slurm_job_id'] != '366251' or second_spec['predecessor'] != '366105'
            or second[0].get('predecessor_job') != '366105'
            or provenance['predecessor_stage_sha256'] != budget.sha(root/HISTORY_FILES[0])
            or provenance['predecessor_models_sha256'] != budget.sha(root/HISTORY_FILES[1])
            or provenance['predecessor_diagnosis_sha256'] != second_spec['diagnosis_sha256']
            or diagnosis['stage_history'] != second
            or diagnosis['script_sha256'] != budget.sha(root/AUDITOR)
            or proof['script_sha256'] != budget.sha(root/VERIFIER)
            or proof['diagnosis_sha256'] != spec['diagnosis_sha256']
            or proof['totals'] != diagnosis['totals']
            or proof['model_endings'] != {e['model']: e['status'] for e in diagnosis['model_endings']}
            or proof['complete_cells'] != diagnosis['complete_cells']
            or proof['known_journaled_usd'] != diagnosis['known_journaled_usd']
            or proof['operations_events'] != 0
            or proof['cost_plan_sha256'] != spec['cost_plan_sha256']):
        raise ValueError('Full initial/v1/v2 history and independent audit required')
    return {p: budget.sha(root/p) for p in HISTORY_FILES}


def checked_scope(root, spec, jid):
    if jid not in WRITER_ADMISSIONS:
        raise ValueError('Unreviewed writer identity')
    admission = json.loads((root/WRITER_ADMISSIONS[jid]).read_text())
    if admission['slurm_job_id'] != jid:
        raise ValueError('Writer admission identity differs')
    reference = json.loads((root/WRITER_ADMISSIONS['366115']).read_text())
    if jid in {'366520', '366545'}:
        run_id = 3 if jid == '366520' else 2
        prior_id = '366118' if run_id == 3 else '366117'
        if (admission['run_id'] != run_id or admission['stage'] != 'core'
                or admission['replica'] != str(root/f'replications/run{run_id:02d}')
                or any(c[0] not in {f'replications/run{run_id:02d}/'+n for n in ('quality','timing_concurrent8','timing_serial')}
                       for c in reference['reviewed_job_write_cells'][prior_id])):
            raise ValueError('Independent core namespace binding differs')
        expected = reference['reviewed_job_write_cells'][prior_id]
    else:
        expected = reference['reviewed_job_write_cells'][jid]
    if spec['reviewed_job_write_cells'].get(jid) != expected:
        raise ValueError('Reviewed writer scope differs from preserved admission')
    return expected


def afterok_ids(value):
    # Slurm lists conjunctive afterok clauses separated by commas, or colon IDs.
    if not re.fullmatch(r'afterok:\d+(?:\((?:unfulfilled|fulfilled|failed)\))?(?::\d+(?:\((?:unfulfilled|fulfilled|failed)\))?)*(?:,afterok:\d+(?:\((?:unfulfilled|fulfilled|failed)\))?)*', value):
        raise ValueError('Only conjunctive afterok dependencies permitted')
    return set(re.findall(r'\d+', value))


def guard(root, spec, own, checked, models, benchmarks, command):
    diagnosis = json.loads((root/spec['diagnosis']).read_text())
    if diagnosis['job'] != '366251' or diagnosis.get('historical_unjournaled_outcomes_resolved') is not False:
        raise ValueError('Fresh drained predecessor audit required')
    for rel, expected in diagnosis['protected_sha256'].items():
        if budget.sha(root/rel) != expected:
            raise ValueError('Predecessor or pinned source changed: '+rel)
    if any((root/rel).exists() for rel in diagnosis['absent_journals']):
        raise ValueError('Previously absent journal appeared')
    history_proof = history(root, spec, diagnosis, models)
    stage = [json.loads(x) for x in (root/STAGE).read_text().splitlines()]
    ends = [json.loads(x) for x in (root/MODELS).read_text().splitlines()]
    validate_history(stage, ends, models, budget.sha(root/ADMISSION))
    if any(o['unknown_outcome_events'] for o in diagnosis['operations']):
        raise ValueError('Unresolved current operational outcomes')
    cost = json.loads((root/spec['cost_plan']).read_text())
    if not cost.get('source_sha256_or_absent'):
        raise ValueError('Fresh raw cost source evidence required')
    for rel, expected in cost['source_sha256_or_absent'].items():
        p = root/rel
        if (budget.sha(p) if p.exists() else None) != expected:
            raise ValueError('Cost source changed')
    for proof in cost['source_proofs']:
        if budget.sha(root/proof['path']) != proof['sha256']:
            raise ValueError('Cost source proof changed')
    config = json.loads((root/'configs/expanded_execution.json').read_text())
    rules, stats = str(config['e2_rules_job']), str(config['statistics_full_job'])
    confirmation, schema = str(config['e2_confirmation_job']), str(config['e2_schema_job'])
    if rules != '365300' or stats != '365355':
        raise ValueError('Join identities changed; fresh integration review required')
    protected = {('quality', m, 'B0', b, 'test') for m in models for b in benchmarks}
    dependencies = {}
    for job in checked['policy']['reviewed_jobs']:
        if job.get('scope') == 'different project':
            continue
        jid = job['id']
        meta = command(['scontrol', 'show', 'job', jid, '-o'], text=True)
        fields = dict(x.split('=', 1) for x in meta.split() if '=' in x)
        dep = fields.get('Dependency', '')
        dependencies[jid] = dep
        if jid == rules:
            if (afterok_ids(dep) != {confirmation, schema} or fields.get('Priority') != '0'
                    or fields.get('JobState') != 'PENDING' or fields.get('StartTime') != 'Unknown'):
                raise ValueError('Held never-started legacy rules or join changed')
        elif jid == stats:
            if afterok_ids(dep) != {confirmation, schema, own, rules}:
                raise ValueError('Incomplete statistics conjunction')
        elif dep != '(null)':
            raise ValueError('Unexpected independent-job dependency')
        if not job.get('self') and jid not in {rules, stats}:
            scope = checked_scope(root, spec, jid)
            if scope is None or any(tuple(c) in protected for c in scope):
                raise ValueError('Unreviewed or overlapping writer scope')
    if not {rules, stats, own} <= dependencies.keys() or dependencies != spec['reviewed_dependencies']:
        raise ValueError('Current all-job dependency inventory differs')
    return {'predecessor_stage_sha256': budget.sha(root/STAGE),
            'predecessor_models_sha256': budget.sha(root/MODELS),
            'predecessor_diagnosis_sha256': spec['diagnosis_sha256'],
            'all_dependencies': dependencies, 'scopes_checked': True, 'history': history_proof}


def make_wrapper(root):
    w = budget.make_wrapper(root, 'variants')
    original = (root/'artifacts/operations/resume_variants_drained.py').read_text()
    parts = {n.name: ast.get_source_segment(original, n) for n in ast.parse(original).body
             if isinstance(n, ast.FunctionDef)}
    validate = parts['validate']
    if validate.count("'predecessor': '365279'") != 1:
        raise ValueError('Predecessor binding anchor drift')
    validate = validate.replace("'predecessor': '365279'", "'predecessor': '366251'")
    validate = validate.replace("'budget_cap_usd': 100", "'budget_cap_usd': 200").replace(' > 100:', ' > 200:')
    execute = parts['execute']
    old = "root/'artifacts/operations/variants_drained_models_v1.jsonl'"
    if execute.count(old) != 1 or execute.count("log('start', invocations_limit=1, verification=checked,") != 1:
        raise ValueError('Exclusive attempt binding anchor drift')
    execute = execute.replace(old, 'root/MODEL_JOURNAL').replace(
        "log('start', invocations_limit=1, verification=checked,",
        "log('start', invocations_limit=1, verification=checked, predecessor_job='366251',")
    w.JOURNAL, w.MODEL_JOURNAL = JOURNAL, MODEL_JOURNAL
    w.REQUIRED |= {SELF, AUDITOR, VERIFIER, PROOF, *HISTORY_FILES, *WRITER_ADMISSIONS.values()}
    exec(compile(validate+'\n\n'+execute, '<variants third bounded attempt>', 'exec'), w.__dict__)
    prior_verify = w.verify_ready
    def verify_ready(project, spec, own, models, benchmarks):
        checked = prior_verify(project, spec, own, models, benchmarks)
        lineage = guard(project, spec, own, checked, models, benchmarks, w.subprocess.check_output)
        for jid in ('365279', '366105'):
            accounting = w.subprocess.check_output(['sacct', '-n', '-j', jid, '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ':'UTC'})
            w.terminal_steps(accounting, jid)
        # Re-read headroom after all source and scheduler work.
        ledger = json.loads((project/'artifacts/budget.json').read_text())
        w.validate(spec, own, ledger)
        cost = json.loads((project/spec['cost_plan']).read_text())
        if ledger['spent']+ledger['reserved']+cost['known_proxy_usd']+cost['historical_unknown_scenario_usd'] > 200:
            raise ValueError('Current remaining scenario exceeds headroom')
        return {**checked, 'predecessor_provenance': lineage, 'final_budget': ledger}
    w.verify_ready = verify_ready
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = json.loads(args.admission.read_text())
    if (spec.get('budget_binding_sha256') != budget.sha(root/SELF)
            or spec.get('budget_authority_sha256') != budget.APPROVAL_SHA
            or spec.get('budget_cap_usd') != 200):
        raise ValueError('Actual exact third-attempt admission required')
    wrapper = make_wrapper(root)
    if args.preflight_output:
        checked = budget.preflight(root, 'variants', 1, spec, wrapper)
        with args.preflight_output.open('x') as f:
            json.dump({'read_only': True, 'stage_execution': False,
                'admission_sha256': budget.sha(args.admission), 'checks': checked}, f, indent=2)
            f.write('\n')
        return 0
    if os.environ.get('SLURM_JOB_ID') != spec['slurm_job_id']:
        raise ValueError('Actual Slurm identity required')
    return wrapper.execute(root, args.admission.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
