"""One-time admission of diagnosed schema successor 365969; leaves job held."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'artifacts/operations'))
from parallel_admission import review
from resume_schema_once import verify_ready, terminal_steps, failure_hashes

JOB = '365969'


def main():
    receipt = ROOT/'artifacts/operations/schema_admission_365969.json'
    if receipt.exists():
        raise ValueError('Admission already recorded; inspect instead of replay')
    spec = json.loads((ROOT/'configs/schema_resume_prepared_20261014T2103Z.json').read_text())
    assert spec['predecessor'] == 365887 and spec['invocations_limit'] == 1
    acct = subprocess.check_output(['sacct', '-n', '-j', '365887', '--format=JobID,State,ExitCode', '-P'], text=True)
    terminal_steps(acct, 365887)
    assert failure_hashes(ROOT) == spec['diagnosed_failure_sha256']
    for rel, h in spec['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == h
    meta = subprocess.check_output(['scontrol', 'show', 'job', JOB, '-o'], text=True)
    assert 'JobState=PENDING ' in meta and 'Reason=JobHeldUser ' in meta
    assert 'JobName=decision-em-schema-resume ' in meta
    assert f'WorkDir={ROOT} ' in meta
    script = subprocess.check_output(['scontrol', 'write', 'batch_script', JOB, '-'], text=True)
    assert script.endswith((ROOT/'artifacts/operations/schema_resume_20261014.sbatch').read_text())
    # Every admitted Run2/3 stage has already started, so its journal/hash stays historical.
    admissions = []
    for path in sorted((ROOT/'configs').glob('run0[23]_*_admission_*.json')):
        adm = json.loads(path.read_text())
        journal = ROOT/f"replications/run{adm['run_id']:02d}/artifacts/stage_execution.jsonl"
        rows = [json.loads(x) for x in journal.read_text().splitlines()]
        assert any(x['stage'] == adm['stage'] and x['status'] == 'started'
                   and str(x['slurm_job_id']) == str(adm['slurm_job_id']) for x in rows), path
        admissions.append({'file': str(path.relative_to(ROOT)), 'action': 'preserve_started_stage_admission',
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    paths = ['configs/parallel_execution_20261014.json', 'configs/expanded_execution.json', 'configs/schema_resume_execution.json']
    for rel in paths:
        backup = ROOT/'artifacts/operations'/('before_schema_365969_'+Path(rel).name)
        with backup.open('xb') as f:
            f.write((ROOT/rel).read_bytes())
    policy_path = ROOT/paths[0]
    policy = json.loads(policy_path.read_text())
    policy['reviewed_project_jobs'][JOB] = {'name': 'decision-em-schema-resume',
        'batch_script_sha256': hashlib.sha256(script.encode()).hexdigest(),
        'scope': 'schema/schema_pilot only; one unchanged continuation after terminal365887 and complete probability-schema diagnosis'}
    policy_path.write_text(json.dumps(policy, indent=2)+'\n')
    execution = json.loads((ROOT/paths[1]).read_text())
    execution.setdefault('schema_resume_history', []).append(execution['e2_schema_job_execution'])
    execution['e2_schema_job'] = int(JOB)
    execution['e2_schema_job_execution'] = {'job_id': int(JOB), 'predecessor': 365887, 'dependency': None,
        'command': '.venv/bin/python -u artifacts/operations/resume_schema_once.py',
        'log': f'logs/schema-resume-{JOB}.log', 'execution_config': paths[2],
        'diagnosis': 'artifacts/operations/schema_terminal_diagnosis_20261014T2103Z.json'}
    deps = {'365300': ['365636', JOB], '365355': ['365636', '365279', JOB, '365300']}
    for job, prereqs in deps.items():
        execution['parallel_execution']['dependencies'][job] = prereqs
        for value in execution.values():
            if isinstance(value, dict) and value.get('job_id') == int(job):
                value['dependency'] = 'afterok:'+':'.join(prereqs)
    execution['parallel_execution']['dependencies'][JOB] = []
    (ROOT/paths[1]).write_text(json.dumps(execution, indent=2)+'\n')
    spec.update(job_id=int(JOB), held_submission_receipt=JOB)
    (ROOT/paths[2]).write_text(json.dumps(spec, indent=2)+'\n')
    joins = {}
    for job, prereqs in deps.items():
        subprocess.run(['scontrol', 'update', f'JobId={job}', 'Dependency=afterok:'+':'.join(prereqs)], check=True)
        after = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
        actual = after.split('Dependency=')[1].split(' ')[0].split(',')
        assert {x.split('(')[0] for x in actual} == {'afterok:'+p for p in prereqs}
        joins[job] = after
    reviewed = review(ROOT, policy_path, JOB)
    budget = verify_ready(ROOT, spec, JOB)
    spec['policy_review'] = reviewed
    spec['budget_before'] = budget
    (ROOT/paths[2]).write_text(json.dumps(spec, indent=2)+'\n')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'job_id': int(JOB), 'status': 'admitted_held_not_released',
        'policy_review': reviewed, 'budget': budget, 'predecessor_accounting': acct,
        'joins_readback': joins, 'replica_admissions': admissions, 'pending_replica_admissions': 0,
        'no_frozen_sources_or_parsers_changed': True, 'new_error_policy': 'Preserve invalid sum0.99 response; no normalization/tolerance change. One unchanged invocation; no automatic further start.'}
    with receipt.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    (ROOT/f'configs/schema_admission_{JOB}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'job': JOB, 'budget': budget, 'policy_sha256': reviewed['policy_sha256'], 'replica_admissions_checked': len(admissions)}))


if __name__ == '__main__':
    main()
