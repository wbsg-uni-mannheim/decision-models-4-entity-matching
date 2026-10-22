"""Correct only missing parent directories; admit fresh held core job identities."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/ceph/aasteine/decision-em-20261013')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True)


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    # Active Run1 jobs must already have finished their original queue check.
    for stage in ('confirmation', 'schema'):
        assert (ROOT/f'artifacts/operations/fast_resume_{stage}_20261015').is_dir()
    proof_rel = 'artifacts/verification/fast_core_start_directory_failure_20261015.json'
    proof = read(ROOT/proof_rel)
    old = read(ROOT/'configs/parallel_fast_20261015.json')
    for rel, digest in old['files_sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    jobs = read(ROOT/'artifacts/operations/fast_jobs_20261015.json')
    for stage in ('core2', 'core3'):
        check = proof[stage]
        assert check['all_steps_terminal'] and check['no_new_inference']
        assert check['attempt_absent'] and check['credit_absent']
        target = ROOT/f'replications/run0{stage[-1]}'
        assert (target/'artifacts/operations').is_dir()
        assert not (target/f'artifacts/operations/fast_resume_{stage}_20261015').exists()
        rows = command('sacct', '-n', '-j', jobs[stage], '--format=JobID,State,ExitCode,End', '-P')
        assert all(row.split('|')[1] in ('FAILED', 'COMPLETED') for row in rows.splitlines() if row.strip())
    journal_path = ROOT/'artifacts/operations/fast_core_directory_submission_20261015.jsonl'
    with journal_path.open('x') as journal:
        for stage in ('core2', 'core3'):
            script = ROOT/f'artifacts/operations/fast_{stage}_directory_20261015.sbatch'
            with script.open('x') as handle:
                handle.write(f'#!/bin/bash\nset -euo pipefail\ncd {ROOT}\nexport PYTHONPATH=artifacts/operations\nexec .venv/bin/python -u artifacts/operations/fast_resume_20261015.py --root {ROOT} --admission {ROOT}/configs/fast_{stage}_directory_admission_20261015.json\n')
            journal.write(json.dumps({'stage': stage, 'event': 'submit_start', 'script_sha256': sha(script)})+'\n')
            journal.flush(); os.fsync(journal.fileno())
            job = command('sbatch', '--parsable', '--hold', '--no-requeue', '--partition=cpu',
                          '--cpus-per-task=2', '--mem=8G', '--time=2-00:00:00',
                          f'--job-name=decision-em-fast-{stage}', '--chdir='+str(ROOT),
                          '--output='+str(ROOT/f'logs/fast-{stage}-%j.log'), str(script)).strip().split(';')[0]
            assert job.isdigit()
            jobs[stage] = job
            journal.write(json.dumps({'stage': stage, 'event': 'submitted_held', 'job': job})+'\n')
            journal.flush(); os.fsync(journal.fileno())
    policy = {**old, 'files_sha256': dict(old['files_sha256']), 'reviewed_project_jobs': {}}
    policy['files_sha256'][proof_rel] = sha(ROOT/proof_rel)
    for stage, job in {**jobs, 'rules': '365300', 'statistics': '365355'}.items():
        meta = command('scontrol', 'show', 'job', job, '-o')
        fields = dict(x.split('=', 1) for x in meta.split() if '=' in x)
        assert fields['WorkDir'] == str(ROOT)
        script = command('scontrol', 'write', 'batch_script', job, '-')
        digest = hashlib.sha256(script.encode()).hexdigest()
        if stage.startswith('core'):
            rel = f'artifacts/operations/fast_{stage}_directory_20261015.sbatch'
            assert fields['Priority'] == '0' and fields['StartTime'] == 'Unknown'
            assert digest == sha(ROOT/rel)
            policy['files_sha256'][rel] = digest
        else:
            assert digest == old['reviewed_project_jobs'][job]['batch_script_sha256']
        policy['reviewed_project_jobs'][job] = {'name': fields['JobName'], 'scope': stage, 'batch_script_sha256': digest}
    policy_rel = 'configs/parallel_fast_core_directory_20261015.json'
    save(ROOT/policy_rel, policy)
    for stage in ('core2', 'core3'):
        spec = read(ROOT/f'configs/fast_{stage}_admission_20261015.json')
        spec.update(slurm_job_id=jobs[stage], new_jobs=jobs, sources=policy['files_sha256'],
                    policy=policy_rel, policy_sha256=sha(ROOT/policy_rel),
                    pre_inference_directory_failure_proof=proof_rel,
                    pre_inference_directory_failure_proof_sha256=sha(ROOT/proof_rel))
        save(ROOT/f'configs/fast_{stage}_directory_admission_20261015.json', spec)
    save(ROOT/'artifacts/operations/fast_jobs_directory_20261015.json', jobs)
    print(json.dumps({'jobs': jobs, 'released': False, 'policy_sha256': sha(ROOT/policy_rel)}))


if __name__ == '__main__':
    main()
