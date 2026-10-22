"""One unchanged schema invocation after a diagnosed, fully terminal predecessor."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from parallel_admission import review

ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}


def terminal_steps(accounting, predecessor):
    rows = [line.split('|') for line in accounting.splitlines() if line.strip()]
    ids = {row[0] for row in rows}
    if not {str(predecessor), str(predecessor)+'.batch', str(predecessor)+'.extern'} <= ids:
        raise ValueError('Missing predecessor step accounting')
    if any(row[1].split()[0] not in TERMINAL for row in rows):
        raise ValueError('Predecessor still has active or unknown steps')


def failure_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for ns in ['schema', 'schema_pilot']
            for p in sorted((root/'runs'/ns).rglob('*.failures.jsonl'))}


def verify_ready(root, spec, own):
    if own != str(spec['job_id']):
        raise ValueError('Not registered schema successor')
    config = json.loads((root/'configs/expanded_execution.json').read_text())
    if config['e2_schema_job'] != int(own):
        raise ValueError('Not current schema successor')
    accounting = subprocess.check_output(
        ['sacct', '-n', '-j', str(spec['predecessor']), '--format=JobID,State,ExitCode', '-P'], text=True)
    terminal_steps(accounting, spec['predecessor'])
    for rel, expected in spec['source_sha256'].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: '+rel)
    if failure_hashes(root) != spec['diagnosed_failure_sha256']:
        raise ValueError('Schema failure journals changed since diagnosis')
    review(root, root/'configs/parallel_execution_20261014.json', own)
    queue = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%T'], text=True)
    for line in queue.splitlines():
        job, state = line.split('|')
        if job == own or state == 'PENDING':
            continue
        script = subprocess.check_output(['scontrol', 'write', 'batch_script', job, '-'], text=True)
        if 'expanded_schema.py' in script or 'resume_schema' in script:
            raise ValueError('Potential overlapping schema writer: '+job)
    budget = json.loads((root/'artifacts/budget.json').read_text())
    if budget['spent']+budget['reserved']+.05 > 100:
        raise ValueError('Shared project spending ceiling reached')
    return budget


def main():
    spec = json.loads((ROOT/'configs/schema_resume_execution.json').read_text())
    own = os.environ['SLURM_JOB_ID']
    journal = ROOT/'artifacts/operations'/('schema_recovery_'+own+'.jsonl')
    with (ROOT/'artifacts/operations/schema_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = verify_ready(ROOT, spec, own)
        sys.path.insert(0, str(ROOT/'src'))
        import expanded_schema
        expanded_schema.verify()
        with journal.open('x') as out:
            def log(event, **values):
                out.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                      'event': event, **values})+'\n')
                out.flush()
                os.fsync(out.fileno())
            log('start', budget=budget, invocations_limit=1)
            result = subprocess.run([sys.executable, '-u', 'src/expanded_schema.py', 'run'], cwd=ROOT)
            log('end', returncode=result.returncode, failure_sha256=failure_hashes(ROOT),
                budget=json.loads((ROOT/'artifacts/budget.json').read_text()))
            return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
