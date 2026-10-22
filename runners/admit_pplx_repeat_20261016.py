"""Create one genuinely held diagnostic repeat; release is a separate full gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import run_pplx_baseline_repeat_20261016 as repeat

ROOT = Path('/ceph/aasteine/decision-em-20261013')
ADMISSION = 'configs/pplx_repeat_admission_20261016.json'
POLICY = 'configs/pplx_repeat_policy_20261016.json'
SCRIPT = 'artifacts/operations/pplx_baseline_repeat_20261016.sbatch'
JOURNAL = 'artifacts/operations/pplx_repeat_submission_20261016.jsonl'
SELF = 'artifacts/operations/admit_pplx_repeat_20261016.py'
NAME = 'pplx-repeat'


def log(event, **data):
    with (ROOT/JOURNAL).open('a') as f:
        f.write(json.dumps({'utc': repeat.now(), 'event': event, **data})+'\n')
        f.flush(); os.fsync(f.fileno())


def gate(kind):
    path = ROOT/f'artifacts/verification/pplx_repeat_{kind}_20261016.json'
    subprocess.run([sys.executable, '-u', str(ROOT/repeat.SELF), '--root', str(ROOT),
                    '--admission', str(ROOT/ADMISSION), '--preflight-output', str(path)], check=True)
    log(kind+'_passed', sha256=repeat.sha(path), admission_sha256=repeat.sha(ROOT/ADMISSION))


def submit():
    import protocol
    assert not (ROOT/JOURNAL).exists() and not (ROOT/ADMISSION).exists()
    with repeat.locks(ROOT):
        assert not (ROOT/repeat.ATTEMPT).exists() and not (ROOT/'runs'/repeat.NAMESPACE).exists()
        queue = repeat.cmd('squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%j')
        jobs = {}
        for line in queue.splitlines():
            job, name = line.split('|', 1)
            meta = repeat.cmd('scontrol', 'show', 'job', job, '-o')
            if str(ROOT) not in meta:
                continue
            assert job in ['365300', '365355'], meta
            fields = dict(x.split('=', 1) for x in meta.split() if '=' in x)
            assert fields['JobState'] == 'PENDING' and fields['StartTime'] == 'Unknown'
            expected = 'afterok:366694:366695' + (':365300' if job == '365355' else '')
            # Slurm appends condition annotations to already failed dependencies.
            import re
            assert re.sub(r'\([^)]*\)', '', fields['Dependency']).replace(',afterok:', ':') == expected
            if job == '365300':
                assert fields['Priority'] == '0'
            jobs[job] = {'name': name, 'dependency': fields['Dependency'],
                         'batch_script_sha256': hashlib.sha256(repeat.cmd('scontrol', 'write', 'batch_script', job, '-').encode()).hexdigest()}
        assert set(jobs) == {'365300', '365355'}
        sources = set(repeat.REQUIRED) | {SCRIPT, SELF}
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT/'src').glob('*.py'))
        protected = {'artifacts/progress.json', 'replications/run03/artifacts/progress.json',
                     'replications/run03/artifacts/stage_execution.jsonl'}
        for folder in ['artifacts/results', 'replications/run03/artifacts/results',
                       'artifacts/verification/core3_commit_anchor_367108_20261016',
                       'manuscript/paper_v2_20261016', 'output/pdf']:
            for directory, dirs, files in os.walk(ROOT/folder):
                dirs[:] = [d for d in dirs if 'checkpoint' not in d]
                protected.update(str((Path(directory)/name).relative_to(ROOT)) for name in files)
        counts = []
        for b in protocol.BENCHMARKS:
            data = 'data/'+b+'/test.json'; sources.add(data)
            cases = repeat.read(ROOT/data); wanted = {c['pair_id']: c for c in cases}
            base = ROOT/'runs/quality/pplx/B0'/b
            raw = [json.loads(x) for x in (base/'test.jsonl').read_text().splitlines()]
            assert len(raw) == len(wanted) and {r['pair_id'] for r in raw} == wanted.keys()
            for suffix in ['jsonl', 'failures.jsonl', 'operations.jsonl', 'executions.jsonl']:
                protected.add(str((base/('test.'+suffix)).relative_to(ROOT)))
            for r in raw:
                c = wanted[r['pair_id']]
                assert r.get('study_run_id', 1) == 1
                assert r['input_hash'] == c['input_hash']
                assert r['request_hash'] == protocol.digest(protocol.payload(c, 'pplx', 'B0'))
                assert protocol.parse_response('pplx', r['attempts'][-1]['response']) == (r['prediction'], r['probability_match'])
            attempts = [a for r in raw for a in r['attempts']]
            fp = base/'test.failures.jsonl'
            if fp.exists():
                attempts += [a for line in fp.read_text().splitlines() for a in json.loads(line)['attempts']]
            counts.append({'benchmark': b, 'responses': len(raw), 'retained_attempts': len(attempts),
                           'known_usd': sum(a['cost'] for a in attempts if a['cost'] is not None),
                           'unknown_attempts': sum(a['cost'] is None for a in attempts)})
        spec = {'admitted': True, 'model': 'pplx', 'variant': 'B0', 'split': 'test', 'run_id': 1,
                'namespace': repeat.NAMESPACE, 'max_invocations': 1, 'project_cap_usd': 200,
                'repeat_cap_usd': repeat.CAP, 'expected_pairs': 30225, 'benchmarks': list(protocol.BENCHMARKS),
                'approval': 'artifacts/operations/approval_pplx_repeat_aaron_20261016.json',
                'source_sha256': {p: repeat.sha(ROOT/p) for p in sorted(sources)},
                'protected_sha256_or_absent': {p: repeat.sha(ROOT/p) if (ROOT/p).exists() else None for p in sorted(protected)},
                'historical_run1_costs_not_quote': counts, 'policy': POLICY}
        repeat.source_check(ROOT, spec)
        budget = repeat.read(ROOT/'artifacts/budget.json')
        assert budget['spent']+budget['reserved']+repeat.CAP <= 200
        repeat.save(ROOT/'artifacts/verification/pplx_repeat_pre_submission_20261016.json',
                    {'utc': repeat.now(), 'budget': budget, 'queue': queue, 'old_run1': counts,
                     'sources': spec['source_sha256'], 'protected': spec['protected_sha256_or_absent']})
        with (ROOT/JOURNAL).open('x') as f:
            f.flush(); os.fsync(f.fileno())
        repeat.fsync_dir((ROOT/JOURNAL).parent)
        log('held_submission_intent', script_sha256=repeat.sha(ROOT/SCRIPT))
        job = repeat.cmd('sbatch', '--hold', '--parsable', '--partition=cpu', '--cpus-per-task=2',
                         '--mem=12G', '--time=2-00:00:00', '--job-name='+NAME,
                         '--output='+str(ROOT/'logs/pplx-repeat-%j.log'), '--chdir='+str(ROOT), str(ROOT/SCRIPT)).strip()
        assert job.isdigit()
        log('held_submission_returned', job=job)
        jobs[job] = {'name': NAME, 'batch_script_sha256': repeat.sha(ROOT/SCRIPT)}
        repeat.save(ROOT/POLICY, {'scope': repeat.NAMESPACE, 'jobs': jobs})
        spec.update(slurm_job_id=job, policy_sha256=repeat.sha(ROOT/POLICY))
        repeat.save(ROOT/ADMISSION, spec)
        repeat.save(ROOT/'artifacts/operations/pplx_repeat_job_20261016.json', {'job': job, 'admission': ADMISSION})
    gate('held_gate')
    print(json.dumps({'job': job, 'released': False}))


def release():
    spec = repeat.read(ROOT/ADMISSION)
    events = [json.loads(x) for x in (ROOT/JOURNAL).read_text().splitlines()]
    assert events[-1]['event'] == 'held_gate_passed'
    assert events[-1]['admission_sha256'] == repeat.sha(ROOT/ADMISSION)
    gate('prerelease_gate')
    log('release_intent', job=spec['slurm_job_id'])
    subprocess.run(['scontrol', 'release', spec['slurm_job_id']], check=True)
    log('release_returned', job=spec['slurm_job_id'], metadata=repeat.cmd('scontrol', 'show', 'job', spec['slurm_job_id'], '-o'))
    print(json.dumps({'job': spec['slurm_job_id'], 'released': True}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['submit', 'release'])
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT/'src'))
    submit() if args.action == 'submit' else release()
