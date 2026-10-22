"""One actual held timing job and two separate full CLI gates. No repeat."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import complete_core3_timing_20261016 as complete

ROOT = Path('/ceph/aasteine/decision-em-20261013')
PREFIX = 'core3_timing'
SCRIPT = 'artifacts/operations/core3_timing_final_20261016.sbatch'
SELF = 'artifacts/operations/admit_core3_timing_20261016.py'
READY = 'artifacts/verification/core3_timing_admission_readiness_20261016.json'
ADMISSION = 'configs/core3_timing_admission_20261016.json'
POLICY = 'configs/parallel_core3_timing_20261016.json'
MAP = 'artifacts/operations/core3_timing_job_20261016.json'
JOURNAL = 'artifacts/operations/core3_timing_submission_20261016.jsonl'
HELD = 'artifacts/verification/core3_timing_held_gate_20261016.json'
PRERELEASE = 'artifacts/verification/core3_timing_prerelease_gate_20261016.json'
NAME = 'decision-em-core3-timing-final'


def log(event, **values):
    row = dict(utc=complete.utc(), event=event, **values)
    with (ROOT/JOURNAL).open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    complete.executor.fsync_dir((ROOT/JOURNAL).parent)


def held_identity(job):
    raw = subprocess.check_output(['scontrol','show','job',job,'-o'], text=True)
    fields = dict(x.split('=',1) for x in raw.split() if '=' in x)
    assert fields['JobState'] == 'PENDING' and fields['Priority'] == '0' and fields['StartTime'] == 'Unknown'
    assert fields['JobName'] == NAME and fields['WorkDir'] == str(ROOT)
    assert fields['Reason'] == 'JobHeldUser'
    actual = subprocess.check_output(['scontrol','write','batch_script',job,'-'], text=True)
    assert actual.encode() == (ROOT/SCRIPT).read_bytes()
    return raw


def cli_gate(output):
    assert not (ROOT/output).exists()
    subprocess.run([sys.executable, '-u', str(ROOT/complete.SELF), '--root', str(ROOT),
                    '--admission', str(ROOT/ADMISSION), '--preflight-output', str(ROOT/output)], check=True)
    receipt = complete.final.read(ROOT/output)
    assert receipt['preflight_only'] is True and receipt['attempt_created'] is False
    assert receipt['inference_executed'] is False
    assert receipt['admission_sha256'] == complete.final.sha(ROOT/ADMISSION)
    return receipt


def hold():
    final, executor = complete.final, complete.executor
    assert all(not (ROOT/p).exists() for p in (ADMISSION,POLICY,MAP,JOURNAL,HELD,PRERELEASE))
    ready = final.read(ROOT/READY)
    assert ready['status'] == 'fresh_full_actual_readiness_before_held_submission'
    assert ready['admitted'] is False and ready['quality']['responses'] == 380475
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(ready['utc'])).total_seconds() < 1200
    prep = final.read(ROOT/'configs/core3_timing_final_preparation_20261016.json')
    assert not prep['admitted'] and not prep['execution_enabled'] and prep['slurm_job_id'] is None
    extra = {SELF,SCRIPT,READY,'artifacts/verification/check_core3_timing_admission_readiness_20261016.py',
             'artifacts/verification/core3_timing_final_preparation_20261016.json',
             'artifacts/operations/approval_runs23_aaron_20261014.json'}
    sources = {**ready['sources'], **{rel:final.sha(ROOT/rel) for rel in extra}}
    for rel,h in sources.items():
        assert final.sha(ROOT/rel) == h, rel
    for rel,h in prep['source_sha256'].items():
        assert final.sha(ROOT/rel) == h, rel
    with ExitStack() as stack:
        target = executor.locks(ROOT, 'core3', stack)
        diagnosis = final.read(ROOT/final.DIAGNOSIS)
        executor.assert_states(target, diagnosis['protected_sha256_or_absent'])
        for rel,h in ready['protected_extra_sha256'].items():
            assert final.sha(ROOT/rel) == h, rel
        assert not any((target/p).exists() for p in ready['required_absent'])
        executor.authority.authority(ROOT)
        old = final.read(ROOT/'configs/parallel_grouped_core3_fourth_20261016.json')
        reviewer = executor.authority.private_module(ROOT,'parallel_admission',[('!= 100','!= 200',1)])
        review = reviewer.review(ROOT, ROOT/'configs/parallel_grouped_core3_fourth_20261016.json')
        assert all(j['scope'] == 'different project' or j['id'] in {'365300','365355'} for j in review['reviewed_jobs'])
        executor.budget_check(ROOT, final.read(ROOT/final.PLAN))
        complete.durable_bytes(ROOT/JOURNAL,b'')
        log('held_submission_intent', script_sha256=final.sha(ROOT/SCRIPT), readiness_sha256=final.sha(ROOT/READY))
        job = subprocess.check_output(['sbatch','--hold','--parsable','--partition=cpu','--cpus-per-task=2',
            '--mem=12G','--time=2-00:00:00','--job-name='+NAME,
            '--output='+str(ROOT/'logs/core3-timing-final-%j.log'),
            '--error='+str(ROOT/'logs/core3-timing-final-%j.log'),
            '--chdir='+str(ROOT),str(ROOT/SCRIPT)],text=True).strip()
        assert job.isdigit(), job
        log('held_submission_returned', job=job)
        metadata = held_identity(job)
        log('held_readback', job=job, metadata=metadata)
        assert '366980' in old['reviewed_project_jobs']
        old['reviewed_project_jobs'].pop('366980')
        old['reviewed_project_jobs'][job] = dict(name=NAME,scope='core3',batch_script_sha256=final.sha(ROOT/SCRIPT))
        old['files_sha256'].update(sources)
        complete.durable_json(ROOT/POLICY,old)
        admission = dict(version='core3_timing_actual_admission_v1',utc=complete.utc(),admitted=True,
            stage='core3_timing',run_id=3,predecessor='366980',slurm_job_id=job,cap_usd=200,max_invocations=1,
            policy=POLICY,policy_sha256=final.sha(ROOT/POLICY),sources=sources,
            attempt=final.ATTEMPT,measurement_plan=final.PLAN,measurement_plan_sha256=final.PLAN_SHA,
            new_timing_responses=420,new_quality_responses=0,quality_credit=False,
            original_per_call_account=True,preparation_is_not_admission=True,
            readiness_sha256=final.sha(ROOT/READY),requires_two_full_actual_cli_gates=True)
        complete.durable_json(ROOT/ADMISSION,admission)
        complete.durable_json(ROOT/MAP,dict(job=job,predecessor='366980',admission=ADMISSION,policy=POLICY,
                                         stage='core3_timing',remaining_jobs={'confirmation':'366694','schema':'366695','core2':'366696','core3':job}))
    cli_gate(HELD)
    log('held_gate_complete',job=job,gate_sha256=final.sha(ROOT/HELD),admission_sha256=final.sha(ROOT/ADMISSION))
    print(json.dumps(dict(job=job,released=False,held_gate_sha256=final.sha(ROOT/HELD))))


def release():
    final = complete.final
    admission = final.read(ROOT/ADMISSION)
    job = admission['slurm_job_id']
    events = final.rows(ROOT/JOURNAL)
    assert events[-1]['event'] == 'held_gate_complete' and events[-1]['job'] == job
    assert events[-1]['admission_sha256'] == final.sha(ROOT/ADMISSION)
    assert events[-1]['gate_sha256'] == final.sha(ROOT/HELD)
    held_identity(job)
    log('separate_prerelease_gate_intent',job=job)
    cli_gate(PRERELEASE)
    held_identity(job)
    log('prerelease_gate_complete',job=job,gate_sha256=final.sha(ROOT/PRERELEASE))
    log('release_intent',job=job,admission_sha256=final.sha(ROOT/ADMISSION))
    subprocess.run(['scontrol','release',job],check=True)
    raw = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    fields = dict(x.split('=',1) for x in raw.split() if '=' in x)
    assert fields['JobState'] in {'PENDING','RUNNING'} and fields['Priority'] != '0'
    log('release_readback',job=job,metadata=raw)
    print(json.dumps(dict(job=job,released=True,metadata=raw)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('hold','release'))
    args = parser.parse_args()
    (hold if args.action == 'hold' else release)()
