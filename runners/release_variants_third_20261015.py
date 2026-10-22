"""Release the single held third variants successor after one fresh full gate."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import authorized_recovery_200 as budget
import resume_variants_third_drained as wrapper

ROOT=Path('/ceph/aasteine/decision-em-20261013')
ADMISSION=ROOT/'configs/variants_third_admission_20261015.json'
PREFLIGHT=ROOT/'artifacts/verification/variants_third_real_admission_20261015.json'
JOURNAL=ROOT/'artifacts/operations/variants_third_release_20261015.jsonl'


def main():
    spec=json.loads(ADMISSION.read_text()); proof=json.loads(PREFLIGHT.read_text())
    if (spec['slurm_job_id']!='366563' or proof['admission_sha256']!=budget.sha(ADMISSION)
            or proof['stage_execution'] or not proof['read_only']):
        raise ValueError('Exact reviewed held job required')
    meta=subprocess.check_output(['scontrol','show','job','366563','-o'],text=True)
    fields=dict(s.split('=',1) for s in meta.split() if '=' in s)
    if (fields['JobState']!='PENDING' or fields['StartTime']!='Unknown' or fields['Priority']!='0'
            or (ROOT/wrapper.JOURNAL).exists() or (ROOT/wrapper.MODEL_JOURNAL).exists()):
        raise ValueError('Job already started or no longer held')
    checked=budget.preflight(ROOT,'variants',1,spec,wrapper.make_wrapper(ROOT))
    for rel, digest in spec['prior_running_admission_and_policy_sha256'].items():
        if budget.sha(ROOT/rel)!=digest: raise ValueError('Active admission or policy changed')
    with JOURNAL.open('x') as f:
        def log(event,**data):
            f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event=event,job='366563',**data))+'\n')
            f.flush();os.fsync(f.fileno())
        log('release_start',admission_sha256=budget.sha(ADMISSION),preflight_sha256=budget.sha(PREFLIGHT),fresh_checks=checked)
        subprocess.check_call(['scontrol','release','366563'])
        meta=subprocess.check_output(['scontrol','show','job','366563','-o'],text=True)
        log('release_readback',scheduler=meta)
        print(meta)


if __name__=='__main__':main()
