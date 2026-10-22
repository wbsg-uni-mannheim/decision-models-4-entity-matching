"""Fresh prerelease verification, then release each admitted job exactly once."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path('/ceph/aasteine/decision-em-20261013')
def read(p):return json.loads(p.read_text())
jobs=read(ROOT/'artifacts/operations/grouped_jobs_20261015.json')
def check(stage):
    p=ROOT/f'artifacts/verification/grouped_{stage}_real_preflight_20261015.json'
    assert p.exists()
    a=ROOT/f'configs/grouped_{stage}_admission_20261015.json'
    spec=read(a);assert spec['slurm_job_id']==jobs[stage]
    r=subprocess.run([sys.executable,'artifacts/operations/grouped_resume_20261015.py','--root',str(ROOT),'--admission',str(a),'--preflight-output',str(ROOT/f'artifacts/verification/grouped_{stage}_prerelease_20261015.json')],cwd=ROOT,text=True,capture_output=True)
    print(stage,r.returncode,r.stdout,r.stderr,flush=True);r.check_returncode()
with ThreadPoolExecutor(4) as pool:list(pool.map(check,jobs))
with (ROOT/'artifacts/operations/grouped_release_20261015.jsonl').open('x') as f:
    for stage,job in jobs.items():
        meta=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        fields=dict(x.split('=',1) for x in meta.split() if '=' in x)
        assert fields['Priority']=='0' and fields['StartTime']=='Unknown' and fields['JobState']=='PENDING'
        row={'stage':stage,'job':job,'event':'release_requested'}
        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        subprocess.run(['scontrol','release',job],check=True)
        meta=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        f.write(json.dumps({**row,'event':'release_readback','metadata':meta})+'\n');f.flush();os.fsync(f.fileno())
        print(stage,job,meta,flush=True)
