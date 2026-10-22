from datetime import datetime,timezone
from pathlib import Path
import json,os,subprocess,fcntl
import resume_fresh_core_run02_third_drained as w3
r=Path('/ceph/aasteine/decision-em-20261013')
a=r/'configs/core_run02_third_admission_20261015.json';s=w3.read(a);job=s['slurm_job_id'];assert job=='366627'
w=w3.make_wrapper(r)
with (r/'artifacts/operations/repetition_writer.lock').open('a') as shared,(r/'replications/run02/artifacts/parallel_writer.lock').open('a') as exclusive:
 fcntl.flock(shared,fcntl.LOCK_SH|fcntl.LOCK_NB);fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
 checked=w3.budget.preflight(r,'core',2,s,w)
 assert not (r/'replications/run02'/w3.ATTEMPT).exists()
 for rel,h in s['preserved_active_admissions_and_policies_sha256'].items():assert w3.budget.sha(r/rel)==h
 meta=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
 fields=dict(t.split('=',1) for t in meta.split() if '=' in t)
 assert fields['JobState']=='PENDING' and fields['Priority']=='0' and fields['StartTime']=='Unknown'
 proof=r/'artifacts/verification/core_run02_third_prerelease_20261015.json'
 with proof.open('x') as f:json.dump(dict(read_only=True,checks=checked,admission_sha256=w3.budget.sha(a)),f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 with (r/'artifacts/operations/core_run02_third_release_20261015.jsonl').open('x') as f:
  def log(event,**kw):
   f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event=event,job=job,**kw))+'\n');f.flush();os.fsync(f.fileno())
  log('release_start',admission_sha256=w3.budget.sha(a),preflight_sha256=w3.budget.sha(proof),budget=w3.read(r/'artifacts/budget.json'))
  subprocess.check_call(['scontrol','release',job])
  log('release_returned',metadata=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True))
 print(json.dumps(dict(job=job,released=True,prerelease_sha256=w3.budget.sha(proof))))
