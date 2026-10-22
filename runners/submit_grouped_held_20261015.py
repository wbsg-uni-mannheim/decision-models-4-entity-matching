from pathlib import Path
import subprocess,json,os,sys,hashlib
root=Path('/ceph/aasteine/decision-em-20261013');sys.path.insert(0,str(root/'artifacts/operations'))
from resume_schema_once import terminal_steps
for j in ['366671','366672','366680','366681']:
 terminal_steps(subprocess.check_output(['sacct','-n','-j',j,'--format=JobID,State,ExitCode,End','-P'],text=True),j)
p=root/'artifacts/operations/grouped_submission_20261015.jsonl'
with p.open('x') as journal:
 jobs={}
 for stage in ['confirmation','schema','core2','core3']:
  script=root/f'artifacts/operations/grouped_{stage}_20261015.sbatch'
  with script.open('x') as f:f.write(f'#!/bin/bash\nset -euo pipefail\ncd {root}\nexport PYTHONPATH=artifacts/operations\nexec .venv/bin/python -u artifacts/operations/grouped_resume_20261015.py --root {root} --admission {root}/configs/grouped_{stage}_admission_20261015.json\n')
  journal.write(json.dumps({'stage':stage,'event':'submit_start','script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()})+'\n');journal.flush();os.fsync(journal.fileno())
  j=subprocess.check_output(['sbatch','--parsable','--hold','--no-requeue','--partition=cpu','--cpus-per-task=2','--mem=8G','--time=2-00:00:00',f'--job-name=decision-em-grouped-{stage}','--chdir='+str(root),'--output='+str(root/f'logs/grouped-{stage}-%j.log'),str(script)],text=True).strip().split(';')[0]
  assert j.isdigit();jobs[stage]=j
  journal.write(json.dumps({'stage':stage,'event':'submitted_held','job':j})+'\n');journal.flush();os.fsync(journal.fileno())
 with (root/'artifacts/operations/grouped_jobs_20261015.json').open('x') as f:json.dump(jobs,f,indent=2)
 print(json.dumps(jobs))
