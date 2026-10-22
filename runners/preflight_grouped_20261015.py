"""Read-only held-job preflight for all four independent successors."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path('/ceph/aasteine/decision-em-20261013')
def work(stage):
    output=ROOT/f'artifacts/verification/grouped_{stage}_real_preflight_20261015.json'
    result=subprocess.run([sys.executable,'artifacts/operations/grouped_resume_20261015.py','--root',str(ROOT),'--admission',str(ROOT/f'configs/grouped_{stage}_admission_20261015.json'),'--preflight-output',str(output)],cwd=ROOT,text=True,capture_output=True)
    print(stage,result.returncode,result.stdout,result.stderr,flush=True);result.check_returncode()
    assert json.loads(output.read_text())['policy']['policy_sha256']
with ThreadPoolExecutor(4) as pool:list(pool.map(work,['confirmation','schema','core2','core3']))
