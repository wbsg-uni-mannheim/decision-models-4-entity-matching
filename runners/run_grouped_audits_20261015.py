"""Wait for terminal steps, reconcile, then audit each independent scope."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from grouped_transition import terminal,reconcile
ROOT=Path('/ceph/aasteine/decision-em-20261013')
for i in range(40):
    try:
        terminal(ROOT)
        break
    except ValueError:
        if i==0: print('Waiting for final predecessor accounting/terminal step',flush=True)
        time.sleep(15)
else:raise RuntimeError('Predecessor not terminal; no reconciliation or admission')
proof=reconcile(ROOT)
print(json.dumps({'event':'reconciled','budget':proof['budget_after'],'controlled_stops':len(proof['resolved_reserve_events'])}),flush=True)
def work(stage):
    code="from pathlib import Path; from contextlib import ExitStack; from grouped_transition import audit,save; from fast_transition_audit import locks; r=Path("+repr(str(ROOT))+"); s="+repr(stage)+"\nwith ExitStack() as stack:\n locks(r,s,stack);d=audit(r,s);save(r/f'artifacts/operations/grouped_audit_{s}_20261015.json',d); print(d['totals'])"
    a=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True)
    print(stage,'audit',a.returncode,a.stdout,a.stderr,flush=True);a.check_returncode()
    a=subprocess.run([sys.executable,'artifacts/verification/verify_grouped_transition_20261015.py','--root',str(ROOT),'--stage',stage],cwd=ROOT,text=True,capture_output=True)
    print(stage,'independent',a.returncode,a.stdout,a.stderr,flush=True);a.check_returncode()
with ThreadPoolExecutor(4) as pool:list(pool.map(work,['confirmation','schema','core2','core3']))
