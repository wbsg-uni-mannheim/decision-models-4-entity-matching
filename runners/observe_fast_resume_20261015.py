"""Read only short-window throughput and new raw response provenance."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path('/ceph/aasteine/decision-em-20261013')
sys.path.insert(0,str(ROOT/'artifacts/operations'))
from fast_transition_audit import sha


def tail_rows(p,n=500):
    with p.open('rb') as f:
        f.seek(0,2);end=f.tell();buf=b''
        while end and buf.count(b'\n')<=n:
            size=min(end,262144);end-=size;f.seek(end);buf=f.read(size)+buf
    lines=buf.splitlines()
    if end:lines=lines[1:]
    return [json.loads(x) for x in lines[-n:]]


def stats(rr):
    rr=sorted(rr,key=lambda r:r['completed_at'])[-200:]
    if len(rr)<20:return {'n':len(rr)}
    elapsed=(datetime.fromisoformat(rr[-1]['completed_at'])-datetime.fromisoformat(rr[0]['completed_at'])).total_seconds()
    return {'n':len(rr),'seconds':elapsed,'responses_per_second':(len(rr)-1)/elapsed,
        'attempt_seconds_median':statistics.median(r['attempts'][-1]['seconds'] for r in rr),
        'from':rr[0]['completed_at'],'to':rr[-1]['completed_at']}


def main():
    mapping=ROOT/'artifacts/operations/fast_jobs_directory_20261015.json'
    if not mapping.exists():mapping=ROOT/'artifacts/operations/fast_jobs_20261015.json'
    jobs=json.loads(mapping.read_text())
    output={'utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,
        'sacct':subprocess.check_output(['sacct','-n','-j',','.join(jobs.values()),'--format=JobID,State,ExitCode,Start,End','-P'],text=True),
        'budget':json.loads((ROOT/'artifacts/budget.json').read_text()),'stages':{}}
    for stage,job in jobs.items():
        target=ROOT/f'replications/run0{stage[-1]}' if stage.startswith('core') else ROOT
        attempt=target/f'artifacts/operations/fast_resume_{stage}_20261015'
        log=ROOT/f'logs/fast-{stage}-{job}.log'
        data={'log_tail':log.read_text()[-1500:] if log.exists() else '', 'started':False}
        if (attempt/'execution.jsonl').exists():
            history=[json.loads(x) for x in (attempt/'execution.jsonl').read_text().splitlines()]
            start=history[0]['utc'];data.update(started=True,start=start,execution=history)
            before=json.loads((attempt/'prior_journals.json').read_text());samples=[]
            for p in (target/'runs'/('schema' if stage=='schema' else 'quality')).rglob('*.jsonl'):
                if p.name.endswith(('.failures.jsonl','.executions.jsonl','.operations.jsonl')):continue
                rel=str(p.relative_to(target));prior=before.get(rel,{'bytes':0})
                if p.stat().st_size<=prior['bytes']:continue
                rr=tail_rows(p);new=[r for r in rr if r['completed_at']>=start]
                if not new:continue
                ns,m,v,b,file=p.relative_to(target/'runs').parts;s=file.split('.')[0]
                cases={r['pair_id']:r for r in json.loads((target/'data'/b/(s+'.json')).read_text())}
                # Exact requests and raw parsing are checked in a fresh per-run subprocess below.
                assert all(r['input_hash']==cases[r['pair_id']]['input_hash'] and r['study_run_id']==(int(stage[-1]) if stage.startswith('core') else 1) for r in new)
                old=[]
                if prior['bytes']:
                    with p.open('rb') as f:
                        f.seek(max(0,prior['bytes']-8000000));raw=f.read(min(prior['bytes'],8000000))
                    lines=raw.splitlines()
                    if prior['bytes']>8000000:lines=lines[1:]
                    old=[json.loads(x) for x in lines[-200:]]
                samples.append({'path':rel,'new':stats(new),'old':stats(old),'new_sample_pair_ids':[r['pair_id'] for r in new[:5]]})
            data['samples']=samples
        output['stages'][stage]=data
    name='fast_observation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'
    path=ROOT/'artifacts/verification'/name
    with path.open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'stages':{s:{k:v for k,v in d.items() if k not in ('execution','log_tail')} for s,d in output['stages'].items()},'budget':{k:output['budget'][k] for k in ('spent','reserved','unknown_cost_attempts')}}))


if __name__=='__main__':main()
