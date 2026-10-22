"""Fresh, full raw audit after the four measured slow writers have drained."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

from diagnose_fresh_baseline import audit_rows
from operational_journal import inspect_operations
from resume_schema_once import terminal_steps

JOBS = {'confirmation': '366115', 'schema': '366116', 'core2': '366627', 'core3': '366644'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def locks(root, stage, stack):
    target = root/f'replications/run0{stage[-1]}' if stage.startswith('core') else root
    paths = [(root/'artifacts/operations/repetition_writer.lock', fcntl.LOCK_SH),
             (target/'artifacts/parallel_writer.lock', fcntl.LOCK_EX)] if stage.startswith('core') else [
             (root/f'artifacts/operations/{stage}_recovery.lock', fcntl.LOCK_EX)]
    for path, mode in paths:
        handle = stack.enter_context(path.open('a')); fcntl.flock(handle, mode | fcntl.LOCK_NB)
    return target


def matrix(root, stage):
    target = root/f'replications/run0{stage[-1]}' if stage.startswith('core') else root
    sys.path.insert(0, str(target/'src'))
    import protocol, orchestrate
    orchestrate.verify()
    cells = []
    if stage.startswith('core'):
        from repeated_runs import verify
        verify(root, int(stage[-1]))
        cells = [('quality', m, 'B0', b, s) for m in protocol.MODELS for b in protocol.BENCHMARKS for s in ('prompt','route','test')]
        cells += [('quality', m, v, b, 'prompt') for m in protocol.ABLATION_MODELS for v in ('P1','P2','S1','O1') for b in protocol.BENCHMARKS]
        selection = target/'configs/selected_variants.json'
        if selection.exists():
            saved = read(selection); assert orchestrate.select() == saved
            cells += [('quality', m, d['variant'], b, 'test') for m,d in saved.items() for b in protocol.BENCHMARKS]
        cells += [(n,m,'B0',b,'prompt') for n in ('timing_serial','timing_concurrent8') for m in protocol.MODELS for b in protocol.BENCHMARKS]
        payload = protocol.payload
        parser = protocol.parse_response
        digest = protocol.digest
    elif stage == 'confirmation':
        import expanded_confirmation as ec, analyze, billiger50
        from confirmation_drained import selected_tests
        from authorized_recovery_200_reviewed import review_development
        billiger50.verify(); selected, _ = selected_tests(root, ec, analyze); review_development(root)
        cells = [('quality', m,v,b,'prompt') for m in ec.MODELS for v in ec.PROMPT_VARIANTS+ec.SERIALIZATIONS for b in ec.BENCHMARKS]
        cells += [('quality',m,v,b,'test') for m,v,b in selected]
        cells += [('quality',m,'B0',b,'test') for m,_,b in selected]
        payload, parser, digest = ec.payload, protocol.parse_response, protocol.digest
    else:
        import expanded_schema as ec, analyze, billiger50
        from schema_drained import selected_tests
        billiger50.verify(); selected, _ = selected_tests(root,ec,analyze)
        cells = [('schema',m,v,b,'prompt') for m in ec.MODELS for v in ec.CONDITIONS for b in ec.BENCHMARKS]
        # Pilot cells use their own selected two cases; selected_tests replays ALL pilot raw rows.
        cells += [('schema',m,v,b,'test') for m,v,b in selected]
        cells += [('quality',m,'B0',b,'test') for m,_,b in selected]
        payload, parser, digest = ec.payload, None, ec.wire_hash
    return target, protocol, sorted(set(cells)), payload, parser, digest


def audit(root, stage, require_terminal=True):
    job=JOBS[stage]
    accounting=subprocess.check_output(['sacct','-n','-j',job,'--format=JobID,State,ExitCode,Start,End','-P'],text=True,env={**os.environ,'TZ':'UTC'}) if require_terminal else ''
    if require_terminal: terminal_steps(accounting,job)
    target, protocol, cells, payload, parser, digest = matrix(root,stage)
    run_id=int(stage[-1]) if stage.startswith('core') else 1
    protected={}
    for base in (target/'src',target/'configs'):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts: protected[str(p.relative_to(target))]=sha(p)
    for ns in {c[0] for c in cells} | ({'schema_pilot'} if stage=='schema' else set()):
        for p in (target/'runs'/ns).rglob('*.jsonl'):
            # Confirmation quality namespace also contains other immutable stages.
            if stage=='confirmation' and tuple(p.relative_to(target/'runs').parts[:4]) not in {c[:4] for c in cells}: continue
            protected[str(p.relative_to(target))]=sha(p)
    attempt = (target/'artifacts/core_drained_recovery_v3') if stage.startswith('core') else root/f'artifacts/operations/{stage}_drained_recovery_v1'
    for p in attempt.rglob('*.jsonl'): protected[str(p.relative_to(target))]=sha(p)
    histories=[target/'artifacts/stage_execution.jsonl',target/'artifacts/progress.json'] if stage.startswith('core') else [root/('artifacts/operations/'+stage+'_recovery_'+('365636' if stage=='confirmation' else '365969')+'.jsonl')]
    for p in histories:
        if p.exists(): protected[str(p.relative_to(target))]=sha(p)
    model_files=sorted((attempt/'phases').glob('*.jsonl')) if stage.startswith('core') else [attempt/'models/models.jsonl']
    endings={}
    for p in model_files:
        rr=rows(p)
        if require_terminal:
            assert rr[0]['event']=='start' and rr[-1]['event']=='end'
            assert rr[-1]['models_drained']==len(rr[0]['models'])
            assert sorted(r['model'] for r in rr if r['event']=='model_end')==sorted(rr[0]['models'])
        endings[str(p.relative_to(target))]=rr
    out=[]
    for ns,m,v,b,s in cells:
        inp=target/'data'/b/(s+'.json'); cases=read(inp); protected[str(inp.relative_to(target))]=sha(inp)
        if ns.startswith('timing_'): cases=cases[:20]
        base=target/'runs'/ns/m/v/b/s
        success,failed=rows(Path(str(base)+'.jsonl')),rows(Path(str(base)+'.failures.jsonl'))
        for suffix in ('.jsonl','.failures.jsonl','.operations.jsonl','.executions.jsonl'):
            path=Path(str(base)+suffix);protected[str(path.relative_to(target))]=sha(path) if path.exists() else None
        if ns=='quality' and stage=='schema':
            pp,dd,parse=protocol.payload,protocol.digest,protocol.parse_response
        else: pp,dd,parse=payload,digest,parser
        if stage=='schema' and ns=='schema':
            import expanded_schema as ec
            parse=lambda alias,raw: ec.parse(alias,raw,v)
        adapter=types.SimpleNamespace(payload=lambda c,a,_:pp(c,a,v),digest=dd,parse_response=parse)
        # Input hash is always canonical records SHA even for schema wire hashes.
        if stage=='schema' and ns=='schema':
            adapter.payload=lambda c,a,_:ec.wire(ec.payload(c,a,v))
            adapter.digest=lambda x:hashlib.sha256(x.encode()).hexdigest() if isinstance(x,str) else protocol.digest(x)
        for r in success+failed:
            if r['variant']!=v or (r.get('study_run_id',1)!=run_id): raise ValueError('Foreign run/variant')
        groups=[[{**r,'variant':'B0','study_run_id':run_id} for r in group] for group in (success,failed)]
        checked=audit_rows(cases,*groups,adapter,m,b,run_id)
        op_adapter=types.SimpleNamespace(payload=pp,digest=dd)
        ops=inspect_operations(Path(str(base)+'.operations.jsonl'),cases,failed,op_adapter,m,v,run_id)
        if ops['unknown_outcome_events']: raise ValueError('Unknown operational outcome blocks transition')
        out.append({'namespace':ns,'model':m,'variant':v,'benchmark':b,'split':s,**checked,'operations':ops})
    for rel,h in protected.items():
        p=target/rel
        if (sha(p) if p.exists() else None)!=h: raise ValueError('Concurrent change '+rel)
    return {'stage':stage,'run_id':run_id,'job':job,'sacct_utc':accounting,'all_steps_terminal':require_terminal,
        'model_journals':endings,'cells':out,'protected_sha256_or_absent':protected,
        'totals':{k:sum(c[k] for c in out) for k in ('expected','responses','missing','attempts','unknown_attempts','failure_rows')},
        'known_journaled_usd':math.fsum(c['known_usd'] for c in out),'operations':sum(c['operations']['events'] for c in out),
        'cost_scope':'All retained scoped attempts, not incremental transition charges or invoice',
        'script_sha256':sha(Path(__file__))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=JOBS,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with ExitStack() as stack:
        locks(a.root,a.stage,stack);result=audit(a.root,a.stage)
        with a.out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'stage':a.stage,'totals':result['totals'],'operations':result['operations'],'sha256':sha(a.out)}))
