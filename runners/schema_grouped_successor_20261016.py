"""Offline Run1 schema successor binding after the complete 366695 end audit.

No CLI, admission, funding, scheduler mutation, locks, reports or retry. The
future wrapper must bind all current gates, persist/fsync the prior BEFORE
funding, and retain both writer locks through exclusive report publication.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import types

import schema_drained as component
from diagnose_schema_terminal import audit_schema_rows
from fresh_rules_grouped import prefix
from fresh_schema_selection import own_path
from fresh_supplement_grouped import model_completions
from grouped_inference import build
from grouped_transition import inspector
from rules_drained import bind
from resume_schema_once import terminal_steps

DIAGNOSIS = 'artifacts/operations/grouped_schema_terminal_366695_20261016.json'
PROOF = 'artifacts/verification/grouped_schema_terminal_366695_20261016.json'
PINS = {
    DIAGNOSIS: '972b42efb39cd3888385bfcb8268e1d78f6d515f1626f87ab532ef1bb1d9eefe',
    PROOF: '2764ba9dc32cbd3418fa67d0625ed12c5a49fb3c1619ad208e41a4e5c8ade640',
    'artifacts/operations/audit_grouped_schema_terminal_20261016.py': '326b40aba95b4a0edd4a7febf90a3194e1ec6745082ca6c8322d9c1b5c161a12',
    'artifacts/verification/verify_grouped_schema_terminal_20261016.py': '6aa206475c37051a15729ca7a1f96b73fbb4c9e63f20e0e88b8f9741752736bd',
}
HISTORY = ('365288','365887','365969','366107','366116','366672','366695')
ATTEMPT = 'artifacts/operations/grouped_second_resume_schema_20261016'
SUFFIXES = ('.jsonl','.failures.jsonl','.operations.jsonl','.executions.jsonl')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def receipt_contract(d, proof):
    if (d['job'] != '366695' or d['run_id'] != 1 or not d['all_steps_terminal']
            or proof['job'] != '366695' or proof['run_id'] != 1
            or proof['totals'] != d['totals']
            or len(d['cells']) != 584 or proof['independent_cells'] != 584
            or not all(proof[k] for k in ('raw_input_request_parser_verified',
                'history_and_admission_verified','credit_exactly_reconciled','credit_already_closed'))
            or proof['unresolved_raw_attempts'] or proof['unknown_allowance_gap_usd']
            or d['unresolved_operational_outcomes']):
        raise ValueError('Full independently reconciled schema end evidence required')
    missing = {(c['model'],c['variant'],c['benchmark']):c['missing'] for c in d['cells'] if c['missing']}
    if any(c['namespace'] != 'schema' or c['split'] != 'test' for c in d['cells'] if c['missing']):
        raise ValueError('Only selected test recovery supported')
    plan = proof['remaining_cost_plan']
    if (len(plan['cells']) != len(missing)
            or {(c['model'],c['variant'],c['benchmark']):c['missing'] for c in plan['cells']} != missing
            or sum(missing.values()) != plan['missing'] or plan['missing'] != d['totals']['missing']
            or any(not math.isclose(math.fsum(c[k] for c in plan['cells']),plan[k],rel_tol=0,abs_tol=1e-10)
                   for k in ('known_proxy_usd','historical_unknown_scenario_usd'))
            or any(plan[k] for k in ('quote_or_hard_bound','future_error_forecast','full_financing_guaranteed'))):
        raise ValueError('Actual selected missing matrix/cost scenario required')
    expected = ['start','end','drained_recovery_start','drained_recovery_inference_end',
        'drained_recovery_end','fast_resume_started','fast_resume_credit_retained',
        'fast_resume_failed','grouped_resume_started','grouped_resume_failed']
    if [e['event'] for e in d['stage_history']] != expected:
        raise ValueError('Entire schema recovery history required')
    for start,end,job in ((2,5,'366116'),(5,8,'366672'),(8,10,'366695')):
        if any(e.get('slurm_job_id',e.get('job_id')) != job for e in d['stage_history'][start:end]):
            raise ValueError('Historical job identity changed')
    return plan


def validate_predecessor(root):
    """Rehash finished end evidence; deliberately does not rerun the full audit."""
    for rel,h in PINS.items():
        if sha(own_path(root,rel)) != h: raise ValueError('Pinned evidence changed '+rel)
    d, proof = read(root/DIAGNOSIS), read(root/PROOF)
    if (proof['diagnosis_sha256'] != PINS[DIAGNOSIS]
            or d['script_sha256'] != PINS['artifacts/operations/audit_grouped_schema_terminal_20261016.py']
            or proof['script_sha256'] != PINS['artifacts/verification/verify_grouped_schema_terminal_20261016.py']):
        raise ValueError('Auditor provenance changed')
    plan = receipt_contract(d,proof)
    protected = {**d['protected_sha256_or_absent'], **d['historical_admission_policy_sha256'],
                 **proof['additional_gate_failure_provenance_sha256']}
    for rel,h in protected.items():
        p=own_path(root,rel)
        if (sha(p) if p.exists() else None) != h: raise ValueError('Audited source/cache/history changed '+rel)
    observed={str(p.relative_to(root)) for ns in ('schema','schema_pilot') for p in (root/'runs'/ns).rglob('*.jsonl')}
    if not observed <= protected.keys(): raise ValueError('Additional schema journal')
    if read(root/'configs/expanded_schema_selection.json') != proof['independent_selection']:
        raise ValueError('Own development selection changed')
    ledger=read(root/'artifacts/budget.json')['inference_credits']['fast-schema-366695']
    if ledger != d['credit_reconciliation_read_only']['ledger'] or ledger['status'] != 'closed':
        raise ValueError('Exactly closed predecessor credit required')
    if own_path(root,ATTEMPT).exists() or (root/'artifacts/expanded_schema_complete.json').exists():
        raise ValueError('Exclusive successor attempt or incomplete schema required')
    accounting={}
    for job in HISTORY:
        raw=subprocess.check_output(['sacct','-n','-j',job,'--format=JobID,State,ExitCode,End','-P'],text=True,env={**os.environ,'TZ':'UTC'})
        terminal_steps(raw,job); accounting[job]=raw
    return {'protected_entries':len(protected),'historical_steps':accounting,'remaining_cost_plan':plan,
            'admission_granted':False,'full_raw_replay_repeated':False}


def layout(root,cells):
    paths={f'runs/schema/{m}/{v}/{b}/test{s}' for m,v,b in cells for s in SUFFIXES}
    observed={str(p.relative_to(root)) for p in (root/'runs/schema').rglob('test*.jsonl')}
    if not observed <= paths: raise ValueError('Unreviewed selected test journal')
    for p in paths: own_path(root,p)
    return sorted(paths)


def inventory(root,schema,cells,inspect):
    checked=[]
    bound=types.SimpleNamespace(payload=schema.payload,digest=schema.wire_hash)
    for m,v,b in cells:
        folder=root/'runs/schema'/m/v/b
        def rows(s):
            p=folder/('test'+s); prefix(p)
            return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        cases=read(root/'data'/b/'test.json'); successes,failures=rows('.jsonl'),rows('.failures.jsonl')
        c=audit_schema_rows(cases,successes,failures,schema,m,v,b)
        ops=inspect(folder/'test.operations.jsonl',cases,failures,bound,m,v,1)
        if ops['unknown_outcome_events']: raise ValueError('Unknown operations retain credit')
        checked.append(c)
    return checked


def capture_prior(root,schema,analysis,inspect):
    """Full pilot/dev selection replay and existing selected-test raw replay."""
    if (root/'artifacts/expanded_schema_complete.json').exists() or own_path(root,ATTEMPT).exists():
        raise ValueError('Exclusive successor attempt required')
    comp_selected=bind(component.selected_tests,inspect_operations=inspect)
    cells,protected=comp_selected(root,schema,analysis)
    paths=layout(root,cells); journals={p:prefix(root/p) for p in paths}
    # Test inputs and all completed pilot/development journals remain immutable.
    for p in (root/'data').rglob('*.json'):
        protected[str(p.relative_to(root))]=sha(p)
    for ns in ('schema','schema_pilot'):
        for p in (root/'runs'/ns).rglob('*.jsonl'):
            if ns=='schema_pilot' or p.name.startswith('prompt'):
                protected[str(p.relative_to(root))]=sha(p)
    checked=inventory(root,schema,cells,inspect)
    if (journals != {p:prefix(root/p) for p in paths}
            or any(sha(root/p)!=h for p,h in protected.items())): raise ValueError('Prior changed during replay')
    return dict(stage='schema',run_id=1,root=str(root),journal_dir=ATTEMPT+'/models',
        cells=cells,protected_sha256=protected,journals=journals,
        prior_responses=sum(c['responses'] for c in checked),admission_granted=False)


def prior_credit(root,path,digest,credit):
    if sha(own_path(root,path)) != digest: raise ValueError('Durable prior changed')
    prior=read(path); row=read(root/'artifacts/budget.json')['inference_credits'][credit.identity]
    expected={'prior_sha256':digest,'run_id':1,'stage':'schema','journal_dir':str(root/ATTEMPT/'models')}
    if (credit.root.resolve()!=root.resolve() or credit.chunk!=8 or credit.closed
            or row['status']!='active' or any(row['provenance'].get(k)!=v for k,v in expected.items())
            or prior['root']!=str(root) or prior['run_id']!=1 or prior['stage']!='schema'
            or prior['journal_dir']!=ATTEMPT+'/models'):
        raise ValueError('Credit must bind exact durable prior/run/attempt')
    for rel,h in prior['protected_sha256'].items():
        if sha(own_path(root,rel))!=h: raise ValueError('Prior source changed')
    if set(prior['journals'])!=set(layout(root,prior['cells'])): raise ValueError('Prior layout changed')
    return prior


def private_component(root,credit,prior_path,prior_sha,inspect=None):
    """Caller must persist prior and fund under all locks/gates; no funding here."""
    inspect=inspector(root) if inspect is None else inspect
    def builder(inference,rid):
        if rid!=1: raise ValueError('Run1 schema only')
        isolated=types.SimpleNamespace(**{**vars(inference),'account':credit.account,
            '_on_journal_failure':lambda:setattr(credit,'journal_io_failed',True)})
        return build(isolated,rid)
    selected=bind(component.selected_tests,inspect_operations=inspect)
    schema_builder=bind(component.build_schema,build=builder)
    execute=bind(component.run_selected_tests,selected_tests=selected,build_schema=schema_builder,inspect_operations=inspect)
    def run(schema,analysis,inference):
        if credit.attempts or credit.settled: raise ValueError('Fresh credit required')
        prior=prior_credit(root,prior_path,prior_sha,credit)
        # Attempt parent belongs to caller, models directory is exclusive to this invocation.
        if (root/ATTEMPT/'models').exists(): raise ValueError('Attempt already used')
        if any(prefix(root/p)!=v for p,v in prior['journals'].items()): raise ValueError('Prior cache changed')
        try: return execute(root,schema,analysis,inference,root/ATTEMPT/'models')
        except OSError:
            credit.journal_io_failed=True
            raise
    return run


def reconcile(root,schema,protocol,prior_path,prior_sha,credit,inspect=None):
    if getattr(credit,'journal_io_failed',False): raise ValueError('Unconfirmed storage; retain credit')
    inspect=inspector(root) if inspect is None else inspect
    prior=prior_credit(root,prior_path,prior_sha,credit)
    ends=model_completions(root/ATTEMPT/'models',protocol)
    observed={p:prefix(root/p) for p in prior['journals']}
    checked=inventory(root,schema,prior['cells'],inspect)
    known=[]; attempts=unknown=0
    for rel,old in prior['journals'].items():
        p=own_path(root,rel); raw=p.read_bytes() if p.exists() else b''
        if (len(raw)<old['bytes'] or (old['exists'] and not p.exists())
                or hashlib.sha256(raw[:old['bytes']]).hexdigest()!=old['sha256']):
            raise ValueError('Prior prefix changed')
        for line in raw[old['bytes']:].splitlines():
            row=json.loads(line)
            if rel.endswith('.executions.jsonl'): continue
            if type(row.get('study_run_id')) is not int or row['study_run_id']!=1:
                raise ValueError('New records require strict Run1')
            if rel.endswith('.operations.jsonl'):
                if row['operational_stop']['outcome_unknown']: raise ValueError('Unknown operation; retain credit')
            else:
                for a in row['attempts']:
                    attempts+=1
                    if a['cost'] is None: unknown+=1
                    else: known.append(a['cost'])
    charged=math.fsum(known)+.05*unknown
    if (credit.outstanding or credit.attempts!=credit.settled or attempts!=credit.settled
            or unknown!=credit.unknown or not math.isclose(charged,credit.charged,rel_tol=0,abs_tol=1e-8)):
        raise ValueError('Durable incremental attempts do not reconcile')
    if observed!={p:prefix(root/p) for p in prior['journals']} or ends!=model_completions(root/ATTEMPT/'models',protocol):
        raise ValueError('Journals changed during reconciliation')
    prior_credit(root,prior_path,prior_sha,credit)
    credit.checkpoint(close=True)
    return dict(attempts=attempts,unknown=unknown,charged=charged,
        complete=ends['status']=='complete' and all(c['complete'] for c in checked),
        prior_responses_excluded=prior['prior_responses'],admission_granted=False,report_written=False)
