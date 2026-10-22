"""Read-only full terminal audit of schema job 366695, including every recovery.

Uses the frozen request/raw audit, but binds the current grouped attempt rather
than the historic transition. Does not close credits or authorize execution.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import types

import fast_transition_audit as base
from grouped_transition import inspector, PROOF
from grouped_resume_20261015 import new_attempt_costs

JOB = '366695'
ATTEMPT = 'artifacts/operations/grouped_resume_schema_20261015'
ADMISSION = 'configs/grouped_schema_admission_20261015.json'


def history(root, target):
    path = root/'artifacts/operations/schema_recovery_365969.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    events = [json.loads(line) for line in lines]
    expected = ['start','end','drained_recovery_start','drained_recovery_inference_end',
        'drained_recovery_end','fast_resume_started','fast_resume_credit_retained','fast_resume_failed',
        'grouped_resume_started','grouped_resume_failed']
    assert [e['event'] for e in events] == expected
    assert hashlib.sha256(b''.join(lines[:2])).hexdigest() == '126a024a544c32e222fad804ed510dbcbd58098f960066d1a0aa1d780e74c705'
    bound = {str(path.relative_to(root)):base.sha(path)}
    model_ends = {}
    configs = {base.sha(p):str(p.relative_to(root)) for p in (root/'configs').glob('*.json')}
    provenance = {}
    for start, stop, folder, job in [(2,5,'schema_drained_recovery_v1','366116'),
            (5,8,'fast_resume_schema_20261015','366672'),
            (8,10,'grouped_resume_schema_20261015','366695')]:
        group = events[start:stop]
        assert all(e.get('slurm_job_id',e.get('job_id')) == job for e in group)
        prefix_key = 'prior_journal_sha256' if start == 2 else 'prior_stage_journal_sha256'
        assert hashlib.sha256(b''.join(lines[:start])).hexdigest() == group[0][prefix_key]
        folder = root/'artifacts/operations'/folder
        assert base.rows(folder/'execution.jsonl') == group
        for p in folder.rglob('*'):
            if p.is_file(): bound[str(p.relative_to(root))] = base.sha(p)
        for p in (folder/'models').glob('*.jsonl'):
            rr=base.rows(p); ends=[e for e in rr if e['event']=='model_end']
            assert rr[0]['event']=='start' and rr[-1]['event']=='end'
            assert sorted(e['model'] for e in ends)==sorted(rr[0]['models'])
            assert rr[-1]['models_drained']==len(ends)==7
            model_ends[str(p.relative_to(root))]=rr
        prior=folder/'prior_journals.json'
        if prior.exists():
            for rel,entry in base.read(prior).items():
                with (root/rel).open('rb') as f: prefix=f.read(entry['bytes'])
                assert hashlib.sha256(prefix).hexdigest()==entry['sha256']
        for event in group:
            for key in ('admission_sha256','parallel_policy_sha256'):
                value=event[key]; assert value in configs
                provenance[configs[value]]=value
    original = root/'artifacts/operations/schema_recovery_365887.jsonl'
    rr = base.rows(original)
    assert [e['event'] for e in rr] == ['start','end'] and rr[-1]['returncode'] == 1
    assert base.sha(original) == '6a00be1b13fd266bcaf0eda7c04ce9cc87eef890e5d14b72271d847c880dcb5f'
    bound[str(original.relative_to(root))] = base.sha(original)
    # These earlier admissions identify the original runner and first recovery.
    for name in ('artifacts/operations/schema_admission_365887.json','configs/schema_admission_365969.json','configs/schema_resume_execution.json','configs/schema_resume_prepared_20261014T2103Z.json','artifacts/verification/schema_recovery_verification_20261014.json','artifacts/verification/schema_recovery_365969_verification_20261014.json'):
        p=root/name
        if not p.exists(): raise ValueError('Missing historical provenance '+name)
        provenance[str(p.relative_to(root))]=base.sha(p)
    return events,bound,model_ends,provenance


def audit(root):
    target = root
    with ExitStack() as stack:
        base.locks(root, 'schema', stack)
        source = Path(base.__file__).read_text()
        if base.sha(Path(base.__file__)) != '22f1f3fc9e61972bbe89fdddacd2273cb772840c814aad10dfcfd78ea4fa2a16':
            raise ValueError('Pinned raw auditor changed')
        old = "attempt = (target/'artifacts/core_drained_recovery_v3') if stage.startswith('core') else root/f'artifacts/operations/{stage}_drained_recovery_v1'"
        if source.count(old) != 1:
            raise ValueError('Audit source drift')
        module = types.ModuleType('terminal_grouped_schema'); module.__file__ = __file__
        exec(compile(source.replace(old, 'attempt = target/'+repr(ATTEMPT)).replace("stage=='schema' and ns=='schema'", "stage=='schema' and ns in ('schema','schema_pilot')"), '<current terminal audit>', 'exec'), module.__dict__)
        original_matrix = module.matrix
        def all_cells(root, stage):
            target, protocol, cells, payload, parser, digest = original_matrix(root, stage)
            import expanded_schema as schema
            manifest = schema.verify()
            cells += [('schema_pilot',m,v,b,'schema_pilot') for m in schema.MODELS
                      for v in schema.CONDITIONS for b in manifest['pilot']['families']]
            return target, protocol, sorted(set(cells)), payload, parser, digest
        module.matrix = all_cells
        module.JOBS = {'schema': JOB}; module.inspect_operations = inspector(root)
        result = module.audit(root, 'schema', True)
        events, protected, ends, provenance = history(root, target)
        result['stage_history'] = events
        result['all_historical_model_journals'] = ends
        result['protected_sha256_or_absent'].update(protected)
        result['historical_admission_policy_sha256'] = provenance
        result['controlled_stop_proof_sha256'] = base.sha(root/PROOF)
        result['admission_sha256'] = base.sha(root/ADMISSION)
        final = base.rows(target/ATTEMPT/'execution.jsonl')[-1]
        credit = base.read(root/'artifacts/budget.json')['inference_credits']['fast-schema-'+JOB]
        observed = new_attempt_costs(target, base.read(target/ATTEMPT/'prior_journals.json'), {'schema'})
        summary = final['credit']
        unresolved = sum(c['operations']['unknown_outcome_events'] for c in result['cells'])
        if credit['status'] != 'closed' or not summary['closed'] or summary['outstanding']:
            raise ValueError('Expected already closed credit')
        if summary['attempts'] != summary['settled'] or unresolved:
            raise ValueError('Unsettled attempts or unknown operations')
        for k in ('attempts','unknown'):
            assert observed[k] == summary[k] == credit[k]
        assert credit['settled'] == summary['settled']
        assert credit['funded'] == summary['funded']
        assert math.isclose(observed['charged'],summary['charged'],abs_tol=1e-8)
        assert math.isclose(observed['charged'],credit['charged'],abs_tol=1e-8)
        result['credit_reconciliation_read_only'] = {'ledger':credit,'summary':summary,'raw':observed,
            'unresolved_raw_gap':{'attempts':0,'unknown':0,'charged':0},
            'exact_durable_reconciliation':True,'already_closed_before_audit':True}
        result['unresolved_operational_outcomes'] = unresolved
        result['recovery_admitted'] = False
        for rel, digest in result['protected_sha256_or_absent'].items():
            p = target/rel
            if (base.sha(p) if p.exists() else None) != digest:
                raise ValueError('Concurrent source/cache/history change')
        import os, subprocess
        jobs = ['365288','365887','365969','366116','366672','366695']
        accounting = subprocess.check_output(['sacct','-n','-j',','.join(jobs),
            '--format=JobIDRaw,State,ExitCode,Start,End','-P'],text=True,env={**os.environ,'TZ':'UTC'})
        for job in jobs: base.terminal_steps(accounting,job)
        result['all_historical_steps_sacct_utc'] = accounting
        result['historical_jobs'] = jobs
        result['script_sha256'] = base.sha(Path(__file__))
        result['no_admission_or_credit_mutation'] = True
        result['historical_unjournaled_outcomes_resolved'] = False
        result['historical_run_id_boundary'] = 'Original Run1 records predate explicit study_run_id; frozen legacy default 1 retained. Current attempt records must explicitly identify Run1.'
        prior = base.read(target/ATTEMPT/'prior_journals.json')
        for p in (target/'runs/schema').rglob('*.jsonl'):
            entry=prior.get(str(p.relative_to(target)),{'bytes':0})
            with p.open('rb') as f:
                f.seek(entry['bytes']); tail=f.read()
            if p.name.endswith('.executions.jsonl'): continue
            for line in tail.splitlines():
                row=json.loads(line)
                assert type(row.get('study_run_id')) is int and row['study_run_id']==1
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n'); f.flush(); __import__('os').fsync(f.fileno())
    print(json.dumps({'totals': result['totals'], 'cells': len(result['cells']),
                      'credit': result['credit_reconciliation_read_only']['raw'], 'sha256': base.sha(args.out)}))
