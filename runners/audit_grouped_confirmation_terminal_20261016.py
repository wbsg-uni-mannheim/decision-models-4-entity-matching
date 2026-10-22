"""Read-only full terminal audit of confirmation job 366694, including every recovery.

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

JOB = '366694'
ATTEMPT = 'artifacts/operations/grouped_resume_confirmation_20261015'
ADMISSION = 'configs/grouped_confirmation_admission_20261015.json'


def history(root, target):
    path = root/'artifacts/operations/confirmation_recovery_365636.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    events = [json.loads(line) for line in lines]
    expected = ['start','end','cooldown','start','end','cooldown','start','end','stopped_for_diagnosis',
        'drained_recovery_start','drained_recovery_inference_end','drained_recovery_end',
        'fast_resume_started','fast_resume_credit_retained','fast_resume_failed',
        'grouped_resume_started','grouped_resume_credit_retained','grouped_resume_failed']
    assert [e['event'] for e in events] == expected
    assert hashlib.sha256(b''.join(lines[:9])).hexdigest() == '124467414c0cac1f86b2b62980e984765a9d634ae24e44ef0c7c914d17496f22'
    bound = {str(path.relative_to(root)):base.sha(path)}
    model_ends = {}
    configs = {base.sha(p):str(p.relative_to(root)) for p in (root/'configs').glob('*.json')}
    provenance = {}
    for start, stop, folder, job in [(9,12,'confirmation_drained_recovery_v1','366115'),
            (12,15,'fast_resume_confirmation_20261015','366671'),
            (15,18,'grouped_resume_confirmation_20261015','366694')]:
        group = events[start:stop]
        assert all(e.get('slurm_job_id',e.get('job_id')) == job for e in group)
        prefix_key = 'prior_journal_sha256' if start == 9 else 'prior_stage_journal_sha256'
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
    return events,bound,model_ends,provenance


def audit(root):
    target = root
    with ExitStack() as stack:
        base.locks(root, 'confirmation', stack)
        source = Path(base.__file__).read_text()
        if base.sha(Path(base.__file__)) != '22f1f3fc9e61972bbe89fdddacd2273cb772840c814aad10dfcfd78ea4fa2a16':
            raise ValueError('Pinned raw auditor changed')
        unknown_guard = "if ops['unknown_outcome_events']: raise ValueError('Unknown operational outcome blocks transition')"
        if source.count(unknown_guard) != 1:
            raise ValueError('Unknown guard drift')
        # Descriptive audit retains every unknown in the output; this module
        # has no admission/reconciliation entry and must never authorize retry.
        source = source.replace(unknown_guard, 'assert ops["unknown_outcome_events"] >= 0')
        old = "attempt = (target/'artifacts/core_drained_recovery_v3') if stage.startswith('core') else root/f'artifacts/operations/{stage}_drained_recovery_v1'"
        if source.count(old) != 1:
            raise ValueError('Audit source drift')
        module = types.ModuleType('terminal_grouped_confirmation'); module.__file__ = __file__
        exec(compile(source.replace(old, 'attempt = target/'+repr(ATTEMPT)), '<current terminal audit>', 'exec'), module.__dict__)
        module.JOBS = {'confirmation': JOB}; module.inspect_operations = inspector(root)
        result = module.audit(root, 'confirmation', True)
        events, protected, ends, provenance = history(root, target)
        result['stage_history'] = events
        result['all_historical_model_journals'] = ends
        result['protected_sha256_or_absent'].update(protected)
        result['historical_admission_policy_sha256'] = provenance
        result['controlled_stop_proof_sha256'] = base.sha(root/PROOF)
        result['admission_sha256'] = base.sha(root/ADMISSION)
        final = base.rows(target/ATTEMPT/'execution.jsonl')[-1]
        credit = base.read(root/'artifacts/budget.json')['inference_credits']['fast-confirmation-'+JOB]
        observed = new_attempt_costs(target, base.read(target/ATTEMPT/'prior_journals.json'), {'quality'})
        summary = final['credit']
        unresolved = sum(c['operations']['unknown_outcome_events'] for c in result['cells'])
        if credit['status'] != 'active' or summary['closed'] or summary['outstanding']:
            raise ValueError('Expected retained zero-outstanding active credit')
        if summary['attempts'] != summary['settled']:
            raise ValueError('Credit has unsettled attempts')
        delta = {'attempts':summary['settled']-observed['attempts'],
                 'unknown':summary['unknown']-observed['unknown'],
                 'charged':summary['charged']-observed['charged']}
        # Known transport exception has no retained raw attempt. Arithmetic
        # explains the conservative allowance but cannot reconstruct its outcome.
        if unresolved != 4 or delta['attempts'] != 4 or delta['unknown'] != 4 or not math.isclose(delta['charged'],.20,abs_tol=1e-8):
            raise ValueError('Additional unexplained raw/credit discrepancy')
        result['credit_reconciliation_read_only'] = {'ledger':credit,'summary':summary,'raw':observed,
            'unresolved_raw_gap':delta,'exact_durable_reconciliation':False,'hold_must_remain':True}
        result['unresolved_operational_outcomes'] = unresolved
        result['recovery_admitted'] = False
        for rel, digest in result['protected_sha256_or_absent'].items():
            p = target/rel
            if (base.sha(p) if p.exists() else None) != digest:
                raise ValueError('Concurrent source/cache/history change')
        result['script_sha256'] = base.sha(Path(__file__))
        result['no_admission_or_credit_mutation'] = True
        result['historical_unjournaled_outcomes_resolved'] = False
        result['historical_run_id_boundary'] = 'Original Run1 records predate explicit study_run_id; frozen legacy default 1 retained. Current attempt records must explicitly identify Run1.'
        prior = base.read(target/ATTEMPT/'prior_journals.json')
        for p in (target/'runs/quality').rglob('*.jsonl'):
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
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'totals': result['totals'], 'cells': len(result['cells']),
                      'credit': result['credit_reconciliation_read_only']['raw'], 'sha256': base.sha(args.out)}))
