"""Read-only full terminal audit of Run3 job 366697, including every recovery.

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

JOB = '366697'
ATTEMPT = 'artifacts/operations/grouped_resume_core3_20261015'
ADMISSION = 'configs/grouped_core3_admission_20261015.json'


def history(root, target):
    path = target/'artifacts/stage_execution.jsonl'
    lines = path.read_bytes().splitlines(keepends=True)
    events = [json.loads(line) for line in lines]
    expected = [('smoke', '365894', ['started', 'complete']),
                ('core', '365895', ['started', 'failed']),
                ('core', '366118', ['started', 'failed']),
                ('core', '366520', ['started', 'failed']),
                ('core', '366644', ['started', 'failed']),
                ('core', '366681', ['started', 'credit_retained', 'failed']),
                ('core', JOB, ['started', 'failed'])]
    flat = [(stage, job, status) for stage, job, states in expected for status in states]
    if [(e['stage'], e['slurm_job_id'], e['status']) for e in events] != flat:
        raise ValueError('Unexpected full initial/v1/v2/v3/fast/grouped history')
    for i, event in enumerate(events):
        if type(event['run_id']) is not int or event['run_id'] != 3:
            raise ValueError('Foreign run history')
        if event['status'] == 'started' and 'prior_stage_journal_sha256' in event:
            if hashlib.sha256(b''.join(lines[:i])).hexdigest() != event['prior_stage_journal_sha256']:
                raise ValueError('Historical stage prefix changed')
    attempts = ['artifacts/core_drained_recovery_v1', 'artifacts/core_drained_recovery_v2',
                'artifacts/core_drained_recovery_v3', 'artifacts/operations/fast_resume_core3_20261015', ATTEMPT]
    bound = {}
    model_ends = {}
    for attempt in attempts:
        folder = target/attempt
        execution = base.rows(folder/'execution.jsonl')
        if not execution or execution != [e for e in events if e['slurm_job_id'] == execution[0]['slurm_job_id']]:
            raise ValueError('Attempt and stage journal disagree')
        for p in folder.rglob('*'):
            if p.is_file():
                bound[str(p.relative_to(target))] = base.sha(p)
        for p in (folder/'phases').glob('*.jsonl'):
            rows = base.rows(p)
            ends = [r for r in rows if r['event'] == 'model_end']
            if (rows[0]['event'] != 'start' or rows[-1]['event'] != 'end'
                    or sorted(r['model'] for r in ends) != sorted(rows[0]['models'])
                    or rows[-1]['models_drained'] != len(ends)):
                raise ValueError('Incomplete historical model endings')
            model_ends[str(p.relative_to(target))] = rows
        prior = folder/'prior_journals.json'
        if prior.exists():
            for rel, entry in base.read(prior).items():
                with (target/rel).open('rb') as f:
                    prefix = f.read(entry['bytes'])
                if hashlib.sha256(prefix).hexdigest() != entry['sha256']:
                    raise ValueError('Prior raw journal prefix changed')
    # Bind every admission/policy actually named by history to its immutable file.
    configs = {base.sha(p): str(p.relative_to(root)) for p in (root/'configs').glob('*.json')}
    provenance = {}
    for event in events:
        for key in ('admission_sha256', 'parallel_policy_sha256', 'launcher_sha256'):
            value = event.get(key)
            if value and key != 'launcher_sha256':
                if value not in configs:
                    # The initial mutable policy was superseded on Oct14. Its
                    # exact digest remains in both immutable initial admissions;
                    # do not pretend the original policy bytes still exist.
                    if (key != 'parallel_policy_sha256' or event['slurm_job_id'] not in ('365894', '365895')
                            or value != 'abca3ad7b5a4ea0835aef50a1877402bcfdfdf05e5bc2f090b969a8bb27a3d3d'):
                        raise ValueError('Historical admission or policy unavailable')
                    name = 'smoke' if event['slurm_job_id'] == '365894' else 'core'
                    p = root/f'configs/run03_{name}_admission_20261014.json'
                    admission = base.read(p)
                    if admission['parallel_policy_sha256'] != value or admission['slurm_job_id'] != event['slurm_job_id']:
                        raise ValueError('Initial policy digest lacks admission provenance')
                    provenance[str(p.relative_to(root))] = base.sha(p)
                    continue
                provenance[configs[value]] = value
    return events, bound, model_ends, provenance


def audit(root):
    target = root/'replications/run03'
    with ExitStack() as stack:
        base.locks(root, 'core3', stack)
        source = Path(base.__file__).read_text()
        old = "attempt = (target/'artifacts/core_drained_recovery_v3') if stage.startswith('core') else root/f'artifacts/operations/{stage}_drained_recovery_v1'"
        if source.count(old) != 1:
            raise ValueError('Audit source drift')
        module = types.ModuleType('terminal_grouped_core3'); module.__file__ = __file__
        exec(compile(source.replace(old, 'attempt = target/'+repr(ATTEMPT)), '<current terminal audit>', 'exec'), module.__dict__)
        module.JOBS = {'core3': JOB}; module.inspect_operations = inspector(root)
        result = module.audit(root, 'core3', True)
        events, protected, ends, provenance = history(root, target)
        result['stage_history'] = events
        result['all_historical_model_journals'] = ends
        result['protected_sha256_or_absent'].update(protected)
        result['historical_admission_policy_sha256'] = provenance
        result['controlled_stop_proof_sha256'] = base.sha(root/PROOF)
        result['admission_sha256'] = base.sha(root/ADMISSION)
        final = base.rows(target/ATTEMPT/'execution.jsonl')[-1]
        credit = base.read(root/'artifacts/budget.json')['inference_credits']['fast-core3-'+JOB]
        observed = new_attempt_costs(target, base.read(target/ATTEMPT/'prior_journals.json'), {'quality'})
        summary = final['credit']
        if (credit['status'] != 'closed' or not summary['closed'] or summary['outstanding']
                or observed['attempts'] != summary['settled'] or summary['attempts'] != summary['settled']
                or observed['unknown'] != summary['unknown']
                or not math.isclose(observed['charged'], summary['charged'], abs_tol=1e-8)
                or any(credit[k] != summary[k] for k in ('attempts', 'settled', 'charged', 'unknown', 'funded'))):
            raise ValueError('Closed credit does not exactly match durable new attempts')
        result['credit_reconciliation_read_only'] = {'ledger': credit, 'summary': summary, 'raw': observed}
        for rel, digest in result['protected_sha256_or_absent'].items():
            p = target/rel
            if (base.sha(p) if p.exists() else None) != digest:
                raise ValueError('Concurrent source/cache/history change')
        result['script_sha256'] = base.sha(Path(__file__))
        result['no_admission_or_credit_mutation'] = True
        result['historical_unjournaled_outcomes_resolved'] = False
        result['initial_policy_bytes_available'] = False
        result['initial_policy_provenance'] = 'Exact digest bound by both preserved initial admissions and append-only stage history; original mutable policy bytes unavailable'
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
