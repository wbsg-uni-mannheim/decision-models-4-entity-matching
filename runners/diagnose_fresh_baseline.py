"""Terminal, read-only audit of a fresh core that stopped in AP1_baseline.

Validates partial caches without fabricating completion or resetting journals.
No admission, inference, reconciliation or automatic recovery is performed.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from repeated_runs import replica_path, sha, verify
from resume_schema_once import terminal_steps


def audit_rows(cases, successes, failures, protocol, model, benchmark, run_id):
    inputs = {c['pair_id']: c for c in cases}
    if len(inputs) != len(cases):
        raise ValueError('Duplicate frozen pairs')
    if any(protocol.digest(c['records']) != c['input_hash'] for c in cases):
        raise ValueError('Frozen input hash mismatch')
    ids = [r['pair_id'] for r in successes]
    if len(set(ids)) != len(ids) or not set(ids) <= inputs.keys():
        raise ValueError('Duplicate or foreign successful response')
    attempts = []
    for rows, failed in ((successes, False), (failures, True)):
        for row in rows:
            c = inputs.get(row['pair_id'])
            if c is None or row.get('study_run_id') != run_id or type(row.get('study_run_id')) is not int:
                raise ValueError('Foreign pair or study run')
            if (row['model_alias'] != model or row['variant'] != 'B0' or row['benchmark'] != benchmark
                    or row['input_hash'] != c['input_hash']
                    or row['request_hash'] != protocol.digest(protocol.payload(c, model, 'B0'))):
                raise ValueError('Input/request/condition mismatch')
            aa = row['attempts']
            if not aa or bool(row.get('error')) != failed:
                raise ValueError('Invalid success/failure placement')
            for a in aa:
                for field in ('seconds', 'cost'):
                    value = a[field]
                    if field == 'cost' and value is None:
                        continue
                    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                        raise ValueError('Invalid duration or cost')
            last = aa[-1]
            if failed:
                if not last.get('error'):
                    raise ValueError('Missing failure error')
                if last['status'] == 200 and last['cost'] is not None:
                    try:
                        protocol.parse_response(model, last['response'])
                    except (ValueError, KeyError, TypeError):
                        pass
                    else:
                        raise ValueError('Recorded parser failure no longer reproduced')
            else:
                if last['status'] != 200 or last.get('error') or last['cost'] is None:
                    raise ValueError('Invalid final success attempt')
                prediction, probability = protocol.parse_response(model, last['response'])
                if (row['prediction'] != prediction or row['probability_match'] != probability
                        or last['response'].get('usage', {}).get('cost') != last['cost']):
                    raise ValueError('Raw parser/usage mismatch')
            attempts.extend(aa)
    missing = sorted(inputs.keys()-set(ids))
    return {'expected': len(cases), 'responses': len(successes), 'missing': len(missing),
            'missing_pair_ids_sha256': protocol.digest(missing), 'failure_rows': len(failures),
            'known_usd': math.fsum(a['cost'] for a in attempts if a['cost'] is not None),
            'unknown_attempts': sum(a['cost'] is None for a in attempts), 'attempts': len(attempts),
            'complete': not missing}


def diagnose(root, run_id, job):
    target = replica_path(root, run_id)
    accounting = subprocess.check_output(['sacct', '-n', '-j', job,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(accounting, job)
    with (target/'artifacts/parallel_writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = verify(root, run_id)
        journal = target/'artifacts/stage_execution.jsonl'
        journal_sha = sha(journal)
        history = [json.loads(line) for line in journal.read_text().splitlines()]
        core = [r for r in history if r['stage'] == 'core']
        if (len(core) != 2 or [r['status'] for r in core] != ['started', 'failed']
                or any(r['slurm_job_id'] != job or r['run_id'] != run_id for r in core)
                or any(r['stage'] not in ('smoke', 'core') for r in history)):
            raise ValueError('Only one diagnosed failed initial core, without later stages')
        progress = json.loads((target/'artifacts/progress.json').read_text())
        if set(progress) != {'smoke', 'AP1_baseline'} or progress['AP1_baseline']['status'] != 'needs_attention':
            raise ValueError('Audit scope is baseline-only failed core')
        if 'protocol' in sys.modules or 'run' in sys.modules:
            raise ValueError('Fresh process required')
        sys.path.insert(0, str(target/'src'))
        import protocol
        import orchestrate
        orchestrate.verify()
        cells, files, errors, expected_files = [], {}, [], set()
        for m in protocol.MODELS:
            for b in protocol.BENCHMARKS:
                for split in ('prompt', 'route', 'test'):
                    cases = json.loads((target/'data'/b/(split+'.json')).read_text())
                    directory = target/'runs/quality'/m/'B0'/b
                    records = []
                    for suffix in ('.jsonl', '.failures.jsonl', '.executions.jsonl'):
                        p = directory/(split+suffix)
                        rel = str(p.relative_to(target))
                        expected_files.add(rel)
                        raw = p.read_bytes() if p.exists() else None
                        rows = [json.loads(line) for line in raw.splitlines()] if raw else []
                        if raw is not None:
                            files[rel] = hashlib.sha256(raw).hexdigest()
                        if suffix != '.executions.jsonl':
                            records.append(rows)
                        if suffix == '.failures.jsonl':
                            errors.extend({'path': rel, 'record': row} for row in rows)
                    cell = audit_rows(cases, *records, protocol, m, b, run_id)
                    cells.append({'model': m, 'benchmark': b, 'split': split, **cell})
        observed_files = {str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')}
        if not observed_files <= expected_files:
            raise ValueError('Out-of-scope response journals: '+str(observed_files-expected_files))
        # Detect any unexpected concurrent writer, in addition to the replica lock.
        if any(sha(target/f) != h for f, h in files.items()) or sha(journal) != journal_sha:
            raise ValueError('Source changed during audit')
        details = json.loads(progress['AP1_baseline']['detail'])
        return {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id, 'predecessor': job,
                'status': 'terminal_partial_baseline_audited_not_recovered', 'sacct_utc': accounting,
                'frozen_files_verified': len(spec['immutable_sha256']), 'stage_journal': history,
                'stage_journal_sha256': sha(journal), 'progress': progress,
                'failure_records': errors, 'worker_errors': details,
                'budget_guard_models': [r['model'] for r in details if r['error'] == 'Project spending ceiling reached'],
                'cells': cells, 'response_files_sha256': files,
                'totals': {k: sum(c[k] for c in cells) for k in
                           ('expected', 'responses', 'missing', 'failure_rows', 'attempts', 'unknown_attempts')},
                'known_journaled_usd': math.fsum(c['known_usd'] for c in cells),
                'complete_cells': sum(c['complete'] for c in cells),
                'budget_snapshot_after_audit': json.loads((root/'artifacts/budget.json').read_text()),
                'limitations': ['Partial B0 inventory is not a study result or complete run.',
                    'Budget reserve failure occurs before HTTP and creates no response failure row.',
                    'Frozen run_cell exits result collection on a raised future; other calls in that '
                    'batch may settle in the shared ledger without a response journal. Exact number '
                    'and run-specific costs cannot be inferred from the shared aggregate ledger.',
                    'No unknown charges or foreign holds released. Any continuation must preserve '
                    'existing journals and explicitly address possibly unjournaled paid outcomes.'],
                'recovery_admitted': False, 'new_paid_calls': 0, 'script_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', type=int, choices=(2, 3), required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = diagnose(args.root.resolve(), args.run_id, args.job)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'totals', 'complete_cells',
                                           'known_journaled_usd', 'budget_guard_models')}))
