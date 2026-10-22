"""Fresh full scoped end audit for drained variants job 366105; read only."""
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from diagnose_fresh_baseline import audit_rows
from operational_journal import inspect_operations
from resume_schema_once import terminal_steps
from authorized_recovery_200 import sha, authority

ROOT = Path(__file__).resolve().parents[2]
JOB = '366105'
STAGE = 'artifacts/operations/variants_drained_recovery_v1.jsonl'
MODELS = 'artifacts/operations/variants_drained_models_v1.jsonl'
ADMISSION = 'configs/variants_budget200_admission_20261015.json'


def validate_history(stage, models, aliases, admission_sha):
    if ([r['event'] for r in stage] != ['start', 'inference_end', 'end']
            or any(r['job_id'] != JOB or r['admission_sha256'] != admission_sha for r in stage)
            or stage[-1].get('status') != 'failed' or stage[-1].get('report_created') is not False
            or stage[0].get('invocations_limit') != 1):
        raise ValueError('Invalid predecessor stage provenance')
    ends = models[1:-1]
    if (models[0].get('event') != 'start' or models[0].get('models') != list(aliases)
            or models[0].get('invocations_per_model') != 1
            or models[-1].get('event') != 'end' or models[-1].get('status') != 'failed'
            or models[-1].get('models_drained') != len(aliases)
            or len(ends) != len(aliases) or {r['model'] for r in ends} != set(aliases)
            or any(r['event'] != 'model_end' for r in ends)):
        raise ValueError('Incomplete or repeated model endings')
    projected = [{k: v for k, v in r.items() if k not in ('event', 'utc')} for r in ends]
    if projected != stage[1]['models']:
        raise ValueError('Model/stage outcomes disagree')
    return ends


def main():
    sys.path.insert(0, str(ROOT/'src'))
    import protocol
    import expanded_variants as variants
    import orchestrate
    authority(ROOT)
    accounting = subprocess.check_output(['sacct', '-n', '-j', JOB,
        '--format=JobID,State,ExitCode,Start,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(accounting, JOB)
    with (ROOT/'artifacts/operations/variants_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = variants.verify()
        orchestrate.verify()
        spec = json.loads((ROOT/ADMISSION).read_text())
        if spec['slurm_job_id'] != JOB:
            raise ValueError('Wrong predecessor admission')
        protected = {**spec['source_sha256'], **{p: sha(ROOT/p) for p in (STAGE, MODELS, ADMISSION)}}
        if any(sha(ROOT/p) != h for p, h in protected.items()):
            raise ValueError('Pinned source drift')
        stage = [json.loads(x) for x in (ROOT/STAGE).read_text().splitlines()]
        models = [json.loads(x) for x in (ROOT/MODELS).read_text().splitlines()]
        ends = validate_history(stage, models, protocol.MODELS, sha(ROOT/ADMISSION))
        cells, files, absent, errors, operations = [], {}, [], [], []
        for model in protocol.MODELS:
            for benchmark in manifest['conditions']:
                cases = json.loads((ROOT/'data'/benchmark/'test.json').read_text())
                directory = ROOT/'runs/quality'/model/'B0'/benchmark
                data = {}
                allowed = {'test.'+s for s in ('jsonl', 'failures.jsonl', 'executions.jsonl', 'operations.jsonl')}
                if not {p.name for p in directory.glob('*.jsonl')} <= allowed:
                    raise ValueError('Unexpected variant journals')
                for suffix in ('jsonl', 'failures.jsonl', 'executions.jsonl', 'operations.jsonl'):
                    p = directory/('test.'+suffix)
                    rel = str(p.relative_to(ROOT))
                    raw = p.read_bytes() if p.exists() else None
                    if raw is None:
                        absent.append(rel)
                    else:
                        files[rel] = sha(p)
                    data[suffix] = [json.loads(x) for x in raw.splitlines()] if raw else []
                for row in data['jsonl']+data['failures.jsonl']:
                    if 'study_run_id' in row and (type(row['study_run_id']) is not int or row['study_run_id'] != 1):
                        raise ValueError('Wrong observed run ID')
                cell = audit_rows(cases, *[[dict(r, study_run_id=1) for r in data[s]]
                    for s in ('jsonl', 'failures.jsonl')], protocol, model, benchmark, 1)
                cells.append(dict(model=model, benchmark=benchmark, split='test', **cell))
                errors.extend({'path': str(directory.relative_to(ROOT))+'/test.failures.jsonl', 'record': r}
                              for r in data['failures.jsonl'])
                operations.append(dict(model=model, benchmark=benchmark,
                    **inspect_operations(directory/'test.operations.jsonl', cases,
                        data['failures.jsonl'], protocol, model, 'B0', 1)))
        op_key = lambda row: (row['model'], row['benchmark'])
        if (len(stage[1]['operations']) != len(operations)
                or sorted(operations, key=op_key) != sorted(stage[1]['operations'], key=op_key)):
            raise ValueError('End-stage operations inventory differs')
        for end in ends:
            complete = all(c['complete'] for c in cells if c['model'] == end['model'])
            if (end['status'] == 'complete') != complete:
                raise ValueError('Model completion disagrees with full raw coverage')
        if any(o['unknown_outcome_events'] for o in operations):
            raise ValueError('Unknown drained outcomes need separate diagnosis')
        if any(sha(ROOT/p) != h for p, h in {**protected, **files}.items()) or any((ROOT/p).exists() for p in absent):
            raise ValueError('Audit scope mutated')
        if (ROOT/'artifacts/expanded_variants_complete.json').exists():
            raise ValueError('Unexpected completed marker')
        result = {'utc': datetime.now(timezone.utc).isoformat(), 'job': JOB,
            'status': 'terminal_scoped_audit_no_recovery', 'sacct_utc': accounting,
            'cells': cells, 'files_sha256': files, 'absent_journals': absent,
            'protected_sha256': protected, 'full_failure_records': errors,
            'operations': operations, 'model_endings': ends, 'stage_history': stage,
            'complete_cells': sum(c['complete'] for c in cells),
            'totals': {k: sum(c[k] for c in cells) for k in
                ('expected', 'responses', 'missing', 'attempts', 'unknown_attempts', 'failure_rows')},
            'known_journaled_usd': math.fsum(c['known_usd'] for c in cells),
            'budget': json.loads((ROOT/'artifacts/budget.json').read_text()),
            'legacy_run_attribution': 'Missing historical run IDs attributed by Run1 namespace only; explicit IDs verified.',
            'historical_unjournaled_outcomes_resolved': False,
            'cost_scope': 'Preserved scoped attempts, not full billing or incremental continuation cost.',
            'recovery_admitted': False, 'new_paid_calls': 0, 'script_sha256': sha(Path(__file__))}
        out = ROOT/'artifacts/operations/variants_terminal_diagnosis_366105_20261015.json'
        with out.open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
        print(json.dumps({k: result[k] for k in ('totals', 'complete_cells', 'known_journaled_usd')}))


if __name__ == '__main__':
    main()
