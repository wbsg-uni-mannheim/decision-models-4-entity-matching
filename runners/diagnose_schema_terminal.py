"""Read-only complete schema scope audit after terminal scheduler steps."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
from diagnose_fresh_baseline import audit_rows
from resume_schema_once import terminal_steps


def audit_schema_rows(cases, successes, failures, schema, model, variant, benchmark):
    for row in successes+failures:
        if row['variant'] != variant or ('study_run_id' in row and
                (type(row['study_run_id']) is not int or row['study_run_id'] != 1)):
            raise ValueError('Foreign schema variant or run')
    # Adapt the generic validator in memory only; no persisted provenance changes.
    adapter = SimpleNamespace(
        digest=lambda value: schema.protocol.digest(value),
        payload=lambda case, alias, _: schema.wire(schema.payload(case, alias, variant)),
        parse_response=lambda alias, raw: schema.parse(alias, raw, variant))
    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest() if isinstance(value, str) else schema.protocol.digest(value)
    adapter.digest = digest
    rows = [[{**row, 'variant': 'B0', 'study_run_id': 1} for row in group]
            for group in (successes, failures)]
    return audit_rows(cases, *rows, adapter, model, benchmark, 1)


def main():
    import expanded_schema as es
    import orchestrate
    import billiger50
    import analyze
    from schema_run_export import selected_matrix
    def denied(*args, **kwargs):
        raise AssertionError('Network prohibited during audit')
    socket.socket = denied
    job = '365969'
    acct = subprocess.check_output(['sacct', '-n', '-j', job,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(acct, job)
    with (ROOT/'artifacts/operations/schema_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = es.verify()
        orchestrate.verify()
        billiger50.verify()
        config = json.loads((ROOT/'configs/expanded_execution.json').read_text())
        for rel, expected in config['source_sha256'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
        frozen = {rel: hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in
                  ['configs/expanded_schema_selection.json', 'configs/expanded_schema_test_cells.json',
                   'configs/expanded_schema_manifest.json', 'artifacts/operations/schema_recovery_365969.jsonl']}
        development = {(m, v, b): es.load(m, v, b, 'prompt') for m in es.MODELS
                       for v in es.CONDITIONS for b in es.BENCHMARKS}
        selected, test_cells = selected_matrix(es, development, analyze.metric)
        assert selected == es.read(ROOT/'configs/expanded_schema_selection.json')
        assert test_cells == es.read(ROOT/'configs/expanded_schema_test_cells.json')
        scope = [('schema_pilot', m, v, b, 'schema_pilot') for m in es.MODELS
                 for v in es.CONDITIONS for b in manifest['pilot']['families']]
        scope += [('schema', m, v, b, 'prompt') for m, v, b in development]
        scope += [('schema', m, v, b, 'test') for m, v, b in test_cells]
        files, expected_files, cells, errors = {}, set(), [], []
        for ns, m, v, b, split in scope:
            cases = es.read(ROOT/'data'/b/(split+'.json'))
            groups = []
            for suffix in ('.jsonl', '.failures.jsonl', '.executions.jsonl'):
                p = ROOT/'runs'/ns/m/v/b/(split+suffix)
                rel = str(p.relative_to(ROOT))
                expected_files.add(rel)
                raw = p.read_bytes() if p.exists() else b''
                rows = [json.loads(line) for line in raw.splitlines()]
                if p.exists():
                    files[rel] = hashlib.sha256(raw).hexdigest()
                if suffix != '.executions.jsonl':
                    groups.append(rows)
                if suffix == '.failures.jsonl':
                    errors.extend({'path': rel, 'record': row} for row in rows)
            cell = audit_schema_rows(cases, *groups, es, m, v, b)
            cells.append({'namespace': ns, 'model': m, 'variant': v, 'benchmark': b, 'split': split, **cell})
        observed = {str(p.relative_to(ROOT)) for ns in ('schema', 'schema_pilot')
                    for p in (ROOT/'runs'/ns).rglob('*.jsonl')}
        assert observed <= expected_files, observed-expected_files
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in {**files, **frozen}.items())
        old = es.read(ROOT/'artifacts/operations/schema_terminal_diagnosis_20261014T2103Z.json')
        assert [x['record'] for x in sorted(errors, key=lambda x: x['path'])] == old['full_failure_records']
        log = (ROOT/'logs/schema-resume-365969.log').read_text()
        assert 'RuntimeError: Project spending ceiling reached' in log
        assert not (ROOT/'artifacts/expanded_schema_complete.json').exists()
        result = {'utc': datetime.now(timezone.utc).isoformat(), 'job': job, 'sacct_utc': acct,
                  'status': 'terminal_partial_schema_audited_no_recovery', 'cells': cells,
                  'files_sha256': files, 'frozen_artifact_sha256': frozen,
                  'source_sha256': config['source_sha256'], 'full_failure_records': errors,
                  'totals': {k: sum(c[k] for c in cells) for k in
                             ('expected', 'responses', 'missing', 'failure_rows', 'attempts', 'unknown_attempts')},
                  'complete_cells': sum(c['complete'] for c in cells),
                  'known_journaled_usd': math.fsum(c['known_usd'] for c in cells),
                  'selection_reconstructed_from_all_development': True,
                  'test_cells': len(test_cells), 'budget_guard_exception_observed': True,
                  'all_worker_exception_types_known': False,
                  'historical_unjournaled_outcomes_resolved': False,
                  'cost_scope': 'Preserved journal attempts only; not a full schema bill. Historical unjournaled outcomes and costs remain unknown.',
                  'legacy_run_attribution': 'Run1 project namespace; missing run IDs were not observed or persisted.',
                  'budget': es.read(ROOT/'artifacts/budget.json'), 'new_paid_calls': 0,
                  'recovery_admitted': False, 'log_sha256': hashlib.sha256(log.encode()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        out = ROOT/'artifacts/operations/schema_terminal_diagnosis_20261015T0133Z.json'
        with out.open('x') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
        print(json.dumps({k: result[k] for k in ('status', 'totals', 'complete_cells', 'test_cells', 'known_journaled_usd')}))


if __name__ == '__main__':
    main()
