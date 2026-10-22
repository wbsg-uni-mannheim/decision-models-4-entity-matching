"""Read-only full terminal E6 audit; no recovery or admission."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
from diagnose_fresh_baseline import audit_rows
from resume_schema_once import terminal_steps


def main():
    import expanded_variants as variants
    import orchestrate
    import protocol
    job = '365279'
    accounting = subprocess.check_output(['sacct', '-n', '-j', job,
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(accounting, job)
    manifest = variants.verify()
    orchestrate.verify()
    config = json.loads((ROOT/'configs/expanded_execution.json').read_text())
    for rel, expected in config['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
    cells, files, errors = [], {}, []
    for model in protocol.MODELS:
        for benchmark in manifest['conditions']:
            cases = json.loads((ROOT/'data'/benchmark/'test.json').read_text())
            directory = ROOT/'runs/quality'/model/'B0'/benchmark
            records = []
            for suffix in ('jsonl', 'failures.jsonl', 'executions.jsonl'):
                p = directory/('test.'+suffix)
                raw = p.read_bytes() if p.exists() else b''
                rows = [json.loads(line) for line in raw.splitlines()]
                if p.exists():
                    files[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
                if suffix == 'failures.jsonl':
                    errors.extend({'path': str(p.relative_to(ROOT)), 'record': r} for r in rows)
                if suffix != 'executions.jsonl':
                    assert all('study_run_id' not in r or r['study_run_id'] == 1 for r in rows)
                    # Legacy Run1 has no run-ID field. Explicit namespace attribution
                    # for reuse of the validator, not an observed ID claim.
                    records.append([{**r, 'study_run_id': 1} for r in rows])
            cell = audit_rows(cases, *records, protocol, model, benchmark, 1)
            cells.append({'model': model, 'benchmark': benchmark, 'split': 'test', **cell})
            assert {p.name for p in directory.glob('*.jsonl')} <= {'test.jsonl', 'test.failures.jsonl', 'test.executions.jsonl'}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in files.items())
    assert len(errors) == 1
    error = errors[0]['record']
    assert error['model_alias'] == 'jev' and error['pair_id'] == '5169253299#2114390938'
    assert error['error'] == 'HTTP 499' and error['attempts'][-1]['status'] == 499
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'job': job, 'sacct_utc': accounting,
              'status': 'terminal_scoped_audit_no_recovery', 'cells': cells, 'files_sha256': files,
              'full_failure_records': errors, 'complete_cells': sum(c['complete'] for c in cells),
              'totals': {k: sum(c[k] for c in cells) for k in ('expected', 'responses', 'missing', 'attempts', 'unknown_attempts')},
              'known_journaled_usd': math.fsum(c['known_usd'] for c in cells),
              'legacy_run_attribution': 'Run1 by project namespace; missing run IDs were not observed or edited.',
              'budget': json.loads((ROOT/'artifacts/budget.json').read_text()),
              'new_paid_calls': 0, 'recovery_admitted': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = ROOT/'artifacts/operations/variants_terminal_diagnosis_20261015T0003Z.json'
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'totals', 'complete_cells', 'known_journaled_usd')}))


if __name__ == '__main__':
    main()
