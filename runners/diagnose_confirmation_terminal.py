"""Read-only audit of all development and selected confirmation cache cells."""
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


def audit_confirmation_rows(cases, successes, failures, confirmation, protocol, model, variant, benchmark):
    for row in successes+failures:
        if row['variant'] != variant or ('study_run_id' in row and
                (type(row['study_run_id']) is not int or row['study_run_id'] != 1)):
            raise ValueError('Foreign confirmation variant or run')
    adapter = SimpleNamespace(digest=protocol.digest, parse_response=protocol.parse_response,
                              payload=lambda case, alias, _: confirmation.payload(case, alias, variant))
    # Legacy Run1 namespace attribution only, without altering persisted rows.
    groups = [[{**r, 'variant': 'B0', 'study_run_id': 1} for r in group]
              for group in (successes, failures)]
    return audit_rows(cases, *groups, adapter, model, benchmark, 1)


def main():
    import expanded_confirmation as ec
    import protocol
    import orchestrate
    import billiger50
    import analyze
    def denied(*args, **kwargs):
        raise AssertionError('Network prohibited during audit')
    socket.socket = denied
    acct = subprocess.check_output(['sacct', '-n', '-j', '365636',
        '--format=JobID,State,ExitCode,End', '-P'], text=True, env={**os.environ, 'TZ': 'UTC'})
    terminal_steps(acct, '365636')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    with (ROOT/'artifacts/operations/confirmation_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ec.verify()
        orchestrate.verify()
        billiger50.verify()
        config = ec.read(ROOT/'configs/expanded_execution.json')
        for rel, expected in config['source_sha256'].items():
            assert sha(ROOT/rel) == expected, rel
        frozen = {rel: sha(ROOT/rel) for rel in [
            'configs/expanded_selected_variants.json', 'configs/expanded_confirmation_cells.json',
            'configs/expanded_confirmation_manifest.json', 'artifacts/operations/confirmation_recovery_365636.jsonl']}
        dev = [(m, v, b, 'prompt') for m in ec.MODELS
               for v in ec.PROMPT_VARIANTS+ec.SERIALIZATIONS for b in ec.BENCHMARKS]
        scores = {}
        analyze.payload = ec.payload
        for m, v, b, split in dev:
            d = analyze.load(m, v, b, split)
            scores[m, b, v] = analyze.metric(d['y'], d['pred'])['f1']
        selected = ec.make_selection(scores)
        assert selected == ec.read(ROOT/'configs/expanded_selected_variants.json')
        test = ec.confirmation_cells(selected)
        assert [list(x) for x in test] == ec.read(ROOT/'configs/expanded_confirmation_cells.json')
        files, cells, errors = {}, [], []
        for m, v, b, split in dev+[(m, v, b, 'test') for m, v, b in test]:
            groups = []
            for suffix in ('.jsonl', '.failures.jsonl', '.executions.jsonl'):
                path = ROOT/'runs/quality'/m/v/b/(split+suffix)
                raw = path.read_bytes() if path.exists() else b''
                rows = [json.loads(line) for line in raw.splitlines()]
                rel = str(path.relative_to(ROOT))
                if path.exists():
                    files[rel] = hashlib.sha256(raw).hexdigest()
                if suffix != '.executions.jsonl':
                    groups.append(rows)
                if suffix == '.failures.jsonl':
                    errors.extend({'path': rel, 'record': row} for row in rows)
            cases = ec.read(ROOT/'data'/b/(split+'.json'))
            cell = audit_confirmation_rows(cases, *groups, ec, protocol, m, v, b)
            cells.append({'namespace': 'quality', 'model': m, 'variant': v, 'benchmark': b, 'split': split, **cell})
        assert all(sha(ROOT/p) == h for p, h in {**files, **frozen}.items())
        assert not (ROOT/'artifacts/expanded_confirmation_complete.json').exists()
        history = [json.loads(line) for line in (ROOT/'artifacts/operations/confirmation_recovery_365636.jsonl').read_text().splitlines()]
        assert history[-1]['event'] == 'stopped_for_diagnosis' and history[-1]['invocation'] == 3
        log = (ROOT/'logs/confirmation-recovery-365636.log').read_text()
        assert 'RuntimeError: Project spending ceiling reached' in log
        result = {'utc': datetime.now(timezone.utc).isoformat(), 'job': '365636', 'sacct_utc': acct,
            'status': 'terminal_partial_confirmation_audited_no_recovery', 'cells': cells,
            'files_sha256': files, 'frozen_artifact_sha256': frozen, 'source_sha256': config['source_sha256'],
            'full_failure_records': errors, 'supervisor_history': history,
            'totals': {k: sum(c[k] for c in cells) for k in
                       ('expected', 'responses', 'missing', 'failure_rows', 'attempts', 'unknown_attempts')},
            'complete_cells': sum(c['complete'] for c in cells), 'test_cells': len(test),
            'known_journaled_usd': math.fsum(c['known_usd'] for c in cells),
            'selection_reconstructed_from_all_development': True,
            'budget_guard_exception_observed': True, 'all_worker_exception_types_known': False,
            'historical_unjournaled_outcomes_resolved': False,
            'cost_scope': 'Preserved unique confirmation-scope cache attempts, including reused B0/S1. Not incremental confirmation cost or full bill; do not add to other stages.',
            'legacy_run_attribution': 'Run1 project namespace; missing run IDs are not independently observed.',
            'budget': ec.read(ROOT/'artifacts/budget.json'), 'new_paid_calls': 0, 'recovery_admitted': False,
            'log_sha256': hashlib.sha256(log.encode()).hexdigest(), 'script_sha256': sha(Path(__file__))}
        with (ROOT/'artifacts/operations/confirmation_terminal_diagnosis_20261015T0233Z.json').open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps({k: result[k] for k in ('status', 'totals', 'complete_cells', 'test_cells', 'known_journaled_usd')}))


if __name__ == '__main__':
    main()
