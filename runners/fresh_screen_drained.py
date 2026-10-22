"""Fresh development screening component; no CLI, admission or reporting.

A separate bounded wrapper must prove actual successful prerequisites, hold
writer locks and review scheduler/policy/scopes/cache/costs/budget/all joins.
This component never grants admission or changes frozen modules or markers.
"""
import hashlib
import json
from pathlib import Path
import types

from authorized_recovery_200 import make_wrapper
from diagnose_fresh_baseline import audit_rows
from drained_inference import drain_model_tasks
from operational_journal import inspect_operations
from repeated_runs import replica_path, shared_account

SCREEN_SHA = '3f7af27bf8b1171f10924bfb4bd745771363c51873e3b88c8d6faf9653e19fb8'
PAYLOAD_SHA = '2dc4d8d1ff24ede5096acd24f7713ef9aa97793bcc6226d6bf57c785dfe320b6'
SPLIT = 'expanded_screen'
STAGE = 'expanded_E2_E3_screen'


def audit_cell(cases, successes, failures, protocol, payload, model, variant, benchmark, run_id):
    # Validate the real variant BEFORE adapting the B0 audit, without modifying
    # any persisted row. Run identity is retained and checked by the raw audit.
    if any(r['variant'] != variant for r in successes+failures):
        raise ValueError('Foreign screening variant')
    adapter = types.SimpleNamespace(digest=protocol.digest, parse_response=protocol.parse_response,
                                   payload=lambda c, m, _: payload(c, m, variant))
    groups = [[{**r, 'variant': 'B0'} for r in group] for group in (successes, failures)]
    return audit_rows(cases, *groups, adapter, model, benchmark, run_id)


def inventory(target, run_id, screen, protocol):
    """All fixed cells, including raw parser, exact payload and operations joins."""
    bound = types.SimpleNamespace(**{**vars(protocol), 'payload': screen.payload})
    cases_by_benchmark = {b: json.loads((target/'data'/b/(SPLIT+'.json')).read_text())
                          for b in screen.SCREEN_BENCHMARKS}
    cells = []
    for model in screen.MODELS:
        for variant in screen.VARIANTS:
            for benchmark in screen.SCREEN_BENCHMARKS:
                directory = target/'runs/quality'/model/variant/benchmark
                def rows(suffix):
                    p = directory/(SPLIT+suffix)
                    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
                cases, failures = cases_by_benchmark[benchmark], rows('.failures.jsonl')
                checked = audit_cell(cases, rows('.jsonl'), failures, protocol, screen.payload,
                                     model, variant, benchmark, run_id)
                operations = inspect_operations(directory/(SPLIT+'.operations.jsonl'), cases,
                                                failures, bound, model, variant, run_id)
                if operations['unknown_outcome_events']:
                    raise ValueError('Unknown screening outcomes require separate terminal review')
                cells.append({'model': model, 'variant': variant, 'benchmark': benchmark,
                              'split': SPLIT, **checked, 'operations': operations})
    return cells


def run_screen(root, target, run_id, inference, screen, expanded, protocol, journal_dir):
    """At most one invocation per model, frozen variant/benchmark order and 8 workers."""
    root, target, journal_dir = Path(root).resolve(), Path(target).resolve(), Path(journal_dir).resolve()
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if inference.PROJECT != target or screen.PROJECT != target:
        raise ValueError('Wrong screening workspace')
    for module, digest in ((screen, SCREEN_SHA), (expanded, PAYLOAD_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen screening source changed')
    if (screen.MODELS != protocol.MODELS or screen.VARIANTS != expanded.VARIANTS or
            screen.payload is not expanded.payload or
            screen.SCREEN_BENCHMARKS != ['abt-buy', 'wdc', 'dblp-scholar']):
        raise ValueError('Frozen screening matrix/payload changed')
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive replica attempt directory required')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('supplement_billiger50', {}).get('status') != 'complete_automated':
        raise ValueError('Supplement prerequisite incomplete')
    if STAGE in progress or (target/'artifacts/results/expanded_screen').exists():
        raise ValueError('Existing screening stage/reports require separate terminal review')
    screen.verify()
    before = inventory(target, run_id, screen, protocol)
    if any(c['failure_rows'] or c['operations']['events'] for c in before):
        raise ValueError('Existing failures require separate terminal review')
    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': screen.payload,
                                       'account': shared_account(inference.account, root)})
    _, cell, adapter = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    journal_dir.mkdir(exist_ok=False)
    def task(model):
        for variant in screen.VARIANTS:
            for benchmark in screen.SCREEN_BENCHMARKS:
                cell(model, variant, benchmark, SPLIT, workers=8)
    outcomes = drain_model_tasks(task, list(screen.MODELS), journal_dir/'models.jsonl', protocol.canonical)
    checked = inventory(target, run_id, screen, protocol)
    return {'run_id': run_id, 'models': outcomes, 'cells': checked, 'adapter': adapter,
            'inference_complete': all(c['complete'] for c in checked) and
                                  all(m['status'] == 'complete' for m in outcomes),
            'admission_granted': False, 'report_written': False, 'completion_marker_written': False,
            'historical_unjournaled_outcomes_resolved': False}
