"""Fresh confirmation development component, never a launcher or admission.

Drains one invocation per model. Only after all development cells are complete
and raw-verified is the frozen selector evaluated in memory. No selection files,
test inference, reports or completion markers are written. A separate wrapper
must hold locks and validate real prerequisites, costs and scheduler admission.
"""
import hashlib
import json
from pathlib import Path
import types

from authorized_recovery_200 import make_wrapper
from confirmation_drained import CONFIRMATION_SHA
from drained_inference import drain_model_tasks
from fresh_screen_drained import PAYLOAD_SHA, audit_cell
from operational_journal import inspect_operations
from repeated_runs import replica_path, shared_account


def matrix(confirmation):
    return [(m, v, b) for m in confirmation.MODELS
            for v in confirmation.PROMPT_VARIANTS+confirmation.SERIALIZATIONS
            for b in confirmation.BENCHMARKS]


def inventory(target, run_id, confirmation, protocol):
    bound = types.SimpleNamespace(**{**vars(protocol), 'payload': confirmation.payload})
    cases = {b: json.loads((target/'data'/b/'prompt.json').read_text())
             for b in confirmation.BENCHMARKS}
    result = []
    for m, v, b in matrix(confirmation):
        directory = target/'runs/quality'/m/v/b
        def rows(suffix):
            p = directory/('prompt'+suffix)
            return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        failures = rows('.failures.jsonl')
        audit = audit_cell(cases[b], rows('.jsonl'), failures, protocol,
                           confirmation.payload, m, v, b, run_id)
        operations = inspect_operations(directory/'prompt.operations.jsonl', cases[b],
                                        failures, bound, m, v, run_id)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown development outcomes need terminal review')
        result.append({'model': m, 'variant': v, 'benchmark': b,
                       'split': 'prompt', **audit, 'operations': operations})
    return result


def selection_in_memory(target, run_id, confirmation, analysis, protocol):
    checked = inventory(target, run_id, confirmation, protocol)
    if not all(c['complete'] for c in checked):
        raise ValueError('Incomplete development; no selection or test permitted')
    load = types.FunctionType(analysis.load.__code__,
        {**analysis.load.__globals__, 'PROJECT': target, 'payload': confirmation.payload},
        argdefs=analysis.load.__defaults__)
    scores, metrics = {}, []
    for m, v, b in matrix(confirmation):
        d = load(m, v, b, 'prompt')
        value = analysis.metric(d['y'], d['pred'])
        scores[m, b, v] = value['f1']
        metrics.append({'model': m, 'variant': v, 'benchmark': b, **value,
                        'known_cost_usd': float(d['costs'].sum()),
                        'unknown_cost_attempts': d['unknown_cost_attempts']})
    selected = confirmation.make_selection(scores)
    return {'selected': selected, 'test_cells': confirmation.confirmation_cells(selected),
            'development_metrics': metrics, 'selection_sha256': protocol.digest(selected),
            'selection_written': False, 'test_used_for_selection': False}


def run_development(root, target, run_id, inference, confirmation, expanded,
                    analysis, protocol, journal_dir):
    root, target, journal_dir = map(lambda p: Path(p).resolve(), (root, target, journal_dir))
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if any(m.PROJECT != target for m in (inference, confirmation, analysis)):
        raise ValueError('Wrong confirmation workspace')
    for module, expected in ((confirmation, CONFIRMATION_SHA), (expanded, PAYLOAD_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen confirmation source changed')
    expected_prompts = ['X_P_'+c+'_'+str(i) for c in expanded.PROMPTS for i in range(3)]
    if (confirmation.MODELS != protocol.MODELS or confirmation.BENCHMARKS != protocol.BENCHMARKS
            or confirmation.PROMPT_VARIANTS != expected_prompts
            or confirmation.SERIALIZATIONS != ['B0', 'S1', 'X_R_aligned']
            or confirmation.payload is not expanded.payload):
        raise ValueError('Frozen confirmation matrix/payload changed')
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive replica attempt required')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('expanded_E2_E3_screen', {}).get('status') != 'complete':
        raise ValueError('Screen prerequisite incomplete')
    protected = [target/'configs'/n for n in
                 ('expanded_selected_variants.json', 'expanded_confirmation_cells.json')]
    protected += [target/'artifacts'/n for n in ('expanded_confirmation_complete.json',
        'expanded_confirmation_development_cost_forecast.json', 'expanded_confirmation_test_cost_forecast.json',
        'results/expanded_confirmation')]
    if any(p.exists() for p in protected):
        raise ValueError('Existing selection/report/stage requires terminal review')
    confirmation.verify()
    before = inventory(target, run_id, confirmation, protocol)
    # B0/S1 are same-run core caches, including retained historical failures.
    # Expanded variants must never be blindly resumed from a failed attempt.
    if any((c['failure_rows'] or c['operations']['events']) and c['variant'] not in ('B0', 'S1')
           for c in before):
        raise ValueError('Existing expanded failures require terminal review')
    if any(not c['complete'] for c in before if c['variant'] == 'B0'):
        raise ValueError('Core B0 development prerequisite incomplete')
    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': confirmation.payload,
                                       'account': shared_account(inference.account, root)})
    _, cell, adapter = make_wrapper(root, 'core').authorized_build(isolated, run_id)
    journal_dir.mkdir(exist_ok=False)
    def task(model):
        for m, v, b in matrix(confirmation):
            if m == model:
                cell(m, v, b, 'prompt', workers=8)
    outcomes = drain_model_tasks(task, list(confirmation.MODELS), journal_dir/'models.jsonl', protocol.canonical)
    checked = inventory(target, run_id, confirmation, protocol)
    complete = all(c['complete'] for c in checked) and all(m['status'] == 'complete' for m in outcomes)
    selection = selection_in_memory(target, run_id, confirmation, analysis, protocol) if complete else None
    if any(p.exists() for p in protected):
        raise ValueError('Unexpected selection/report writer')
    return {'run_id': run_id, 'models': outcomes, 'cells': checked, 'adapter': adapter,
            'development_complete': complete, 'selection_candidate': selection,
            'selection_written': False, 'test_calls_started': False, 'admission_granted': False,
            'report_written': False, 'completion_marker_written': False,
            'historical_unjournaled_outcomes_resolved': False}
