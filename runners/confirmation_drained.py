"""Execution component for a separately admitted Run1 confirmation recovery.

No CLI, scheduler admission or completion marker. A future wrapper must hold
the confirmation writer lock and verify terminal diagnosis, scopes, source/cache
hashes, all dependency joins, policy and a complete remaining-cost plan first.
Only the already selected test matrix is continued, once per model. Complete
development and frozen selection are independently rechecked before any call.
"""
import hashlib
import json
from pathlib import Path
import types

from drained_inference import build, drain_model_tasks
from operational_journal import inspect_operations

CONFIRMATION_SHA = 'dcf0323b256ded6ec57cd8d92dd3724cdb7a293fe94b82a66e33184c0a326b7d'


def selected_tests(root, confirmation, analysis):
    """Read-only replay; never call frozen select(), which writes artifacts."""
    if confirmation.PROJECT != root or analysis.PROJECT != root:
        raise ValueError('Wrong confirmation workspace')
    if hashlib.sha256(Path(confirmation.__file__).read_bytes()).hexdigest() != CONFIRMATION_SHA:
        raise ValueError('Frozen confirmation source changed')
    confirmation.verify()
    # Bind only this private load function to expanded payloads. Imported modules
    # and their shared globals remain untouched.
    load = types.FunctionType(analysis.load.__code__,
        {**analysis.load.__globals__, 'PROJECT': root, 'payload': confirmation.payload},
        argdefs=analysis.load.__defaults__)
    scores = {}
    for model in confirmation.MODELS:
        for benchmark in confirmation.BENCHMARKS:
            for variant in confirmation.PROMPT_VARIANTS+confirmation.SERIALIZATIONS:
                data = load(model, variant, benchmark, 'prompt')
                scores[model, benchmark, variant] = analysis.metric(data['y'], data['pred'])['f1']
    selected = confirmation.make_selection(scores)
    selection_file = root/'configs/expanded_selected_variants.json'
    cells_file = root/'configs/expanded_confirmation_cells.json'
    cells = confirmation.confirmation_cells(selected)
    if (selected != json.loads(selection_file.read_text()) or
            [list(c) for c in cells] != json.loads(cells_file.read_text())):
        raise ValueError('Frozen selection or test matrix mismatch')
    return cells, {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (selection_file, cells_file)}


def run_selected_tests(root, confirmation, analysis, inference, protocol, journal_dir):
    """One test continuation; caller owns actual admission and exclusive lock."""
    if inference.PROJECT != root:
        raise ValueError('Wrong inference workspace')
    if (root/'artifacts/expanded_confirmation_complete.json').exists():
        raise ValueError('Confirmation already complete')
    if journal_dir.exists():
        raise ValueError('Already attempted; new terminal review required')
    cells, hashes = selected_tests(root, confirmation, analysis)
    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': confirmation.payload})
    _, cell, adapter = build(isolated, 1)
    journal_dir.mkdir(exist_ok=False)
    def task(model):
        for alias, variant, benchmark in cells:
            if alias == model:
                cell(alias, variant, benchmark, 'test', workers=8)
    outcomes = drain_model_tasks(task, list(confirmation.MODELS), journal_dir/'models.jsonl', protocol.canonical)
    bound_protocol = types.SimpleNamespace(**{**vars(protocol), 'payload': confirmation.payload})
    operations = []
    for model, variant, benchmark in cells:
        directory = root/'runs/quality'/model/variant/benchmark
        failure = directory/'test.failures.jsonl'
        rows = [json.loads(line) for line in failure.read_text().splitlines()] if failure.exists() else []
        cases = json.loads((root/'data'/benchmark/'test.json').read_text())
        operations.append({'model': model, 'variant': variant, 'benchmark': benchmark,
            **inspect_operations(directory/'test.operations.jsonl', cases, rows,
                                 bound_protocol, model, variant, 1)})
    if any(hashlib.sha256((root/p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise ValueError('Selection changed during execution')
    return {'models': outcomes, 'operations': operations, 'adapter': adapter,
            'selection_file_sha256': hashes, 'test_cells': len(cells),
            'completion_marker_written': False, 'recovery_admitted': False,
            'historical_unjournaled_outcomes_resolved': False}
