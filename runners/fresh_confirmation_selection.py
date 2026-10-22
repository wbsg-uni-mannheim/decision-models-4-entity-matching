"""Durable own-run development freeze; no admission, test calls or final report.

A future admitted wrapper must hold the global shared and replica exclusive
locks throughout development, this freeze and selected tests. A partial write
is deliberately not resumable: only the final receipt makes a freeze usable.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path

from confirmation_drained import CONFIRMATION_SHA
from fresh_confirmation_development_drained import matrix, selection_in_memory
from fresh_screen_drained import PAYLOAD_SHA
from repeated_runs import replica_path, verify

ATTEMPT = 'artifacts/fresh_confirmation_selection_freeze'
RECEIPT = ATTEMPT+'/committed.json'
OUTPUTS = ('configs/expanded_selected_variants.json',
           'configs/expanded_confirmation_cells.json',
           'artifacts/results/expanded_confirmation/development_metrics.json',
           'artifacts/results/expanded_confirmation/development_metrics.csv')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def identity(root, target, run_id, confirmation, expanded, analysis, protocol):
    root, target = Path(root).resolve(), Path(target).resolve()
    if type(run_id) is not int or run_id not in (2, 3) or target != replica_path(root, run_id):
        raise ValueError('Fresh replica identity required')
    if confirmation.PROJECT != target or analysis.PROJECT != target:
        raise ValueError('Wrong confirmation workspace')
    verify(root, run_id)
    for module, expected in ((confirmation, CONFIRMATION_SHA), (expanded, PAYLOAD_SHA)):
        if sha(Path(module.__file__)) != expected:
            raise ValueError('Frozen source changed')
    if (confirmation.MODELS != protocol.MODELS or confirmation.BENCHMARKS != protocol.BENCHMARKS
            or confirmation.PROMPT_VARIANTS != ['X_P_'+c+'_'+str(i) for c in expanded.PROMPTS for i in range(3)]
            or confirmation.SERIALIZATIONS != ['B0', 'S1', 'X_R_aligned']
            or confirmation.payload is not expanded.payload):
        raise ValueError('Frozen matrix/payload changed')
    confirmation.verify()
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('expanded_E2_E3_screen', {}).get('status') != 'complete':
        raise ValueError('Screen prerequisite incomplete')
    return root, target


def development_provenance(target, confirmation, journal):
    journal = Path(journal).resolve()
    if not journal.is_relative_to(target/'artifacts') or journal.name != 'models.jsonl':
        raise ValueError('Own-run development model journal required')
    events = [json.loads(s) for s in journal.read_text().splitlines()]
    ends = events[1:-1]
    if (not events or events[0].get('event') != 'start'
            or events[0].get('models') != list(confirmation.MODELS)
            or events[0].get('invocations_per_model') != 1
            or events[-1].get('event') != 'end' or events[-1].get('status') != 'complete'
            or events[-1].get('models_drained') != len(confirmation.MODELS)
            or len(ends) != len(confirmation.MODELS)
            or {e.get('model') for e in ends} != set(confirmation.MODELS)
            or any(e.get('event') != 'model_end' or e.get('status') != 'complete' for e in ends)):
        raise ValueError('All development model completions required')
    paths = {journal}
    for m, v, b in matrix(confirmation):
        paths.add(target/'data'/b/'prompt.json')
        for suffix in ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl'):
            paths.add(target/'runs/quality'/m/v/b/('prompt'+suffix))
    return {str(p.relative_to(target)): sha(p) for p in sorted(paths)}


def output_bytes(candidate, confirmation, protocol):
    # Match the frozen select() report's exact row and column order.
    indexed = {(r['model'], r['variant'], r['benchmark']): r for r in candidate['development_metrics']}
    rows = []
    for m in confirmation.MODELS:
        for b in confirmation.BENCHMARKS:
            for v in confirmation.PROMPT_VARIANTS+confirmation.SERIALIZATIONS:
                row = indexed[m, v, b]
                rows.append({'model': m, 'benchmark': b, 'variant': v,
                             **{k: val for k, val in row.items() if k not in ('model', 'benchmark', 'variant')}})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    values = (candidate['selected'], candidate['test_cells'], rows)
    return dict(zip(OUTPUTS, [(protocol.canonical(v)+'\n').encode() for v in values]+[stream.getvalue().encode()], strict=True))


def durable_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze_development(root, target, run_id, confirmation, expanded, analysis, protocol, model_journal):
    _, target = identity(root, target, run_id, confirmation, expanded, analysis, protocol)
    if ((target/ATTEMPT).exists() or any((target/p).exists() for p in OUTPUTS)
            or (target/'artifacts/expanded_confirmation_complete.json').exists()
            or (target/'artifacts/results/expanded_confirmation').exists()):
        raise ValueError('Existing freeze or report requires separate review')
    before = development_provenance(target, confirmation, model_journal)
    candidate = selection_in_memory(target, run_id, confirmation, analysis, protocol)
    if before != development_provenance(target, confirmation, model_journal):
        raise ValueError('Development changed during selection')
    rendered = output_bytes(candidate, confirmation, protocol)
    # Claim an exclusive attempt before publishing any part of the freeze.
    (target/ATTEMPT).mkdir()
    for name, raw in rendered.items():
        durable_write(target/name, raw)
    # Persist newly created directory entries as well as file contents. The
    # final receipt is published only after every ancestor is synchronized.
    directories = {target/ATTEMPT}
    for name in rendered:
        directory = (target/name).parent
        while directory.is_relative_to(target):
            directories.add(directory)
            if directory == target:
                break
            directory = directory.parent
    directories.add(target/'artifacts')
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    if before != development_provenance(target, confirmation, model_journal):
        raise ValueError('Development changed during freeze; no commit')
    receipt = {'run_id': run_id, 'committed_at': datetime.now(timezone.utc).isoformat(),
               'status': 'development_selection_frozen', 'model_journal': str(Path(model_journal).resolve().relative_to(target)),
               'development_sha256': before, 'selection_sha256': candidate['selection_sha256'],
               'outputs_sha256': {n: hashlib.sha256(raw).hexdigest() for n, raw in rendered.items()},
               'selected_regime_rows': len(candidate['selected']), 'unique_test_cells': len(candidate['test_cells']),
               'development_cells': len(candidate['development_metrics']),
               'test_used_for_selection': False, 'admission_granted': False,
               'test_calls_started': False, 'final_report_written': False}
    durable_write(target/RECEIPT, (protocol.canonical(receipt)+'\n').encode())
    return receipt


def verify_frozen(root, target, run_id, confirmation, expanded, analysis, protocol):
    """Read-only prerequisite for a future test component; never authorizes calls."""
    _, target = identity(root, target, run_id, confirmation, expanded, analysis, protocol)
    if not (target/RECEIPT).exists():
        raise ValueError('No committed development freeze')
    receipt = json.loads((target/RECEIPT).read_text())
    if (type(receipt.get('run_id')) is not int or receipt['run_id'] != run_id
            or receipt.get('status') != 'development_selection_frozen'
            or receipt.get('test_used_for_selection') is not False
            or receipt.get('admission_granted') is not False
            or receipt.get('test_calls_started') is not False
            or receipt.get('final_report_written') is not False):
        raise ValueError('Invalid freeze identity or selection provenance')
    journal = target/receipt['model_journal']
    if receipt['development_sha256'] != development_provenance(target, confirmation, journal):
        raise ValueError('Frozen development provenance changed')
    candidate = selection_in_memory(target, run_id, confirmation, analysis, protocol)
    rendered = output_bytes(candidate, confirmation, protocol)
    expected = {n: hashlib.sha256(raw).hexdigest() for n, raw in rendered.items()}
    if (receipt['outputs_sha256'] != expected or receipt['selection_sha256'] != candidate['selection_sha256']
            or any((target/n).read_bytes() != raw for n, raw in rendered.items())
            or receipt['selected_regime_rows'] != len(candidate['selected'])
            or receipt['unique_test_cells'] != len(candidate['test_cells'])
            or receipt['development_cells'] != len(candidate['development_metrics'])):
        raise ValueError('Frozen selection/report differs from full development reconstruction')
    if receipt['development_sha256'] != development_provenance(target, confirmation, journal):
        raise ValueError('Development changed during verification')
    return candidate
