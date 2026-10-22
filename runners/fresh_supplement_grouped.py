"""Private grouped transport and exact durable supplement credit reconciliation.

Offline component only: no CLI, scheduler admission, funding, locks, report or
completion marker. A future separate wrapper must prove the full successful
core history and all current gates, persist/fsync this prior before funding,
bind its digest/run/attempt in the credit, and hold both locks through reporting.
The completed per-call component and wrapper remain unchanged.
"""
import hashlib
import json
import math
from pathlib import Path
import types

import fresh_supplement_drained as component
import fresh_schema_selection as files
from fresh_rules_grouped import identity, prefix
from grouped_inference import build as grouped_build
from rules_drained import bind

SUFFIXES = ('.jsonl', '.failures.jsonl', '.executions.jsonl', '.operations.jsonl')


def layout(target, protocol):
    paths = {f'runs/quality/{m}/B0/{component.BENCHMARK}/test{s}'
             for m in protocol.MODELS for s in SUFFIXES}
    observed = {str(p.relative_to(target)) for p in (target/'runs/quality').glob(
        '*/B0/'+component.BENCHMARK+'/*.jsonl')}
    if not observed <= paths:
        raise ValueError('Unreviewed supplement journal')
    for rel in paths:
        files.own_path(target, rel)
    return sorted(paths)


def sources(target, supplement, protocol):
    if supplement.PROJECT != target or supplement.BENCHMARK != component.BENCHMARK or supplement.MODELS != protocol.MODELS:
        raise ValueError('Wrong supplement workspace or protocol')
    if files.sha(Path(supplement.__file__)) != component.SUPPLEMENT_SHA:
        raise ValueError('Frozen supplement source changed')
    supplement.verify()
    return {p: files.sha(target/p) for p in (
        'configs/billiger50_manifest.json', 'data/'+component.BENCHMARK+'/test.json')}


def capture_prior(root, target, rid, supplement, protocol, journal_dir):
    """Read only; this receipt is not a scheduler/history admission."""
    identity(root, target, rid)
    journal_dir = files.own_path(target, journal_dir)
    if not journal_dir.is_relative_to(target/'artifacts') or journal_dir.exists():
        raise ValueError('Exclusive supplement attempt required')
    progress = json.loads((target/'artifacts/progress.json').read_text())
    if progress.get('AP3_AP4_analysis', {}).get('status') != 'complete_automated':
        raise ValueError('Core analysis prerequisite incomplete')
    if 'supplement_billiger50' in progress or (target/'artifacts/results/billiger50').exists():
        raise ValueError('Existing supplement stage/report requires review')
    protected = sources(target, supplement, protocol)
    journals = {p: prefix(target/p) for p in layout(target, protocol)}
    checked = component.inventory(target, rid, protocol)
    if any(c['failure_rows'] or c['operations']['events'] for c in checked):
        raise ValueError('Historical supplement failures/operations require review')
    if (protected != sources(target, supplement, protocol)
            or journals != {p: prefix(target/p) for p in layout(target, protocol)}):
        raise ValueError('Prior changed during raw replay')
    return dict(run_id=rid, replica=str(target), journal_dir=str(journal_dir.relative_to(target)),
        stage='supplement', protected_sha256=protected, journals=journals,
        valid_prior_responses=sum(c['responses'] for c in checked), admission_granted=False)


def read_prior(root, target, rid, prior_path, prior_sha, journal_dir, supplement, protocol):
    identity(root, target, rid)
    prior_path = files.own_path(target, prior_path)
    journal_dir = files.own_path(target, journal_dir)
    if (not prior_path.is_relative_to(target/'artifacts')
            or not journal_dir.is_relative_to(target/'artifacts')
            or files.sha(prior_path) != prior_sha):
        raise ValueError('Durable prior receipt changed')
    prior = json.loads(prior_path.read_text())
    if (prior.get('run_id') != rid or prior.get('replica') != str(target)
            or prior.get('stage') != 'supplement'
            or prior.get('journal_dir') != str(journal_dir.relative_to(target))
            or prior.get('protected_sha256') != sources(target, supplement, protocol)
            or set(prior.get('journals', {})) != set(layout(target, protocol))):
        raise ValueError('Prior identity, sources or journal layout changed')
    return prior


def credit_identity(root, target, rid, prior_sha, journal_dir, credit):
    if credit.root.resolve() != root.resolve() or credit.chunk != 8 or credit.closed:
        raise ValueError('Wrong or closed bounded process credit')
    ledger = json.loads((root/'artifacts/budget.json').read_text())
    row = ledger['inference_credits'][credit.identity]
    provenance = row['provenance']
    if (row['status'] != 'active' or provenance.get('prior_sha256') != prior_sha
            or provenance.get('run_id') != rid or provenance.get('stage') != 'supplement'
            or provenance.get('journal_dir') != str(journal_dir.relative_to(target))):
        raise ValueError('Credit does not bind this durable prior and attempt')


def private_component(credit, prior_path, prior_sha):
    """Bind grouped writes privately, leaving imported frozen functions intact."""
    if credit.closed or credit.attempts or credit.settled or credit.chunk != 8:
        raise ValueError('Fresh eight dollar process credit required')
    def builder(inference, rid):
        isolated = types.SimpleNamespace(**{**vars(inference), 'account': credit.account,
            '_on_journal_failure': lambda: setattr(credit, 'journal_io_failed', True)})
        return grouped_build(isolated, rid)
    original = bind(component.run_supplement,
        make_wrapper=lambda *_: types.SimpleNamespace(authorized_build=builder))
    def run(root, target, rid, inference, supplement, protocol, journal_dir):
        identity(root, target, rid)
        credit_identity(root, target, rid, prior_sha, journal_dir, credit)
        if credit.attempts or credit.settled:
            raise ValueError('Credit already used')
        prior = read_prior(root, target, rid, prior_path, prior_sha, journal_dir, supplement, protocol)
        if prior != capture_prior(root, target, rid, supplement, protocol, journal_dir):
            raise ValueError('Prior cache changed before invocation')
        try:
            return original(root, target, rid, inference, supplement, protocol, journal_dir)
        except OSError:
            credit.journal_io_failed = True
            raise
    return run


def model_completions(journal_dir, protocol):
    path = journal_dir/'models.jsonl'
    if not path.exists():
        raise ValueError('Complete model ends required')
    prefix(path)
    events = [json.loads(s) for s in path.read_text().splitlines()]
    ends = events[1:-1]
    if (len(events) != len(protocol.MODELS)+2 or events[0].get('event') != 'start'
            or events[0].get('models') != list(protocol.MODELS)
            or events[0].get('invocations_per_model') != 1
            or events[-1].get('event') != 'end'
            or events[-1].get('status') not in ('complete', 'failed')
            or events[-1].get('models_drained') != len(protocol.MODELS)
            or {e.get('model') for e in ends} != set(protocol.MODELS)
            or any(e.get('event') != 'model_end' or e.get('status') not in ('complete','failed') for e in ends)
            or ((events[-1]['status']=='complete') != all(e['status']=='complete' for e in ends))):
        raise ValueError('Complete model ends required')
    return dict(status=events[-1]['status'], sha256=files.sha(path))


def reconcile(root, target, rid, prior_path, prior_sha, journal_dir, credit, supplement, protocol):
    """Close only after full raw replay, all ends and exact new-attempt equality."""
    if getattr(credit, 'journal_io_failed', False):
        raise ValueError('Journal durability failed; keep credit')
    credit_identity(root, target, rid, prior_sha, journal_dir, credit)
    prior = read_prior(root, target, rid, prior_path, prior_sha, journal_dir, supplement, protocol)
    ends = model_completions(journal_dir, protocol)
    observed = {p: prefix(target/p) for p in layout(target, protocol)}
    checked = component.inventory(target, rid, protocol)
    known = []; attempts = unknown = 0
    for rel, old in prior['journals'].items():
        path = files.own_path(target, rel)
        raw = path.read_bytes() if path.exists() else b''
        if (type(old['bytes']) is not int or old['bytes'] < 0 or len(raw) < old['bytes']
                or (old['exists'] and not path.exists())
                or hashlib.sha256(raw[:old['bytes']]).hexdigest() != old['sha256']
                or (old['bytes'] and raw[old['bytes']-1:old['bytes']] != b'\n')
                or (raw and not raw.endswith(b'\n'))):
            raise ValueError('Durable prior prefix changed')
        for line in raw[old['bytes']:].splitlines():
            row = json.loads(line)
            if rel.endswith('.operations.jsonl'):
                if row['operational_stop']['outcome_unknown']:
                    raise ValueError('Unknown operation; keep credit')
            elif not rel.endswith('.executions.jsonl'):
                for attempt in row['attempts']:
                    attempts += 1
                    if attempt['cost'] is None: unknown += 1
                    else: known.append(attempt['cost'])
    charged = math.fsum(known)+.05*unknown
    if (credit.outstanding or credit.attempts != credit.settled or attempts != credit.settled
            or unknown != credit.unknown or not math.isclose(charged, credit.charged, rel_tol=0, abs_tol=1e-8)):
        raise ValueError('Unreconciled durable incremental attempts; keep credit')
    if (observed != {p: prefix(target/p) for p in layout(target, protocol)}
            or ends != model_completions(journal_dir, protocol)):
        raise ValueError('Raw or model journals changed during reconciliation')
    read_prior(root, target, rid, prior_path, prior_sha, journal_dir, supplement, protocol)
    credit_identity(root, target, rid, prior_sha, journal_dir, credit)
    credit.checkpoint(close=True)
    return dict(attempts=attempts, unknown=unknown, charged=charged, models=ends,
        prior_responses_excluded=prior['valid_prior_responses'],
        inference_complete=all(c['complete'] for c in checked) and ends['status']=='complete',
        admission_granted=False, report_written=False)
