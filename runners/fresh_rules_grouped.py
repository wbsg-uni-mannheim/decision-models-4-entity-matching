"""Offline-prepared transport, durable delta reconciliation and frozen report.

No CLI, job submission, admission, funding, locks or retry. The future wrapper
must check the actual entire prerequisite histories/current gates and hold root
shared plus replica exclusive locks through the report. A supplied process credit
is never closed from outstanding=0 alone. Existing manual answers are preserved
and excluded from incremental charges by exact durable byte prefixes.
"""
import hashlib
import json
import math
from pathlib import Path
import types

import fresh_rules_drained as component
import fresh_schema_selection as files
from grouped_inference import build as grouped_build
from repeated_runs import replica_path, verify
from rules_drained import bind

REPORT_DIR = 'artifacts/results/expanded_rules'
MARKER = 'artifacts/expanded_rules_complete.json'
REPORT_NAMES = ('metrics.json', 'intervals.json', 'costs.json', 'macro.json', 'rules.json',
                'provenance.json', 'metrics.csv', 'COST_README.txt')


def identity(root, target, rid):
    if type(rid) is not int or rid not in (2, 3) or target != replica_path(root, rid):
        raise ValueError('Exact fresh replica identity required')


def prefix(path):
    raw = path.read_bytes() if path.exists() else b''
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Incomplete durable journal line')
    return {'exists': path.exists(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def layout(target, cells):
    paths = {str((Path('runs')/c['namespace']/c['model']/c['variant']/c['benchmark']/(c['split']+suffix)))
             for c in cells for suffix in component.SUFFIXES}
    for rel in paths:
        files.own_path(target, rel)
    observed = {str(p.relative_to(target)) for c in cells
        for p in (target/'runs'/c['namespace']/c['model']/c['variant']/c['benchmark']).glob(c['split']+'*.jsonl')}
    if not observed <= paths:
        raise ValueError('Unreviewed rule/manual/control journal')
    return sorted(paths)


def capture_prior(root, target, rid, rules, protocol):
    """Read-only full own-control/manual replay before a wrapper funds credit."""
    identity(root, target, rid)
    component.prerequisites(target)
    component.untouched(target, protocol)
    frozen, _, cells, protected = component.prepare(target, rules, protocol)
    controls = [dict(c, variant='B0', namespace='quality') for c in cells if c['variant'] == 'R_rules']
    layout(target, cells+controls)
    journals = {p: prefix(files.own_path(target, p)) for p in layout(target, cells)}
    control_hashes = component.hashes(target, controls)
    checked = component.audit(target, rid, frozen, protocol, cells)
    component.audit(target, rid, frozen, protocol, controls, require_complete=True)
    if journals != {p: prefix(target/p) for p in journals} or control_hashes != component.hashes(target, controls):
        raise ValueError('Prior cache changed during raw replay')
    component.unchanged(target, protected)
    return dict(run_id=rid, replica=str(target), cells=cells, journals=journals,
        protected_sha256=protected, control_sha256_or_absent=control_hashes,
        valid_prior_responses=sum(c['responses'] for c in checked), admission_granted=False)


def private_component(credit):
    """Privately replace the completed component's accounting/writer transport."""
    if credit.closed or credit.attempts or credit.settled or credit.chunk != 8:
        raise ValueError('Fresh bounded eight dollar process credit required')
    def builder(inference, rid):
        isolated = types.SimpleNamespace(**{**vars(inference), 'account': credit.account,
            '_on_journal_failure': lambda: setattr(credit, 'journal_io_failed', True)})
        return grouped_build(isolated, rid)
    original = bind(component.run_rules,
        make_wrapper=lambda *_: types.SimpleNamespace(authorized_build=builder))
    def run(root, target, rid, inference, rules, protocol, journal_dir):
        identity(root, target, rid)
        if credit.root.resolve() != root.resolve() or credit.closed or credit.attempts:
            raise ValueError('Credit belongs to another root or was already used')
        try:
            return original(root, target, rid, inference, rules, protocol, journal_dir)
        except OSError:
            # Includes an unconfirmed model journal write. Readable bytes alone
            # cannot establish durable storage after fsync reports failure.
            credit.journal_io_failed = True
            raise
    return run


def model_completions(journal_dir, protocol, successful=False):
    result = {}
    for phase in ('prompt', 'test'):
        path = journal_dir/(phase+'_models.jsonl')
        if not path.exists():
            if phase == 'test' and not successful and result['prompt']['status'] == 'failed':
                continue
            raise ValueError('Complete phase model ends required')
        prefix(path)
        events = component.rows(path)
        ends = events[1:-1]
        if (len(events) != len(protocol.MODELS)+2 or events[0].get('event') != 'start'
                or events[0].get('models') != list(protocol.MODELS)
                or events[0].get('invocations_per_model') != 1
                or events[-1].get('event') != 'end'
                or events[-1].get('status') not in ('complete', 'failed')
                or events[-1].get('models_drained') != len(protocol.MODELS)
                or {e.get('model') for e in ends} != set(protocol.MODELS)
                or any(e.get('event') != 'model_end' or e.get('status') not in ('complete', 'failed') for e in ends)):
            raise ValueError('Complete phase model ends required')
        complete = all(e['status'] == 'complete' for e in ends)
        if ((events[-1]['status'] == 'complete') != complete or (successful and not complete)
                or (phase == 'test' and result['prompt']['status'] != 'complete')):
            raise ValueError('Complete development required before test model ends')
        result[phase] = dict(status=events[-1]['status'], sha256=files.sha(path))
    return result


def read_prior(root, target, rid, prior_path, prior_sha, rules, protocol):
    identity(root, target, rid)
    files.own_path(target, prior_path)
    if files.sha(prior_path) != prior_sha:
        raise ValueError('Durable prior receipt changed')
    prior = json.loads(prior_path.read_text())
    frozen, _, cells, protected = component.prepare(target, rules, protocol)
    controls = [dict(c, variant='B0', namespace='quality') for c in cells if c['variant'] == 'R_rules']
    if (prior.get('replica') != str(target) or prior.get('run_id') != rid or prior.get('cells') != cells
            or prior.get('protected_sha256') != protected
            or set(prior['journals']) != set(layout(target, cells))
            or prior.get('control_sha256_or_absent') != component.hashes(target, controls)):
        raise ValueError('Prior identity, sources or controls changed')
    layout(target, cells+controls)
    component.unchanged(target, protected)
    return prior, frozen, cells, controls


def reconcile(root, target, rid, prior_path, prior_sha, journal_dir, credit, rules, protocol):
    if credit.root.resolve() != root.resolve() or credit.chunk != 8 or credit.closed:
        raise ValueError('Wrong or closed process credit')
    if getattr(credit, 'journal_io_failed', False):
        raise ValueError('Journal durability failed; keep credit')
    provenance = json.loads((root/'artifacts/budget.json').read_text())['inference_credits'][credit.identity]['provenance']
    if provenance.get('prior_sha256') != prior_sha or provenance.get('run_id') != rid:
        raise ValueError('Credit does not bind the durable prior receipt')
    ends = model_completions(journal_dir, protocol)
    prior, frozen, cells, controls = read_prior(root, target, rid, prior_path, prior_sha, rules, protocol)
    observed = component.hashes(target, cells)
    # Replaying all old and new raw records catches foreign run IDs, request or
    # input drift, parser/prediction corruption, and unknown operations.
    component.audit(target, rid, frozen, protocol, cells)
    component.audit(target, rid, frozen, protocol, controls, require_complete=True)
    known = []; unknown = attempts = 0
    for rel, old in prior['journals'].items():
        path = files.own_path(target, rel)
        raw = path.read_bytes() if path.exists() else b''
        if (type(old['bytes']) is not int or old['bytes'] < 0 or len(raw) < old['bytes']
                or (old['exists'] and not path.exists())
                or hashlib.sha256(raw[:old['bytes']]).hexdigest() != old['sha256']
                or (old['bytes'] and raw[old['bytes']-1:old['bytes']] != b'\n')
                or (raw and not raw.endswith(b'\n'))):
            raise ValueError('Durable prior prefix or current journal changed')
        for line in raw[old['bytes']:].splitlines():
            row = json.loads(line)
            if rel.endswith('.operations.jsonl'):
                if row['operational_stop']['outcome_unknown']:
                    raise ValueError('Unknown operational outcome; keep credit')
            elif not rel.endswith('.executions.jsonl'):
                for a in row['attempts']:
                    attempts += 1
                    if a['cost'] is None: unknown += 1
                    else: known.append(a['cost'])
    charged = math.fsum(known)+.05*unknown
    if (credit.outstanding or credit.attempts != credit.settled or attempts != credit.settled
            or unknown != credit.unknown or not math.isclose(charged, credit.charged, rel_tol=0, abs_tol=1e-8)):
        raise ValueError('Unreconciled durable incremental attempts; keep credit')
    if observed != component.hashes(target, cells) or ends != model_completions(journal_dir, protocol):
        raise ValueError('Raw or model journals changed during reconciliation')
    read_prior(root, target, rid, prior_path, prior_sha, rules, protocol)
    credit.checkpoint(close=True)
    return dict(attempts=attempts, unknown=unknown, charged=charged,
        prior_responses_excluded=prior['valid_prior_responses'], phases=ends)


def finish_report(root, target, rid, rules, protocol, analysis, journal_dir, staging):
    """Caller must hold both writer locks through this exclusive publication."""
    identity(root, target, rid)
    staging = files.own_path(target, staging)
    if not staging.is_relative_to(target/'artifacts') or any((target/p).exists() for p in (REPORT_DIR, MARKER)) or staging.exists():
        raise ValueError('Existing report/partial publication requires review')
    verify(root, rid)
    frozen, spec, cells, protected = component.prepare(target, rules, protocol)
    controls = [dict(c, variant='B0', namespace='quality') for c in cells if c['variant'] == 'R_rules']
    ends = model_completions(journal_dir, protocol, successful=True)
    before = component.hashes(target, cells+controls)
    layout(target, cells+controls)
    component.audit(target, rid, frozen, protocol, cells+controls, require_complete=True)
    staging.mkdir(exist_ok=False)
    load = bind(analysis.load, PROJECT=target, payload=frozen.payload)
    def load_cell(m, v, b, split):
        return load(m, v, b, split, namespace=frozen.CONDITIONS.get(v, 'quality'))
    markers = []
    def capture(path, value):
        if path != staging/MARKER: raise ValueError('Unexpected frozen marker')
        markers.append(value)
    def read(path):
        if path != staging/'configs/learned_rules.json': raise ValueError('Unexpected frozen report read')
        return json.loads((target/'configs/learned_rules.json').read_text())
    report = bind(rules.report, PROJECT=staging, load_cell=load_cell, read=read, write_frozen=capture)
    report(spec)
    expected = dict(status='complete', test_cells_per_condition=63, development_cells_per_condition=56,
        evaluated_conditions=['R_rules', 'X_P_domain_0'], controls=['B0', 'X_P_domain_0'],
        human_rule_quality_review='not performed', full_expanded_study_complete=False)
    out = staging/REPORT_DIR
    if markers != [expected] or {p.name for p in out.iterdir()} != set(REPORT_NAMES):
        raise ValueError('Frozen report output mismatch')
    verify(root, rid); component.unchanged(target, protected)
    layout(target, cells+controls)
    if before != component.hashes(target, cells+controls) or ends != model_completions(journal_dir, protocol, True):
        raise ValueError('Report sources changed')
    for name in REPORT_NAMES:
        files.durable_write(target/REPORT_DIR/name, (out/name).read_bytes())
    files.sync_directory(target/REPORT_DIR)
    files.durable_write(target/MARKER, (protocol.canonical(expected)+'\n').encode())
    return {p: files.sha(target/p) for p in [MARKER, *(REPORT_DIR+'/'+n for n in REPORT_NAMES)]}
