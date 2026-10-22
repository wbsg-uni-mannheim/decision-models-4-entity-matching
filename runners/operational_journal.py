"""Validate drain metadata without counting the same paid attempts twice."""
from collections import Counter
import hashlib
import json


def key(row):
    return tuple(row[k] for k in ('pair_id', 'request_hash', 'completed_at'))


def inspect_operations(path, cases, failures, protocol, model, variant, run_id):
    raw = path.read_bytes() if path.exists() else b''
    rows = [json.loads(line) for line in raw.splitlines()]
    inputs = {c['pair_id']: c for c in cases}
    failure_ops = [r for r in failures if r.get('operational_stop')]
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate operational event')
    matched = []
    for row in rows:
        c = inputs.get(row['pair_id'])
        if (c is None or row.get('study_run_id') != run_id or type(row.get('study_run_id')) is not int
                or row['model_alias'] != model or row['variant'] != variant
                or row['benchmark'] != c['benchmark'] or row['input_hash'] != c['input_hash']
                or row['request_hash'] != protocol.digest(protocol.payload(c, model, variant))
                or 'attempts' in row or not row.get('error')):
            raise ValueError('Invalid operational identity or duplicate attempt payload')
        stop = row['operational_stop']
        n = stop['previous_attempts_preserved']
        if type(n) is not int or not 0 <= n <= 2:
            raise ValueError('Invalid preserved retry count')
        ceiling = stop['budget_reservation_rejected']
        unknown = stop['outcome_unknown']
        if (type(ceiling) is not bool or type(unknown) is not bool or unknown == ceiling
                or stop['phase'] not in ('reserve', 'future')
                or (ceiling and (stop['phase'] != 'reserve' or stop['exception_type'] != 'RuntimeError'
                                 or row['error'] != 'Project spending ceiling reached'))):
            raise ValueError('Invalid operational classification')
        linked = [f for f in failure_ops if key(f) == key(row)]
        if n:
            if len(linked) != 1 or len(linked[0]['attempts']) != n:
                raise ValueError('Preserved attempts missing or duplicated')
            if {k: v for k, v in linked[0].items() if k != 'attempts'} != row:
                raise ValueError('Operational/failure record mismatch')
            matched.append(key(row))
        elif linked:
            raise ValueError('Unattempted reserve event must not create a failure charge row')
    if Counter(matched) != Counter(key(f) for f in failure_ops):
        raise ValueError('Orphan operational failure record')
    return {'events': len(rows), 'budget_rejections': sum(r['operational_stop']['budget_reservation_rejected'] for r in rows),
            'unknown_outcome_events': sum(r['operational_stop']['outcome_unknown'] for r in rows),
            'previous_attempts_in_failure_journal': sum(r['operational_stop']['previous_attempts_preserved'] for r in rows),
            'additional_paid_attempts_counted': 0,
            'sha256': hashlib.sha256(raw).hexdigest() if path.exists() else None,
            'cost_scope': 'Recorded attempts only; operational metadata is not a charge. Unknown outcomes remain unpriced.'}
