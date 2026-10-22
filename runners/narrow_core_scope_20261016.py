"""Pure narrow-scope guard for future execution integration.

No inference, admission, selection, cache validation, locks, funding or CLI.
The caller must prove own-run development selection and source provenance.
This module bounds what that separately admitted caller may request; it does
not make an old broad wrapper or a blocked historical attempt runnable.
"""

import hashlib
from pathlib import Path

PINS = {
    'configs/narrow_three_runs_completion_20261016.json':
        'a853c587254db5c8fe031b6236a91b08681e55650b25d8ef30bf2fc9893790a2',
    'src/protocol.py': '29f262bdac311c69353143c994ae14e906b29e002a1edf00cbfe7733878659ec',
    'src/expanded_protocol.py': '2dc4d8d1ff24ede5096acd24f7713ef9aa97793bcc6226d6bf57c785dfe320b6',
    'src/expanded_schema.py': 'aaf916eb209e125b340fe30d08456b4d6fa58de756708414a0c1224538e737b1',
    'src/expanded_confirmation.py': 'dcf0323b256ded6ec57cd8d92dd3724cdb7a293fe94b82a66e33184c0a326b7d',
    'src/expanded_calibration.py': '18c9ba45a4f30f806c5b8adc225386a3b9ab47ec2a6947d5a9e555e5de16111d',
}
MODELS = ('jev', 'clef', 'clef_flash', 'luna_decisions', 'pplx', 'kev', 'luna')
BENCHMARKS = ('abt-buy', 'wdc', 'dblp-scholar', 'walmart-amazon',
              'dn7-walmart-amazon', 'billiger-de', 'dblp-acm', 'semi-heter')
PROMPTS = tuple(f'X_P_{category}_{i}' for category in
                ('minimal', 'identity', 'detailed', 'domain', 'conservative', 'evidence')
                for i in range(3))
BINARY = ('C_match', 'C_match_reverse', 'C_same', 'C_same_reverse')
SCHEMAS = BINARY + ('C_review',)


def verify_sources(root):
    """Read-only source pins, not scheduler/history/cache admission."""
    root = Path(root)
    for relative, expected in PINS.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Narrow scope source changed: {relative}')
    return dict(PINS)


def check_run(run_id):
    if type(run_id) is not int or run_id not in (1, 2, 3):
        raise ValueError('Strict original integer Run-ID 1/2/3 required')


def cell(run_id, namespace, model, variant, benchmark, split):
    check_run(run_id)
    if model not in MODELS or benchmark not in BENCHMARKS:
        raise ValueError('Outside narrow model/benchmark scope')
    return (run_id, namespace, model, variant, benchmark, split)


def development_cells(run_id):
    """Candidate scopes only. Existing complete own-run cells must be reused.

    B0 development is a read-only prerequisite, not a new-call candidate.
    No screening, pilot, aligned representation, timing or supplementary split.
    """
    check_run(run_id)
    return frozenset(
        cell(run_id, ns, m, v, b, 'prompt')
        for ns, variants in [('quality', PROMPTS + ('S1',)), ('schema', SCHEMAS)]
        for m in MODELS for v in variants for b in BENCHMARKS
    )


def selected_test_cells(run_id, prompt_choices, binary_choices):
    """Validate complete narrow choice rows; never select from test scores.

    Each input row needs run_id/model/benchmark/regime/variant. A separate
    verified immutable receipt must establish these are own-run DEV winners.
    No choice is generated here, and global consistency is checked explicitly.
    """
    check_run(run_id)
    result = set()
    expected = {(m, b, r) for m in MODELS for b in BENCHMARKS
                for r in ('global', 'per_benchmark')}
    for rows, variants, namespace in [(prompt_choices, PROMPTS, 'quality'),
                                      (binary_choices, BINARY, 'schema')]:
        seen = set()
        globals_by_model = {m: set() for m in MODELS}
        for row in rows:
            check_run(row['run_id'])
            key = (row['model'], row['benchmark'], row['regime'])
            if (row['run_id'] != run_id or key not in expected or key in seen
                    or row['variant'] not in variants):
                raise ValueError('Foreign, duplicate, broad or invalid selected condition')
            seen.add(key)
            if row['regime'] == 'global':
                globals_by_model[row['model']].add(row['variant'])
            result.add(cell(run_id, namespace, row['model'], row['variant'],
                            row['benchmark'], 'test'))
        if seen != expected or any(len(v) != 1 for v in globals_by_model.values()):
            raise ValueError('Complete own-run global/local choice matrix required')
    for m in MODELS:
        for b in BENCHMARKS:
            result.add(cell(run_id, 'quality', m, 'S1', b, 'test'))
            result.add(cell(run_id, 'schema', m, 'C_review', b, 'test'))
    return frozenset(result)


def calibration_partition(pool, route, extra):
    """Exact first100 frozen pool, alternating50/50; no new data or reordering.

    Caller supplies hash-verified own-replica frozen inputs. Return source IDs,
    not requests. Unavailable label partitions need no new calibration calls.
    """
    def indexed(rows):
        result = {}
        hashes = set()
        for row in rows:
            key = row['pair_id']
            if (type(key) is not str or key in result or row['input_hash'] in hashes
                    or type(row['label']) is not int or row['label'] not in (0, 1)):
                raise ValueError('Nonunique or invalid calibration input')
            result[key] = row
            hashes.add(row['input_hash'])
        return result

    all_pool, old, new = indexed(pool), indexed(route), indexed(extra)
    if (len(pool) < 100 or set(old) & set(new)
            or {r['input_hash'] for r in route} & {r['input_hash'] for r in extra}):
        raise ValueError('Incomplete pool or overlapping source blocks')
    merged = {**old, **new}
    if any(key not in merged or merged[key] != row for key, row in all_pool.items()):
        raise ValueError('Pool case differs from original route/extra source')
    fit, tune = pool[:100:2], pool[1:100:2]
    available = all({r['label'] for r in rows} == {0, 1} for rows in (fit, tune))
    chosen = {r['pair_id'] for r in pool[:100]} if available else set()
    return {
        'available': available,
        'fit_pair_ids': [r['pair_id'] for r in fit],
        'tune_pair_ids': [r['pair_id'] for r in tune],
        'route_pair_ids': [r['pair_id'] for r in route if r['pair_id'] in chosen],
        'extra_pair_ids': [r['pair_id'] for r in extra if r['pair_id'] in chosen],
        'label_budget': 100,
    }


def validate_call(run_id, candidate, pair_ids, expected_missing_ids, allowed_cells):
    """Final scope assertion only; no substitution for financial/runtime gates.

    expected_missing_ids must come from independent current cache validation.
    Equality, rather than subset acceptance, prevents silent shortening. For
    calibration, allowed_cells and IDs must be built from calibration_partition.
    """
    check_run(run_id)
    if (len(candidate) != 6 or type(candidate[0]) is not int
            or candidate[0] != run_id or candidate not in allowed_cells):
        raise ValueError('Invocation outside exact own-run cell matrix')
    _, ns, model, variant, benchmark, split = candidate
    cell(run_id, ns, model, variant, benchmark, split)
    valid = ((ns == 'quality' and variant in PROMPTS + ('S1',)
              and split in ('prompt', 'test'))
             or (ns == 'schema' and variant in SCHEMAS and split in ('prompt', 'test'))
             or (ns == 'quality' and variant == 'B0' and split == 'calibration_extra'
                 and model != 'luna'))
    if not valid:
        raise ValueError('Baseline repeat, broad stage or unsupported invocation')
    if (not pair_ids or any(type(p) is not str for p in pair_ids)
            or len(set(pair_ids)) != len(pair_ids)
            or list(pair_ids) != list(expected_missing_ids)):
        raise ValueError('Exact ordered current missing-pair scope required')
    if run_id == 1 and candidate[1:] == (
            'quality', 'luna', 'X_P_conservative_0', 'wdc', 'test'):
        raise ValueError('Unresolved original raw-gap cell remains blocked')
    return True
