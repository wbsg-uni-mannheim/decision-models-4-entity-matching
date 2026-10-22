"""Read-only, failure-triggered timing measurement plan. No execution/admission.

One interrupted 20-pair measurement is replaced in a new namespace. Its raw
answers, failures and costs stay immutable; they are not part of the replacement
latency/throughput sample. No quality answer is requested again. A future wrapper
must enforce all scheduler/history/budget/lock gates and publish an explicit
analysis source map. The frozen analysis cannot be run blindly on old paths.
"""
import hashlib
import json
import math
from pathlib import Path

PINS = {
    'artifacts/operations/grouped_core3_terminal_366980_20261016.json':
        'b90b868f88c69d5575889e917431953f8717dd477a74cf9f4f0a447c5dc14d36',
    'artifacts/verification/grouped_core3_terminal_366980_20261016.json':
        '077004d108488b8b3451aef72175367f5c82d4008a661751e80b753914677969',
    'artifacts/verification/core3_timing_terminal_366980_20261016.json':
        'f5967a4b746b4a5b282dfd8807ffa660cde3b07dde93e7f1eba56397f4a6bfcc',
    'artifacts/verification/core3_366980_ledger_delta_20261016.json':
        '92a804ec28385a614147977f912591d0dc8819af06d10143cb7cf2a1c9db7cb5',
    'artifacts/verification/core_phase_cost_scenarios_20261015.json':
        'f2bf640134521fdabceb520d934d9b400115aab7cf99cdbef26886a9745d806f',
}
REPLACEMENT = 'timing_remeasurement_366980_v1_concurrent8'
GAP = ('timing_concurrent8', 'kev', 'billiger-de')
ATTEMPT = 'artifacts/operations/core3_timing_recovery_366980_v1'
SUFFIXES = ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def base(target, namespace, model, benchmark):
    return target/'runs'/namespace/model/'B0'/benchmark


def plan_from_evidence(timing, remaining, models, benchmarks):
    """Pure plan construction; fixed failure criterion, never speed selection."""
    cells = {(c['namespace'], c['model'], c['benchmark']): c for c in timing['cells']}
    expected = {(ns, m, b) for m in models for b in benchmarks
                for ns in ('timing_serial', 'timing_concurrent8')}
    require(len(cells) == len(timing['cells']) == 112 and set(cells) == expected,
            'Exact 112-cell timing matrix required')
    prices = {tuple(c['key'][i] for i in (0, 1, 3)): c for c in remaining['cells']}
    require(len(prices) == len(remaining['cells']) == 21, 'Exact remaining cost matrix required')
    require(remaining['missing'] == 411 and not remaining['quote_or_hard_bound'], 'Missing-cost proof drift')
    require({k for k, c in cells.items() if c['missing']} == set(prices), 'Cost coverage mismatch')
    require([k for k, c in cells.items() if c['responses_without_execution_walltime']] == [GAP],
            'Only the independently diagnosed walltime gap is eligible')
    planned = []
    for m in models:
        for b in benchmarks:
            for ns, workers in (('timing_serial', 1), ('timing_concurrent8', 8)):
                key = (ns, m, b)
                c = cells[key]
                require(c['responses'] + c['missing'] == 20, 'Fixed first-20 sample required')
                if key == GAP:
                    require((c['responses'], c['missing'], c['attempts'], c['failure_rows'],
                             c['unknown_attempts'], c['execution_records'],
                             c['responses_covered_by_execution_records']) == (9, 11, 30, 7, 21, 0, 0),
                            'Interrupted measurement evidence drift')
                    action, source, count = 'replace_interrupted_measurement', REPLACEMENT, 20
                elif c['missing']:
                    require(c['responses'] == c['attempts'] == c['failure_rows'] == c['unknown_attempts']
                            == c['execution_records'] == 0 and c['missing'] == 20,
                            'Other prospective timing cells must be wholly unstarted')
                    action, source, count = 'measure_unstarted_cell', ns, 20
                else:
                    require(c['responses'] == c['responses_covered_by_execution_records'] == 20
                            and c['execution_records'] == 1 and c['failure_rows'] == 0,
                            'Retained cell must have one complete measurement')
                    action, source, count = 'retain_complete_measurement', ns, 0
                price = prices.get(key)
                if price:
                    require(price['missing'] == c['missing'], 'Price denominator mismatch')
                    for name in ('known_proxy_usd', 'historical_unknown_scenario_usd'):
                        require(math.isfinite(price[name]) and price[name] >= 0, 'Invalid historical price')
                planned.append({'original_namespace': ns, 'analysis_namespace': source,
                    'model': m, 'benchmark': b, 'variant': 'B0', 'split': 'prompt',
                    'workers': workers, 'limit': 20, 'action': action, 'new_responses': count,
                    'known_proxy_usd': price['known_proxy_usd'] * count / price['missing'] if price else 0,
                    'historical_unknown_scenario_usd':
                        price['historical_unknown_scenario_usd'] * count / price['missing'] if price else 0})
    require(sum(c['new_responses'] for c in planned) == 420, 'Unexpected recovery scope')
    require(math.isclose(math.fsum(c['known_proxy_usd'] for c in remaining['cells']),
                         remaining['known_proxy_usd'], abs_tol=1e-12), 'Cost total mismatch')
    return {'version': 'core3_timing_measurement_recovery_v1', 'run_id': 3, 'predecessor': '366980',
        'admitted': False, 'slurm_job_id': None, 'cells': planned,
        'new_responses': 420, 'previously_valid_timing_responses_remeasured': 9,
        'new_quality_responses': 0, 'retained_complete_measurements': 91,
        'original_missing_responses': 411, 'attempt': ATTEMPT,
        'known_proxy_usd': math.fsum(c['known_proxy_usd'] for c in planned),
        'historical_unknown_scenario_usd': math.fsum(c['historical_unknown_scenario_usd'] for c in planned),
        'quote_or_hard_bound': False, 'full_financing_guaranteed': False,
        'historical_failure_attempts_retained': 21,
        'historical_interrupted_measurement_known_usd': cells[GAP]['known_usd'],
        'historical_interrupted_measurement_unknown_allowance_usd': 1.05,
        'measurement_policy': 'Preserve 91 complete cells. Complete the 20 unstarted cells. Replace exactly '
            'one failed measurement with all original first 20 pairs in a new empty namespace, once. '
            'Failure-triggered, independent of observed speed or predictions; no best-of attempts. '
            'Any further failure stops and requires a new terminal audit, never automatic remeasurement.',
        'analysis_policy': 'Both latency records and execution walltime must use the same explicitly '
            'mapped measurement. Old nine successes and 21 failed attempts remain diagnostic/cost '
            'evidence, never mixed with replacement latency or throughput. All spending is retained. '
            'Different collection times and failure-contingent replacement must be disclosed.',
        'remaining_integration': ['bounded timing wrapper', 'full FOURTH history and current gates',
            'global shared and replica exclusive locks', 'fresh shared budget check and original per-call account',
            'exclusive durable attempt and per-cell start/end provenance',
            'private frozen analysis with consistent explicit source mapping',
            'independent final raw/execution/cost/report validation before Core completion']}


def prepare(root, protocol):
    """Read saved proofs and current hashes only; no writes or inference."""
    root = Path(root)
    target = root/'replications/run03'
    for rel, expected in PINS.items():
        require(sha(root/rel) == expected, 'Terminal proof/source drift: '+rel)
    timing = read(root/'artifacts/verification/core3_timing_terminal_366980_20261016.json')
    proof = read(root/'artifacts/verification/grouped_core3_terminal_366980_20261016.json')
    require(all(proof[k] is True for k in ('full_raw_input_request_parser_operations_verified',
                'history_and_admission_verified', 'credit_exactly_reconciled')), 'Terminal proof incomplete')
    require(proof['run_id'] == 3 and proof['job'] == '366980', 'Terminal identity mismatch')
    require(proof['remaining_cost_plan']['historical_source_sha256'] == PINS[
        'artifacts/verification/core_phase_cost_scenarios_20261015.json'], 'Historical source mismatch')
    for name, key in (('run.py', 'frozen_run_source_sha256'), ('analyze.py', 'analysis_source_sha256')):
        require(sha(target/'src'/name) == timing[key], 'Frozen timing source drift')
    for rel, expected in timing['protected_sha256_or_absent'].items():
        path = target/rel
        require((sha(path) if path.exists() else None) == expected, 'Timing/input state drift: '+rel)
    require(not (target/ATTEMPT).exists(), 'Exclusive future attempt already exists')
    require(not (target/'runs'/REPLACEMENT).exists(), 'Replacement namespace already exists')
    plan = plan_from_evidence(timing, proof['remaining_cost_plan'], protocol.MODELS, protocol.BENCHMARKS)
    inputs = {}
    for b in protocol.BENCHMARKS:
        cases = read(target/'data'/b/'prompt.json')[:20]
        require(len(cases) == 20 and len({c['pair_id'] for c in cases}) == 20, 'Invalid fixed timing input')
        require(all(c['benchmark'] == b and c['input_hash'] == protocol.digest(c['records']) for c in cases),
                'Timing input identity drift')
        inputs[b] = cases
    for cell in plan['cells']:
        cases = inputs[cell['benchmark']]
        cell['ordered_pair_ids'] = [c['pair_id'] for c in cases]
        cell['ordered_input_hashes'] = [c['input_hash'] for c in cases]
        cell['ordered_request_hashes'] = [protocol.digest(protocol.payload(c, cell['model'], 'B0')) for c in cases]
    plan['proof_sha256'] = dict(PINS)
    plan['current_timing_input_states_checked'] = len(timing['protected_sha256_or_absent'])
    plan['full_raw_replay_repeated'] = False
    plan['scheduler_admission_gate_executed'] = False
    return plan


def validate_measurement(target, cell, protocol):
    """Read-only exact full-cell acceptance; cannot accept a cached 9+11 splice.

    This does not replace admission, source pinning, durable-write confirmation,
    raw cost reconciliation, stage/model ends, or the final report verifier.
    """
    target = Path(target)
    require(cell['model'] in protocol.MODELS and cell['benchmark'] in protocol.BENCHMARKS,
            'Unknown model or benchmark')
    require(cell['original_namespace'] in ('timing_serial', 'timing_concurrent8'), 'Unknown timing regime')
    key = (cell['original_namespace'], cell['model'], cell['benchmark'])
    expected_namespace = REPLACEMENT if key == GAP else cell['original_namespace']
    require(cell['analysis_namespace'] == expected_namespace, 'Analysis source mapping drift')
    cases = read(target/'data'/cell['benchmark']/'prompt.json')[:20]
    require(len(cases) == 20 and len({c['pair_id'] for c in cases}) == 20, 'Twenty distinct cases required')
    require(cell['workers'] == (1 if cell['original_namespace'] == 'timing_serial' else 8), 'Worker drift')
    require(cell['limit'] == 20 and cell['variant'] == 'B0' and cell['split'] == 'prompt', 'Cell scope drift')
    for field, values in (
        ('ordered_pair_ids', [c['pair_id'] for c in cases]),
        ('ordered_input_hashes', [c['input_hash'] for c in cases]),
        ('ordered_request_hashes', [protocol.digest(protocol.payload(c, cell['model'], 'B0')) for c in cases]),
    ):
        if field in cell:
            require(cell[field] == values, 'Frozen ordered input/request mapping drift')
    path = base(target, cell['analysis_namespace'], cell['model'], cell['benchmark'])
    success, failed, operations, executions = [rows(path/('prompt'+s)) for s in SUFFIXES]
    require(not operations and not failed, 'Failed/unknown invocation is not a complete measurement')
    by = {c['pair_id']: c for c in cases}
    require(len(success) == len({r['pair_id'] for r in success}) == 20
            and {r['pair_id'] for r in success} == set(by), 'Full response identity mismatch')
    for row in success:
        case = by[row['pair_id']]
        require(type(row.get('study_run_id')) is int and row['study_run_id'] == 3, 'Strict RunID3 required')
        require(row['model_alias'] == cell['model'] and row['benchmark'] == cell['benchmark']
                and row['variant'] == 'B0' and not row.get('error'), 'Response scope mismatch')
        require(row['input_hash'] == case['input_hash'] == protocol.digest(case['records']), 'Input drift')
        require(row['request_hash'] == protocol.digest(protocol.payload(case, cell['model'], 'B0')), 'Request drift')
        require(bool(row['attempts']), 'No durable request attempts')
        for attempt in row['attempts']:
            require(math.isfinite(attempt['seconds']) and attempt['seconds'] >= 0, 'Invalid request seconds')
            require(attempt['cost'] is None or (math.isfinite(attempt['cost']) and attempt['cost'] >= 0),
                    'Invalid retained cost')
        last = row['attempts'][-1]
        require(last['status'] == 200 and last['cost'] is not None, 'Missing accepted final attempt')
        require(protocol.parse_response(cell['model'], last['response']) ==
                (row['prediction'], row['probability_match']), 'Frozen parser mismatch')
        require(last['cost'] == last['response']['usage']['cost'], 'Usage mismatch')
    require(len(executions) == 1, 'Exactly one uninterrupted cell execution required')
    e = executions[0]
    require(type(e['new_pairs']) is int and e['new_pairs'] == 20 and e['workers'] == cell['workers'],
            'Partial cache resume is not a full timing measurement')
    require(math.isfinite(e['wall_seconds']) and e['wall_seconds'] > 0, 'Invalid execution walltime')
    require(math.isclose(e['throughput_pairs_per_second'], 20/e['wall_seconds'], rel_tol=1e-12),
            'Execution throughput mismatch')
    attempts = [a for row in success for a in row['attempts']]
    return {'responses': 20, 'attempts': len(attempts), 'wall_seconds': e['wall_seconds'],
            'pairs_per_second': 20/e['wall_seconds'],
            'known_usd': math.fsum(a['cost'] for a in attempts if a['cost'] is not None),
            'unknown_attempts': sum(a['cost'] is None for a in attempts),
            'source_namespace': cell['analysis_namespace']}
