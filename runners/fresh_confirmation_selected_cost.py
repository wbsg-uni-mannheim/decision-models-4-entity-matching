"""Read-only post-freeze cost gate for the future bounded confirmation wrapper.

Reconstruct every own-run development cell before pricing the actual selected
missing tests. No CLI, admission, reservation, inference or output writes. The
caller must hold global shared/replica exclusive locks and check all current
scheduler, policy, prerequisite and scope gates before using this result.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from authorized_recovery_200 import APPROVAL_SHA, authority
import fresh_confirmation_development_drained as development
import fresh_confirmation_selection as freeze
import fresh_confirmation_test_drained as tests

UNKNOWN_ALLOWANCE_USD = .05


def calculate(development_cells, test_cells):
    """Own-run development attempt mean times actual missing test pair count."""
    sources = {}
    for cell in development_cells:
        key = (cell['model'], cell['variant'], cell['benchmark'])
        if key in sources or not cell['complete'] or cell['missing'] != 0:
            raise ValueError('Unique complete development sources required')
        if type(cell['responses']) is not int or cell['responses'] <= 0:
            raise ValueError('Positive development source size required')
        if (type(cell['known_usd']) not in (int, float) or not math.isfinite(cell['known_usd'])
                or cell['known_usd'] < 0 or type(cell['unknown_attempts']) is not int
                or cell['unknown_attempts'] < 0):
            raise ValueError('Invalid development cost source')
        sources[key] = cell
    result, seen = [], set()
    for cell in test_cells:
        key = (cell['model'], cell['variant'], cell['benchmark'])
        if key in seen:
            raise ValueError('Duplicate selected test cell')
        seen.add(key)
        if (any(type(cell[k]) is not int or cell[k] < 0 for k in ('expected', 'responses', 'missing'))
                or cell['responses'] + cell['missing'] != cell['expected']):
            raise ValueError('Invalid selected test coverage')
        benchmark = 'billiger-de' if cell['benchmark'] == 'billiger-de-50un' else cell['benchmark']
        source_key = (cell['model'], cell['variant'], benchmark)
        if source_key not in sources:
            raise ValueError('Missing matching own-run development source')
        source = sources[source_key]
        ratio = cell['missing']/source['responses']
        result.append({'model': key[0], 'variant': key[1], 'benchmark': key[2],
            'source_benchmark': benchmark, 'source_split': 'prompt',
            'source_responses': source['responses'], 'source_known_usd': source['known_usd'],
            'source_unknown_attempts': source['unknown_attempts'],
            'test_expected': cell['expected'], 'test_valid_cached': cell['responses'],
            'test_missing': cell['missing'], 'missing_pair_ids_sha256': cell['missing_pair_ids_sha256'],
            'known_proxy_usd': source['known_usd']*ratio,
            'historical_unknown_scenario_usd': source['unknown_attempts']*UNKNOWN_ALLOWANCE_USD*ratio})
    if not result:
        raise ValueError('Nonempty selected test matrix required')
    return {'cells': result, 'unique_test_cells': len(result),
            'missing_test_responses': sum(c['test_missing'] for c in result),
            'valid_cached_test_responses': sum(c['test_valid_cached'] for c in result),
            'known_proxy_usd': math.fsum(c['known_proxy_usd'] for c in result),
            'historical_unknown_scenario_usd': math.fsum(c['historical_unknown_scenario_usd'] for c in result)}


def budget_review(plan, budget):
    values = [budget.get('spent'), budget.get('reserved'), plan['known_proxy_usd'],
              plan['historical_unknown_scenario_usd']]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < -1e-8 for v in values):
        raise ValueError('Invalid current budget/planning values')
    total = math.fsum(max(0, v) for v in values)
    return {'budget_snapshot': budget, 'budget_cap_usd': 200,
            'budget_authority_sha256': APPROVAL_SHA, 'planned_total_usd': total,
            'scenario_fits_current_headroom': total <= 200,
            'reservation_created': False, 'admission_granted': False}


def test_snapshot(target, cells):
    expected = tests.snapshot(target, cells)
    for m, v, b in cells:
        directory = target/'runs/quality'/m/v/b
        if any(str(p.relative_to(target)) not in expected for p in directory.glob('test*.jsonl')):
            raise ValueError('Unreviewed selected-test journal')
    return expected


def review(root, target, run_id, confirmation, expanded, analysis, protocol):
    """Full raw replay, actual missing scope and fresh budget snapshot; no writes."""
    root, target = Path(root).resolve(), Path(target).resolve()
    authority(root)
    candidate = freeze.verify_frozen(root, target, run_id, confirmation, expanded, analysis, protocol)
    tests.no_report(target)
    if (target/tests.ATTEMPT).exists():
        raise ValueError('Existing selected-test attempt requires separate terminal review')
    receipt = json.loads((target/freeze.RECEIPT).read_text())
    model_journal = target/receipt['model_journal']
    before_dev = freeze.development_provenance(target, confirmation, model_journal)
    cells = candidate['test_cells']
    controls = {(m, 'B0', b) for m, _, b in cells}
    before_test = test_snapshot(target, set(cells) | controls)
    dev = development.inventory(target, run_id, confirmation, protocol)
    if len(dev) != len(development.matrix(confirmation)) or not all(c['complete'] for c in dev):
        raise ValueError('Every own-run development cell must be complete')
    audited_controls = tests.inventory(target, run_id, confirmation, protocol, controls)
    if not all(c['complete'] for c in audited_controls):
        raise ValueError('Complete own-run B0 test controls required')
    audited_tests = tests.inventory(target, run_id, confirmation, protocol, cells)
    if any((c['responses'] or c['failure_rows'] or c['operations']['events'])
           and c['variant'] not in ('B0', 'S1') for c in audited_tests):
        raise ValueError('Existing expanded test outcomes require terminal review')
    plan = calculate(dev, audited_tests)
    # Reconstruct all development and selection again, never trust a receipt alone.
    replay = freeze.verify_frozen(root, target, run_id, confirmation, expanded, analysis, protocol)
    if (replay != candidate or before_dev != freeze.development_provenance(target, confirmation, model_journal)
            or before_test != test_snapshot(target, set(cells) | controls)):
        raise ValueError('Development/selection/test sources changed during cost review')
    budget = json.loads((root/'artifacts/budget.json').read_text())
    return {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
            'selection_sha256': candidate['selection_sha256'],
            'selection_receipt_sha256': freeze.sha(target/freeze.RECEIPT),
            'development_cells_raw_verified': len(dev), 'b0_test_control_cells': len(audited_controls),
            'development_sha256_or_absent': before_dev, 'test_sha256_or_absent': before_test,
            **plan, **budget_review(plan, budget), 'new_api_calls': 0,
            'test_used_for_selection': False, 'test_response_cost_used_for_forecast': False,
            'quote_or_hard_bound': False, 'historical_unjournaled_outcomes_resolved': False,
            'limits': 'Own-run development historical mean including retained retry/failure costs; '
                      'billiger50 transfers billiger-de. Separate unknown allowance scenario, not a future '
                      'error forecast. No test-token adjustment, provider quote or funding guarantee. '
                      'Concurrent spend may change after this snapshot; atomic per-request USD200 reserve '
                      'remains mandatory. Caller must persist provenance and revalidate scheduler/policy/'
                      'prerequisites/scopes/all joins under global SH and replica EX locks.'}


def require_headroom(root, reviewed):
    """Reread the shared budget; a saved fit boolean never authorizes spending.

    This is still only a planning gate, not a reservation or a source/scheduler
    revalidation. The caller must just have run review() under both locks.
    """
    root = Path(root).resolve()
    authority(root)
    current = budget_review(reviewed, json.loads((root/'artifacts/budget.json').read_text()))
    if not current['scenario_fits_current_headroom']:
        raise ValueError('Actual selected-test scenario exceeds current headroom')
    return current
