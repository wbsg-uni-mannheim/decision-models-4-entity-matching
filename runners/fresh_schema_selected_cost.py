"""Read-only own-run schema selected-test planning after a durable freeze.

No launcher, admission, reservation, inference or output writes. The future
wrapper must hold global SH / replica EX locks and validate current scheduler,
prerequisites, sources, scopes and all joins. Only first test attempts qualify.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from authorized_recovery_200 import authority
from fresh_confirmation_selected_cost import calculate, budget_review, require_headroom  # noqa: F401
import fresh_schema_development_drained as development
import fresh_schema_selection as freeze
import fresh_schema_test_drained as tests


def test_snapshot(target, cells):
    expected = tests.snapshot(target, cells)
    for m, v, b in cells:
        namespace = 'quality' if v == 'B0' else 'schema'
        directory = target/'runs'/namespace/m/v/b
        if any(str(p.relative_to(target)) not in expected for p in directory.glob('test*.jsonl')):
            raise ValueError('Unreviewed selected-test/control journal')
    return expected


def review(root, target, run_id, schema, analysis):
    """Replay all pilot/Dev raw data, price own selection, then reread budget."""
    root, target = Path(root).resolve(), Path(target).resolve()
    authority(root)
    candidate = freeze.verify_frozen(root, target, run_id, schema, analysis)
    tests.no_report(target)
    if freeze.own_path(target, target/tests.ATTEMPT).exists():
        raise ValueError('Existing selected-test attempt requires separate terminal review')
    receipt = json.loads((target/freeze.RECEIPT).read_text())
    model_journal = freeze.own_path(target, receipt['model_journal'])
    before_dev = freeze.provenance(target, run_id, schema, model_journal)
    cells = set(map(tuple, candidate['test_cells']))
    controls = {(m, 'B0', b) for m, _, b in cells}
    before_test = test_snapshot(target, cells | controls)
    # Same first-attempt contract as run_selected_tests, including empty journals.
    if any(p.exists() for p in tests.paths(target, cells) if p.suffix == '.jsonl'):
        raise ValueError('Existing schema test journals require terminal review')
    pilot = development.inventory(target, run_id, schema, True)
    dev = development.inventory(target, run_id, schema)
    if (len(pilot) != len(development.matrix(schema, True))
            or len(dev) != len(development.matrix(schema))
            or not all(c['complete'] for c in pilot + dev)):
        raise ValueError('Every own-run pilot/development cell must be complete')
    audited_controls = tests.inventory(target, run_id, schema, controls)
    if not all(c['complete'] for c in audited_controls):
        raise ValueError('Complete own-run B0 test controls required')
    audited_tests = tests.inventory(target, run_id, schema, cells)
    plan = calculate(dev, audited_tests)
    replay = freeze.verify_frozen(root, target, run_id, schema, analysis)
    if (candidate != replay
            or before_dev != freeze.provenance(target, run_id, schema, model_journal)
            or before_test != test_snapshot(target, cells | controls)):
        raise ValueError('Pilot/development/selection/test sources changed during cost review')
    budget = json.loads((root/'artifacts/budget.json').read_text())
    return {'utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
            'selection_sha256': {p: freeze.sha(target/p) for p in freeze.OUTPUTS},
            'selection_receipt_sha256': freeze.sha(target/freeze.RECEIPT),
            'pilot_cells_raw_verified': len(pilot), 'development_cells_raw_verified': len(dev),
            'b0_test_control_cells': len(audited_controls),
            'development_sha256_or_absent': before_dev, 'test_sha256_or_absent': before_test,
            **plan, **budget_review(plan, budget), 'new_api_calls': 0,
            'test_used_for_selection': False, 'test_response_cost_used_for_forecast': False,
            'quote_or_hard_bound': False, 'historical_unjournaled_outcomes_resolved': False,
            'limits': 'Own-run development means include retained retry/failure costs; billiger50 '
                      'transfers billiger-de. Pilot costs are not added to already completed development '
                      'or selected-test means. Unknown allowance scenario is separate, not a future error '
                      'forecast. No B0/cross-run cache credit for schema tests. No test-token adjustment, '
                      'provider quote or funding guarantee. Concurrent spend may change after this '
                      'snapshot; current shared USD200 reservation/credit controls remain mandatory. '
                      'Caller must persist provenance and revalidate scheduler/policy/prerequisites/'
                      'sources/scopes/all joins under global SH and replica EX locks.'}
