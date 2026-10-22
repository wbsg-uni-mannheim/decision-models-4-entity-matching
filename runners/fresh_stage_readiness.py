"""Read-only extension-stage preflight. This never submits or admits a job.

The frozen launcher has deliberately broad prerequisites. Here we additionally
check prerequisites enforced by the frozen entry points before preparing an
actual scheduler admission. Stage journals are not scheduler liveness evidence.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

from repeated_runs import STAGES, replica_path, sha, verify
from repeated_runs_parallel import PREREQUISITES, validate_history


EXTENSIONS = tuple(s for s in STAGES if s not in ('smoke', 'core'))
REQUIRES = {s: list(PREREQUISITES[s]) for s in EXTENSIONS}
# These assertions occur in the actual frozen module entry points.
REQUIRES['screen'] = ['core', 'supplement']
REQUIRES['calibration'] = ['core', 'screen']
MARKERS = {
    'core': ('artifacts/progress.json', 'AP3_AP4_analysis', 'complete_automated'),
    'supplement': ('artifacts/progress.json', 'supplement_billiger50', 'complete_automated'),
    'screen': ('artifacts/progress.json', 'expanded_E2_E3_screen', 'complete'),
    'calibration': ('artifacts/expanded_calibration_complete.json', None,
                    'complete_with_explicit_unavailable_fits'),
    'confirmation': ('artifacts/expanded_confirmation_complete.json', None, 'complete'),
    'schema': ('artifacts/expanded_schema_complete.json', None, 'complete'),
    'stability': ('artifacts/expanded_stability_complete.json', None, 'complete'),
    'cascades': ('artifacts/expanded_cascades_offline_complete.json', None,
                 'complete_with_explicit_unavailable_routers'),
}


def readiness(target, run_id, stage, rows, budget):
    """Inspect prerequisites without writing journals or granting execution."""
    if stage not in EXTENSIONS or run_id not in (2, 3):
        raise ValueError('Only original fresh-run extension stages')
    for key in ('spent', 'reserved'):
        if type(budget.get(key)) not in (int, float) or not math.isfinite(budget[key]) or budget[key] < -1e-8:
            raise ValueError('Invalid shared budget snapshot')
    blockers = []
    try:
        validate_history(rows, run_id, stage)
    except (KeyError, ValueError) as exc:
        blockers.append(str(exc))
    markers = []
    for prerequisite in REQUIRES[stage]:
        history = [r.get('status') for r in rows if r.get('stage') == prerequisite]
        if not history or history[-1] != 'complete':
            blockers.append('Unfinished required stage: '+prerequisite)
        rel, key, expected = MARKERS[prerequisite]
        path = target/rel
        status = None
        if path.exists():
            value = json.loads(path.read_text())
            status = value.get(key, {}).get('status') if key else value.get('status')
        if status != expected:
            blockers.append('Missing or incomplete result marker: '+prerequisite)
        markers.append({'stage': prerequisite, 'path': rel, 'expected_status': expected,
                        'observed_status': status, 'sha256': sha(path) if path.exists() else None})
    paid = stage not in ('cascades', 'errors')
    # The frozen launcher currently applies this guard to cache-only stages too.
    if budget['spent'] + budget['reserved'] + .05 > 100:
        blockers.append('Frozen launcher shared-budget preflight would reject')
    return {'stage': stage, 'run_id': run_id, 'command': list(STAGES[stage]),
            'prerequisites': REQUIRES[stage], 'markers': markers,
            'paid_inference': paid, 'blockers': list(dict.fromkeys(blockers)),
            'local_prerequisites_satisfied': not blockers,
            'execution_admitted': False,
            'remaining_admission_checks': ['All prior writer Slurm steps terminal',
                'Actual new job and batch-script identity', 'Current parallel policy and all joins',
                'Exclusive replica lock and fresh execution-time preflight',
                'Budget plan; per-request atomic shared reservations for paid stages']}


def verify_modules(target):
    """Run each real frozen manifest verifier in a separate offline process."""
    modules = sorted({STAGES[s][0][:-3] for s in EXTENSIONS})
    results = []
    for module in modules:
        code = '''import importlib, json, socket, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ssl
def forbidden(*args, **kwargs):
    raise AssertionError('Network forbidden in manifest preflight')
socket.socket = forbidden
socket.create_connection = forbidden
m = importlib.import_module(sys.argv[2])
assert m.PROJECT == Path(sys.argv[1]).parent
m.verify()
print(json.dumps({'module': sys.argv[2], 'manifest_verified': True}))
'''
        result = subprocess.run([sys.executable, '-c', code, str(target/'src'), module],
                                text=True, capture_output=True, check=True, timeout=120)
        record = json.loads(result.stdout)
        record['source_sha256'] = sha(target/'src'/(module+'.py'))
        results.append(record)
    return results


def inspect(root, check_manifests=True):
    root = root.resolve()
    approval_path = root/'artifacts/operations/approval_runs23_aaron_20261014.json'
    approval = json.loads(approval_path.read_text())
    if (approval.get('approved_by') != 'Aaron' or approval.get('authorized_run_ids') != [2, 3]
            or approval.get('budget_cap_usd') != 100 or not approval.get('quoted_user_authorization')):
        raise ValueError('Fresh runs authorization absent or changed')
    budget = json.loads((root/'artifacts/budget.json').read_text())
    runs = []
    for run_id in (2, 3):
        spec = verify(root, run_id)
        target = replica_path(root, run_id)
        journal = target/'artifacts/stage_execution.jsonl'
        raw = journal.read_bytes() if journal.exists() else b''
        rows = [json.loads(line) for line in raw.splitlines()]
        runs.append({'run_id': run_id, 'replica': str(target),
                     'replica_manifest_sha256': sha(target/'replica.json'),
                     'frozen_files_verified': len(spec['immutable_sha256']),
                     'journal_snapshot': rows,
                     'module_manifest_checks': verify_modules(target) if check_manifests else [],
                     'stages': [readiness(target, run_id, s, rows, budget) for s in EXTENSIONS]})
    return {'utc': datetime.now(timezone.utc).isoformat(),
            'status': 'read_only_preparation_not_admission',
            'approval_sha256': sha(approval_path), 'budget_snapshot': budget,
            'launcher_sha256': sha(root/'artifacts/operations/repeated_runs_parallel.py'),
            'preflight_sha256': sha(Path(__file__)), 'runs': runs,
            'scope': 'Original extension stages. Peeters has a separate frozen runner. '
                     'No study result, full-run funding promise or scheduler admission.',
            'readiness_limit': 'Sequential live snapshots can change; rerun and inspect actual '
                               'scheduler state immediately before any admission.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'runs': len(result['runs']),
                      'stage_rows': sum(len(r['stages']) for r in result['runs'])}))
