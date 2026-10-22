"""One user-authorized fresh B0 test repeat, eight families; original Run1 retained."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

from grouped_inference import build
from leased_budget import Credit

NAMESPACE = 'pplx_run01_b0_repeat_20261016'
ATTEMPT = 'artifacts/operations/pplx_run01_b0_repeat_20261016'
SELF = 'artifacts/operations/run_pplx_baseline_repeat_20261016.py'
CAP = 3.0
REQUIRED = {SELF, 'src/run.py', 'src/protocol.py', 'configs/manifest.json',
            'artifacts/operations/grouped_inference.py', 'artifacts/operations/drained_inference.py',
            'artifacts/operations/leased_budget.py',
            'artifacts/operations/approval_budget200_parallel_aaron_20261015.json',
            'artifacts/operations/approval_pplx_repeat_aaron_20261016.json'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    fd = os.open(Path(p).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    require(read(p) == value, 'Readback failed')


@contextmanager
def locks(root):
    with (root/'artifacts/operations/repetition_writer.lock').open('a') as a, \
            (root/'artifacts/operations/pplx_repeat_writer.lock').open('a') as b:
        fcntl.flock(a, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(b, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def cmd(*args):
    return subprocess.check_output(args, text=True)


def source_check(root, spec):
    require(REQUIRED <= spec['source_sha256'].keys(), 'Missing required source pins')
    require(spec['source_sha256'].get(SELF) == sha(root/SELF), 'Runner not pinned')
    for rel, digest in spec['source_sha256'].items():
        p = root/rel
        require(not p.is_symlink() and sha(p) == digest, 'Source drift: '+rel)
    for rel, digest in spec['protected_sha256_or_absent'].items():
        p = root/rel
        require((sha(p) if p.exists() else None) == digest, 'Original changed: '+rel)


def scheduler(root, spec, held):
    own = spec['slurm_job_id']; policy = read(root/spec['policy'])
    require(sha(root/spec['policy']) == spec['policy_sha256'], 'Policy drift')
    queue = cmd('squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%j')
    seen = set(); metadata = {}
    for line in queue.splitlines():
        job, name = line.split('|', 1)
        meta = cmd('scontrol', 'show', 'job', job, '-o')
        if job not in policy['jobs']:
            require(str(root) not in meta and not name.startswith(('decision-em', 'pplx-repeat')), 'Unreviewed project writer')
            continue
        expected = policy['jobs'][job]; seen.add(job)
        fields = dict(x.split('=', 1) for x in meta.split() if '=' in x)
        require(name == expected['name'] and fields['WorkDir'] == str(root), 'Job identity drift')
        batch = cmd('scontrol', 'write', 'batch_script', job, '-')
        require(hashlib.sha256(batch.encode()).hexdigest() == expected['batch_script_sha256'], 'Batch drift')
        if job == own:
            require(fields['JobState'] == ('PENDING' if held else 'RUNNING'), 'Unexpected own job state')
            if held:
                require(fields['Priority'] == '0' and fields['StartTime'] == 'Unknown', 'Job must actually be held')
        else:
            require(fields['JobState'] == 'PENDING' and fields['StartTime'] == 'Unknown', 'Legacy job started')
            require(fields['Dependency'] == expected['dependency'], 'Legacy join changed')
            if job == '365300':
                require(fields['Priority'] == '0', 'Rules must stay held')
        metadata[job] = meta
    require(seen == set(policy['jobs']), 'Missing reviewed job')
    accounts = {}
    for job in ['366694', '366695', '366696', '367108']:
        raw = cmd('sacct', '-n', '-P', '-j', job, '--format=JobIDRaw,State,ExitCode')
        rows = [x.split('|') for x in raw.splitlines() if x.strip()]
        require({job, job+'.batch', job+'.extern'} <= {x[0] for x in rows}, 'Missing Slurm steps')
        require(all(x[1] in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL') for x in rows), 'Active predecessor step')
        if job == '367108':
            require(all(x[1:3] == ['COMPLETED', '0:0'] for x in rows), 'Core3 success changed')
        accounts[job] = raw
    return {'queue': queue, 'metadata': metadata, 'all_steps': accounts}


def readiness(root, spec, held=True):
    require(spec.get('admitted') is True and spec['model'] == 'pplx' and spec['variant'] == 'B0'
            and spec['split'] == 'test' and spec['run_id'] == 1 and spec['namespace'] == NAMESPACE
            and spec['max_invocations'] == 1 and spec['project_cap_usd'] == 200
            and spec['repeat_cap_usd'] == CAP, 'Exact repeat admission required')
    source_check(root, spec)
    require(not (root/ATTEMPT).exists() and not (root/'runs'/NAMESPACE).exists(), 'Repeat already attempted')
    approval = read(root/spec['approval'])
    require(approval['approved_by'] == 'Aaron' and approval['scope'] == 'pplx_B0_test_eight_core_benchmarks_once', 'Explicit repeat authority missing')
    budget = read(root/'artifacts/budget.json')
    require(all(math.isfinite(budget[k]) and budget[k] >= 0 for k in ['spent', 'reserved']), 'Invalid budget')
    require(budget['spent']+budget['reserved']+CAP <= 200, 'Insufficient shared budget')
    require(not any(x['active'] for x in budget.get('maintenance_barriers', {}).values()), 'Maintenance barrier')
    import protocol
    require(spec['benchmarks'] == list(protocol.BENCHMARKS) and spec['expected_pairs'] == 30225, 'Wrong matrix')
    manifest = read(root/'configs/manifest.json')
    for b in spec['benchmarks']:
        cases = read(root/'data'/b/'test.json')
        require(protocol.digest(cases) == manifest['benchmarks'][b]['splits']['test']['sha256'], 'Frozen input mismatch')
        require(len(cases) == protocol.BENCHMARKS[b]['count'], 'Wrong cell size')
    return {'utc': now(), 'budget': budget, 'scheduler': scheduler(root, spec, held), 'pairs': 30225}


def validate_results(root, spec, credit):
    import protocol
    totals = {'attempts': 0, 'unknown': 0, 'charged': 0.0}
    metrics = []
    for b in spec['benchmarks']:
        cases = read(root/'data'/b/'test.json'); wanted = {c['pair_id']: c for c in cases}
        folder = root/'runs'/NAMESPACE/'pplx/B0'/b
        rows = [json.loads(x) for x in (folder/'test.jsonl').read_text().splitlines()]
        require(len(rows) == len(wanted) and {r['pair_id'] for r in rows} == wanted.keys(), 'Incomplete/duplicate cell')
        require(not (folder/'test.operations.jsonl').exists(), 'Operational outcome unresolved')
        failures = folder/'test.failures.jsonl'
        failed = [json.loads(x) for x in failures.read_text().splitlines()] if failures.exists() else []
        tp = fp = fn = tn = 0
        for r in rows+failed:
            c = wanted[r['pair_id']]
            require(type(r['study_run_id']) is int and r['study_run_id'] == 1
                    and r['model_alias'] == 'pplx' and r['variant'] == 'B0'
                    and r['input_hash'] == c['input_hash']
                    and r['request_hash'] == protocol.digest(protocol.payload(c, 'pplx', 'B0')), 'Raw identity mismatch')
            for a in r['attempts']:
                totals['attempts'] += 1; totals['unknown'] += a['cost'] is None
                totals['charged'] += .05 if a['cost'] is None else a['cost']
        for r in rows:
            require(protocol.parse_response('pplx', r['attempts'][-1]['response']) == (r['prediction'], r['probability_match']), 'Raw parser mismatch')
            y = wanted[r['pair_id']]['label']; pred = r['prediction']
            tp += y == pred == 1; fp += y == 0 and pred == 1
            fn += y == 1 and pred == 0; tn += y == pred == 0
        metrics.append({'benchmark': b, 'n': len(rows), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                        'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0})
        executions = [json.loads(x) for x in (folder/'test.executions.jsonl').read_text().splitlines()]
        require(len(executions) == 1 and executions[0]['new_pairs'] == len(cases) and executions[0]['workers'] == 16, 'Incomplete cell execution')
    require(credit.attempts == credit.settled == totals['attempts'] and credit.unknown == totals['unknown']
            and credit.outstanding == 0 and math.isclose(credit.charged, totals['charged'], abs_tol=1e-9), 'Durable accounting mismatch')
    source_check(root, spec)
    return {'cells': metrics, 'totals': totals, 'macro_f1': sum(x['f1'] for x in metrics)/len(metrics),
            'original_run1_preserved': True, 'diagnostic_repeat_not_silent_replacement': True}


def execute(root, spec, admission):
    require(os.environ.get('SLURM_JOB_ID') == spec['slurm_job_id'], 'Real admitted Slurm job required')
    with locks(root):
        prior = readiness(root, spec, held=False)
        attempt = root/ATTEMPT; attempt.mkdir()
        fsync_dir(attempt.parent)
        save(attempt/'prior.json', prior)
        credit = Credit(root, 'pplx-repeat-'+spec['slurm_job_id'],
                        {'prior_sha256': sha(attempt/'prior.json'), 'admission_sha256': sha(admission),
                         'run_id': 1, 'namespace': NAMESPACE, 'attempt': ATTEMPT}, chunk=CAP)
        storage_failed = []
        try:
            import run
            def account(action, amount=0):
                with credit.lock:
                    if action == 'reserve' and credit.charged+.05*(credit.outstanding+1) > CAP:
                        raise RuntimeError('Repeat spending ceiling reached')
                    return credit.account(action, amount)
            isolated = types.SimpleNamespace(**{**vars(run), 'account': account,
                                                '_on_journal_failure': lambda: storage_failed.append(True)})
            _, cell, binding = build(isolated, 1)
            save(attempt/'started.json', {'utc': now(), 'binding': binding, 'admission_sha256': sha(admission)})
            for b in spec['benchmarks']:
                source_check(root, spec)
                cell('pplx', 'B0', b, 'test', workers=16, namespace=NAMESPACE)
            require(not storage_failed, 'Unconfirmed storage writes')
            proof = validate_results(root, spec, credit)
            save(attempt/'final_validation.json', proof)
            save(attempt/'credit_closed.json', credit.checkpoint(close=True))
            save(attempt/'complete.json', {'utc': now(), 'proof_sha256': sha(attempt/'final_validation.json'),
                                          'macro_f1': proof['macro_f1'], 'slurm_steps_must_be_checked_separately': True})
        except Exception as exc:
            save(attempt/'failed.json', {'utc': now(), 'exception_type': type(exc).__name__,
                                       'credit': credit.summary(), 'hold_retained': not credit.closed})
            raise


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--admission', type=Path, required=True); ap.add_argument('--preflight-output', type=Path)
    args = ap.parse_args(); root = args.root.resolve()
    sys.path.insert(0, str(root/'src')); spec = read(args.admission)
    if args.preflight_output:
        with locks(root):
            save(args.preflight_output, {'admission_sha256': sha(args.admission), 'checks': readiness(root, spec)})
    else:
        execute(root, spec, args.admission)


if __name__ == '__main__':
    main()
