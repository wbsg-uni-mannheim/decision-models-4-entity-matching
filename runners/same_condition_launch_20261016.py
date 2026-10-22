"""One bounded two-replica repetition admission; no selection or legacy recovery."""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import leased_budget as old_budget
import same_condition_repetition_runtime_20261016 as runtime

SELF = 'artifacts/operations/same_condition_launch_20261016.py'
AUTH = 'artifacts/operations/approval_same_condition_250_aaron_20261016.json'
BUDGET_POLICY = 'configs/same_condition_budget250_20261016.json'
PREP = 'artifacts/verification/same_condition_launch_readiness_20261016.json'
ADMISSION = 'configs/same_condition_admission_20261016.json'
POLICY = 'configs/same_condition_scheduler_20261016.json'
BATCH = 'artifacts/operations/same_condition_runs23_20261016.sbatch'
JOURNAL = 'artifacts/operations/same_condition_submission_20261016.jsonl'
ATTEMPT = 'artifacts/operations/same_condition_repetition_20261016_v1'
NAME = 'em-identical-runs23'
CAP = 250
RUNTIME_SHA = '52e9b55b3f65ef6b331ae6213db50e2f3b9f1634e9a42732086fa48c5272eb80'
ANCHOR_SHA = '7c3597f032f727362ec107f514e418b697f22ae0c7f7866392d67de5a80a3954'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    return runtime.sha(Path(p))


def save(p, value):
    p = Path(p)
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    runtime.sync_dir(p.parent)
    require(read(p) == value, 'Durable readback mismatch')


def cmd(*args):
    return subprocess.check_output(args, text=True)


@contextmanager
def locks(root, ids=(2, 3), exclusive=False):
    with ExitStack() as stack:
        for p, mode in [(root/'artifacts/operations/repetition_writer.lock', fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)] + [
                (root/f'replications/run{rid:02d}/artifacts/parallel_writer.lock', fcntl.LOCK_EX) for rid in ids]:
            f = stack.enter_context(p.open('a'))
            fcntl.flock(f, mode | fcntl.LOCK_NB)
        yield


def authority(root):
    a = read(root/AUTH); p = read(root/BUDGET_POLICY)
    require(a['approved_by'] == 'Aaron' and a['project_cap_usd'] == CAP
            and a['same_conditions_only'] is True and a['new_selection'] is False
            and p['project_cap_usd'] == CAP and p['approval_sha256'] == sha(root/AUTH), 'Budget authority mismatch')
    return p


def budget_check(root, available=0):
    authority(root)
    b = read(root/'artifacts/budget.json')
    require(b.get('project_cap_usd') == CAP and b.get('budget_authority_sha256') == sha(root/AUTH), 'Ledger ceiling authority')
    require(all(type(b[k]) in (float, int) and math.isfinite(b[k]) and b[k] >= 0 for k in ('spent', 'reserved')), 'Invalid ledger')
    require(b['spent'] + b['reserved'] + available <= CAP + 1e-8, 'Budget exhausted')
    require(not any(x['active'] for x in b.get('maintenance_barriers', {}).values()), 'Maintenance barrier')
    return b


def credit_class(root):
    """Private method globals and defaults; original 200-dollar module untouched."""
    authority(root)
    require(sha(Path(old_budget.__file__)) == runtime.PINS['leased_budget'], 'Budget source drift')
    @contextmanager
    def checked_ledger(directory):
        require(Path(directory).resolve() == root.resolve(), 'Foreign ledger')
        authority(root)
        with old_budget.ledger(directory) as obj:
            require(obj.get('project_cap_usd') == CAP and obj.get('budget_authority_sha256') == sha(root/AUTH), 'Ledger policy drift')
            yield obj
    namespace = {**vars(old_budget), 'CAP': CAP, 'ledger': checked_ledger}
    methods = {}
    for name, function in vars(old_budget.Credit).items():
        if isinstance(function, types.FunctionType):
            defaults = (0, CAP) if name == 'account' else function.__defaults__
            methods[name] = types.FunctionType(function.__code__, namespace, function.__name__, defaults, function.__closure__)
    return type('Authorized250Credit', (), methods)


def load_modules(target):
    # Called in one fresh process per replica only; never share global modules.
    sys.path.insert(0, str(target/'src'))
    import run
    import protocol
    import expanded_protocol
    import expanded_schema
    require(Path(run.__file__).resolve() == target/'src/run.py', 'Stale replica import')
    return run, protocol, expanded_protocol, expanded_schema


def own_scope(root, rid):
    target = root/f'replications/run{rid:02d}'
    return {str(p.relative_to(root)) for p in runtime.paths(target, runtime.matrix(rid))}


def steps(root, jobs, expected=None):
    result = {}
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
    for job in jobs:
        raw = cmd('sacct', '-n', '-P', '-j', job, '--format=JobIDRaw,State,ExitCode,End')
        rows = [line.split('|')[:4] for line in raw.splitlines() if line.strip()]
        require({job, job+'.batch', job+'.extern'} <= {r[0] for r in rows}, 'Missing historical steps')
        require(all(r[1].split()[0] in terminal and r[3] not in ('', 'Unknown', 'N/A') for r in rows), 'Nonterminal historical step')
        result[job] = rows
    if expected is not None:
        require(result == expected, 'Historical step state drift')
    return result


def queue(root, policy=None, running=False):
    found = {}; raw = cmd('squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%j')
    for line in raw.splitlines():
        job, name = line.split('|', 1)
        meta = cmd('scontrol', 'show', 'job', job, '-o')
        fields = dict(x.split('=', 1) for x in meta.split() if '=' in x)
        if str(root) not in meta and not name.startswith(('decision-em', 'em-identical', 'pplx-repeat')):
            continue
        require(job in (policy['jobs'] if policy else {'365300': {}, '365355': {}}), 'Unreviewed project job')
        require(fields['WorkDir'] == str(root), 'Job workdir drift')
        batchsha = hashlib.sha256(cmd('scontrol', 'write', 'batch_script', job, '-').encode()).hexdigest()
        if job in ('365300', '365355'):
            deps = re.sub(r'\([^)]*\)', '', fields['Dependency']).replace(',afterok:', ':')
            require(deps == 'afterok:366694:366695' + (':365300' if job == '365355' else ''), 'Legacy ALLjoins drift')
            require(fields['JobState'] == 'PENDING' and fields['StartTime'] == 'Unknown', 'Legacy writer active')
            if job == '365300':
                require(fields['Priority'] == '0', 'Rules not held')
        else:
            require(name == NAME and fields['JobState'] == ('RUNNING' if running else 'PENDING'), 'Own job state')
            if not running:
                require(fields['Priority'] == '0' and fields['StartTime'] == 'Unknown', 'Own job not held')
        entry = dict(name=name, batch_script_sha256=batchsha, dependency=fields['Dependency'])
        if policy:
            require(entry == policy['jobs'][job], 'Job batch/dependency drift')
        found[job] = entry
    require(set(found) == (set(policy['jobs']) if policy else {'365300', '365355'}), 'Missing reviewed jobs')
    return found


def file_state(path):
    return sha(path) if path.exists() else None


def protected(root, prep, final=False, rid=None):
    for rel, h in prep['source_sha256'].items():
        require(file_state(root/rel) == h, 'Source drift: '+rel)
    for rel, h in prep['protected_sha256_or_absent'].items():
        require(file_state(root/rel) == h, 'Original drift: '+rel)
    for run in (2, 3):
        target = root/f'replications/run{run:02d}'
        allowed = own_scope(root, run)
        observed = {str(p.relative_to(root)) for p in (target/'runs').rglob('*.jsonl')}
        require(observed <= set(prep['original_raw'][str(run)]) | allowed, 'Foreign raw namespace')
        # During concurrent execution only immutable bytes outside either scoped set are protected above.
        if final and run == rid:
            before = prep['priors'][str(run)]['journals']
            for rel, old in before.items():
                path = target/rel
                if old['exists']:
                    require(path.exists(), 'Old raw disappeared')
                raw = path.read_bytes() if path.exists() else b''
                require(len(raw) >= old['bytes'] and hashlib.sha256(raw[:old['bytes']]).hexdigest() == old['sha256'], 'Old raw prefix drift')


def foreign_credits(root, prep):
    b = budget_check(root)
    for key, value in prep['budget_before']['inference_credits'].items():
        require(b['inference_credits'][key] == value, 'Foreign credit changed')
    require(b.get('direct_openai_commitments') == prep['budget_before'].get('direct_openai_commitments')
            and b.get('ensemble_pilot_commitments') == prep['budget_before'].get('ensemble_pilot_commitments'), 'Foreign commitment changed')


def gate(root, spec, running=False, rid=None):
    require(spec.get('admitted') is True and spec.get('scope') == runtime.STAGE and spec.get('run_ids') == [2, 3]
            and spec.get('max_invocations_per_run') == 1 and spec.get('project_cap_usd') == CAP
            and spec.get('new_selection') is False and spec.get('calibration_calls') == 0, 'Exact admission required')
    require(spec['admission_path'] == ADMISSION and read(root/ADMISSION) == spec, 'Admission path drift')
    require(sha(root/PREP) == spec['readiness_sha256'] and sha(root/POLICY) == spec['policy_sha256'], 'Admission pins drift')
    prep = read(root/PREP)
    protected(root, prep)
    foreign_credits(root, prep)
    if not running:
        budget_check(root, 16)
    sched = queue(root, read(root/POLICY), running)
    history = steps(root, prep['history_steps'].keys(), prep['history_steps'])
    for run in ((rid,) if rid else (2, 3)):
        require(not (root/f'replications/run{run:02d}'/ATTEMPT).exists(), 'Attempt already exists; no reuse')
    return dict(utc=now(), scheduler=sched, history_steps=history, budget=read(root/'artifacts/budget.json'))


def inventory_process(root, rid, output):
    target = root/f'replications/run{rid:02d}'
    modules = load_modules(target)
    with locks(root, (rid,)):
        prior = runtime.capture_prior(root, target, rid, *modules, target/ATTEMPT/'inference')
        save(output, prior)


def preflight(root, spec, output):
    # Both replica locks remain held while fresh per-replica child processes audit.
    with locks(root):
        result = gate(root, spec)
        prep = read(root/PREP)
        for rid in (2, 3):
            child = subprocess.run([sys.executable, __file__, '--root', str(root), '--action', 'inventory-unlocked', '--run-id', str(rid)], capture_output=True, text=True, check=True)
            require(json.loads(child.stdout) == prep['priors'][str(rid)], 'Scoped cache changed before release')
        protected(root, prep)
        save(output, {'admission_sha256': sha(root/ADMISSION), 'checks': result, 'full_scoped_raw_readback': True})


def execute(root, rid):
    spec = read(root/ADMISSION)
    require(os.environ.get('SLURM_JOB_ID') == spec['slurm_job_id'], 'Real admitted job required')
    target = root/f'replications/run{rid:02d}'
    modules = load_modules(target)
    with locks(root, (rid,)):
        startgate = gate(root, spec, True, rid)
        prior = runtime.capture_prior(root, target, rid, *modules, target/ATTEMPT/'inference')
        require(prior == read(root/PREP)['priors'][str(rid)], 'Prior changed after release')
        attempt = target/ATTEMPT
        attempt.mkdir(); runtime.sync_dir(attempt.parent)
        save(attempt/'prior.json', prior)
        save(attempt/'start_gate.json', startgate)
        credit = credit_class(root)(root, f'same-condition-{rid}-'+spec['slurm_job_id'],
            {'prior_sha256': sha(attempt/'prior.json'), 'run_id': rid, 'stage': runtime.STAGE,
             'journal_dir': str((attempt/'inference').relative_to(target)), 'admission_sha256': sha(root/ADMISSION)}, chunk=8)
        save(attempt/'started.json', {'utc': now(), 'run_id': rid, 'job': spec['slurm_job_id'], 'credit': credit.summary(),
            'prior_sha256': sha(attempt/'prior.json'), 'admission_sha256': sha(root/ADMISSION)})
        try:
            result = runtime.execute(root, target, rid, *modules, attempt/'prior.json', sha(attempt/'prior.json'), attempt/'inference', credit)
            save(attempt/'execution_result.json', result)
            protected(root, read(root/PREP), True, rid)
            closed = runtime.reconcile(root, target, rid, *modules, attempt/'prior.json', sha(attempt/'prior.json'), attempt/'inference', credit)
            save(attempt/'credit_reconciliation.json', closed)
            foreign_credits(root, read(root/PREP))
            require(result['inference_complete'] and closed['inference_complete'], 'Scoped inference incomplete')
            save(attempt/'inference_complete.json', {'utc': now(), 'run_id': rid, 'result_sha256': sha(attempt/'execution_result.json'),
                'credit_sha256': sha(attempt/'credit_reconciliation.json'), 'independent_report_and_slurm_steps_pending': True})
        except Exception as exc:
            save(attempt/'failed.json', {'utc': now(), 'exception_type': type(exc).__name__, 'credit': credit.summary(), 'hold_retained': not credit.closed})
            raise


def launch(root):
    spec = read(root/ADMISSION)
    require(os.environ.get('SLURM_JOB_ID') == spec['slurm_job_id'], 'Wrong real launch')
    processes = [subprocess.Popen([sys.executable, '-u', __file__, '--root', str(root), '--action', 'execute', '--run-id', str(rid)]) for rid in (2, 3)]
    codes = [p.wait() for p in processes]
    require(codes == [0, 0], 'One or more replica processes failed; no automatic retry')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--action', choices=['inventory', 'inventory-unlocked', 'preflight', 'execute', 'launch'], required=True)
    p.add_argument('--run-id', type=int, choices=(2, 3)); p.add_argument('--output', type=Path)
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'inventory':
        inventory_process(root, a.run_id, a.output)
    elif a.action == 'inventory-unlocked':
        target = root/f'replications/run{a.run_id:02d}'
        print(json.dumps(runtime.capture_prior(root, target, a.run_id, *load_modules(target), target/ATTEMPT/'inference')))
    elif a.action == 'preflight':
        preflight(root, read(root/ADMISSION), a.output)
    elif a.action == 'execute':
        execute(root, a.run_id)
    else:
        launch(root)


if __name__ == '__main__':
    main()
