"""Create exactly one held third drained variant successor; never release it."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import authorized_recovery_200 as budget
import resume_variants_third_drained as third
from resume_schema_once import terminal_steps

ROOT = Path('/ceph/aasteine/decision-em-20261013')
POLICY = 'configs/parallel_variants_third_20261015.json'
ADMISSION = 'configs/variants_third_admission_20261015.json'
DIAG = 'artifacts/operations/variants_terminal_diagnosis_366251_20261015.json'
COST = 'artifacts/verification/variant_remaining_cost_plan_366251_20261015.json'
SCRIPT = 'artifacts/operations/variants_third_20261015.sbatch'
ACTIVE = {'366115', '366116', '366545', '366520', '365300', '365355'}


def run(*args):
    return subprocess.check_output(args, text=True, env={**os.environ, 'TZ': 'UTC'})


def read(rel):
    return json.loads((ROOT/rel).read_text())


def save(rel, obj):
    with (ROOT/rel).open('x') as f:
        json.dump(obj, f, indent=2); f.write('\n')


def fields(job):
    return dict(x.split('=', 1) for x in run('scontrol', 'show', 'job', job, '-o').split() if '=' in x)


def main():
    budget.authority(ROOT)
    w = third.make_wrapper(ROOT)
    acct = run('sacct', '-n', '-j', '366251', '--format=JobID,State,ExitCode,End', '-P')
    terminal_steps(acct, '366251')
    d, cost = read(DIAG), read(COST)
    assert cost['diagnosis_sha256'] == budget.sha(ROOT/DIAG)
    for p, h in {**d['files_sha256'], **d['protected_sha256']}.items():
        assert budget.sha(ROOT/p) == h, p
    assert not any((ROOT/p).exists() for p in d['absent_journals'])
    assert not (ROOT/third.JOURNAL).exists() and not (ROOT/third.MODEL_JOURNAL).exists()
    for rel in (POLICY, ADMISSION, SCRIPT):
        assert not (ROOT/rel).exists()
    old = read('configs/parallel_core_run02_second_reviewed_20261015.json')
    checked = w.review(ROOT, ROOT/'configs/parallel_core_run02_second_reviewed_20261015.json', None)
    assert {j['id'] for j in checked['reviewed_jobs'] if j.get('scope') != 'different project'} == ACTIVE
    meta = {j: fields(j) for j in ACTIVE}
    assert third.afterok_ids(meta['365355']['Dependency']) == {'366115', '366116', '366251', '365300'}
    assert third.afterok_ids(meta['365300']['Dependency']) == {'366115', '366116'}
    assert meta['365300']['Priority'] == '0' and meta['365300']['StartTime'] == 'Unknown'
    assert all(meta[j]['JobState'] == 'RUNNING' and meta[j]['Dependency'] == '(null)'
               for j in ACTIVE-{'365300', '365355'})
    prior = read('configs/confirmation_budget200_reviewed_admission_20261015.json')
    scopes = {j: prior['reviewed_job_write_cells'][{'366545':'366117','366520':'366118'}.get(j,j)] for j in ACTIVE-{'365300', '365355'}}
    for j in scopes: third.checked_scope(ROOT, {'reviewed_job_write_cells':scopes}, j)
    import protocol
    third.history(ROOT, {'diagnosis_sha256':budget.sha(ROOT/DIAG), 'cost_plan_sha256':budget.sha(ROOT/COST)}, d, protocol.MODELS)
    for rel, digest in cost['source_sha256_or_absent'].items():
        assert (budget.sha(ROOT/rel) if (ROOT/rel).exists() else None) == digest, rel
    for source in cost['source_proofs']:
        assert budget.sha(ROOT/source['path']) == source['sha256']
    protected = {('quality', c['model'], 'B0', c['benchmark'], 'test') for c in d['cells']}
    assert not any(tuple(c) in protected for cc in scopes.values() for c in cc)
    ledger = read('artifacts/budget.json')
    assert ledger['spent']+ledger['reserved']+cost['known_proxy_usd']+cost['historical_unknown_scenario_usd'] <= 200
    # Preserve current admission/policy identities, including all four running jobs.
    protected_live = {p: budget.sha(ROOT/p) for p in [
        'configs/parallel_budget200_reviewed_20261015.json', 'configs/parallel_budget200_20261015.json',
        'configs/parallel_core_run02_second_reviewed_20261015.json',
        'configs/parallel_core_run03_second_20261015.json',
        'configs/parallel_variants_second_20261015.json',
        'configs/core_run02_second_reviewed_admission_20261015.json',
        'configs/core_run03_second_admission_20261015.json',
        'configs/variants_second_admission_20261015.json',
        *[f'configs/{tag}_budget200_reviewed_admission_20261015.json' for tag in
          ('confirmation', 'schema', 'core_run02', 'core_run03')]]}
    with (ROOT/'artifacts/operations/variants_third_submission_20261015.jsonl').open('x') as journal:
        def log(event, **kw):
            journal.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **kw})+'\n')
            journal.flush(); os.fsync(journal.fileno())
        with (ROOT/SCRIPT).open('x') as f:
            f.write(f'#!/bin/bash\nset -euo pipefail\ncd {ROOT}\nexport PYTHONPATH=src:artifacts/operations\nexec .venv/bin/python -u {third.SELF} --root {ROOT} --admission {ROOT/ADMISSION}\n')
        log('submit_start', script=SCRIPT, sha256=budget.sha(ROOT/SCRIPT))
        job = run('sbatch', '--parsable', '--hold', '--partition=cpu', '--cpus-per-task=2', '--mem=8G',
                  '--time=2-00:00:00', '--job-name=decision-em-variants-third200', '--chdir='+str(ROOT),
                  '--output='+str(ROOT/'logs/variants-third200-%j.log'), str(ROOT/SCRIPT)).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Uncertain submission outcome; do not repeat')
        log('submitted_held', job=job)
        config = read('configs/expanded_execution.json')
        assert config['e6_variants_job'] == 366251
        save('artifacts/operations/before_variants_third_expanded_execution.json', config)
        config['e6_variants_job'] = int(job)
        (ROOT/'configs/expanded_execution.json').write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n')
        run('scontrol', 'update', 'JobId=365355', 'Dependency=afterok:366115:'+job+':366116:365300')
        log('statistics_join_updated', successor=job)
        queue = set()
        # Review every current job against the already-checked old identity or this exact submitted script.
        policy = {**old, 'reviewed_project_jobs': {}, 'files_sha256': dict(old['files_sha256'])}
        for rel in w.REQUIRED:
            policy['files_sha256'][rel] = budget.sha(ROOT/rel)
        deps = {}
        for line in run('squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%j|%Z').splitlines():
            jid, name, cwd = line.split('|', 2)
            if cwd != str(ROOT):
                continue
            queue.add(jid)
            assert jid in ACTIVE|{job}
            sh = hashlib.sha256(run('scontrol', 'write', 'batch_script', jid, '-').encode()).hexdigest()
            if jid in ACTIVE:
                assert old['reviewed_project_jobs'][jid]['name'] == name
                assert old['reviewed_project_jobs'][jid]['batch_script_sha256'] == sh
            else:
                assert sh == budget.sha(ROOT/SCRIPT)
            policy['reviewed_project_jobs'][jid] = {'name': name, 'scope': 'variants third' if jid == job else old['reviewed_project_jobs'][jid]['scope'], 'batch_script_sha256': sh}
            deps[jid] = fields(jid)['Dependency']
        assert queue == ACTIVE|{job}
        save(POLICY, policy)
        spec = read('configs/variants_second_admission_20261015.json')
        spec.update(predecessor='366251', slurm_job_id=job, diagnosis=DIAG, diagnosis_sha256=budget.sha(ROOT/DIAG),
                    cost_plan=COST, cost_plan_sha256=budget.sha(ROOT/COST), parallel_policy=POLICY,
                    parallel_policy_sha256=budget.sha(ROOT/POLICY), budget_binding_sha256=budget.sha(ROOT/third.SELF),
                    reviewed_dependencies=deps, reviewed_job_write_cells={**scopes, job: sorted(protected)},
                    prior_running_admission_and_policy_sha256=protected_live)
        for rel in w.REQUIRED:
            spec['source_sha256'][rel] = budget.sha(ROOT/rel)
        save(ADMISSION, spec)
        assert all(budget.sha(ROOT/p) == h for p, h in protected_live.items())
        save('artifacts/operations/variants_third_held_plan_20261015.json',
             {'job': job, 'admission': ADMISSION, 'policy': POLICY, 'dependencies': deps,
              'preserved_live_sha256': protected_live, 'held_not_released': True})
        log('prepared_held', job=job, dependencies=deps, admission_sha256=budget.sha(ROOT/ADMISSION))
        print(json.dumps({'job': job, 'dependencies': deps, 'held_not_released': True}))


if __name__ == '__main__':
    main()
