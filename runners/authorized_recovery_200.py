"""Explicitly authorized USD200 binding of unchanged bounded recovery modules.

All original source files and scientific functions stay byte-identical. Only
private budget validation constants and the private account default change.
Real scheduler, scope, cache, diagnosis, locks and join checks remain mandatory.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import types

APPROVAL = 'artifacts/operations/approval_budget200_parallel_aaron_20261015.json'
APPROVAL_SHA = 'f4e7f5255c74a18b856a30fb0abc4b44dfb9d81aa98db3e9468b313871dae41b'
SELF = 'artifacts/operations/authorized_recovery_200.py'
PINNED = {
 'resume_variants_drained': '20186df914b77656d466965c63d7b846871a18d40022cb1731d078ed143a1e61',
 'resume_confirmation_drained': '3ef29ad0d9fea82fc16c11b8c0e4bade776f080f7caec4623310769152311e98',
 'resume_schema_drained': 'a104b107df14df045bdd5b6dc3796541bc595b58d1768b200fdcba136c69529c',
 'resume_fresh_core_drained': '37c93ffca7a7338fa61498c70c85f84b7af2eb29c6bda317cb9a560f4973e309',
 'parallel_admission': 'd74c16978b393e38f1b142afbcba8159c716dc24bcc4b807a4f8c3eae82f2c46',
 'confirmation_drained': '96727d4eba97a9619edd9717ddada7eb21667b10ed03b4c815b09cb60f60117e',
 'schema_drained': 'a52d18e0124e3c5e03ac0c898a403b74b8d8e3d392fbdfc0e93b017e8ab18e5c',
 'fresh_core_drained': '73ae8c556063b853af1296fe5298fdac8c26e0ffa851096a94c935ea983c540f',
 'drained_inference': '28e7ce2b2020a1d2b8a9def38c1b3e70e2b202938e5564a6a45cd981330a85b2'}
MODULES = {'variants': 'resume_variants_drained', 'confirmation': 'resume_confirmation_drained',
           'schema': 'resume_schema_drained', 'core': 'resume_fresh_core_drained'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authority(root):
    path = root/APPROVAL
    if sha(path) != APPROVAL_SHA:
        raise ValueError('Budget authorization changed')
    obj = json.loads(path.read_text())
    if obj['approved_by'] != 'Aaron' or obj['project_cap_usd'] != 200 or not obj['parallel_execution_authorized']:
        raise ValueError('USD200 parallel authorization required')
    return obj


def private_module(root, name, replacements=()):
    path = root/'artifacts/operations'/f'{name}.py'
    if sha(path) != PINNED[name]:
        raise ValueError('Prepared source drift: '+name)
    source = path.read_text()
    for old, new, count in replacements:
        if source.count(old) != count:
            raise ValueError('Budget binding anchor drift')
        source = source.replace(old, new)
    module = types.ModuleType('private_authorized_'+name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def make_wrapper(root, stage):
    authority(root)
    for name, expected in PINNED.items():
        if sha(root/'artifacts/operations'/f'{name}.py') != expected:
            raise ValueError('Prepared dependency drift: '+name)
    drain = importlib.import_module('drained_inference')
    def build(inference, run_id):
        # The frozen function uses the same shared fcntl lock and ledger. Fresh
        # core has already rebound PROJECT to the common root before this point.
        fn = inference.account
        if fn.__name__ != 'account' or fn.__defaults__ != (0, 100) or fn.__closure__:
            raise ValueError('Unexpected account binding')
        account = types.FunctionType(fn.__code__, dict(fn.__globals__), fn.__name__, (0, 200))
        if account.__globals__['PROJECT'].resolve() != root.resolve():
            raise ValueError('Account must use the shared root')
        isolated = types.SimpleNamespace(**{**vars(inference), 'account': account})
        request, cell, provenance = drain.build(isolated, run_id)
        return request, cell, {**provenance, 'project_cap_usd': 200,
            'budget_authority_sha256': APPROVAL_SHA, 'budget_binding_sha256': sha(root/SELF)}
    wrapper = private_module(root, MODULES[stage], [
        ("'budget_cap_usd': 100", "'budget_cap_usd': 200", 1),
        (' > 100:', ' > 200:', 1 if stage == 'core' else 2)])
    policy = private_module(root, 'parallel_admission', [("!= 100", "!= 200", 1)])
    def review(project, path, own):
        spec = json.loads(path.read_text())
        if spec['approval_file'] != APPROVAL or spec['approval_sha256'] != APPROVAL_SHA or spec['project_cap_usd'] != 200:
            raise ValueError('Current parallel policy must reference explicit USD200 authority')
        return policy.review(project, path, own)
    wrapper.review = review
    wrapper.REQUIRED |= {SELF, APPROVAL} | {f'artifacts/operations/{n}.py' for n in PINNED}
    if stage == 'variants':
        wrapper.build = build
    elif stage == 'confirmation':
        component = private_module(root, 'confirmation_drained')
        component.build = build
        wrapper.run_selected_tests = component.run_selected_tests
    elif stage == 'schema':
        component = private_module(root, 'schema_drained')
        component.build = build
        wrapper.run_selected_tests = component.run_selected_tests
    else:
        component = private_module(root, 'fresh_core_drained')
        component.build = build
        wrapper.run_phases = component.run_phases
    wrapper.authorized_build = build
    return wrapper


def preflight(root, stage, run_id, spec, wrapper):
    """Real read-only gates for held Slurm jobs; never create stage journals."""
    import sys
    own = spec['slurm_job_id']
    if stage == 'core':
        target = root/f'replications/run{run_id:02d}'
        replica = wrapper.verify(root, run_id)
        return wrapper.verify_ready(root, target, run_id, spec, own, replica)
    sys.path.insert(0, str(root/'src'))
    if stage == 'variants':
        import expanded_variants as ev
        import protocol
        import orchestrate
        ev.verify(); orchestrate.verify()
        return wrapper.verify_ready(root, spec, own, protocol.MODELS, ev.VARIANTS)
    module = importlib.import_module('expanded_'+stage)
    analysis = importlib.import_module('analyze')
    cells, _ = wrapper.selected_tests(root, module, analysis)
    return wrapper.verify_ready(root, spec, own, cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=MODULES, required=True)
    parser.add_argument('--run-id', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = json.loads(args.admission.read_text())
    if (spec.get('budget_authority_sha256') != APPROVAL_SHA or spec.get('budget_cap_usd') != 200
            or spec.get('budget_binding_sha256') != sha(root/SELF)
            or spec.get('stage') != args.stage or spec.get('run_id') != args.run_id):
        raise ValueError('Exact admitted USD200 binding required')
    wrapper = make_wrapper(root, args.stage)
    if args.preflight_output:
        result = preflight(root, args.stage, args.run_id, spec, wrapper)
        with args.preflight_output.open('x') as f:
            json.dump({'admission_sha256': sha(args.admission), 'read_only': True,
                       'stage_execution': False, 'checks': result}, f, indent=2); f.write('\n')
        return 0
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Real Slurm execution required')
    return (wrapper.execute(root, args.run_id, args.admission) if args.stage == 'core'
            else wrapper.execute(root, args.admission))


if __name__ == '__main__':
    raise SystemExit(main())
