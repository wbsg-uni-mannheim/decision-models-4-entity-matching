"""Private frozen analysis binding for the explicit Run3 measurement source map.

Never edits frozen sources or old measurements. The interrupted measurement's
cost remains separate from the replacement sample, in an additional cost proof.
Full analysis writes only a caller-owned exclusive staging directory. This is
not a scientific completion marker or a substitute for independent final audit.
"""
import ast
import copy
import math
from pathlib import Path
import types

import core3_timing_recovery_plan_20261016 as plan

PLAN = 'configs/core3_timing_recovery_measurement_plan_20261016.json'
PLAN_SHA = '6e7835eb42b741aa953a1a1d4c4c7c5002aa54debaf6f4bfdbece28f7db1e5d3'
ANALYZE_SHA = '1b98194d769020d772d8e21200cbfe7caa1fe8d14ca6681eeca10c1769282f37'


def validate_map(spec, protocol):
    cells = spec['cells']
    expected = [(m, b, ns) for m in protocol.MODELS for b in protocol.BENCHMARKS
                for ns in ('timing_serial', 'timing_concurrent8')]
    plan.require([(c['model'], c['benchmark'], c['original_namespace']) for c in cells] == expected,
                 'Exact ordered timing matrix required')
    plan.require(len(cells) == 112 and spec['run_id'] == 3 and type(spec['run_id']) is int,
                 'Exact Run3 timing map required')
    for c in cells:
        gap = (c['original_namespace'], c['model'], c['benchmark']) == plan.GAP
        plan.require(c['analysis_namespace'] == (plan.REPLACEMENT if gap else c['original_namespace']),
                     'Response and execution sources must share the canonical mapping')
        plan.require(c['variant'] == 'B0' and c['split'] == 'prompt' and c['limit'] == 20
                     and c['workers'] == (1 if c['original_namespace'] == 'timing_serial' else 8),
                     'Timing scope changed')
    return {(c['model'], c['benchmark'], c['original_namespace']): copy.deepcopy(c) for c in cells}


def bind(frozen, target, spec, protocol, output):
    """Bind BOTH frozen load and execution path; all other analysis stays frozen."""
    target, output = Path(target), Path(output)
    source = Path(frozen.__file__).read_text()
    plan.require(plan.sha(Path(frozen.__file__)) == ANALYZE_SHA, 'Frozen analyzer changed')
    mapping = validate_map(spec, protocol)
    accepted = {k: plan.validate_measurement(target, c, protocol) for k, c in mapping.items()}
    namespace = dict(vars(frozen))
    namespace.update(PROJECT=target, OUT=output)
    # Clone every module function, so save/load/main all use private globals.
    for name, fn in vars(frozen).items():
        if isinstance(fn, types.FunctionType) and fn.__globals__ is vars(frozen):
            namespace[name] = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    original_load = namespace['load']

    def resolve(m, b, ns):
        plan.require((m, b, ns) in mapping, 'Unmapped timing access')
        return mapping[m, b, ns]['analysis_namespace']

    def load(alias, variant, b, split, namespace='quality'):
        if namespace.startswith('timing_'):
            plan.require(variant == 'B0' and split == 'prompt', 'Unexpected timing query')
            namespace = resolve(alias, b, namespace)
        return original_load(alias, variant, b, split, namespace)

    def executions(m, b, ns):
        return target/'runs'/resolve(m, b, ns)/m/'B0'/b/'prompt.executions.jsonl'

    namespace.update(load=load, _execution_path=executions)
    old = "execution_path=PROJECT/'runs'/('timing_'+regime)/m/'B0'/b/'prompt.executions.jsonl'"
    plan.require(source.count(old) == 1, 'Frozen execution path anchor changed')
    tree = ast.parse(source.replace(old, "execution_path=_execution_path(m,b,'timing_'+regime)"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    exec(compile(ast.Module(body=[main], type_ignores=[]), '<private mapped frozen analysis>', 'exec'), namespace)
    # Expose exactly the frozen timing block for isolated integration tests and
    # final report comparison, without running expensive quality bootstraps.
    loop = next(n for n in main.body if isinstance(n, ast.For)
                and "timings.append" in ast.unparse(n))
    code = 'def timing_table():\n    timings = []\n' + '\n'.join('    '+s for s in ast.unparse(loop).splitlines()) + '\n    return timings\n'
    exec(compile(code, '<frozen timing table>', 'exec'), namespace)
    return types.SimpleNamespace(**namespace), accepted


def cost_proof(target, spec, accepted):
    """Every original/replacement raw attempt, including excluded failed cell."""
    target = Path(target)
    sampled = []
    for c in spec['cells']:
        key = (c['model'], c['benchmark'], c['original_namespace'])
        sampled.append(accepted[key])
    old = plan.base(target, *plan.GAP)
    # GAP tuple is namespace/model/benchmark, exactly base's argument order.
    success = plan.rows(old/'prompt.jsonl')
    failure = plan.rows(old/'prompt.failures.jsonl')
    attempts = [a for r in success+failure for a in r['attempts']]
    plan.require(len(success) == 9 and len(failure) == 7 and len(attempts) == 30
                 and not (old/'prompt.executions.jsonl').exists(), 'Old gap diagnosis changed')
    excluded = {'responses': 9, 'attempts': 30,
        'known_usd': math.fsum(a['cost'] for a in attempts if a['cost'] is not None),
        'unknown_attempts': sum(a['cost'] is None for a in attempts),
        'execution_walltime': None, 'included_in_latency_or_throughput': False}
    selected = {k: math.fsum(c[k] for c in sampled) for k in ('known_usd',)}
    selected.update({k: sum(c[k] for c in sampled) for k in ('responses', 'attempts', 'unknown_attempts')})
    return {'analysis_sample': selected, 'excluded_interrupted_diagnostic': excluded,
        'all_retained_timing_attempts': selected['attempts']+excluded['attempts'],
        'all_retained_known_usd': selected['known_usd']+excluded['known_usd'],
        'all_retained_unknown_attempts': selected['unknown_attempts']+excluded['unknown_attempts'],
        'cost_scope': 'Retained original and new timing attempts; not incremental billing or an invoice',
        'measurement_policy': spec['measurement_policy'], 'analysis_policy': spec['analysis_policy']}
