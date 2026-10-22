"""Private Run1 rules/manual execution component, with no CLI or admission.

A future bounded wrapper must hold all shared-cache writer locks and validate
actual scheduler, policy, scopes, sources, cache diagnosis, costs, budget and
dependency joins. This component neither releases the old pending job nor writes
a scientific report/completion marker. Ordinary exceptions are drained; process
kill/storage failure and historical unjournaled outcomes remain unresolved.
"""
import hashlib
import json
from pathlib import Path
import types

from diagnose_confirmation_terminal import audit_confirmation_rows
from drained_inference import build, drain_model_tasks
from operational_journal import inspect_operations
from rules_remaining_cost_plan import matrix, validate_matrix

RULES_SHA = 'c2e3ff5f2287d049d2469514ca0addc8a63c5021019642a5b0b61cf71f2044ee'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(function, **values):
    return types.FunctionType(function.__code__, {**function.__globals__, **values},
                              argdefs=function.__defaults__, closure=function.__closure__)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def prepare(root, rules, protocol):
    """Read-only private frozen verification; imported RULES stays untouched."""
    if rules.PROJECT != root:
        raise ValueError('Wrong rules workspace')
    if sha(Path(rules.__file__)) != RULES_SHA:
        raise ValueError('Frozen rules source changed')
    if (root/'artifacts/expanded_rules_complete.json').exists():
        raise ValueError('Rules already complete')
    private_rules = {}
    spec = bind(rules.verify, PROJECT=root, RULES=private_rules)()
    payload = bind(rules.payload, RULES=private_rules)
    frozen = types.SimpleNamespace(**{**vars(rules), 'PROJECT': root,
                                      'RULES': private_rules, 'payload': payload})
    expected = matrix(protocol.MODELS, protocol.BENCHMARKS)
    validate_matrix(spec['cells'], expected)
    if rules.CONDITIONS != {'R_rules': 'rules', 'X_P_domain_0': 'quality'}:
        raise ValueError('Wrong fixed conditions')
    templates = {b: {m: {v: payload({'benchmark': b, 'records': {'left': {}, 'right': {}}}, m, v)
                            for v in ('B0', 'R_rules', 'X_P_domain_0')} for m in protocol.MODELS}
                 for b in [*protocol.BENCHMARKS, 'billiger-de-50un']}
    if templates != spec['request_templates']:
        raise ValueError('Frozen request templates changed')
    paths = ['src/expanded_rules.py', 'src/run.py', 'src/protocol.py', 'src/expanded_protocol.py',
             'configs/expanded_rules_manifest.json', 'configs/learned_rules.json',
             'configs/rule_agent_training_manifest.json', 'artifacts/astra_agent/acceptance.json']
    training = json.loads((root/'configs/rule_agent_training_manifest.json').read_text())
    paths += list(training['excluded_split_hashes'])
    paths += [f'data/{b}/rules_train.json' for b in protocol.BENCHMARKS]
    paths += [f'data/{c["benchmark"]}/{c["split"]}.json' for c in expected]
    protected = {p: sha(root/p) for p in sorted(set(paths))}
    return frozen, spec, expected, protected


def audit(root, frozen, protocol, cells, require_complete=False):
    """Validate raw responses, prior retries and operational joins once each."""
    result = []
    bound_protocol = types.SimpleNamespace(**{**vars(protocol), 'payload': frozen.payload})
    for c in cells:
        m, v, b, split, ns = (c[k] for k in ('model', 'variant', 'benchmark', 'split', 'namespace'))
        directory = root/'runs'/ns/m/v/b
        cases = json.loads((root/'data'/b/(split+'.json')).read_text())
        failures = rows(directory/(split+'.failures.jsonl'))
        info = audit_confirmation_rows(cases, rows(directory/(split+'.jsonl')), failures,
                                       frozen, protocol, m, v, b)
        operations = inspect_operations(directory/(split+'.operations.jsonl'), cases, failures,
                                        bound_protocol, m, v, 1)
        if operations['unknown_outcome_events']:
            raise ValueError('Unknown operational outcome requires terminal review')
        if require_complete and not info['complete']:
            raise ValueError('Incomplete rules/control phase')
        result.append({**c, **info, 'operations': operations})
    return result


def unchanged(root, protected):
    if any(sha(root/p) != h for p, h in protected.items()):
        raise ValueError('Frozen sources or inputs changed during execution')


def run_fixed_matrix(root, rules, inference, protocol, journal_dir):
    """At most one invocation per model PER PHASE, no retries or report.

    All seven development workers drain, then every development cell is raw
    validated as complete before the first test invocation. Caller owns admission
    and exclusive shared-cache locks for the entire call and subsequent report.
    """
    if inference.PROJECT != root:
        raise ValueError('Wrong inference workspace')
    if journal_dir.exists():
        raise ValueError('Already attempted; terminal review required')
    frozen, spec, cells, protected = prepare(root, rules, protocol)
    controls = [dict(c, variant='B0', namespace='quality') for c in cells if c['variant'] == 'R_rules']
    # Partial manual caches can originate in confirmation; check actual raw
    # requests, not just the variant/namespace name, before any new call.
    audit(root, frozen, protocol, cells)
    audit(root, frozen, protocol, controls, require_complete=True)
    unchanged(root, protected)
    isolated = types.SimpleNamespace(**{**vars(inference), 'payload': frozen.payload})
    _, cell, adapter = build(isolated, 1)
    journal_dir.mkdir(exist_ok=False)
    result = {'phases': [], 'adapter': adapter, 'protected_sha256': protected,
              'cells': len(cells), 'completion_marker_written': False, 'admitted': False,
              'historical_unjournaled_outcomes_resolved': False}
    for split in ('prompt', 'test'):
        phase_cells = [c for c in cells if c['split'] == split]
        def task(model):
            for b in [*protocol.BENCHMARKS, *(['billiger-de-50un'] if split == 'test' else [])]:
                for v, ns in frozen.CONDITIONS.items():
                    cell(model, v, b, split, workers=8, namespace=ns)
        outcomes = drain_model_tasks(task, list(protocol.MODELS),
                                    journal_dir/(split+'_models.jsonl'), protocol.canonical)
        audited = audit(root, frozen, protocol, phase_cells)
        unchanged(root, protected)
        complete = all(x['status'] == 'complete' for x in outcomes) and all(x['complete'] for x in audited)
        result['phases'].append({'split': split, 'models': outcomes, 'cells': audited, 'complete': complete})
        if not complete:
            result['status'] = 'failed'
            return result
    # A wrapper must recheck this again while retaining its locks, immediately
    # before privately binding the frozen report and writing its completion marker.
    audit(root, frozen, protocol, controls, require_complete=True)
    result['status'] = 'inference_complete_report_pending'
    return result
