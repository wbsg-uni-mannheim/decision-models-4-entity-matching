"""Read-only full fresh schema cost scenarios; no selection or admission.

All Run1 pilot/development sources are replayed. Test scenarios enumerate one
shared global binary condition and one local condition per family, plus Review.
The maxima bound these historical proxies only, never future provider charges.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

from diagnose_schema_terminal import audit_schema_rows
from operational_journal import inspect_operations
from repeated_runs import verify

FIELDS = ('known_proxy_usd', 'historical_unknown_scenario_usd')
SUFFIXES = ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def total(values):
    return {k: math.fsum(v[k] for v in values) for k in FIELDS}


def source(root, schema, model, variant, benchmark, pilot=False):
    namespace, split = ('schema_pilot', 'schema_pilot') if pilot else ('schema', 'prompt')
    base = root/'runs'/namespace/model/variant/benchmark/split
    paths = {s: Path(str(base)+s) for s in SUFFIXES}
    before = {str(p.relative_to(root)): sha(p) for p in paths.values()}
    ip = root/'data'/benchmark/(split+'.json')
    before[str(ip.relative_to(root))] = sha(ip)
    cases, successes, failures = read(ip), rows(paths['.jsonl']), rows(paths['.failures.jsonl'])
    audited = audit_schema_rows(cases, successes, failures, schema, model, variant, benchmark)
    bound = SimpleNamespace(payload=schema.payload, digest=schema.wire_hash)
    ops = inspect_operations(paths['.operations.jsonl'], cases, failures, bound, model, variant, 1)
    if not audited['complete'] or not audited['responses'] or ops['unknown_outcome_events']:
        raise ValueError('Complete source without unknown operational outcomes required')
    if any(sha(root/p) != h for p, h in before.items()):
        raise ValueError('Source changed while reading')
    return dict(model=model, variant=variant, benchmark=benchmark, namespace=namespace, split=split,
                path=str(paths['.jsonl'].relative_to(root)), **audited, operations=ops,
                known_proxy_usd=audited['known_usd'],
                historical_unknown_scenario_usd=.05*audited['unknown_attempts'],
                protected_sha256_or_absent=before)


def scenarios(sources, models, benchmarks, binary, pilot_families, test_counts):
    if (len(set(models)) != len(models) or len(set(benchmarks)) != len(benchmarks)
            or len(set(binary)) != len(binary) or len(binary) != 4 or 'C_review' in binary
            or 'billiger-de' not in benchmarks or not set(pilot_families) <= set(benchmarks)):
        raise ValueError('Invalid schema matrix')
    expected = {(ns, m, v, b) for ns, families in
                [('schema_pilot', pilot_families), ('schema', benchmarks)]
                for m in models for v in binary+['C_review'] for b in families}
    indexed = {(s['namespace'], s['model'], s['variant'], s['benchmark']): s for s in sources}
    if len(indexed) != len(sources) or set(indexed) != expected:
        raise ValueError('Complete unique pilot/development matrix required')
    for s in sources:
        if type(s['responses']) is not int or s['responses'] <= 0:
            raise ValueError('Invalid source count')
        if s['namespace'] == 'schema_pilot' and s['responses'] != 2:
            raise ValueError('Frozen pilot requires two responses per cell')
        if any(type(s[k]) not in (int, float) or not math.isfinite(s[k]) or s[k] < 0 for k in FIELDS):
            raise ValueError('Invalid source cost')
    if (set(test_counts) != set(benchmarks)|{'billiger-de-50un'}
            or any(type(n) is not int or n <= 0 for n in test_counts.values())):
        raise ValueError('Complete positive test sizes required')
    phases = {}
    for ns in ('schema_pilot', 'schema'):
        cells = [s for s in sources if s['namespace'] == ns]
        phases[ns] = dict(cells=len(cells), prospective_responses=sum(s['responses'] for s in cells), **total(cells))
    candidates, by_key = [], {}
    for m in models:
        for b in benchmarks:
            targets = [b, 'billiger-de-50un'] if b == 'billiger-de' else [b]
            n = sum(test_counts[t] for t in targets)
            for v in binary+['C_review']:
                s = indexed['schema', m, v, b]
                row = dict(model=m, variant=v, source_benchmark=b, target_benchmarks=targets,
                           source_responses=s['responses'], target_responses=n,
                           **{k: s[k]*n/s['responses'] for k in FIELDS})
                candidates.append(row); by_key[m, v, b] = row
    # Keep global choice identical across all families, including transfer.
    # Local and global can coincide: count each selected cell only once.
    evaluations, maxima = [], []
    for m in models:
        for field in FIELDS:
            choices = []
            for g in binary:
                locals_, family_costs = {}, []
                for b in benchmarks:
                    options = [(math.fsum(by_key[m, v, b][field] for v in {g, l, 'C_review'}), l)
                               for l in binary]
                    cost, local = sorted(options, key=lambda x: (-x[0], x[1]))[0]
                    locals_[b] = local; family_costs.append(cost)
                candidate = dict(model=m, cost_field=field, global_variant=g, local_variants=locals_,
                                 value=math.fsum(family_costs))
                choices.append(candidate); evaluations.append(candidate)
            maxima.append(sorted(choices, key=lambda x: (-x['value'], x['global_variant']))[0])
    test_total = {k: math.fsum(x['value'] for x in maxima if x['cost_field'] == k) for k in FIELDS}
    development_total = total(list(phases.values()))
    combined = {k: development_total[k]+test_total[k] for k in FIELDS}
    return dict(pilot_and_development=phases, pilot_and_development_total=development_total,
                test_candidates=candidates, global_choice_scenarios=evaluations, test_maxima=maxima,
                test_scenario_maximum=test_total,
                maximum_distinct_test_cells=3*len(models)*len(test_counts),
                maximum_test_responses=3*len(models)*sum(test_counts.values()),
                per_run_total=combined, two_runs_total={k: 2*v for k, v in combined.items()})


def build(root, output):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root/'src'))
    import expanded_schema as es
    import orchestrate
    import billiger50
    def denied(*args, **kwargs):
        raise AssertionError('Network forbidden in schema cost planning')
    socket.socket = denied; socket.create_connection = denied
    if es.PROJECT != root:
        raise ValueError('Wrong workspace')
    manifest = es.verify(); orchestrate.verify(); billiger50.verify()
    families = manifest['pilot']['families']
    if families != ['abt-buy', 'dblp-scholar'] or manifest['pilot']['pairs_per_family'] != 2:
        raise ValueError('Frozen pilot changed')
    specs = {n: verify(root, n) for n in (2, 3)}
    matrix = [(ns, m, v, b, split) for ns, split, bb in
              [('schema_pilot', 'schema_pilot', families), ('schema', 'prompt', es.BENCHMARKS)]
              for m in es.MODELS for v in es.CONDITIONS for b in bb]
    protected = set()
    for ns, m, v, b, split in matrix:
        protected.add(f'data/{b}/{split}.json')
        protected |= {f'runs/{ns}/{m}/{v}/{b}/{split}'+s for s in SUFFIXES}
    protected |= {str(p.relative_to(root)) for pattern in ('*admission*.json', 'parallel_*.json')
                  for p in (root/'configs').glob(pattern)}
    protected.add('artifacts/schema_pilot_status.json')
    # Selection is protected but never read to choose or price fresh conditions.
    protected |= {'configs/expanded_schema_selection.json', 'configs/expanded_schema_test_cells.json'}
    counts = {}
    for b in [*es.BENCHMARKS, 'billiger-de-50un']:
        rel = f'data/{b}/test.json'; protected.add(rel); cases = read(root/rel)
        if len({c['pair_id'] for c in cases}) != len(cases):
            raise ValueError('Duplicate test pairs')
        counts[b] = len(cases)
    for n, spec in specs.items():
        prefix = f'replications/run{n:02d}/'
        protected |= {prefix+p for p in spec['immutable_sha256']}
        protected |= {p for p in spec['immutable_sha256'] if p.startswith(('src/', 'configs/'))}
        for rel in [p for p in protected if p.startswith('data/')]:
            if sha(root/(prefix+rel)) != sha(root/rel):
                raise ValueError('Replica input differs')
            protected.add(prefix+rel)
        for ns, split, bb in [('schema_pilot', 'schema_pilot', families),
                              ('schema', 'prompt', es.BENCHMARKS),
                              ('schema', 'test', list(counts))]:
            for m in es.MODELS:
                for v in es.CONDITIONS:
                    for b in bb:
                        protected |= {prefix+f'runs/{ns}/{m}/{v}/{b}/{split}'+s for s in SUFFIXES}
        for ns in ('schema', 'schema_pilot'):
            if any((root/prefix/'runs'/ns).rglob('*')):
                raise ValueError('Fresh schema already attempted; actual missing plan required')
        for rel in ('artifacts/schema_development_attempt', 'artifacts/fresh_schema_selection_freeze',
                    'artifacts/fresh_schema_selected_test', 'artifacts/fresh_schema_final_report',
                    'artifacts/expanded_schema_complete.json', 'artifacts/schema_pilot_status.json',
                    'artifacts/results/expanded_schema', 'configs/expanded_schema_selection.json',
                    'configs/expanded_schema_test_cells.json'):
            if (root/(prefix+rel)).exists():
                raise ValueError('Fresh schema already attempted')
            protected.add(prefix+rel)
    before = {p: sha(root/p) for p in sorted(protected)}
    sources = [source(root, es, m, v, b, ns == 'schema_pilot') for ns, m, v, b, split in matrix]
    pilot = read(root/'artifacts/schema_pilot_status.json')
    expected = [dict(model=s['model'], condition=s['variant'], benchmark=s['benchmark'], status='passed',
                     known_cost_usd=s['known_usd'], unknown_cost_attempts=s['unknown_attempts'])
                for s in sources if s['namespace'] == 'schema_pilot']
    if pilot.get('complete') is not True or pilot['cells'] != expected:
        raise ValueError('Pilot status/raw provenance differs')
    value = scenarios(sources, list(es.MODELS), es.BENCHMARKS, es.BINARY, families, counts)
    if any(sha(root/p) != h for p, h in before.items()):
        raise ValueError('Protected source changed during planning')
    result = dict(utc=datetime.now(timezone.utc).isoformat(), status='fresh_schema_historical_scenarios_not_admission',
                  sources=sources, test_counts=counts, **value, protected_sha256_or_absent=before,
                  source_responses=sum(s['responses'] for s in sources), source_attempts=sum(s['attempts'] for s in sources),
                  replicas_frozen_files_verified={str(n): len(s['immutable_sha256']) for n, s in specs.items()},
                  script_sha256=sha(Path(__file__)), admitted=False, new_paid_calls=0, new_reservations=0,
                  fresh_choices_made=False, quote_or_hard_bound=False, budget_snapshot=read(root/'artifacts/budget.json'),
                  limitations=[
                      'Historical complete Run1 source means include retained retries/failures; no actual fresh schema choices or test costs.',
                      'All fresh pilot/development cells are new; schema namespaces receive no B0 cache credit and no cross-run reuse.',
                      'Known and unknown maxima are separate, potentially different choices. One global binary per model, one local per family, Review fixed; billiger50 choice/cost source tied to billiger-de.',
                      'Maxima bound only these historical scenarios, not real charges, token budgets, quotes or representative future error rates.',
                      'Test input/output lengths may differ; no same-pair or token-length pricing. Unknown allowances are scenarios, not invoices.',
                      'After own freeze and full pilot/development replay, recompute actual selected missing tests and current budget before admission.',
                      'Active core/schema-test/confirmation journals, progress and budget excluded from unchanged guarantee. No scheduler/admission or full-funding claim.',
                      'Historical unjournaled outcomes and all other study stages remain outside this cost scope.'])
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    r = build(args.root.resolve(), args.output)
    print(json.dumps({k: r[k] for k in ('source_responses', 'source_attempts', 'pilot_and_development',
                                      'test_scenario_maximum', 'maximum_test_responses', 'two_runs_total')}))
