"""Read-only fresh confirmation scenarios, without choosing prompts or admission.

Every candidate uses its own complete Run1 development source. Unknown fresh
choices are bounded only within these historical scenarios, never in real USD.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import sys

from core_phase_cost_plan import aggregate
from fixed_extension_cost_plan import source, SUFFIXES
from fresh_confirmation_selection import sha
from repeated_runs import verify

FIELDS = ('known_proxy_usd', 'historical_unknown_scenario_usd')


def scenarios(sources, models, benchmarks, prompts, serializations, ablation_models, test_counts):
    expected = {(m, v, b) for m in models for v in prompts+serializations for b in benchmarks}
    indexed = {(s['model'], s['variant'], s['benchmark']): s for s in sources}
    if len(indexed) != len(sources) or set(indexed) != expected:
        raise ValueError('Complete unique development source matrix required')
    if set(test_counts) != set(benchmarks)|{'billiger-de-50un'}:
        raise ValueError('Complete test family sizes required')
    for s in sources:
        if type(s['responses']) is not int or s['responses'] <= 0:
            raise ValueError('Invalid source count')
        if any(type(s[k]) not in (int, float) or not math.isfinite(s[k]) or s[k] < 0 for k in FIELDS):
            raise ValueError('Invalid source cost')
    if any(type(n) is not int or n <= 0 for n in test_counts.values()):
        raise ValueError('Invalid test count')
    development = {}
    for name, rows in {
        'gross_all_cells_before_reuse': sources,
        'conditional_complete_B0': [s for s in sources if s['variant'] != 'B0'],
        'conditional_successful_core': [s for s in sources if s['variant'] != 'B0'
            and not (s['variant'] == 'S1' and s['model'] in ablation_models)]}.items():
        development[name] = {'cells': len(rows), 'prospective_responses': sum(s['responses'] for s in rows), **aggregate(rows)}
    candidates, envelopes = [], []
    # Four prompt regimes and two serialization regimes can select at most
    # four / two distinct cells per model and target. Same-cell reuse is exact.
    # billiger50 choices are tied to billiger-de: price both sizes together.
    for m in models:
        for b in benchmarks:
            targets = [b, 'billiger-de-50un'] if b == 'billiger-de' else [b]
            n = sum(test_counts[t] for t in targets)
            for family, variants, maximum in (('prompt', prompts, 4), ('serialization', serializations, 2)):
                group = []
                for v in variants:
                    s = indexed[m, v, b]
                    row = {'model': m, 'source_benchmark': b, 'target_benchmarks': targets,
                           'variant': v, 'family': family, 'source_responses': s['responses'],
                           'target_responses': n, 'new_responses_if_selected': 0 if v == 'B0' else n,
                           **{k: 0.0 if v == 'B0' else s[k]*n/s['responses'] for k in FIELDS}}
                    group.append(row); candidates.append(row)
                envelope = {'model': m, 'source_benchmark': b, 'family': family,
                            'maximum_distinct_choices': maximum, 'target_benchmarks': targets,
                            'new_responses_scenario_max': sum(sorted((r['new_responses_if_selected'] for r in group), reverse=True)[:maximum])}
                for field in FIELDS:
                    ranked = sorted(group, key=lambda r: (-r[field], r['variant']))[:maximum]
                    envelope[field] = math.fsum(r[field] for r in ranked)
                    envelope[field+'_variants'] = [r['variant'] for r in ranked]
                envelopes.append(envelope)
    total = aggregate(envelopes)
    combined = {k: development['conditional_successful_core'][k]+total[k] for k in FIELDS}
    return {'development_alternatives_not_additive': development, 'test_candidates': candidates,
            'test_envelopes': envelopes, 'test_envelope_total': total,
            'new_test_response_scenario_max': sum(r['new_responses_scenario_max'] for r in envelopes),
            'conditional_per_run_total': combined,
            'conditional_two_runs_total': {k: 2*v for k, v in combined.items()}}


def build(root, output):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(root/'src'))
    import protocol
    import expanded_confirmation as ec
    import expanded_protocol as ep
    def denied(*a, **kw):
        raise AssertionError('Network forbidden in cost planning')
    socket.socket = denied; socket.create_connection = denied
    if ec.PROJECT != root:
        raise ValueError('Wrong source workspace')
    ec.verify()
    specs = {n: verify(root, n) for n in (2, 3)}
    cells = [dict(namespace='quality', model=m, variant=v, benchmark=b, split='prompt')
             for m in ec.MODELS for v in ec.PROMPT_VARIANTS+ec.SERIALIZATIONS for b in ec.BENCHMARKS]
    protected = set()
    for c in cells:
        base = f"runs/quality/{c['model']}/{c['variant']}/{c['benchmark']}/prompt"
        protected |= {base+suffix for suffix in SUFFIXES}
    frozen = ['configs/expanded_confirmation_manifest.json','configs/manifest.json','configs/billiger50_manifest.json',
              'configs/parallel_budget200_20261015.json','configs/parallel_budget200_reviewed_20261015.json',
              'configs/parallel_variants_second_20261015.json']
    protected |= set(frozen)
    for n, spec in specs.items():
        target = root/f'replications/run{n:02d}'
        protected |= {f'replications/run{n:02d}/{p}' for p in spec['immutable_sha256']}
        protected |= {p for p in spec['immutable_sha256'] if p.startswith(('src/','configs/'))}
        # These new stages must remain untouched. Active B0/S1/progress/budget
        # are explicitly outside the unchanged guarantee and reuse credit.
        for b in [*ec.BENCHMARKS, 'billiger-de-50un']:
            for m in ec.MODELS:
                for v in ec.PROMPT_VARIANTS+['X_R_aligned']:
                    for split in ('prompt','test'):
                        for suffix in SUFFIXES:
                            rel=f'replications/run{n:02d}/runs/quality/{m}/{v}/{b}/{split}{suffix}'
                            if (root/rel).exists():
                                raise ValueError('Expanded fresh cache exists; need actual missing-pair plan')
                            protected.add(rel)
        for rel in ('artifacts/fresh_confirmation_selection_freeze','artifacts/fresh_confirmation_selected_test',
                    'artifacts/fresh_confirmation_final_report','artifacts/expanded_confirmation_complete.json',
                    'configs/expanded_selected_variants.json','configs/expanded_confirmation_cells.json'):
            if (target/rel).exists():
                raise ValueError('Fresh confirmation already attempted')
            protected.add(str((target/rel).relative_to(root)))
    for b in ec.BENCHMARKS:
        protected.add(f'data/{b}/prompt.json')
    test_counts={}
    for b in [*ec.BENCHMARKS,'billiger-de-50un']:
        rel=f'data/{b}/test.json';protected.add(rel)
        cases=json.loads((root/rel).read_text())
        if len({c['pair_id'] for c in cases}) != len(cases):
            raise ValueError('Duplicate test pairs')
        test_counts[b]=len(cases)
    before={p:sha(root/p) for p in sorted(protected)}
    sources=[source(root,c,ep,protocol) for c in cells]
    value=scenarios(sources,list(ec.MODELS),list(ec.BENCHMARKS),ec.PROMPT_VARIANTS,
                    ec.SERIALIZATIONS,list(protocol.ABLATION_MODELS),test_counts)
    if any(sha(root/p) != h for p,h in before.items()):
        raise ValueError('Source or protected fresh scope changed during planning')
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'fresh_confirmation_historical_scenarios_not_admission',
            'sources':sources,'test_counts':test_counts,**value,'protected_sha256_or_absent':before,
            'source_responses':sum(s['responses'] for s in sources),'source_attempts':sum(s['attempts'] for s in sources),
            'replicas_frozen_files_verified':{str(n):len(s['immutable_sha256']) for n,s in specs.items()},
            'script_sha256':sha(Path(__file__)),'admitted':False,'new_paid_calls':0,'new_reservations':0,
            'budget_snapshot':json.loads((root/'artifacts/budget.json').read_text()),
            'quote_or_hard_bound':False,'fresh_choices_made':False,
            'method':'All complete own-variant/model/family Run1 development costs include retained retries/failures. Full development alternatives; successful-core conditional credit excludes B0 and four core S1 development models only. Unknown tests use full target sizes times matching development per-response costs; B0 free only conditional on complete own core+supplement. Sum largest four prompt and two serialization distinct candidate scenarios per model/family; billiger50 tied to billiger-de. Known and unknown maxima computed separately, not necessarily same variants. No S1 test credit assumed.',
            'limitations':['Scenario envelopes bound calculated historical proxies only, not actual future charges or a provider/token hard bound.',
                'Full successful core/supplement is a condition, not an observed completion. No current active fresh cache is credited.',
                'After actual prerequisites and own development selection, recompute exact missing selected-test scope and actual costs before test admission.',
                'Unknown allowances are separate historical scenarios, not provider fees or representative future error forecasts.',
                'No Run1 winners or test quality used to choose fresh variants; actual selection remains own-run development only.',
                'Test lengths/output tokens may differ from development; billiger50 transfers billiger-de cost means.',
                'No response reuse across runs; shared cache credit only for byte-identical requests in the same run.',
                'Active core B0/S1 journals, progress and budget excluded from unchanged guarantee. No full scheduler/admission gate.',
                'Other stages and historical unjournaled outcomes excluded; no full financing, billing or zero-duplicate guarantee.']}
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();r=build(args.root.resolve(),args.output)
    print(json.dumps({k:r[k] for k in ('source_responses','source_attempts','development_alternatives_not_additive','test_envelope_total','new_test_response_scenario_max','conditional_two_runs_total')}))
