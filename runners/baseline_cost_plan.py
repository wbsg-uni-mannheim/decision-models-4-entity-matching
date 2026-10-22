"""Offline B0-only cost planning; verifies all required Run1 source cells."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

from fresh_run_cost_plan import inventory, sha


def build(root, proof_path, output_path):
    root = root.resolve()
    if any(name in sys.modules for name in ['run', 'protocol', 'analyze']):
        raise RuntimeError('Use a fresh process')
    sys.path.insert(0, str(root/'src'))
    import analyze
    import billiger50
    import orchestrate
    from protocol import MODELS, BENCHMARKS
    if analyze.PROJECT != root:
        raise ValueError('Wrong source workspace')
    orchestrate.verify()
    billiger50.verify()
    matrix = [(m, b, s) for m in MODELS for b in BENCHMARKS
              for s in ['prompt', 'route', 'test']]
    matrix += [(m, billiger50.BENCHMARK, 'test') for m in MODELS]
    sources, costs, unknowns = [], [], []
    for model, benchmark, split in matrix:
        data = analyze.load(model, 'B0', benchmark, split)
        path = root/'runs/quality'/model/'B0'/benchmark/(split+'.jsonl')
        failures = path.with_name(split+'.failures.jsonl')
        sources.append({'model':model, 'variant':'B0', 'benchmark':benchmark, 'split':split,
                        'namespace':'quality', 'cache_sha256':sha(path),
                        'failure_sha256':sha(failures) if failures.exists() else None,
                        'request_hashes_verified':True, 'response_count':len(data['responses'])})
        costs.append(float(data['costs'].sum()))
        unknowns.append(data['unknown_cost_attempts'])
    proof = {'run_id':1, 'stages':['baseline_only_including_development'],
             'verified_source_cells':len(sources), 'sources':sources}
    with proof_path.open('x') as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)
        handle.write('\n')
    # Independent aggregation from raw responses AND failure journals. This also
    # refuses unmatched failed requests and detects changes since verification.
    cells, proofs = inventory(root, [proof_path.resolve()])
    known = math.fsum(cell['known_usd'] for cell in cells)
    unknown = sum(cell['unknown_attempts'] for cell in cells)
    if not math.isclose(known, math.fsum(costs), abs_tol=1e-9) or unknown != sum(unknowns):
        raise ValueError('Independent cost reconciliation failed')
    budget_bytes = (root/'artifacts/budget.json').read_bytes()
    budget = json.loads(budget_bytes)
    headroom = 100-budget['spent']-budget['reserved']
    report = {'utc':datetime.now(timezone.utc).isoformat(),
              'scope':'B0 quality, seven models, eight core benchmarks prompt/route/test plus billiger50 test',
              'source_proofs':proofs, 'verified_source_cells':len(cells),
              'responses_per_run':sum(cell['responses'] for cell in cells),
              'observed_run1_known_usd':known, 'observed_run1_unknown_attempts':unknown,
              'two_fresh_runs_known_proxy_usd':2*known,
              'two_fresh_runs_historical_unknown_allowance_scenario_usd':2*unknown*.05,
              'budget_snapshot':budget, 'uncommitted_headroom_usd':headroom,
              'known_proxy_excess_over_headroom_usd':max(0, 2*known-headroom),
              'independent_cost_aggregation':'matched analyze.load and raw journal inventory',
              'limitations':['Planning proxy, not provider quote or hard bound',
                             'Historical unknown allowances are not actual charges or future error frequency',
                             'Smoke answers reused only inside the same fresh run; no separate double counting',
                             'Excludes remaining Run1 work, timing, AP2, other extensions, Peeters and Astra'],
              'execution_admitted':False, 'new_paid_calls':0, 'cells':cells}
    with output_path.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.proof, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'cells'}))
