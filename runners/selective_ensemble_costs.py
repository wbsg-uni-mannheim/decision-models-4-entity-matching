"""Read-only routing inventory and pilot-derived cost projection; no API calls."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

from strong_ensemble import MEMBERS


def union_by_fraction(rows):
    unions = defaultdict(set)
    for row in rows:
        unions[row['fraction']].update((row['benchmark'], p) for p in row['pair_ids'])
    return {str(f): {'pairs': len(pairs), 'member_calls_per_run': 3*len(pairs),
        'member_calls_three_run_projection': 9*len(pairs)} for f, pairs in sorted(unions.items())}


def inventory(root, output):
    sys.path.insert(0, str(root/'src'))
    import analyze
    import orchestrate
    import billiger50
    import expanded_cascades as cascades
    from protocol import BENCHMARKS, digest
    orchestrate.verify(); billiger50.verify(); cascades.verify()
    selected_path = root/'configs/expanded_cascade_selection.json'
    selected = json.loads(selected_path.read_text())
    entries = [e for e in selected if e['strategy']=='symmetric' and e['status']=='frozen']
    provenance = []
    for e in entries:
        dev = analyze.load(e['model'], 'B0', e['benchmark'], 'route')
        n = round(len(dev['cases'])*e['fraction'])
        expected = cascades.cutoff(dev['cases'], cascades.score(e, dev['cases'], dev['prob']), n)
        if expected != e['cutoff']:
            raise ValueError('Existing routing cutoff disagrees with frozen development')
    rows = []
    for b in [*BENCHMARKS, 'billiger-de-50un']:
        source = 'billiger-de' if b=='billiger-de-50un' else b
        for m in cascades.DECISIONS:
            d = analyze.load(m, 'B0', b, 'test')
            path = root/'runs/quality'/m/'B0'/b/'test.jsonl'
            provenance.append({'benchmark':b,'model':m,'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'requests_verified':True,'known_cost_usd':float(d['costs'].sum()),'unknown_cost_attempts':d['unknown_cost_attempts']})
            for e in entries:
                if e['benchmark']!=source or e['model']!=m:
                    continue
                mask=cascades.route(e,d['cases'],d['prob'])
                pairs=[c['pair_id'] for c,yes in zip(d['cases'],mask) if yes]
                rows.append({'benchmark':b,'model':m,'fraction':e['fraction'],'test_n':len(mask),
                    'routed_n':len(pairs),'actual_fraction':len(pairs)/len(mask),'pair_ids':pairs,
                    'development_cutoff':e['cutoff'],'mask_sha256':digest(pairs)})
    if len(rows)!=162:
        raise ValueError('Incomplete six-model/nine-split/three-quota routing inventory')
    unions = union_by_fraction(rows)
    all_pairs = {(r['benchmark'],p) for r in rows for p in r['pair_ids']}
    result={'created_at':datetime.now(timezone.utc).isoformat(),'source_run':1,'rows':rows,'unions_by_fraction':unions,
        'all_quotas_union':{'pairs':len(all_pairs),'member_calls_per_run':3*len(all_pairs),'member_calls_three_run_projection':9*len(all_pairs)},
        'sources':provenance,'selection_sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        'full_evaluation_authorized':False,'paid_calls':0,
        'three_run_projection_caveat':'Run01 masks multiplied by three for planning only; real runs02/03 require independently fitted fresh development masks. No caches copied across runs.',
        'scope':'Only symmetric uncertainty routing at existing development targets; no test-metric threshold selection, no direct full-test member/ensemble controls.',
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({'routing_cells':len(rows),'unions_by_fraction':unions,'all_quotas_union':result['all_quotas_union']}))


def pilot_costs(package):
    manifest=json.loads((package/'manifest.json').read_text())
    requests_path=package/'requests.jsonl'
    if hashlib.sha256(requests_path.read_bytes()).hexdigest()!=manifest['requests_sha256']:
        raise ValueError('Pilot request bytes changed')
    requests={r['request_id']:r for r in map(json.loads,requests_path.read_text().splitlines())}
    events=list(map(json.loads,(package/'execution.jsonl').read_text().splitlines()))
    if not events or events[-1]['event']!='complete_technical_pilot_only':
        raise ValueError('Pilot incomplete; no reliable completed-pilot cost projection')
    observations={}
    for e in events:
        if e['event']=='valid_vote':
            rid=e['request_id']
            if rid in observations or rid not in requests:
                raise ValueError('Duplicate or unexpected pilot vote')
            observations[rid]=e['cost_usd']
    if set(observations)!=set(requests) or len(observations)!=48:
        raise ValueError('Missing pilot cost rows')
    pair_costs=defaultdict(dict)
    for rid,cost in observations.items():
        r=requests[rid]
        pair_costs[r['benchmark'],r['pair_id']][r['member']]=cost
    by_benchmark=defaultdict(list)
    for (b,p),members in pair_costs.items():
        if set(members)!=set(MEMBERS):
            raise ValueError('Incomplete ensemble pilot pair')
        by_benchmark[b].append(sum(members.values()))
    return {b:{'pilot_pairs':len(costs),'mean_usd':statistics.mean(costs),'min_usd':min(costs),'max_usd':max(costs)} for b,costs in by_benchmark.items()}


def quote(inventory_path, package, output):
    inv=json.loads(inventory_path.read_text())
    costs=pilot_costs(package)
    result={'created_at':datetime.now(timezone.utc).isoformat(),'pilot_costs_by_family':costs,'cells':[], 'unions_by_fraction':{},
        'three_run_projection_only':True,'not_budget_authorization':True,
        'limitations':['Two pilot pairs per family selected by hash/length, not a representative random sample; min/max is a scenario range, not a confidence interval or cost bound.',
            'billiger50 uses billiger100 pilot costs as an unvalidated transfer estimate.',
            'Routing may differ in fresh runs02/03; this uses run01 masks multiplied by three.',
            'Exclude extra retries, missing usage, live latency experiments and separate existing-study costs. Decision B0 costs shown separately.'],
        'inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        'pilot_journal_sha256':hashlib.sha256((package/'execution.jsonl').read_bytes()).hexdigest()}
    def estimate(pairs):
        totals={k:0. for k in ['mean_usd','min_usd','max_usd']}
        for b,_ in pairs:
            source='billiger-de' if b=='billiger-de-50un' else b
            for k in totals:
                totals[k]+=3*costs[source][k]
        return totals
    for r in inv['rows']:
        result['cells'].append({k:v for k,v in r.items() if k not in ['pair_ids','development_cutoff']} |
            {'ensemble_cost_three_runs':estimate((r['benchmark'],p) for p in r['pair_ids'])})
    for fraction in sorted({r['fraction'] for r in inv['rows']}):
        pairs={(r['benchmark'],p) for r in inv['rows'] if r['fraction']==fraction for p in r['pair_ids']}
        result['unions_by_fraction'][str(fraction)]={'pairs_run01':len(pairs),'ensemble_cost_three_runs':estimate(pairs)}
    pairs={(r['benchmark'],p) for r in inv['rows'] for p in r['pair_ids']}
    result['all_quotas_union']={'pairs_run01':len(pairs),'ensemble_cost_three_runs':estimate(pairs)}
    result['decision_b0_known_cost_three_run_projection']=3*sum(r['known_cost_usd'] for r in inv['sources'])
    result['decision_b0_unknown_attempts_run01']=sum(r['unknown_cost_attempts'] for r in inv['sources'])
    with output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('inventory'); a.add_argument('--root',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('quote'); a.add_argument('--inventory',type=Path,required=True); a.add_argument('--package',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='inventory': inventory(a.root,a.output)
    else: quote(a.inventory,a.package,a.output)
