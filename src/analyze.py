"""Regenerate metrics, selective prediction, cascades, figures and audit packets."""
import csv
import json
from pathlib import Path
import numpy as np
from protocol import MODELS, BENCHMARKS, ABLATION_MODELS, canonical, digest, payload

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT/'artifacts/results'

def read(path):
    return json.loads(path.read_text())

def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def csv_save(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def metric(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    return {'n': len(y), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn) if tp+fn else 0.,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'accuracy': (tp+tn)/len(y) if len(y) else None}

def load(alias, variant, b, split, namespace='quality'):
    cases = read(PROJECT/'data'/b/(split+'.json'))
    path = PROJECT/'runs'/namespace/alias/variant/b/(split+'.jsonl')
    data = [json.loads(x) for x in path.read_text().splitlines()]
    by_id = {r['pair_id']: r for r in data}
    assert len(data) == len(by_id), 'Duplicate response'
    if namespace.startswith('timing_'):
        cases = cases[:20]
    assert {r['pair_id'] for r in cases} == by_id.keys(), ('Incomplete cell', alias, variant, b, split)
    ordered = [by_id[r['pair_id']] for r in cases]
    for c,r in zip(cases, ordered):
        assert r['request_hash'] == digest(payload(c, alias, variant))
    y = np.array([r['label'] for r in cases])
    pred = np.array([r['prediction'] for r in ordered])
    prob = None if alias == 'luna' else np.array([r['probability_match'] for r in ordered], dtype=float)
    costs = np.array([sum(a['cost'] for a in r['attempts'] if a['cost'] is not None) for r in ordered])
    times = np.array([sum(a['seconds'] for a in r['attempts']) for r in ordered])
    unknown_by_pair = np.array([sum(a['cost'] is None for a in r['attempts']) for r in ordered])
    failure_path = path.with_name(split+'.failures.jsonl')
    if failure_path.exists():
        positions = {c['pair_id']: i for i,c in enumerate(cases)}
        for failure in map(json.loads, failure_path.read_text().splitlines()):
            if failure['pair_id'] in positions:
                i = positions[failure['pair_id']]
                assert failure['request_hash'] == ordered[i]['request_hash']
                costs[i] += sum(a['cost'] for a in failure['attempts'] if a['cost'] is not None)
                unknown_by_pair[i] += sum(a['cost'] is None for a in failure['attempts'])
    return {'cases': cases, 'responses': ordered, 'y': y, 'pred': pred, 'prob': prob,
            'costs': costs, 'times': times, 'unknown_cost_attempts': int(unknown_by_pair.sum()),
            'unknown_cost_by_pair': unknown_by_pair}

def calibration(y, prob):
    clipped = np.clip(prob, 1e-12, 1-1e-12)
    bins = []
    ece = 0.
    for i in range(10):
        mask = (prob >= i/10) & ((prob < (i+1)/10) if i < 9 else (prob <= 1))
        n = int(mask.sum())
        if n:
            mean_p, mean_y = float(prob[mask].mean()), float(y[mask].mean())
            ece += n/len(y)*abs(mean_p-mean_y)
        else:
            mean_p = mean_y = None
        bins.append({'lower': i/10, 'upper': (i+1)/10, 'n': n, 'mean_p': mean_p, 'positive_rate': mean_y})
    return {'brier': float(np.mean((prob-y)**2)),
            'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
            'log_loss_clip': 1e-12, 'ece_10_equal_width': ece, 'bins': bins}

def risk_curve(y, pred, prob):
    # Threshold curves include all ties together rather than arbitrary favorable ordering.
    confidence = np.maximum(prob, 1-prob)
    thresholds = sorted(set(np.quantile(confidence, np.linspace(0, 1, 51)).tolist()))
    out = []
    for t in thresholds:
        accept = confidence >= t
        accepted_pos = accept & (pred == 1)
        accepted_neg = accept & (pred == 0)
        out.append({'threshold': float(t), 'coverage': float(accept.mean()),
                    'risk': float(np.mean(y[accept] != pred[accept])),
                    'positive_predictions_accepted': int(accepted_pos.sum()),
                    'match_prediction_coverage': float(accepted_pos.sum()/max(1, (pred == 1).sum())),
                    'accepted_match_precision': float(y[accepted_pos].mean()) if accepted_pos.any() else None,
                    'nonmatch_predictions_accepted': int(accepted_neg.sum()),
                    'accepted_nonmatch_error': float(y[accepted_neg].mean()) if accepted_neg.any() else None})
    return out

def routing_mask(dev, test, fraction):
    def keys(data):
        return [(float(min(p, 1-p)), digest(['route-tie-v1', c['pair_id']])) for p,c in zip(data['prob'], data['cases'])]
    dev_keys = sorted(keys(dev), reverse=True)
    n = round(len(dev_keys)*fraction)
    cutoff = dev_keys[n-1]
    mask = np.array([k >= cutoff for k in keys(test)])
    return mask, {'uncertainty_cutoff': cutoff[0], 'tie_hash_cutoff': cutoff[1], 'dev_routed': n, 'dev_n': len(dev_keys)}

def components(cases):
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root
    for c in cases:
        a,b = [('left',c['entity_ids']['left']), ('right',c['entity_ids']['right'])]
        a,b = find(a),find(b)
        parent[a] = b
    index = {}
    ids = []
    for c in cases:
        r = find(('left',c['entity_ids']['left']))
        ids.append(index.setdefault(r,len(index)))
    return np.array(ids)

def bootstrap_delta(cases, y, pred, control, repeats=1000):
    def run(groups):
        count = int(groups.max())+1
        sizes = np.bincount(groups)
        effective = float(sizes.sum()**2/np.sum(sizes**2))
        if count < 20 or effective < 10:
            return {'groups': count, 'effective_groups': effective, 'ci95': None,
                    'reason': 'Too few effective independent components for a credible cluster interval'}
        stats = []
        for prediction in [pred,control]:
            stats.append(np.stack([np.bincount(groups, weights=v, minlength=count) for v in [
                (y==1)&(prediction==1), (y==0)&(prediction==1), (y==1)&(prediction==0)]]))
        rng = np.random.default_rng(20261013)
        values = []
        for _ in range(repeats):
            draws = rng.integers(0,count,size=count)
            scores = []
            for table in stats:
                tp,fp,fn = table[:,draws].sum(axis=1)
                scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            values.append(scores[0]-scores[1])
        return {'groups': count, 'effective_groups': effective, 'ci95': np.quantile(values,[.025,.975]).tolist()}
    return {'difference': metric(y,pred)['f1']-metric(y,control)['f1'],
            'entity_component_bootstrap': run(components(cases)),
            'pair_bootstrap_sensitivity': run(np.arange(len(cases))), 'replicates': repeats}

def generate_audit(cells):
    output = []
    diagnostic = []
    for b in BENCHMARKS:
        cases = cells[(b,'luna')]['cases']
        ordered = sorted(range(len(cases)), key=lambda i: digest(['audit-random',b,cases[i]['pair_id']]))
        random_ids = set(ordered[:10])
        candidates = []
        for i,c in enumerate(cases):
            predictions = {m: int(cells[(b,m)]['pred'][i]) for m in MODELS}
            high_wrong = sum(cells[(b,m)]['pred'][i] != c['label'] and
                             max(cells[(b,m)]['prob'][i],1-cells[(b,m)]['prob'][i]) >= .9 for m in MODELS if m != 'luna')
            candidates.append((int(high_wrong), len(set(predictions.values())), digest(['audit-diagnostic',b,c['pair_id']]), i))
        chosen = [('random', i) for i in ordered[:10]] + [('diagnostic',t[-1]) for t in sorted(candidates,reverse=True) if t[-1] not in random_ids][:10]
        for stratum,i in chosen:
            c = cases[i]
            audit_id = digest(['audit-id',b,c['pair_id']])[:12]
            output.append({'audit_id': audit_id, 'benchmark': b, 'record_left': canonical(c['records']['left']),
                           'record_right': canonical(c['records']['right']), 'human_label': '', 'category': '',
                           'ambiguous': '', 'evidence_note': ''})
            diagnostic.append({'audit_id': audit_id, 'benchmark': b, 'pair_id': c['pair_id'],
                               'stratum': stratum, 'gold_label': c['label'],
                               'model_predictions': {m: int(cells[(b,m)]['pred'][i]) for m in MODELS}})
    assert len(output) == 160
    csv_save(OUT/'audit_blind_160.csv',output)
    second = []
    for b in BENCHMARKS:
        subset = [r for r in output if r['benchmark'] == b]
        second.extend(sorted(subset,key=lambda r:digest(['second-review',r['audit_id']]))[:5])
    csv_save(OUT/'audit_second_reviewer_40.csv',second)
    save('audit_key_private.json', diagnostic)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells, main_rows, reliability, cascades, intervals, ablations, timings = {}, [], [], [], [], [], []
    for b in BENCHMARKS:
        for m in MODELS:
            data = load(m,'B0',b,'test')
            cells[(b,m)] = data
            result = {'benchmark': b, 'model': m, **metric(data['y'],data['pred']),
                      'known_cost_usd': float(data['costs'].sum()), 'unknown_cost_attempts': data['unknown_cost_attempts'],
                      'cost_per_1000': float(data['costs'].mean()*1000),
                      'quality_run_latency_p50': float(np.quantile(data['times'],.5)),
                      'quality_run_latency_p95': float(np.quantile(data['times'],.95))}
            main_rows.append(result)
            if m != 'luna':
                reliability.append({'benchmark': b, 'model': m, **calibration(data['y'],data['prob']),
                                    'risk_coverage': risk_curve(data['y'],data['pred'],data['prob'])})
        control = cells[(b,'luna')]
        for m in MODELS:
            if m == 'luna':
                continue
            test = cells[(b,m)]
            dev = load(m,'B0',b,'route')
            intervals.append({'benchmark': b, 'model': m, 'control': 'luna',
                              **bootstrap_delta(test['cases'],test['y'],test['pred'],control['pred'])})
            for q in [.10,.25,.50]:
                mask, thresholds = routing_mask(dev,test,q)
                pred = np.where(mask,control['pred'],test['pred'])
                costs = test['costs']+np.where(mask,control['costs'],0.)
                random_results = []
                for seed in range(20):
                    random_mask = np.zeros(len(mask),dtype=bool)
                    random_mask[np.random.default_rng(seed).choice(len(mask),int(mask.sum()),replace=False)] = True
                    rp = np.where(random_mask,control['pred'],test['pred'])
                    random_results.append({'f1': metric(test['y'],rp)['f1'],
                                           'cost_usd': float(np.sum(test['costs']+np.where(random_mask,control['costs'],0.)))})
                cascades.append({'benchmark': b, 'model': m, 'target_fraction': q,
                                 'actual_fraction': float(mask.mean()), 'thresholds': thresholds,
                                 **metric(test['y'],pred), 'known_cost_usd': float(costs.sum()),
                                 'unknown_cost_attempts': int(test['unknown_cost_by_pair'].sum()+control['unknown_cost_by_pair'][mask].sum()),
                                 'luna_control_unknown_cost_attempts': control['unknown_cost_attempts'],
                                 'cost_relative_to_luna': float(costs.sum()/control['costs'].sum()),
                                 'corrected_errors': int(np.sum(mask&(test['pred']!=test['y'])&(control['pred']==test['y']))),
                                 'introduced_errors': int(np.sum(mask&(test['pred']==test['y'])&(control['pred']!=test['y']))),
                                 'random_f1_mean': float(np.mean([r['f1'] for r in random_results])),
                                 'random_f1_sd': float(np.std([r['f1'] for r in random_results],ddof=1)),
                                 'random_cost_mean': float(np.mean([r['cost_usd'] for r in random_results])),
                                 'paired_f1_difference_vs_luna': bootstrap_delta(test['cases'],test['y'],pred,control['pred'])})
    selection = read(PROJECT/'configs/selected_variants.json')
    for m in ABLATION_MODELS:
        for b in BENCHMARKS:
            for split,variants in [('prompt',['P1','P2','S1','O1']), ('test',[selection[m]['variant']])]:
                base = load(m,'B0',b,split)
                for variant in variants:
                    alt = load(m,variant,b,split)
                    ablations.append({'model':m,'benchmark':b,'split':split,'variant':variant,
                                      **metric(alt['y'],alt['pred']),
                                      'delta_f1':metric(alt['y'],alt['pred'])['f1']-metric(base['y'],base['pred'])['f1'],
                                      'decision_flip_rate':float(np.mean(base['pred']!=alt['pred'])),
                                      'mean_abs_probability_change':None if m=='luna' else float(np.mean(np.abs(base['prob']-alt['prob'])))})
    for m in MODELS:
        for b in BENCHMARKS:
            for regime in ['serial','concurrent8']:
                data=load(m,'B0',b,'prompt',namespace='timing_'+regime)
                execution_path=PROJECT/'runs'/('timing_'+regime)/m/'B0'/b/'prompt.executions.jsonl'
                executions=[json.loads(line) for line in execution_path.read_text().splitlines()]
                timings.append({'model':m,'benchmark':b,'regime':regime,'n':len(data['y']),
                                'latency_p50':float(np.quantile(data['times'],.5)),
                                'latency_p95':float(np.quantile(data['times'],.95)),
                                'known_cost_usd':float(data['costs'].sum()),
                                'pairs_per_second':sum(r['new_pairs'] for r in executions)/sum(r['wall_seconds'] for r in executions)})
    csv_save(OUT/'main_metrics.csv',main_rows)
    csv_save(OUT/'ablations.csv',ablations)
    csv_save(OUT/'timing.csv',timings)
    save('reliability.json',reliability)
    save('cascades.json',cascades)
    save('paired_intervals.json',intervals)
    save('main_metrics.json',main_rows)
    save('summary.json',{'macro_f1':{m:float(np.mean([r['f1'] for r in main_rows if r['model']==m])) for m in MODELS},
                         'baseline_pairs':sum(r['n'] for r in main_rows), 'cells':len(main_rows),
                         'independent_human_audit':'pending', 'inference_budget_ledger':read(PROJECT/'artifacts/budget.json')})
    generate_audit(cells)
    print('ANALYSIS COMPLETE', len(main_rows), 'cells; human audit pending', flush=True)
    # Plotting and manuscript generation are separate reproducible steps.
    from report import build
    build(OUT)

if __name__=='__main__':
    main()
