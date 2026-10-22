"""Frozen, cache-only expanded contrasts; no selection or new inference calls."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import analyze
import similarity_baseline as similarity
import expanded_calibration as calibration
import expanded_cascades as cascades
import expanded_confirmation as confirmation
import expanded_protocol
import expanded_rules as rules
import expanded_schema as schema
import expanded_stability as stability
import expanded_variants as variants
from prepare import write_frozen
from protocol import BENCHMARKS, MODELS, canonical, digest

PROJECT = Path(__file__).resolve().parents[1]
REPLICATES = 9999
SEED = 20261013
STAGES = ['baseline', 'similarity', 'screen', 'confirmation', 'stability', 'calibration', 'cascades', 'variants', 'schema', 'rules']


def read(path):
    return json.loads(path.read_text())


def holm(values):
    """Unavailable tests stay in the planned family, conservatively as p=1."""
    p = np.array([1. if v is None else v for v in values], dtype=float)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    result = np.empty(len(p));result[order] = adjusted
    return [None if v is None else float(result[i]) for i, v in enumerate(values)]


def f1_parts(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    assert y.shape == pred.shape and np.isin(y, [0, 1]).all() and np.isin(pred, [0, 1]).all()
    tp = (y == 1) & (pred == 1)
    return np.stack([2*tp.astype(float), 2*tp+((y == 0) & (pred == 1))+((y == 1) & (pred == 0))])


def mean_parts(values, accepted=None):
    values = np.asarray(values, dtype=float)
    accepted = np.ones(len(values), dtype=bool) if accepted is None else np.asarray(accepted, dtype=bool)
    assert values.shape == accepted.shape and np.isfinite(values).all()
    return np.stack([np.where(accepted, values, 0.), accepted.astype(float)])


def estimate(tables, weights, zero_denominator):
    total = tables.sum(axis=-1)
    ratio = np.divide(total[:, 0], total[:, 1], out=np.full(len(total), zero_denominator), where=total[:, 1] != 0)
    return float(np.dot(weights, ratio))


def resample(tables, weights, groups, zero_denominator, repeats, seed, test_null):
    _, groups = np.unique(groups, return_inverse=True)
    count = int(groups.max())+1
    sizes = np.bincount(groups)
    effective = float(sizes.sum()**2/np.sum(sizes**2))
    base = {'groups': count, 'effective_groups': effective}
    if count < 20 or effective < 10:
        return {**base, 'ci95': None, 'p_approx': None, 'reason': 'Fewer than20components or10effective components; interval not reported'}
    grouped = np.array([[np.bincount(groups, weights=a, minlength=count) for a in t] for t in tables])
    observed = estimate(tables, weights, zero_denominator)
    if not np.isfinite(observed):
        return {**base, 'ci95': None, 'p_approx': None, 'reason': 'Undefined denominator in observed statistic'}
    rng = np.random.default_rng(seed)
    samples = []
    # Bound memory independently of number of pairs and contrast terms.
    for _ in range(repeats):
        draws = rng.integers(0, count, size=count)
        samples.append(estimate(grouped[:, :, draws], weights, zero_denominator))
    samples = np.asarray(samples);valid = np.isfinite(samples)
    if valid.mean() < .99:
        return {**base, 'ci95': None, 'p_approx': None, 'valid_replicates': int(valid.sum()), 'reason': 'More than1percent resamples have undefined denominator'}
    samples = samples[valid]
    # Centered bootstrap null approximation, NOT an exact randomization test.
    p = (1+int((np.abs(samples-observed) >= abs(observed)-1e-12).sum()))/(len(samples)+1) if test_null else None
    return {**base, 'ci95': np.quantile(samples, [.025, .975]).tolist(), 'p_approx': p,
            'valid_replicates': len(samples), 'p_resolution': 1/(len(samples)+1) if test_null else None}


def contrast(cases, tables, weights, zero_denominator=0., repeats=REPLICATES, seed=SEED, test_null=True):
    tables = np.asarray(tables, dtype=float);weights = np.asarray(weights, dtype=float)
    assert tables.shape == (len(weights), 2, len(cases)) and len(cases) > 0
    assert np.isfinite(tables).all() and np.all(tables[:, 1] >= 0)
    value = estimate(tables, weights, zero_denominator)
    return {'estimate': value if np.isfinite(value) else None,
            'entity_component': resample(tables, weights, analyze.components(cases), zero_denominator, repeats, seed, test_null),
            'pair_sensitivity': resample(tables, weights, np.arange(len(cases)), zero_denominator, repeats, seed, test_null),
            'replicates': repeats, 'test': 'Two-sided null-centered bootstrap approximation' if test_null else 'Interval only'}


def align(*data):
    reference = data[0]
    ids = [c['pair_id'] for c in reference['cases']]
    assert len(ids) == len(set(ids))
    result = []
    for d in data:
        mapping = {c['pair_id']: i for i, c in enumerate(d['cases'])}
        assert len(mapping) == len(d['cases']) and set(mapping) == set(ids), 'Cannot pair different full splits'
        order = [mapping[k] for k in ids]
        assert np.array_equal(reference['y'], d['y'][order]), 'Labels differ'
        for i, j in enumerate(order):
            assert reference['cases'][i]['entity_ids'] == d['cases'][j]['entity_ids'], 'Entity alignment differs'
        result.append({**d, 'cases': [d['cases'][i] for i in order], 'y': d['y'][order], 'pred': d['pred'][order]})
    return result


@lru_cache(maxsize=64)
def load(m, v, b, split='test', namespace='quality'):
    previous = analyze.payload
    try:
        analyze.payload = rules.payload if v == 'R_rules' else stability.payload
        return analyze.load(m, v, b, split, namespace=namespace)
    finally:
        analyze.payload = previous


def f1_row(stage, model, b, condition, control, *data, weights=None, extra=None):
    aligned = align(*data)
    return ({'stage': stage, 'model': model, 'benchmark': b, 'condition': condition, 'control': control,
             'metric': 'match_f1', **(extra or {})}, aligned[0]['cases'],
            [f1_parts(d['y'], d['pred']) for d in aligned], weights or [1, -1], 0., True)


def comparisons(stage):
    """Declare contrasts using frozen manifests/selections, never observed test scores."""
    targets = [*BENCHMARKS, 'billiger-de-50un']
    if stage == 'baseline':
        for b in targets:
            for m in MODELS:
                if m != 'luna':
                    yield f1_row(stage, m, b, 'B0', 'luna_B0', load(m, 'B0', b), load('luna', 'B0', b))
    elif stage == 'similarity':
        thresholds=read(PROJECT/'configs/similarity_selection.json')['thresholds']
        for b in targets:
            d=load('luna','B0',b);p=similarity.scores(d['cases']);source='billiger-de' if b=='billiger-de-50un' else b
            for name,t in [('fixed_0.5',.5),('development_F1',thresholds[source])]:
                baseline={**d,'pred':(p>=t).astype(int)}
                yield f1_row(stage,'jaccard',b,name,'luna_B0',baseline,d)
    elif stage == 'screen':
        for b in stability.SCREEN_BENCHMARKS:
            for m in MODELS:
                baseline=load(m,'B0',b,'expanded_screen')
                for v in expanded_protocol.VARIANTS:
                    if v!='B0':
                        yield f1_row(stage,m,b,v,'B0',load(m,v,b,'expanded_screen'),baseline,
                                     extra={'split':'development_screen','contrast_family':'prompt' if v.startswith('X_P_') else 'representation'})
    elif stage == 'confirmation':
        selected = read(PROJECT/'configs/expanded_selected_variants.json')
        for m, v, b in confirmation.confirmation_cells(selected):
            if v != 'B0':
                regimes = [e['family']+'/'+e['regime'] for e in selected if (e['model'], e['variant'], e['benchmark']) == (m,v,b)]
                yield f1_row(stage, m, b, v, 'B0', load(m,v,b), load(m,'B0',b), extra={'regimes': regimes})
    elif stage == 'stability':
        factorial = ['I3_domain_json', 'X_P_domain_0', 'I3_identity_json', 'X_P_identity_0']
        for b in stability.SCREEN_BENCHMARKS:
            reference = [load('luna',v,b,'expanded_screen') for v in factorial]
            for m in MODELS:
                data = [load(m,v,b,'expanded_screen') for v in factorial]
                yield f1_row(stage,m,b,'prompt_x_format','zero_interaction',*data,weights=[1,-1,-1,1],extra={'split':'development_screen'})
                if m != 'luna':
                    yield f1_row(stage,m,b,'prompt_x_format_vs_luna','luna_interaction',*data,*reference,
                                 weights=[1,-1,-1,1,-1,1,1,-1],extra={'split':'development_screen'})
                base = load(m,'B0',b,'expanded_screen')
                first, second = [load(m,'B0',b,'expanded_repeat',n) for n in ['e3_repeat_1','e3_repeat_2']]
                wanted = {c['pair_id'] for c in first['cases']};ii=[i for i,c in enumerate(base['cases']) if c['pair_id'] in wanted]
                subset = {**base, 'cases':[base['cases'][i] for i in ii], 'y':base['y'][ii], 'pred':base['pred'][ii]}
                a,c,d=align(subset,first,second)
                for name, x, y in [('B0_vs_repeat1',a,c),('B0_vs_repeat2',a,d),('repeat1_vs_repeat2',c,d)]:
                    meta={'stage':stage,'model':m,'benchmark':b,'condition':name,'control':'agreement', 'metric':'prediction_flip_rate','split':'development_repeat'}
                    yield meta,a['cases'],[mean_parts(x['pred']!=y['pred'])],[1],np.nan,False
    elif stage == 'calibration':
        for e in read(PROJECT/'configs/calibration_selection.json'):
            if e['status'] != 'frozen':
                for method in (calibration.METHODS if e['method']=='all' else [e['method']]):
                    for metric,rule in [('match_f1','fixed_0.5'),('match_f1','development_F1'),('brier_difference','')]:
                        name=f"{e['regime']}/{method}/{e['labels_per_source']}"+('/'+rule if rule else '')
                        meta={'stage':stage,'model':e['model'],'benchmark':e['benchmark'],'condition':name,'control':'B0','metric':metric,'unavailable_reason':e['reason']}
                        yield meta,None,None,None,0.,True
                continue
            m,b=e['model'],e['benchmark'];d=load(m,'B0',b);p=calibration.transform(e['parameters'],d['prob'])
            name=f"{e['regime']}/{e['method']}/{e['labels_per_source']}"
            for rule,t in [('fixed_0.5',.5),('development_F1',e['threshold'])]:
                pred={**d,'pred':(p>=t).astype(int)}
                yield f1_row(stage,m,b,name+'/'+rule,'B0',pred,d)
            meta={'stage':stage,'model':m,'benchmark':b,'condition':name,'control':'B0','metric':'brier_difference'}
            yield meta,d['cases'],[mean_parts((p-d['y'])**2),mean_parts((d['prob']-d['y'])**2)],[1,-1],np.nan,True
    elif stage == 'cascades':
        for e in read(PROJECT/'configs/expanded_cascade_selection.json'):
            if e['status'] != 'frozen':
                for b in [e['benchmark']]+(['billiger-de-50un'] if e['benchmark']=='billiger-de' else []):
                    for fraction in cascades.FRACTIONS:
                        for control in ['B0','luna_B0']:
                            meta={'stage':stage,'model':e['model'],'benchmark':b,'condition':f"{e['strategy']}/{fraction}",'control':control,'metric':'match_f1','unavailable_reason':e['reason']}
                            yield meta,None,None,None,0.,True
                continue
            for b in [e['benchmark']]+(['billiger-de-50un'] if e['benchmark']=='billiger-de' else []):
                m=e['model'];d=load(m,'B0',b);luna=load('luna','B0',b)
                peer=load(e['peer'],'B0',b)['prob'] if e['peer'] else None
                mask=cascades.route(e,d['cases'],d['prob'],peer)
                cascade={**d,'pred':np.where(mask,luna['pred'],d['pred'])}
                for control,base in [('B0',d),('luna_B0',luna)]:
                    yield f1_row(stage,m,b,f"{e['strategy']}/{e['fraction']}",control,cascade,base)
    elif stage == 'variants':
        for b in variants.LANGUAGES:
            for m in MODELS:
                yield f1_row(stage,m,b,'B0','billiger_de100_B0',load(m,'B0',b),load(m,'B0','billiger-de'))
        # WDC full splits have different pairs: deliberately no paired inference.
    elif stage == 'schema':
        for m,v,b in read(PROJECT/'configs/expanded_schema_test_cells.json'):
            d=schema.load(m,v,b,'test');base=load(m,'B0',b)
            if np.any(d['pred']==-1) or v == 'C_review':
                a,base=align(d,base);accepted=a['pred']!=-1
                meta={'stage':stage,'model':m,'benchmark':b,'condition':v,'control':'B0_on_same_accepted_pairs','metric':'accepted_error_difference'}
                yield meta,a['cases'],[mean_parts(a['pred']!=a['y'],accepted),mean_parts(base['pred']!=base['y'],accepted)],[1,-1],np.nan,True
                yield {**meta,'metric':'coverage','control':'all_pairs'},a['cases'],[mean_parts(accepted)],[1],np.nan,False
            else:
                yield f1_row(stage,m,b,v,'B0',d,base)
    elif stage == 'rules':
        rules.verify()
        for b in targets:
            for m in MODELS:
                d=load(m,'R_rules',b,namespace='rules')
                for v in ['B0',rules.MANUAL]:
                    yield f1_row(stage,m,b,'R_rules',v,d,load(m,v,b))
    else:
        raise ValueError(stage)


def prepare():
    inputs = [p for p in (PROJECT/'configs').glob('*manifest.json') if p.name!='expanded_statistics_manifest.json']
    spec={'registration':'Expanded inference plan frozen after core B0 inspection, before expanded test outcomes; exploratory post-registration, no test-driven contrast selection',
          'stages':STAGES,'models':MODELS,'replicates':REPLICATES,'seed':SEED,
          'primary_unit':'Observed bipartite entity-connected components; require >=20components and >=10effective components',
          'sensitivity':'Pair bootstrap ignores shared-entity dependence; never substitutes for unavailable component inference',
          'intervals':'Pointwise percentile95%; no simultaneous coverage claim',
          'p_values':'Approximate two-sided null-centered bootstrap p, add-one correction; not exact, not equivalence or causal evidence; Holm inherits approximation error, no finite-sample familywise guarantee',
          'multiplicity':'Holm per declared stage/metric/core-or-supplement family; screen separates prompt/representation; calibration separates regime and threshold rule; cascades separate route quota and control. Unavailable fits and component p retained as1. No claims across all families jointly',
          'selection_uncertainty':'Conditional on frozen development choices and one rule-induction run, not reselected or refit in test bootstrap',
          'benchmark_heterogeneity':'Per-benchmark effects and descriptive range; no pooled independent-benchmark or random-effects claim because families overlap and eight benchmarks are purposively selected',
          'unavailable_fits':'Calibration/cascade unavailable entries explicitly attached, never interpreted as null findings',
          'abstention':'Error difference on identical accepted pairs with full-pair resampling; undefined zero-denominator replicates handled explicitly',
          'excluded_inference':['WDC disjoint full-split paired tests','historical teacher aggregate significance','mixed-effects fits with unverified independence','causal architecture effects'],
          'source_sha256':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROJECT/'src').glob('*.py'))},
          'manifest_sha256':{str(p.relative_to(PROJECT)):digest(read(p)) for p in inputs},
          'new_api_calls':0}
    write_frozen(PROJECT/'configs/expanded_statistics_manifest.json',spec)


def verify():
    spec=read(PROJECT/'configs/expanded_statistics_manifest.json')
    for name,value in spec['source_sha256'].items():assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==value,name
    for name,value in spec['manifest_sha256'].items():assert digest(read(PROJECT/name))==value,name
    return spec


def report(stage, spec):
    out=PROJECT/'artifacts/results/expanded_statistics';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    checkpoint=out/(stage+'_checkpoint');checkpoint.mkdir(exist_ok=True)
    for meta,cases,tables,weights,zero,test in comparisons(stage):
        key=digest(meta);path=checkpoint/(key+'.json')
        fingerprint=digest({'cases':cases,'tables':None if tables is None else np.asarray(tables).tolist(),'weights':weights})
        if path.exists():
            row=read(path);assert row['input_sha256']==fingerprint and row['spec_sha256']==digest(spec)
        else:
            if cases is None:
                missing={'ci95':None,'p_approx':None,'reason':meta['unavailable_reason']}
                result={'estimate':None,'entity_component':missing,'pair_sensitivity':missing,'test':'Unavailable planned contrast'}
            else:
                result=contrast(cases,tables,weights,zero,test_null=test)
            row={**meta,'input_sha256':fingerprint,'spec_sha256':digest(spec),**result}
            write_frozen(path,row)
        rows.append(row)
        if len(rows)%50==0:print('STATISTICS',stage,len(rows),flush=True)
    # Families fixed by method/split role, not observed effect or p.
    groups={}
    for i,r in enumerate(rows):
        group=stage+'/'+r['metric']+('/core' if r['benchmark'] in BENCHMARKS else '/supplement')
        if stage=='screen':group+='/'+r['contrast_family']
        if stage=='calibration':group+='/'+r['condition'].split('/')[0]+('/'+r['condition'].split('/')[-1] if r['metric']=='match_f1' else '')
        if stage=='cascades':group+='/'+r['condition'].split('/')[-1]+'/'+r['control']
        groups.setdefault(group,[]).append(i)
    for group, indices in groups.items():
        tested=[i for i in indices if rows[i]['test']!='Interval only']
        adjusted=holm([rows[i]['entity_component']['p_approx'] for i in tested])
        for i,p in zip(tested,adjusted):
            rows[i].update(holm_family=group,holm_family_size=len(tested),p_holm_component=p)
    (out/(stage+'.json')).write_text(canonical(rows)+'\n')
    # Report empirical model/benchmark heterogeneity, without pretending the selected families are independent random draws.
    heterogeneity=[];by_condition={}
    for r in rows:
        if r['benchmark'] in BENCHMARKS and r['estimate'] is not None:
            key=(r['model'],r['condition'],r['control'],r['metric'])
            by_condition.setdefault(key,[]).append(r)
    for (m,condition,control,metric), group in by_condition.items():
        values=[r['estimate'] for r in group]
        heterogeneity.append({'model':m,'condition':condition,'control':control,'metric':metric,
                              'benchmarks':[r['benchmark'] for r in group],'effects':values,
                              'descriptive_mean':float(np.mean(values)),'min':min(values),'max':max(values),
                              'range':max(values)-min(values),'pooled_inference':'Not claimed; purposive/partly overlapping benchmark families'})
    (out/(stage+'_heterogeneity.json')).write_text(canonical(heterogeneity)+'\n')
    unavailable=[]
    for name in ({'calibration':['configs/calibration_selection.json'],'cascades':['configs/expanded_cascade_selection.json']}.get(stage,[])):
        unavailable.extend(e for e in read(PROJECT/name) if e['status']!='frozen')
    summary={'stage':stage,'contrasts':len(rows),'component_intervals_available':sum(r['entity_component']['ci95'] is not None for r in rows),
             'explicitly_unavailable_fits':unavailable,'post_registered':True,'human_audit_complete':False}
    write_frozen(out/(stage+'_complete.json'),summary)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','analyze']);ap.add_argument('--stage',choices=STAGES);args=ap.parse_args()
    if args.mode=='prepare':prepare();return
    spec=verify();summaries=[report(s,spec) for s in ([args.stage] if args.stage else STAGES)]
    if not args.stage:
        write_frozen(PROJECT/'artifacts/expanded_statistics_complete.json',{'status':'complete','stages':summaries,'new_api_calls':0,'full_study_complete':False})


if __name__=='__main__':main()
