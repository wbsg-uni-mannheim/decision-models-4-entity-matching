"""Independent read-only validation of staged Run3 timing recovery results.

Does not import the executor, measurement validator, analysis adapter, frozen
analyzer, terminal auditor or credit implementation. Uses the frozen protocol
only for exact requests and response parsing. No admission, HTTP or writes.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ATTEMPT = 'artifacts/operations/core3_timing_recovery_366980_v1'
PLAN = 'configs/core3_timing_recovery_measurement_plan_20261016.json'
PLAN_SHA = '6e7835eb42b741aa953a1a1d4c4c7c5002aa54debaf6f4bfdbece28f7db1e5d3'
DIAGNOSIS = 'artifacts/operations/grouped_core3_terminal_366980_20261016.json'
DIAGNOSIS_SHA = 'b90b868f88c69d5575889e917431953f8717dd477a74cf9f4f0a447c5dc14d36'
REPLACEMENT = 'timing_remeasurement_366980_v1_concurrent8'
SUFFIXES = ('.jsonl', '.failures.jsonl', '.operations.jsonl', '.executions.jsonl')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def equal(actual, expected, path='value'):
    """Complete recursive comparison: no omitted keys or unchecked numbers."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), path+' keys')
        for k in expected:
            equal(actual[k], expected[k], path+'/'+str(k))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path+' length')
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, path+'/'+str(i))
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11), path+' number')
    else:
        require(type(actual) is type(expected) and actual == expected, path+' identity')


def csv_equal(path, expected):
    with path.open() as f:
        table = list(csv.DictReader(f))
    require(len(table) == len(expected), path.name+' row count')
    for i, (actual, wanted) in enumerate(zip(table, expected)):
        require(actual.keys() == wanted.keys(), path.name+' columns')
        for k, v in wanted.items():
            value = actual[k]
            if v is None:
                require(value == '', path.name+' null')
            elif type(v) in (int, float):
                equal(float(value), v, path.name+'/'+str(i)+'/'+k)
            else:
                require(value == v, path.name+' string')


def cell(target, protocol, ns, model, variant, benchmark, split, *, timing=False,
         complete=True, resolved=None):
    cases = read(target/'data'/benchmark/(split+'.json'))
    if timing:
        cases = cases[:20]
    require(cases and len({c['pair_id'] for c in cases}) == len(cases), 'Input IDs')
    by_id = {c['pair_id']:c for c in cases}
    directory = target/'runs'/ns/model/variant/benchmark
    success, failure, operations = [rows(directory/(split+s)) for s in SUFFIXES[:3]]
    require(len({r['pair_id'] for r in success}) == len(success), 'Duplicate success')
    require({r['pair_id'] for r in success} <= by_id.keys(), 'Unexpected success')
    if complete:
        require({r['pair_id'] for r in success} == by_id.keys(), 'Incomplete raw cell')
    for records, failed in ((success, False), (failure, True)):
        for r in records:
            c = by_id[r['pair_id']]
            require(c['benchmark'] == benchmark and digest(c['records']) == c['input_hash'] == r['input_hash'], 'Raw input')
            require(type(r.get('study_run_id')) is int and r['study_run_id'] == 3, 'Strict RunID')
            require((r['model_alias'], r['variant'], r['benchmark']) == (model, variant, benchmark), 'Raw identity')
            require(r['request_hash'] == digest(protocol.payload(c, model, variant)), 'Exact request')
            require(bool(r.get('error')) == failed and r['attempts'], 'Raw failure classification')
            for a in r['attempts']:
                require(type(a['seconds']) in (int, float) and math.isfinite(a['seconds']) and a['seconds'] >= 0, 'Attempt time')
                require(a['cost'] is None or (type(a['cost']) in (int, float) and math.isfinite(a['cost']) and a['cost'] >= 0), 'Attempt cost')
                if a['cost'] is not None and a['status'] == 200:
                    require(a['response'].get('usage', {}).get('cost') == a['cost'], 'Usage cost')
            last = r['attempts'][-1]
            if not failed:
                require(last['status'] == 200 and not last.get('error') and last['cost'] is not None, 'Success attempt')
                pred, prob = protocol.parse_response(model, last['response'])
                require((pred, prob) == (r['prediction'], r['probability_match']), 'Frozen raw parse')
            elif last['status'] == 200 and last['cost'] is not None:
                try:
                    protocol.parse_response(model, last['response'])
                except (ValueError, KeyError, TypeError):
                    pass
                else:
                    raise ValueError('Failure unexpectedly parses')
    require(len({(o['pair_id'], o['request_hash'], o['completed_at']) for o in operations}) == len(operations), 'Duplicate operation')
    for o in operations:
        c = by_id[o['pair_id']]
        require(type(o.get('study_run_id')) is int and o['study_run_id'] == 3, 'Operation RunID')
        require(o['input_hash'] == digest(c['records']) and o['request_hash'] == digest(protocol.payload(c, model, variant)), 'Operation request')
        require((o['model_alias'], o['variant'], o['benchmark']) == (model, variant, benchmark) and 'attempts' not in o, 'Operation identity')
        stop = o['operational_stop']
        linked = [r for r in failure if r.get('operational_stop') and
                  (r['pair_id'], r['request_hash'], r['completed_at']) == (o['pair_id'], o['request_hash'], o['completed_at'])]
        if stop['previous_attempts_preserved']:
            require(len(linked) == 1 and len(linked[0]['attempts']) == stop['previous_attempts_preserved']
                    and {k:v for k,v in linked[0].items() if k != 'attempts'} == o, 'Operation failure join')
        else:
            require(not linked, 'Spurious failure join')
        if stop['outcome_unknown']:
            allowed = (resolved or {}).get(digest(o), {})
            require(allowed.get('stage') == 'core3' and allowed.get('path') == str((directory/(split+'.operations.jsonl')).relative_to(target))
                    and stop['phase'] == 'reserve' and stop['exception_type'] == 'ValueError', 'Unresolved outcome')
        else:
            require(stop['budget_reservation_rejected'] and stop['phase'] == 'reserve'
                    and stop['exception_type'] == 'RuntimeError', 'Unclassified operation')
    if timing:
        require(not operations, 'Timing operation')
    attempts = [a for r in success+failure for a in r['attempts']]
    stats = {'responses':len(success), 'attempts':len(attempts), 'failure_rows':len(failure),
             'unknown_attempts':sum(a['cost'] is None for a in attempts),
             'known_usd':math.fsum(a['cost'] for a in attempts if a['cost'] is not None)}
    by_response = {r['pair_id']:r for r in success}
    ordered = [by_response[c['pair_id']] for c in cases if c['pair_id'] in by_response]
    ordered_cases = [c for c in cases if c['pair_id'] in by_response]
    costs, unknown = [], []
    for r in ordered:
        aa = r['attempts']+[a for f in failure if f['pair_id'] == r['pair_id'] for a in f['attempts']]
        costs.append(math.fsum(a['cost'] for a in aa if a['cost'] is not None))
        unknown.append(sum(a['cost'] is None for a in aa))
    return {'cases':ordered_cases, 'rows':ordered, 'stats':stats, 'directory':directory,
            'y':np.array([c['label'] for c in ordered_cases]), 'p':np.array([r['prediction'] for r in ordered]),
            'prob':None if model == 'luna' else np.array([r['probability_match'] for r in ordered]),
            'cost':np.array(costs), 'unknown':np.array(unknown),
            'time':np.array([math.fsum(a['seconds'] for a in r['attempts']) for r in ordered])}


def metrics(y, prediction):
    counts = [sum(int(a == x and b == z) for a, b in zip(y, prediction)) for x,z in ((1,1),(0,1),(1,0),(0,0))]
    tp, fp, fn, tn = counts
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                accuracy=(tp+tn)/len(y) if len(y) else None)


def interval(data, prediction, control):
    # Independent graph traversal, rather than the analyzer's union-find.
    graph = {}
    for c in data['cases']:
        left, right = ('L', c['entity_ids']['left']), ('R', c['entity_ids']['right'])
        graph.setdefault(left, set()).add(right); graph.setdefault(right, set()).add(left)
    component = {}; number = 0
    for c in data['cases']:
        node = ('L', c['entity_ids']['left'])
        if node in component:
            continue
        pending = [node]
        while pending:
            item = pending.pop()
            if item in component:
                continue
            component[item] = number
            pending.extend(graph[item]-component.keys())
        number += 1
    cluster = np.array([component['L', c['entity_ids']['left']] for c in data['cases']])
    def bootstrap(groups):
        count = int(max(groups))+1
        sizes = np.bincount(groups)
        effective = float(len(groups)**2/np.dot(sizes, sizes))
        if count < 20 or effective < 10:
            return {'groups':count, 'effective_groups':effective, 'ci95':None,
                    'reason':'Too few effective independent components for a credible cluster interval'}
        tables = []
        for pp in (prediction, control):
            tables.append(np.array([np.bincount(groups, weights=(data['y'] == a)&(pp == b), minlength=count)
                                    for a,b in ((1,1),(0,1),(1,0))]))
        rng = np.random.default_rng(20261013)
        values = []
        for _ in range(1000):
            draw = rng.integers(count, size=count)
            a, b = [table[:,draw].sum(axis=1) for table in tables]
            def f(c):
                den = 2*c[0]+c[1]+c[2]
                return 2*c[0]/den if den else 0.
            values.append(f(a)-f(b))
        return {'groups':count, 'effective_groups':effective, 'ci95':np.quantile(values,[.025,.975]).tolist()}
    return {'difference':metrics(data['y'],prediction)['f1']-metrics(data['y'],control)['f1'],
            'entity_component_bootstrap':bootstrap(cluster),
            'pair_bootstrap_sensitivity':bootstrap(np.arange(len(cluster))), 'replicates':1000}


def reliability(data):
    y, pred, prob = data['y'], data['p'], data['prob']
    bins = []; ece = 0.
    for i in range(10):
        mask = (prob >= i/10)&((prob < (i+1)/10) if i < 9 else (prob <= 1))
        n = int(sum(mask)); p = float(prob[mask].mean()) if n else None; rate = float(y[mask].mean()) if n else None
        if n:
            ece += n/len(y)*abs(p-rate)
        bins.append(dict(lower=i/10, upper=(i+1)/10, n=n, mean_p=p, positive_rate=rate))
    confidence = np.maximum(prob, 1-prob); curve = []
    for t in sorted(set(np.quantile(confidence, np.linspace(0,1,51)).tolist())):
        accepted = confidence >= t; pos = accepted&(pred == 1); neg = accepted&(pred == 0)
        curve.append(dict(threshold=t, coverage=float(accepted.mean()), risk=float((y[accepted] != pred[accepted]).mean()),
            positive_predictions_accepted=int(sum(pos)), match_prediction_coverage=float(sum(pos)/max(1,sum(pred == 1))),
            accepted_match_precision=float(y[pos].mean()) if pos.any() else None,
            nonmatch_predictions_accepted=int(sum(neg)), accepted_nonmatch_error=float(y[neg].mean()) if neg.any() else None))
    clipped = np.clip(prob,1e-12,1-1e-12)
    return dict(brier=float(np.mean((prob-y)**2)), log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
                log_loss_clip=1e-12, ece_10_equal_width=ece, bins=bins, risk_coverage=curve)


def report_values(target, protocol, quality, timings, output):
    """Rebuild every numerical JSON/CSV result from independently parsed rows."""
    main, rel, cascades, intervals, ablations = [], [], [], [], []
    for b in protocol.BENCHMARKS:
        for m in protocol.MODELS:
            d = quality[m,'B0',b,'test']
            main.append(dict(benchmark=b, model=m, **metrics(d['y'],d['p']), known_cost_usd=float(d['cost'].sum()),
                unknown_cost_attempts=int(d['unknown'].sum()), cost_per_1000=float(d['cost'].mean()*1000),
                quality_run_latency_p50=float(np.quantile(d['time'],.5)), quality_run_latency_p95=float(np.quantile(d['time'],.95))))
            if m != 'luna':
                rel.append(dict(benchmark=b, model=m, **reliability(d)))
        control = quality['luna','B0',b,'test']
        for m in protocol.MODELS:
            if m == 'luna':
                continue
            d = quality[m,'B0',b,'test']; dev = quality[m,'B0',b,'route']
            intervals.append(dict(benchmark=b, model=m, control='luna', **interval(d,d['p'],control['p'])))
            def keys(data):
                return [(min(float(p),1-float(p)), digest(['route-tie-v1',c['pair_id']])) for p,c in zip(data['prob'],data['cases'])]
            for q in (.1,.25,.5):
                n = round(len(dev['cases'])*q); cutoff = sorted(keys(dev), reverse=True)[n-1]
                mask = np.array([key >= cutoff for key in keys(d)])
                prediction = np.where(mask,control['p'],d['p']); costs = d['cost']+mask*control['cost']
                random_f1, random_cost = [], []
                for seed in range(20):
                    idx = np.random.default_rng(seed).choice(len(mask),int(sum(mask)),replace=False)
                    rp = d['p'].copy(); rp[idx] = control['p'][idx]
                    random_f1.append(metrics(d['y'],rp)['f1']); random_cost.append(float(d['cost'].sum()+control['cost'][idx].sum()))
                cascades.append(dict(benchmark=b, model=m, target_fraction=q, actual_fraction=float(mask.mean()),
                    thresholds=dict(uncertainty_cutoff=cutoff[0],tie_hash_cutoff=cutoff[1],dev_routed=n,dev_n=len(dev['cases'])),
                    **metrics(d['y'],prediction), known_cost_usd=float(costs.sum()),
                    unknown_cost_attempts=int(d['unknown'].sum()+control['unknown'][mask].sum()),
                    luna_control_unknown_cost_attempts=int(control['unknown'].sum()), cost_relative_to_luna=float(costs.sum()/control['cost'].sum()),
                    corrected_errors=int(sum(mask&(d['p'] != d['y'])&(control['p'] == d['y']))),
                    introduced_errors=int(sum(mask&(d['p'] == d['y'])&(control['p'] != d['y']))),
                    random_f1_mean=float(np.mean(random_f1)), random_f1_sd=float(np.std(random_f1,ddof=1)),
                    random_cost_mean=float(np.mean(random_cost)), paired_f1_difference_vs_luna=interval(d,prediction,control['p'])))
    selected = read(target/'configs/selected_variants.json')
    for m in protocol.ABLATION_MODELS:
        for b in protocol.BENCHMARKS:
            for split, variants in [('prompt',('P1','P2','S1','O1')),('test',(selected[m]['variant'],))]:
                base = quality[m,'B0',b,split]
                for v in variants:
                    alt = quality[m,v,b,split]
                    ablations.append(dict(model=m,benchmark=b,split=split,variant=v,**metrics(alt['y'],alt['p']),
                        delta_f1=metrics(alt['y'],alt['p'])['f1']-metrics(base['y'],base['p'])['f1'],
                        decision_flip_rate=float(np.mean(alt['p'] != base['p'])),
                        mean_abs_probability_change=None if m == 'luna' else float(np.mean(abs(alt['prob']-base['prob'])))))
    for name, value in [('main_metrics',main),('reliability',rel),('cascades',cascades),('paired_intervals',intervals)]:
        equal(read(output/(name+'.json')), value, name)
    for name, value in [('main_metrics',main),('ablations',ablations),('timing',timings)]:
        csv_equal(output/(name+'.csv'), value)
    summary = read(output/'summary.json')
    # Ledger is a historical analysis-time snapshot, not an invoice or a pure
    # incremental timing charge. Compare its captured bytes through report SHA.
    equal({k:v for k,v in summary.items() if k != 'inference_budget_ledger'},
          dict(macro_f1={m:float(np.mean([r['f1'] for r in main if r['model'] == m])) for m in protocol.MODELS},
               baseline_pairs=sum(r['n'] for r in main), cells=56, independent_human_audit='pending'), 'summary')
    equal(read(output/'data_provenance.json'), read(target/'configs/manifest.json')['benchmarks'], 'data provenance')
    table = (output/'main_table.md').read_text().splitlines()
    require(len(table) == 11 and table[1] == '| --- | '+' | '.join(['---:']*7)+' |', 'Main table structure')
    for line, b in zip(table[2:10], protocol.BENCHMARKS):
        expected_line = '| '+b+' | '+' | '.join(f"{next(r['f1'] for r in main if r['model'] == m and r['benchmark'] == b)*100:.2f}" for m in protocol.MODELS)+' |'
        require(line == expected_line, 'Main table values')
    require(table[-1] == '| Macro average | '+' | '.join(f"{summary['macro_f1'][m]*100:.2f}" for m in protocol.MODELS)+' |', 'Main table macro')
    audit_packets(protocol, quality, output)
    return {'baseline':len(main), 'ablation':len(ablations), 'reliability':len(rel), 'cascade':len(cascades),
            'paired_intervals':len(intervals), 'timing':len(timings), 'bootstrap_replicates':1000,
            'all_numerical_json_csv_fields_reconstructed':True, 'human_audit':'pending'}


def audit_packets(protocol, quality, output):
    blind, private, second = [], [], []
    for b in protocol.BENCHMARKS:
        cases = quality['luna','B0',b,'test']['cases']
        random = sorted(range(len(cases)),key=lambda i:digest(['audit-random',b,cases[i]['pair_id']]))[:10]
        candidates = []
        for i,c in enumerate(cases):
            pp = {m:int(quality[m,'B0',b,'test']['p'][i]) for m in protocol.MODELS}
            high = sum(quality[m,'B0',b,'test']['p'][i] != c['label'] and
                       max(quality[m,'B0',b,'test']['prob'][i],1-quality[m,'B0',b,'test']['prob'][i]) >= .9
                       for m in protocol.MODELS if m != 'luna')
            candidates.append((int(high),len(set(pp.values())),digest(['audit-diagnostic',b,c['pair_id']]),i))
        chosen = [('random',i) for i in random]+[('diagnostic',x[-1]) for x in sorted(candidates,reverse=True) if x[-1] not in random][:10]
        for stratum,i in chosen:
            c = cases[i]; aid = digest(['audit-id',b,c['pair_id']])[:12]
            canonical = lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
            blind.append(dict(audit_id=aid,benchmark=b,record_left=canonical(c['records']['left']),record_right=canonical(c['records']['right']),
                              human_label='',category='',ambiguous='',evidence_note=''))
            private.append(dict(audit_id=aid,benchmark=b,pair_id=c['pair_id'],stratum=stratum,gold_label=c['label'],
                                model_predictions={m:int(quality[m,'B0',b,'test']['p'][i]) for m in protocol.MODELS}))
        second.extend(sorted([r for r in blind if r['benchmark'] == b],key=lambda r:digest(['second-review',r['audit_id']]))[:5])
    csv_equal(output/'audit_blind_160.csv',blind); csv_equal(output/'audit_second_reviewer_40.csv',second)
    equal(read(output/'audit_key_private.json'),private,'audit key')


def verify_evidence(target, protocol, spec, diagnosis, attempt):
    """Full data/report proof; callers additionally bind immutable authority."""
    prior = read(attempt/'prior.json'); staged = read(attempt/'staged_result.json')
    require(staged['status'] == 'measured_analysis_staged_pending_independent_final_audit'
            and staged['core_complete'] is False and staged['reports_published'] is False
            and staged['new_timing_responses'] == 420 and staged['new_quality_responses'] == 0, 'Staging status')
    workspace = attempt/'analysis_staging'; output = workspace/'artifacts/results'
    observed = {str(p.relative_to(workspace)):sha(p) for p in workspace.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for p in workspace.rglob('*')), 'Symlink in staging')
    require(observed == staged['reports_sha256'], 'Staged output changed')
    for name in ('manifest.json','selected_variants.json'):
        require((workspace/'configs'/name).read_bytes() == (target/'configs'/name).read_bytes(), 'Staged config')
    expected = [(m,b,ns) for m in protocol.MODELS for b in protocol.BENCHMARKS for ns in ('timing_serial','timing_concurrent8')]
    require([(c['model'],c['benchmark'],c['original_namespace']) for c in spec['cells']] == expected
            and len(expected) == 112 and type(spec['run_id']) is int and spec['run_id'] == 3, 'Ordered source matrix')
    immutable = dict(diagnosis['protected_sha256_or_absent'])
    for c in spec['cells']:
        gap = (c['model'],c['benchmark'],c['original_namespace']) == ('kev','billiger-de','timing_concurrent8')
        require(c['analysis_namespace'] == (REPLACEMENT if gap else c['original_namespace']), 'Both sources must map together')
        require(c['variant'] == 'B0' and c['split'] == 'prompt' and c['limit'] == 20
                and c['workers'] == (1 if c['original_namespace'] == 'timing_serial' else 8), 'Fixed measurement definition')
        if c['action'] == 'measure_unstarted_cell':
            require(c['new_responses'] == 20 and not gap, 'Unstarted definition')
            for suffix in SUFFIXES:
                path = target/'runs'/c['analysis_namespace']/c['model']/'B0'/c['benchmark']/('prompt'+suffix)
                require(immutable.pop(str(path.relative_to(target)), None) is None, 'Previously started measurement')
        else:
            require(c['action'] == ('replace_interrupted_measurement' if gap else 'retain_complete_measurement')
                    and c['new_responses'] == (20 if gap else 0), 'Recovery action')
    immutable.pop('artifacts/stage_execution.jsonl',None)
    require(prior['protected_sha256_or_absent'] == immutable, 'Prior protection scope')
    for rel, expected_hash in immutable.items():
        p = target/rel
        require((sha(p) if p.exists() else None) == expected_hash, 'Protected evidence drift: '+rel)
    root = target.parents[1]
    stop_path = root/'artifacts/verification/grouped_stop_reconciled_20261015.json'
    require(sha(stop_path) == diagnosis['controlled_stop_proof_sha256'], 'Controlled operation proof')
    resolved = read(stop_path)['resolved_reserve_events']
    selected = read(target/'configs/selected_variants.json')
    keys = {(m,'B0',b,s) for m in protocol.MODELS for b in protocol.BENCHMARKS for s in ('prompt','route','test')}
    keys |= {(m,v,b,'prompt') for m in protocol.ABLATION_MODELS for v in ('P1','P2','S1','O1') for b in protocol.BENCHMARKS}
    keys |= {(m,selected[m]['variant'],b,'test') for m in protocol.ABLATION_MODELS for b in protocol.BENCHMARKS}
    dcells = [c for c in diagnosis['cells'] if c['namespace'] == 'quality']
    require(len(keys) == len(dcells) == 328 and {tuple(c[k] for k in ('model','variant','benchmark','split')) for c in dcells} == keys, 'Full quality scope')
    quality = {}
    for c in dcells:
        key = tuple(c[k] for k in ('model','variant','benchmark','split'))
        data = cell(target,protocol,'quality',*key,resolved=resolved)
        equal(data['stats'], {k:c[k] for k in data['stats']}, 'Quality terminal counts/costs')
        quality[key] = data
    own_selection = {}
    for m in protocol.ABLATION_MODELS:
        scores = {v:sum(metrics(quality[m,v,b,'prompt']['y'],quality[m,v,b,'prompt']['p'])['f1'] for b in protocol.BENCHMARKS)/8
                  for v in ('P1','P2','S1')}
        equal(selected[m]['development_macro_match_f1'],scores,'Own development scores')
        own_selection[m] = max(scores,key=scores.get)
        require(selected[m]['variant'] == own_selection[m], 'Own selection')
    timing_rows, measures, new, old = [], [], [], []
    for c in spec['cells']:
        d = cell(target,protocol,c['analysis_namespace'],c['model'],'B0',c['benchmark'],'prompt',timing=True)
        require(d['stats']['responses'] == 20 and d['stats']['failure_rows'] == 0, 'Complete timing without failure')
        cases = d['cases']
        require(c['ordered_pair_ids'] == [r['pair_id'] for r in cases]
                and c['ordered_input_hashes'] == [r['input_hash'] for r in cases]
                and c['ordered_request_hashes'] == [digest(protocol.payload(r,c['model'],'B0')) for r in cases], 'Ordered timing inputs/requests')
        ee = rows(d['directory']/'prompt.executions.jsonl')
        require(len(ee) == 1, 'Exactly one whole execution')
        e = ee[0]
        require(type(e['new_pairs']) is int and e['new_pairs'] == 20 and type(e['workers']) is int
                and e['workers'] == c['workers'] and type(e['wall_seconds']) in (int,float)
                and math.isfinite(e['wall_seconds']) and e['wall_seconds'] > 0, 'Full measurement walltime')
        equal(e['throughput_pairs_per_second'],20/e['wall_seconds'],'Execution throughput')
        accepted = dict(responses=20, attempts=d['stats']['attempts'], known_usd=d['stats']['known_usd'],
            unknown_attempts=d['stats']['unknown_attempts'], wall_seconds=e['wall_seconds'],
            pairs_per_second=20/e['wall_seconds'], source_namespace=c['analysis_namespace'])
        measures.append(accepted); (new if c['new_responses'] else old).append(accepted)
        timing_rows.append(dict(model=c['model'],benchmark=c['benchmark'],regime=c['original_namespace'].removeprefix('timing_'),n=20,
            latency_p50=float(np.quantile(d['time'],.5)),latency_p95=float(np.quantile(d['time'],.95)),
            known_cost_usd=float(d['cost'].sum()),pairs_per_second=20/e['wall_seconds']))
    require(len(old) == 91 and len(new) == 21, 'Old/new measurement scope')
    gap = cell(target,protocol,'timing_concurrent8','kev','B0','billiger-de','prompt',timing=True,complete=False)
    require(gap['stats']['responses'] == 9 and gap['stats']['attempts'] == 30 and gap['stats']['failure_rows'] == 7
            and gap['stats']['unknown_attempts'] == 21 and not (gap['directory']/'prompt.executions.jsonl').exists(), 'Old incomplete diagnostic')
    sample = {k:sum(v[k] for v in measures) for k in ('responses','attempts','unknown_attempts')}
    sample['known_usd'] = math.fsum(v['known_usd'] for v in measures)
    cost = dict(analysis_sample=sample,excluded_interrupted_diagnostic={k:v for k,v in gap['stats'].items() if k != 'failure_rows'})
    cost['excluded_interrupted_diagnostic'].update(execution_walltime=None,included_in_latency_or_throughput=False)
    cost.update(all_retained_timing_attempts=sample['attempts']+30,
        all_retained_known_usd=sample['known_usd']+gap['stats']['known_usd'],all_retained_unknown_attempts=sample['unknown_attempts']+21,
        cost_scope='Retained original and new timing attempts; not incremental billing or an invoice',
        measurement_policy=spec['measurement_policy'],analysis_policy=spec['analysis_policy'])
    equal(read(attempt/'timing_cost_proof.json'),cost,'All old/new costs')
    actual = dict(attempts=sum(v['attempts'] for v in new),settled=sum(v['attempts'] for v in new),
        unknown=sum(v['unknown_attempts'] for v in new),known_usd=math.fsum(v['known_usd'] for v in new),outstanding=0,account_io_failed=False)
    equal(staged['account'],actual,'Per-call observer')
    measurement_proof = read(attempt/'timing_measurements.json')
    equal(measurement_proof['new'],new,'Measurement proof'); equal(measurement_proof['account'],actual,'Measurement account')
    events = rows(attempt/'cells.jsonl'); planned_new = [(i,c) for i,c in enumerate(spec['cells']) if c['new_responses']]
    require(len(events) == 42, 'All cell starts/ends')
    allowed = set(diagnosis['protected_sha256_or_absent'])
    for j, ((i,c), accepted) in enumerate(zip(planned_new,new)):
        start, end = events[2*j:2*j+2]
        require(start['event'] == 'start' and end['event'] == 'end' and end['status'] == 'complete', 'Cell event order')
        for e in (start,end):
            require(e['cell_index'] == i and e['cell'] == c and e['prior_sha256'] == sha(attempt/'prior.json')
                    and type(e['run_id']) is int and e['run_id'] == 3, 'Cell event provenance')
        equal(end['accepted'],accepted,'Cell end')
        allowed.update(str((target/'runs'/c['analysis_namespace']/c['model']/'B0'/c['benchmark']/('prompt'+s)).relative_to(target)) for s in SUFFIXES)
    require({str(p.relative_to(target)) for p in (target/'runs').rglob('*.jsonl')} <= allowed, 'Unexpected journal scope')
    report = report_values(target,protocol,quality,timing_rows,output)
    for name in ('quality_cost','reliability','risk_coverage','match_precision_coverage','cascade_quality_cost'):
        for ext, magic in [('png',b'\x89PNG\r\n\x1a\n'),('pdf',b'%PDF-'),('svg',b'<?xml')]:
            p = output/'figures'/(name+'.'+ext)
            require(p.read_bytes().startswith(magic) and p.stat().st_size > 1000, 'Missing/malformed figure')
    # Figures are source-generated and byte-bound; this is not visual inspection.
    return dict(quality_cells=328,quality_responses=sum(d['stats']['responses'] for d in quality.values()),
        own_selection=own_selection,timing_cells=112,timing_responses=2240,new_timing_responses=420,new_quality_responses=0,
        all_timing_costs=cost,new_per_call_account=actual,report=report,staged_files_sha256=observed,
        figure_files=15,figures_visually_reviewed=False,private_legacy_draft_publishable=False,
        protected_sha256_or_absent=immutable,
        raw_sha256={str(p.relative_to(target)):sha(p) for p in (target/'runs').rglob('*.jsonl')})


def verify(root, admission_path, protocol):
    root = Path(root).resolve(); target = root/'replications/run03'; attempt = target/ATTEMPT
    require(sha(root/PLAN) == PLAN_SHA and sha(root/DIAGNOSIS) == DIAGNOSIS_SHA, 'Frozen authority')
    spec, diagnosis = read(root/PLAN), read(root/DIAGNOSIS)
    admission = read(admission_path)
    require(admission.get('admitted') is True and admission.get('stage') == 'core3_timing'
            and type(admission.get('run_id')) is int and admission['run_id'] == 3
            and admission['predecessor'] == '366980' and admission['cap_usd'] == 200
            and admission['max_invocations'] == 1, 'Separate timing admission required')
    require(admission['sources'].get('artifacts/operations/core3_final_validation_20261016.py') == sha(Path(__file__)), 'Final validator not admitted')
    for rel, h in {**diagnosis['admission_sources_verified'],**diagnosis['historical_admission_policy_sha256'],**admission['sources']}.items():
        require(sha(root/rel) == h, 'Source/history hash drift: '+rel)
    prior = read(attempt/'prior.json')
    require(prior['run_id'] == 3 and type(prior['run_id']) is int and prior['attempt'] == ATTEMPT
            and prior['plan_sha256'] == PLAN_SHA and prior['admission_sha256'] == sha(admission_path), 'Durable prior authority')
    lines = (target/'artifacts/stage_execution.jsonl').read_bytes().splitlines(keepends=True)
    old = diagnosis['stage_history']; prefix = lines[:len(old)]
    require([json.loads(x) for x in prefix] == old and hashlib.sha256(b''.join(prefix)).hexdigest() == prior['prior_stage_sha256'], 'Whole old history prefix')
    events = rows(attempt/'execution.jsonl')
    require([e['status'] for e in events] == ['started','measured_pending_audit']
            and [json.loads(x) for x in lines[len(old):]] == events, 'Explicit new timing history')
    for e in events:
        require(e['stage'] == 'core' and type(e['run_id']) is int and e['run_id'] == 3
                and e['event'] == 'timing_recovery_'+e['status'] and e['recovery'] == 'timing_remeasurement_v1'
                and e['predecessor'] == '366980' and e['slurm_job_id'] == admission['slurm_job_id']
                and e['admission_sha256'] == sha(admission_path) and e['parallel_policy_sha256'] == admission['policy_sha256']
                and e['prior_stage_journal_sha256'] == prior['prior_stage_sha256'] and e['prior_sha256'] == sha(attempt/'prior.json'), 'New stage provenance')
    require(events[-1]['result'] == read(attempt/'staged_result.json'), 'Stage result provenance')
    for rel, historical in diagnosis['all_historical_model_journals'].items():
        require(rows(target/rel) == historical, 'Historical model journal drift')
        ends = [e for e in historical if e['event'] == 'model_end']
        require(historical[0]['event'] == 'start' and historical[-1]['event'] == 'end'
                and sorted(e['model'] for e in ends) == sorted(historical[0]['models'])
                and historical[-1]['models_drained'] == len(ends), 'Historical model ends')
    budget = read(root/'artifacts/budget.json')
    require(budget['inference_credits']['fast-core3-366980'] == diagnosis['credit_reconciliation_read_only']['ledger'], 'Old quality credit drift')
    source = read(root/'artifacts/verification/core_phase_cost_scenarios_20261015.json')
    for rel,h in {**source['source_files_sha256'],**source['frozen_source_sha256']}.items():
        require(sha(root/rel) == h, 'Historical cost source drift')
    proof = verify_evidence(target,protocol,spec,diagnosis,attempt)
    require(proof['quality_responses'] == 380475, 'Complete real quality count')
    proof.update(admission_sha256=sha(admission_path),prior_sha256=sha(attempt/'prior.json'),plan_sha256=PLAN_SHA,
                 diagnosis_sha256=DIAGNOSIS_SHA,validator_sha256=sha(Path(__file__)),
                 staged_result_sha256=sha(attempt/'staged_result.json'),run_id=3,
                 status='independent_raw_execution_cost_numeric_report_verified',
                 current_scheduler_gate_required=True,exclusive_publication_required=True,core_complete=False)
    return proof
