"""Read-only verified adapters for inference-cell metrics across study runs.

Each invocation uses one run workspace in a fresh process. Never evaluates partial
cells, creates selections, changes a cache or treats one run as three replicates.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

from repeated_runs import replica_path, verify
from stability_run_diagnostics import export_diagnostics, export_probability_diagnostics
from language_run_diagnostics import LANGUAGES, export_language_diagnostics
from calibration_run_export import export_calibration
from schema_run_export import export_schema
from live_cascade_export import export_live
from offline_cascade_export import export_cascades
from peeters_run_export import export_peeters
from operational_journal import inspect_operations


def read(path):
    return json.loads(path.read_text())


def check_response_origin(responses, run_id, prepared_utc=None):
    for row in responses:
        if run_id == 1:
            if row.get('study_run_id', 1) != 1:
                raise ValueError('Wrong run identity in historical workspace')
        else:
            if type(row.get('study_run_id')) is not int or row['study_run_id'] != run_id:
                raise ValueError('Fresh run response ID missing or inconsistent')
            if datetime.fromisoformat(row['completed_at']) < datetime.fromisoformat(prepared_utc):
                raise ValueError('Response predates fresh workspace preparation')


def normalized_rows(run_id, model, condition, benchmark, metrics, known_cost, unknown, seconds, variant):
    n = metrics['n']
    if n <= 0:
        raise ValueError('Empty cells are unavailable, never scored zero')
    values = {k: metrics[k] for k in ['f1', 'precision', 'recall']}
    values.update(known_cost_usd_per_1000=1000*known_cost/n,
                  unknown_attempts_per_1000=1000*unknown/n,
                  request_attempt_seconds_per_pair=seconds/n)
    return [{'run_id': run_id, 'model': model, 'condition': condition, 'benchmark': benchmark,
             'metric': metric, 'value': value, 'selected_variant': variant, 'n': n} for metric, value in values.items()]


def confirmation_policy(family, regime):
    return 'confirmation/'+family+'/'+regime


def export_fixed_extension(workspace, stage, module, models, analyze, emit):
    """Require completion and frozen inputs before reading extension caches."""
    complete = read(workspace/'artifacts'/('expanded_'+stage+'_complete.json'))
    if complete.get('status') != 'complete':
        raise ValueError(stage+' incomplete')
    manifest = module.verify()
    previous_payload = analyze.payload
    try:
        # Interaction requests combine a prompt with JSON serialization. Using
        # the ordinary expanded payload would verify the wrong request bytes.
        analyze.payload = module.payload
        if stage == 'variants':
            for benchmark in manifest['conditions']:
                for model in models:
                    emit(model, 'B0', benchmark, 'test', 'variants/B0')
        elif stage == 'stability':
            for model in models:
                for benchmark in module.SCREEN_BENCHMARKS:
                    for variant in [*module.INTERACTIONS, *module.INTERACTIONS.values()]:
                        emit(model, variant, benchmark, 'expanded_screen', 'stability/factorial/'+variant)
                    for repeat in [1, 2]:
                        # These are within-run diagnostics, never study run IDs.
                        emit(model, 'B0', benchmark, 'expanded_repeat',
                             'stability/diagnostic_repeat_'+str(repeat), 'e3_repeat_'+str(repeat))
        else:
            raise ValueError('Unsupported fixed extension')
    finally:
        analyze.payload = previous_payload


def export_rules(workspace, module, models, benchmarks, analyze, cell, emit):
    """Check all fixed rule/control development cells before exporting test metrics."""
    complete = read(workspace/'artifacts/expanded_rules_complete.json')
    if complete.get('status') != 'complete':
        raise ValueError('Rules incomplete')
    manifest = module.verify()
    conditions = {'R_rules': 'rules', 'X_P_domain_0': 'quality'}
    if module.CONDITIONS != conditions or complete.get('evaluated_conditions') != list(conditions):
        raise ValueError('Rule conditions changed')
    if complete.get('controls') != ['B0', 'X_P_domain_0']:
        raise ValueError('Rule controls incomplete')
    expected = [{'model': m, 'variant': v, 'benchmark': b, 'split': split, 'namespace': ns}
                for m in models for v, ns in conditions.items()
                for b in [*benchmarks, 'billiger-de-50un']
                for split in (['test'] if b == 'billiger-de-50un' else ['prompt', 'test'])]
    if manifest['cells'] != expected:
        raise ValueError('Rule development/test matrix differs from fixed protocol')
    if (complete.get('test_cells_per_condition') != len(models)*(len(benchmarks)+1)
            or complete.get('development_cells_per_condition') != len(models)*len(benchmarks)):
        raise ValueError('Rule completion counts disagree with full matrix')
    previous_payload = analyze.payload
    try:
        # verify() loads the frozen learned rules before request reconstruction.
        analyze.payload = module.payload
        for entry in expected:
            cell(entry['model'], entry['variant'], entry['benchmark'], entry['split'], entry['namespace'])
        for m in models:
            for b in [*benchmarks, 'billiger-de-50un']:
                for v in ['B0', *conditions]:
                    emit(m, v, b, 'test', 'rules/'+v, conditions.get(v, 'quality'))
    finally:
        analyze.payload = previous_payload


def export_paired_languages(workspace, module, models, analyze, cell, run_id):
    marker = read(workspace/'artifacts/expanded_variants_complete.json')
    if marker.get('status') != 'complete' or marker.get('language_comparisons') != 3*len(models):
        raise ValueError('Complete language comparison matrix required')
    if set(module.LANGUAGES) != LANGUAGES:
        raise ValueError('Language condition matrix changed')
    previous_payload = analyze.payload
    rows = []
    try:
        analyze.payload = module.payload
        for model in models:
            reference = cell(model, 'B0', 'billiger-de', 'test')[0]
            for benchmark in sorted(LANGUAGES):
                variant = cell(model, 'B0', benchmark, 'test')[0]
                rows.extend(export_language_diagnostics(run_id, model, benchmark, reference, variant))
    finally:
        analyze.payload = previous_payload
    return rows


def export(root, run_id, stages, output):
    root = root.resolve()
    workspace = root if run_id == 1 else replica_path(root, run_id)
    spec = None if run_id == 1 else verify(root, run_id)
    if any(m in sys.modules for m in ['analyze', 'protocol', 'expanded_confirmation']):
        raise RuntimeError('Use a fresh process for each run workspace')
    sys.path.insert(0, str(workspace/'src'))
    import analyze
    import expanded_protocol
    from protocol import MODELS, BENCHMARKS, ABLATION_MODELS, digest
    assert analyze.PROJECT == workspace
    # Request hashes cover sent records, but labels also need frozen-data checks.
    import orchestrate
    orchestrate.verify()
    if any(stage in stages for stage in ['baseline', 'confirmation', 'rules', 'schema', 'peeters']):
        import billiger50
        billiger50.verify()
    analyze.payload = expanded_protocol.payload
    rows, provenance = [], []
    peeters_summary = None
    live_summaries = []
    loaded = {}
    def cell(model, variant, benchmark, split, namespace='quality'):
        key = (model, variant, benchmark, split, namespace)
        if key not in loaded:
            d = analyze.load(model, variant, benchmark, split, namespace)
            check_response_origin(d['responses'], run_id, None if spec is None else spec['created_at'])
            seconds = float(d['times'].sum())
            path = workspace/'runs'/namespace/model/variant/benchmark/(split+'.jsonl')
            fail = path.with_name(split+'.failures.jsonl')
            failures = []
            if fail.exists():
                for item in map(json.loads, fail.read_text().splitlines()):
                    failures.append(item)
                    seconds += sum(a['seconds'] for a in item['attempts'])
            import protocol
            operations = inspect_operations(path.with_name(split+'.operations.jsonl'), d['cases'],
                                            failures, protocol, model, variant, run_id)
            loaded[key] = (d, seconds)
            provenance.append({'model': model, 'variant': variant, 'benchmark': benchmark, 'split': split,
                'namespace': namespace, 'cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'failure_sha256': hashlib.sha256(fail.read_bytes()).hexdigest() if fail.exists() else None,
                'operational_journal': operations,
                'cost_completeness': 'journaled attempts; unrecorded historical outcomes cannot be ruled out',
                'request_hashes_verified': True, 'response_count': len(d['responses'])})
        return loaded[key]
    def emit(model, variant, benchmark, split, condition, namespace='quality'):
        d, seconds = cell(model, variant, benchmark, split, namespace)
        rows.extend(normalized_rows(run_id, model, condition, benchmark, analyze.metric(d['y'], d['pred']),
                    float(d['costs'].sum()), d['unknown_cost_attempts'], seconds, variant))
    for stage in stages:
        if stage == 'baseline':
            for b in [*BENCHMARKS, 'billiger-de-50un']:
                for m in MODELS:
                    emit(m, 'B0', b, 'test', 'baseline/B0')
        elif stage == 'ap2':
            # Recompute the frozen tie rule without invoking the selection writer.
            selected = read(workspace/'configs/selected_variants.json')
            for m in ABLATION_MODELS:
                scores = {}
                for v in ['P1', 'P2', 'S1']:
                    fs = []
                    for b in BENCHMARKS:
                        d, _ = cell(m, v, b, 'prompt')
                        fs.append(analyze.metric(d['y'], d['pred'])['f1'])
                    scores[v] = sum(fs)/len(fs)
                winner = max(['P1', 'P2', 'S1'], key=lambda v: scores[v])
                if selected[m]['variant'] != winner:
                    raise ValueError('Core AP2 frozen selection disagrees with development')
                for b in BENCHMARKS:
                    emit(m, winner, b, 'test', 'core_ap2/development_selected')
        elif stage == 'screen':
            import expanded_screen
            expanded_screen.verify()
            for m in MODELS:
                for v in expanded_screen.VARIANTS:
                    for b in expanded_screen.SCREEN_BENCHMARKS:
                        emit(m, v, b, 'expanded_screen', 'screen/'+v)
        elif stage == 'confirmation':
            import expanded_confirmation as ec
            ec.verify()
            selected = read(workspace/'configs/expanded_selected_variants.json')
            complete = read(workspace/'artifacts/expanded_confirmation_complete.json')
            if complete['status'] != 'complete' or complete['selection_sha256'] != digest(selected):
                raise ValueError('Confirmation incomplete or changed selection')
            scores = {}
            for m in MODELS:
                for b in BENCHMARKS:
                    for v in ec.PROMPT_VARIANTS+ec.SERIALIZATIONS:
                        d, _ = cell(m, v, b, 'prompt')
                        scores[m, b, v] = analyze.metric(d['y'], d['pred'])['f1']
            if ec.make_selection(scores) != selected:
                raise ValueError('Frozen confirmation selection differs from full development matrix')
            for entry in selected:
                emit(entry['model'], entry['variant'], entry['benchmark'], 'test',
                     confirmation_policy(entry['family'], entry['regime']))
        elif stage in ['variants', 'stability']:
            import importlib
            module = importlib.import_module('expanded_'+stage)
            export_fixed_extension(workspace, stage, module, MODELS, analyze, emit)
            if stage == 'variants':
                rows.extend(export_paired_languages(workspace, module, MODELS, analyze, cell, run_id))
            if stage == 'stability':
                previous_payload = analyze.payload
                try:
                    analyze.payload = module.payload
                    for m in MODELS:
                        for b in module.SCREEN_BENCHMARKS:
                            factorial = {v: cell(m, v, b, 'expanded_screen')[0]
                                         for v in [*module.INTERACTIONS, *module.INTERACTIONS.values()]}
                            baseline = cell(m, 'B0', b, 'expanded_screen')[0]
                            repeats = [cell(m, 'B0', b, 'expanded_repeat', ns)[0]
                                       for ns in ['e3_repeat_1', 'e3_repeat_2']]
                            rows.extend(export_diagnostics(run_id, m, b, factorial, baseline, *repeats))
                            rows.extend(export_probability_diagnostics(run_id, m, b, baseline, *repeats))
                finally:
                    analyze.payload = previous_payload
        elif stage == 'peeters':
            import protocol
            origin = lambda responses: check_response_origin(responses, run_id, None if spec is None else spec['created_at'])
            peeters_rows, peeters_summary = export_peeters(root, workspace, protocol, analyze.metric,
                normalized_rows, origin, provenance, run_id)
            rows.extend(peeters_rows)
        elif stage == 'cascades':
            import expanded_cascades
            import protocol
            origin = lambda responses: check_response_origin(responses, run_id, None if spec is None else spec['created_at'])
            rows.extend(export_cascades(workspace, expanded_cascades, analyze, protocol, cell, origin, run_id))
        elif stage == 'live':
            import expanded_cascades
            import protocol
            origin = lambda responses: check_response_origin(responses, run_id, None if spec is None else spec['created_at'])
            live_rows, live_summaries = export_live(workspace, expanded_cascades, protocol, origin, provenance, run_id)
            rows.extend(live_rows)
        elif stage == 'schema':
            import expanded_schema
            origin = lambda responses: check_response_origin(responses, run_id, None if spec is None else spec['created_at'])
            rows.extend(export_schema(workspace, expanded_schema, analyze, cell, origin, provenance, run_id))
        elif stage == 'calibration':
            import expanded_calibration
            rows.extend(export_calibration(workspace, expanded_calibration, analyze, cell, run_id, digest))
        elif stage == 'rules':
            import expanded_rules
            export_rules(workspace, expanded_rules, MODELS, BENCHMARKS, analyze, cell, emit)
        else:
            raise ValueError('Unsupported stage')
    output.mkdir(parents=True, exist_ok=False)
    (output/'metrics.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    (output/'verification.json').write_text(json.dumps({'run_id': run_id, 'workspace': str(workspace),
        'stages': stages, 'normalized_rows': len(rows), 'verified_source_cells': len(provenance), 'sources': provenance,
        'cost_scope': 'Known request costs include failure journals; unknown counts separate. Policy cost rows may share cells and must not be summed as project spending.',
        'peeters_summary': peeters_summary,
        'peeters_export_sha256': hashlib.sha256(Path(__file__).with_name('peeters_run_export.py').read_bytes()).hexdigest(),
        'peeters_scope': 'Complete stage journal required; all fixed development and test cells checked against frozen inputs, request hashes, run IDs and raw parsers. B0 comparisons use the same run. Values serialization jointly changes markers, quoting, field names and null representation. No historical full replication or three-run SD from one run. Reused baseline costs are not additional project charges.',
        'time_scope': 'For non-live request duration metrics: sum attempt durations including failures, excluding scheduling/cooldown. Live latency uses complete-chain wall time; see live_scope.',
        'selection_check': 'Non-live selection adapters reconstruct development choices; live alone verifies selection digest and executed routes. No independent historical chronology reconstruction.',
        'extension_scope': 'Variants stay outside the original eight-family macro. Paired language test contrasts require matching pair/entity IDs and labels; F1 differences, flips, correctness fractions and probability shifts are descriptive, not inferential. Translation is not human-validated; no paired WDC full-split comparisons. Stability includes fixed F1 factorial contrast, prediction agreement/flip rates and probability ranges across three within-run development measurements, not test performance. Generative Luna probability diagnostics are explicitly unavailable. Small repeats never become study run IDs.',
        'rules_scope': 'Fixed learned-rule, manual-domain and B0 test conditions; full rule/manual development matrix request-checked first. Reused B0/manual costs are not additional project charges. No human rule validation or repeated rule induction claimed; paired contrasts remain separate.',
        'cascade_export_sha256': hashlib.sha256(Path(__file__).with_name('offline_cascade_export.py').read_bytes()).hexdigest(),
        'cascade_scope': 'Full development fits and cutoffs reconstructed before test. Fixed peer overhead, routed fallback costs, all 20 random seeds and retrospective label-aware oracle checked; unavailable routers retained as null. Cached costs are reused, not additional project spending; random seeds are not study runs.',
        'live_summaries': live_summaries,
        'live_scope': 'Fresh complete-chain p50/p95 with six concurrent configured model chains. Small fixed 20-pair samples per cell, not production SLA or full-test latency; routing selection hash checked, development fit reconstruction remains in offline cascade adapter. Raw requests/parsers/journals/routes/costs verified; failed/orphaned call charges retained.',
        'live_export_sha256': hashlib.sha256(Path(__file__).with_name('live_cascade_export.py').read_bytes()).hexdigest(),
        'schema_scope': 'Exact wire-order request verification and raw reparse for pilot/development/test. Full development selection and test matrix checked before test. Review is abstention; zero coverage produces null accepted metrics. Global/local costs can reuse the same cells. Order effects are descriptive development diagnostics.',
        'schema_export_sha256': hashlib.sha256(Path(__file__).with_name('schema_run_export.py').read_bytes()).hexdigest(),
        'calibration_scope': 'All frozen development fits and thresholds reconstructed before test; unavailable budgets retained as null. Saved parameters evaluated without refitting on test. Source costs in provenance are shared and not per-condition charges.',
        'calibration_export_sha256': hashlib.sha256(Path(__file__).with_name('calibration_run_export.py').read_bytes()).hexdigest(),
        'three_runs_complete': False, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'stability_diagnostics_sha256': hashlib.sha256(Path(__file__).with_name('stability_run_diagnostics.py').read_bytes()).hexdigest(),
        'language_diagnostics_sha256': hashlib.sha256(Path(__file__).with_name('language_run_diagnostics.py').read_bytes()).hexdigest()}, indent=2)+'\n')
    print(json.dumps({'run_id': run_id, 'rows': len(rows), 'source_cells': len(provenance)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--run-id', required=True, type=int, choices=[1, 2, 3])
    p.add_argument('--stages', nargs='+', choices=['baseline', 'ap2', 'screen', 'confirmation', 'variants', 'stability', 'rules', 'calibration', 'schema', 'live', 'cascades', 'peeters'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    export(args.root, args.run_id, args.stages, args.output)
