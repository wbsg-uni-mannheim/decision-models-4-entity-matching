"""Facts for the labelled-data chapter in one file. Read-only, no inference, no paid calls.

bootstrap_vs_pplx_b0: paired bootstrap of every labelled-data method against PPLX v1.2 B0 Run 1
(runs/prompt12_run01_20261019), method of inputs_b0_eval.py: stratified per benchmark, 10,000 replicates, seed 0,
same resample for all arms, mean F1 difference over the 8 benchmarks, 95% CI; differences and CIs in F1 points.
Mean arms (Ditto seed mean, Claude Code 3-run mean) take the mean F1 of the runs per replicate. Point F1 values are
checked against lr/results.json, ditto/results.json, claudecode_v3*/scores.json, claudecode_v3_reps_20261029.json,
labeled_rules_20261028.json and labeled_fewshot_20261028.json.
vs_pplx_b0: the same numbers in table form (method -> size -> diff, ci95). lr_gold_control: lr/gold_control.json.
label_sets: frozen sets and splits, how the pairs were chosen, Sol-B0 labels vs the official training labels on the
overlapping pairs, Sol cost. method_facts, inference_cost_test, leakage: see the keys.

    /path/to/envs/ditto-modern/bin/python src/labeled_chapter_facts.py \
        --astra-timing ~/promptval_scratch/astra_timing_20261030.json --out artifacts/results/labeled_chapter_facts_20261030.json
"""
import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
import protocol  # noqa: E402
from inputs_b0_eval import f1, journal  # noqa: E402

LC = ROOT/'runs/labeled_chapter_20261028'
RES = ROOT/'artifacts/results'
B0_PPLX = ROOT/'runs/prompt12_run01_20261019/pplx_v12/B0'
B0_JEV = ROOT/'runs/quality/jev/B0'
RULES_RUN, FS_RUN = ROOT/'runs/labeledch_rules_run01_20261028', ROOT/'runs/labeledch_fewshot_run01_20261028'
CC = {1: LC/'claudecode_v3', 2: LC/'claudecode_v3_rep2', 3: LC/'claudecode_v3_rep3'}
SIZES, SEEDS, REPS = ['200', '2000', '4000'], [42, 52, 62], 10_000
MODELS = {'pplx': 'pplx_v12', 'jev': 'jev'}
NOTES_PATTERN = r'abt|buy\.com|walmart|amazon|dblp|acm|scholar|wdc|billiger|semi-heter|dn7'  # grep -il -E of 29 Oct


def read_gz(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(x) for x in f]


def mean(xs):
    xs = list(xs)
    return sum(xs)/len(xs)


def prf(pairs):
    tp = sum(p and y for p, y in pairs); fp = sum(p and not y for p, y in pairs); fn = sum(not p and y for p, y in pairs)
    pr, rc = (tp/(tp+fp) if tp+fp else None), (tp/(tp+fn) if tp+fn else None)
    return {'n': len(pairs), 'precision': pr, 'recall': rc, 'f1': 2*tp/(2*tp+fp+fn) if tp else 0., 'tp': tp, 'fp': fp, 'fn': fn,
            'tn': len(pairs)-tp-fp-fn}


# ------------------------------------------------------------------ predictions per arm

def ditto_preds(b, size, seed, ids):
    s = json.loads((LC/f'ditto/output/{b}_set{size}_seed{seed}/summary.json').read_text())['results'][0]
    with open(Path(s['run_dir'])/'predictions.csv') as f:
        pred = {r['pair_id']: int(r['pred']) for r in csv.DictReader(f)}
    assert pred.keys() == set(ids), (b, size, seed)
    return np.array([pred[i] for i in ids], dtype=np.int8), s['metrics']['test']['f1']


def arms(b, ids, lr):
    """name -> {size: [(predictions, reference F1 or None), ...]} for one benchmark; several entries = mean arm."""
    cc_manifest = {r: json.loads((w/'manifest.json').read_text())['runs'] for r, w in CC.items()}
    cc_scores = {r: json.loads((w/'scores.json').read_text()) for r, w in CC.items()}
    rules = json.loads((RES/'labeled_rules_20261028.json').read_text())['cells']
    fs = json.loads((RES/'labeled_fewshot_20261028.json').read_text())['cells']
    out = {'jev_b0': {'all': [(journal(B0_JEV/b/'test.jsonl', ids), rules['jev'][b]['200']['f1_b0'])]}}
    out['lr'] = {s: [(np.array([lr[b][s]['predictions'][i] for i in ids], dtype=np.int8), lr[b][s]['test_f1'])] for s in SIZES}
    dit = {s: [ditto_preds(b, s, seed, ids) for seed in SEEDS] for s in SIZES}
    for k, seed in enumerate(SEEDS):
        out[f'ditto_seed{seed}'] = {s: [dit[s][k]] for s in SIZES}
    out['ditto_seed_mean'] = dit
    cc = {}
    for r in CC:
        for s in SIZES:
            key = f'{b}/{s}'
            sc = cc_scores[r][key]
            assert 'error' not in sc and sc['order_invariant'] and not sc['flagged_events'], (r, key)
            p = json.loads((CC[r]/'apply'/cc_manifest[r][key]['rid']/'test_predictions.json').read_text())
            cc.setdefault(s, []).append((np.array([p[i] for i in ids], dtype=np.int8), sc['f1']))
    for k, r in enumerate(CC):
        out[f'claudecode_v3_run{r}'] = {s: [cc[s][k]] for s in SIZES}
    out['claudecode_v3_mean'] = cc
    for name, m in MODELS.items():
        out[f'{name}_rules'] = {s: [(journal(RULES_RUN/m/f'RULES_{s}'/b/'test.jsonl', ids), rules[m][b][s]['f1'])] for s in SIZES}
        out[f'{name}_fewshot'] = {k: [(journal(FS_RUN/m/k.upper()/b/'test.jsonl', ids), fs[m][b][k.upper()]['f1'])] for k in ('fs10', 'fs20')}
    return out


def bootstrap(lr):
    rng = np.random.default_rng(0)
    data = {}
    for b in protocol.BENCHMARKS:
        cases = json.loads((ROOT/'data'/b/'test.json').read_text())
        ids = [c['pair_id'] for c in cases]
        data[b] = (ids, np.array([c['label'] for c in cases], dtype=np.int8), rng.integers(0, len(ids), size=(REPS, len(ids))))
    rules = json.loads((RES/'labeled_rules_20261028.json').read_text())['cells']
    acc = {}
    for b in protocol.BENCHMARKS:
        ids, y, sample = data[b]
        full = np.arange(len(ids))
        b0 = journal(B0_PPLX/b/'test.jsonl', ids)
        f0, f0_s = float(f1(y, b0, full)), f1(y, b0, sample)
        assert abs(f0-rules['pplx_v12'][b]['200']['f1_b0']) < 1e-12, b
        for name, sizes in arms(b, ids, lr).items():
            for s, runs in sizes.items():
                fs = []
                for p, ref in runs:
                    fs.append(float(f1(y, p, full)))
                    assert ref is None or abs(fs[-1]-ref) < 1e-9, (name, b, s, fs[-1], ref)
                d = mean(f1(y, p, sample) for p, _ in runs)-f0_s
                cell = {'f1': mean(fs), 'f1_pplx_b0': f0, 'diff': 100*(mean(fs)-f0),
                        'ci95': [100*float(np.percentile(d, 2.5)), 100*float(np.percentile(d, 97.5))]}
                if len(runs) == 1:
                    p = runs[0][0]
                    cell |= {'correct_to_wrong': int(((b0 == y) & (p != y)).sum()), 'wrong_to_correct': int(((b0 != y) & (p == y)).sum())}
                else:
                    cell['f1_runs'] = fs
                a = acc.setdefault(name, {}).setdefault(s, {'cells': {}, 'd': np.zeros(REPS)})
                a['cells'][b] = cell; a['d'] += d/len(protocol.BENCHMARKS)
    out = {}
    for name, sizes in acc.items():
        for s, a in sizes.items():
            cells = a.pop('cells'); d = a.pop('d')
            m = {'f1': mean(c['f1'] for c in cells.values()), 'f1_pplx_b0': mean(c['f1_pplx_b0'] for c in cells.values()),
                 'diff': mean(c['diff'] for c in cells.values()),
                 'ci95': [100*float(np.percentile(d, 2.5)), 100*float(np.percentile(d, 97.5))]}
            if 'correct_to_wrong' in next(iter(cells.values())):
                m |= {k: sum(c[k] for c in cells.values()) for k in ('correct_to_wrong', 'wrong_to_correct')}
            out.setdefault(name, {})[s] = {'mean': m, 'cells': cells}
    # point checks against the existing result files (fractions there, points here)
    reps = json.loads((RES/'claudecode_v3_reps_20261029.json').read_text())['mean']
    fsr = json.loads((RES/'labeled_fewshot_20261028.json').read_text())['cells']
    for s in SIZES:
        assert abs(out['claudecode_v3_mean'][s]['mean']['diff']-100*reps[s]['diff']) < 1e-9, s
        for name, m in MODELS.items():
            assert abs(out[f'{name}_rules'][s]['mean']['f1']-mean(rules[m][b][s]['f1'] for b in protocol.BENCHMARKS)) < 1e-12
    for name, m in MODELS.items():
        for k in ('fs10', 'fs20'):
            assert abs(out[f'{name}_fewshot'][k]['mean']['f1']-mean(fsr[m][b][k.upper()]['f1'] for b in protocol.BENCHMARKS)) < 1e-12
    return out


# ------------------------------------------------------------------ label sets

def gold_train(b):
    """Official training labels (both orientations), loader of labelsets_sol_eval_20261028.py."""
    import labelsets_sol_20261028 as L
    L.lab_modules()
    import prepare
    names = {'abt-buy': 'abt-buy-train.json', 'wdc': 'wdcproducts80cc20rnd000un_train_large.json.gz'}
    p = L.LAB/'benchmarks'/b/names.get(b, f'{b}-train.json.gz')
    text = gzip.open(p, 'rt').read() if p.suffix == '.gz' else None
    if text is not None and text.lstrip().startswith('{') and '\n{' not in text.strip():
        data = json.loads(text); rows = [{k: v[i] for k, v in data.items()} for i in next(iter(data.values()))]
    else:
        rows = prepare.read_rows(p)
    gold = {}
    for r in rows:
        a, c, y = str(r['id_left']), str(r['id_right']), int(r['label'])
        gold[(a, c)] = y; gold.setdefault((c, a), y)
    return gold, len(rows)


def label_sets():
    sets = json.loads((ROOT/'configs/labeled_chapter_sets_20261028.json').read_text())['benchmarks']
    report = json.loads((ROOT/'runs/labelsets_sol_b0_20261028/report_20261028T003030Z.json').read_text())
    calls = sum(report[b]['calls_total'] for b in protocol.BENCHMARKS)
    led = json.loads((ROOT/'artifacts/budget.json').read_text())['direct_openai_commitments']
    usd = {k: led[k]['actual_usd'] for k in led if k.startswith('labelsets-sol-')}
    b0_usd = usd['labelsets-sol-382926-1793140992']+usd['labelsets-sol-382946-1793142175']
    per_call = b0_usd/calls
    out = {'selection': {
        '200': 'Similarity-search seed of the labelling paper source run (per-query seed rule, 30/70 positive/negative '
               'target), cleaned of official test and validation pairs, topped up to 100 pairs with the same seed rule, plus '
               'one AL-ML round of 100 pairs (committee LR, RF, ET, HistGBDT, XGBoost and rule scorers; 20,000 pool '
               'candidates; top-5 committee variance). Labels: GPT-6.1 Sol with the B0 request.',
        '2000': 'Nested in the 4000 set: the first k positives and 2000-k negatives of the 4000 set in labelling order, '
                'k at the 4000 set positive share (profile mechanism of the labelling repo); the seed is inside.',
        '4000': 'Nested-profile selection on the source AL run label list in labelling order (existing AL runs of the '
                'labelling paper), seed first, positives at the share of the benchmark small profile (dn7: medium, '
                'semi-heter: large). Semi-HETER: its 1,880 labelled pairs continued with AL-ML rounds of 500 on the same '
                'pool to 4,000. Pairs are relabelled by Sol (the GPT-5.2 labels only pick the pairs).',
        'cleaning': 'Every official test and validation pair is removed from seed, label lists and pool before selection.',
        'split': 'Stratified 80/20 train/validation per set; within each class pairs ranked by '
                 "sha256('labeled-chapter-split|<benchmark>|<pair_id>'), first round(0.2 n) to validation.",
        'source': 'artifacts/operations/labelsets_sol_20261028.py, artifacts/operations/labeled_chapter_20261028.py'},
        'benchmarks': {}}
    for b in protocol.BENCHMARKS:
        gold, n_gold = gold_train(b)
        r = report[b]
        comp = {'200': {'seed_after_cleaning': r['topup'].get('seed_sol'), 'seed_topup': r['topup'].get('added', 0),
                        'al_round': r['al']['added'], 'in_4000_set': r['export']['set200_in_set4000']},
                '4000': ({'existing_al_pairs': r['semi']['base_sol'], 'new_al_rounds_of_500': len(r['semi']['rounds']),
                          'new_al_pairs': r['semi']['missing']} if 'semi' in r else {'existing_al_pairs': r['known']['asked']})}
        info = {'official_train_pairs': n_gold, 'sizes': {}}
        for s in SIZES:
            f = sets[b][s]['files']
            rows = read_gz(ROOT/f['all']['path'])
            ov = [(bool(x['label']), bool(gold[(x['id_left'], x['id_right'])])) for x in rows if (x['id_left'], x['id_right']) in gold]
            info['sizes'][s] = {'pairs': f['all']['n'], 'matches': f['all']['pos'], 'pos_share': sets[b][s]['pos_share'],
                                'train': {'n': f['train']['n'], 'matches': f['train']['pos']},
                                'valid': {'n': f['valid']['n'], 'matches': f['valid']['pos']}, 'seed_inside': sets[b][s]['seed_inside'],
                                'composition': comp.get(s), 'sol_vs_official_train_labels': prf(ov)}
        info['sol_calls'] = r['calls_total']
        out['benchmarks'][b] = info
    out['sol_cost'] = {
        'ledger_usd': usd, 'b0_label_calls': calls, 'b0_label_usd': b0_usd, 'usd_per_call': per_call,
        'overall_usd_incl_listing1_pilot': sum(usd.values()),
        'per_size_usd_at_mean_call_cost': {s: 8*int(s)*per_call for s in SIZES},
        'note': 'One Sol call per unique pair (cache); the sets share pairs (2000 in 4000, the seed and part of the AL round '
                'of the 200 set in the 4000 set), so per-size costs are the stand-alone cost of 8 x size labels at the mean '
                'call cost; ledger_usd is what was spent. labelsets-sol-382880 is the Listing-1 pilot, whose labels are not used.'}
    return out


# ------------------------------------------------------------------ method facts

def sacct_minutes(job):
    names = (LC/'ditto/runs.txt').read_text().split()
    rows = subprocess.run(['sacct', '-j', str(job), '-X', '-n', '-P', '--format=JobID,ElapsedRaw,NodeList,AllocTRES'],
                          capture_output=True, text=True, check=True).stdout.split('\n')
    out = {}
    for r in filter(None, rows):
        jid, sec, node, tres = r.split('|')
        out[names[int(jid.split('_')[1])-1]] = {'minutes': int(sec)/60, 'node': node, 'gpu': re.search(r'gres/gpu:([^=]+)=', tres).group(1)}
    return out


def method_facts(astra_timing):
    lr = json.loads((LC/'lr/results.json').read_text())
    import yaml
    dcfg = yaml.safe_load((LC/'ditto/configs/abt-buy_set200_seed42.yaml').read_text())['defaults']
    dres = json.loads((LC/'ditto/results.json').read_text())
    wall = sacct_minutes(384000)
    cc_runs = {}
    for r, w in CC.items():
        for key, m in json.loads((w/'manifest.json').read_text())['runs'].items():
            res = json.loads((w/m['rid']/'state/done.json').read_text())['summary']['result']
            cc_runs[(r, key)] = {'turns': res['num_turns'], 'minutes': res['duration_ms']/6e4, 'cli_usd': res['total_cost_usd']}
    ws = json.loads((CC[1]/'manifest.json').read_text())['runs']['abt-buy/200']
    rules = json.loads((ROOT/'configs/labeled_chapter_rules_20261028.json').read_text())
    timing = json.loads(Path(astra_timing).read_text())
    fsc = json.loads((ROOT/'configs/labeled_chapter_fewshot_20261028.json').read_text())
    return {
        'lr': {'features_per_benchmark': {b: lr[b]['200']['features'] for b in lr},
               'feature_definition': 'Labelling paper AL-ML committee features (active_learning_ml._build_feature_matrix): per '
                                     'configured field one value: numeric closeness 1-|a-b|/max(|a|,|b|) if both values parse as '
                                     'numbers, else 1 for case-insensitive exact text match, 0 if both empty, else token Jaccard; '
                                     'plus the cosine of the text-embedding-3-small record embeddings (256 d; WDC 1536 d).',
               'model': 'sklearn LogisticRegression(max_iter=1000, random_state=42, class_weight=balanced)',
               'tuning': 'C in [0.01, 0.1, 1, 10, 100] x threshold in 0.05..0.95 (step 0.05), maximum validation F1 on the '
                         'frozen validation split, first best kept; train on the train split only.',
               'chosen': {b: {s: {'C': lr[b][s]['C'], 'threshold': lr[b][s]['threshold']} for s in SIZES} for b in lr},
               'source': 'artifacts/operations/labeled_chapter_20261028.py (lr)'},
        'ditto': {'config': dcfg, 'backbone': dcfg['model_name'], 'epochs_max': dcfg['epochs'], 'batch_size': dcfg['batch_size'],
                  'max_len': dcfg['max_len'], 'lr': dcfg['lr'], 'early_stopping_patience': dcfg['early_stopping_patience'],
                  'seeds': SEEDS,
                  'selection': 'Per epoch the threshold in [0, 1] (step 0.01) with the highest validation F1 '
                               '(ditto_modern.metrics.tune_threshold_for_f1); the epoch with the highest tuned validation F1 is '
                               'kept, its threshold is applied to test. Train/validation = frozen splits.',
                  'gpu': sorted({v['gpu'] for v in wall.values()}), 'array_job': 384000,
                  'mean_minutes_per_run_train_plus_test': {s: mean(v['minutes'] for k, v in wall.items() if f'_set{s}_' in k) for s in SIZES},
                  'mean_best_epoch': {s: mean(v['best_epoch'] for k, v in dres.items() if f'_set{s}_' in k) for s in SIZES},
                  'source': 'artifacts/operations/ditto_labeled_20261028.py; labelling repo scripts/training/train_ditto.py'},
        'claudecode': {'model': 'claude-opus-5-5', 'auth': 'subscription', 'budget_cap_usd_cli_estimate': 10.0, 'wallclock_cap_h': 2,
                       'isolation': 'bwrap jail, only the workspace (and state dir) bound; no web tools; no OpenAI; no PyDI; '
                                    'CPU; Python with numpy, pandas, scikit-learn',
                       'workspace': sorted(ws['files'])+['output/ (empty)'],
                       'workspace_note': 'input/train.json and input/valid.json: labelled pairs of the frozen split; no test '
                                         'pairs. Run directories are neutral ids rNN; TASK.md names only "dataset rNN".',
                       'applier_guards': [
                           'parent process holds the test pairs; a fresh child (python -I -B) per random order loads only the '
                           'matcher from a frozen copy of output/ and gets one pair at a time',
                           'file-open audit hook: only output/ and the Python installation may be opened',
                           'subprocess, exec, fork, spawn and socket events refused; ctypes events reported',
                           'predictions in two random orders must agree (order_invariant)'],
                       'runs': len(cc_runs),
                       'mean_turns_per_run': mean(v['turns'] for v in cc_runs.values()),
                       'mean_minutes_per_run': mean(v['minutes'] for v in cc_runs.values()),
                       'mean_cli_usd_per_run': mean(v['cli_usd'] for v in cc_runs.values()),
                       'per_size': {s: {'mean_turns': mean(v['turns'] for (r, k), v in cc_runs.items() if k.endswith('/'+s)),
                                        'mean_minutes': mean(v['minutes'] for (r, k), v in cc_runs.items() if k.endswith('/'+s))}
                                    for s in SIZES},
                       'per_repetition': {r: {'cli_usd': sum(v['cli_usd'] for (q, k), v in cc_runs.items() if q == r),
                                              'turns': sum(v['turns'] for (q, k), v in cc_runs.items() if q == r),
                                              'agent_hours': sum(v['minutes'] for (q, k), v in cc_runs.items() if q == r)/60}
                                          for r in CC},
                       'source': 'artifacts/operations/claudecode_labeled_20261028.py, claudecode_labeled_20261028_v3.py'},
        'astra_rules': {'model': rules['generator_model'], 'reasoning_effort': 'high', 'harness': 'Codex CLI exec, workspace-write '
                        'sandbox, web search disabled, local Mac, ChatGPT login (no API keys)',
                        'per_size': {s: {'mean_rules': mean(v['n_rules'] for k, v in rules['sources'].items() if k.endswith('/'+s)),
                                         'mean_chars': mean(v['chars'] for k, v in rules['sources'].items() if k.endswith('/'+s)),
                                         'mean_minutes': mean(v['minutes'] for k, v in timing['runs'].items() if k.endswith('/'+s))}
                                     for s in SIZES},
                        'per_run': {k: {**rules['sources'][k], **timing['runs'][k]} for k in rules['sources']},
                        'timing_source': timing['source'],
                        'prompt': 'B0 instruction + "General rules derived from a separate labeled training sample:" + one "- rule" line per rule'},
        'fewshot': {'k': fsc['k'], 'rrf_k': fsc['rrf_k'], 'min_class_share': fsc['min_class_share'],
                    'retrieval': 'From the train split of the 4000 set (3,200 pairs) per test pair: BM25 over the serialized '
                                 'pair (B0 serialization, tokens [a-z0-9]+) and embedding cosine (mean of left-left and '
                                 'right-right record cosines, text-embedding-3-small), fused by RRF (k=60); at least k/5 '
                                 'examples per class; ordered by fused rank.',
                    'placement': 'Examples go into the instruction text after B0 ("Labeled example pairs from a separate '
                                 'training sample ...", each "Example i (match|no match):" + the pair in B0 serialization): '
                                 'Jev noul instructions (Decisions API, no examples field) and the PPLX system message. '
                                 'Test-pair serialization and answer schema as B0.',
                    'source': 'artifacts/operations/fewshot_labeled_20261028.py, src/labeled_chapter.py'},
    }


# ------------------------------------------------------------------ cost and leakage

def journal_cost(path):
    usd = unknown = n = 0
    for line in Path(path).read_text().splitlines():
        r = json.loads(line); n += 1
        for a in r['attempts']:
            usd += a['cost'] or 0.; unknown += a['cost'] is None
    return usd, unknown, n


def inference_cost():
    dirs = {'pplx': {'B0': B0_PPLX}, 'jev': {'B0': B0_JEV}}
    for name, m in MODELS.items():
        dirs[name] |= {f'RULES_{s}': RULES_RUN/m/f'RULES_{s}' for s in SIZES} | {v: FS_RUN/m/v for v in ('FS10', 'FS20')}
    out = {}
    for name, vs in dirs.items():
        for v, d in vs.items():
            c = [journal_cost(d/b/'test.jsonl') for b in protocol.BENCHMARKS]
            usd, pairs = sum(x[0] for x in c), sum(x[2] for x in c)
            out.setdefault(name, {})[v] = {'usd_journals': usd, 'pairs': pairs, 'usd_per_1m_pairs': usd/pairs*1e6,
                                           'unknown_cost_attempts': sum(x[1] for x in c)}
    led = json.loads((ROOT/'artifacts/budget.json').read_text())['inference_credits']
    ledger = {k: led[k]['charged'] for k in led if k.startswith('labeledch-')}
    return {'per_model_variant': out, 'ledger_charged_usd': ledger,
            'ledger_total_usd': {'rules': sum(v for k, v in ledger.items() if 'rules' in k),
                                 'fewshot': sum(v for k, v in ledger.items() if 'fewshot' in k)},
            'note': 'usd_journals = attempts kept in the test journals (sums over 8 benchmarks); the ledger also holds the '
                    'archived failed attempts. B0: Jev runs/quality (13 Oct), PPLX prompt12_run01_20261019.'}


def leakage():
    import labeled_chapter
    labeled_chapter.load_fewshot()
    out = {'fewshot_shared_record': {}, 'test_pair_duplicates_in_splits': {}, 'claudecode_notes_naming_benchmark': {}}
    for b in protocol.BENCHMARKS:
        tests = json.loads((ROOT/'data'/b/'test.json').read_text())
        res = {}
        for k in ('k10', 'k20'):
            anyr = match = 0
            for c in tests:
                ex = [labeled_chapter.TRAIN[b][p] for p in labeled_chapter.FEWSHOT[b][k][c['pair_id']]]
                shared = [r for r in ex if str(r['id_left']) == str(c['entity_ids']['left']) or str(r['id_right']) == str(c['entity_ids']['right'])]
                anyr += bool(shared); match += any(r['label'] for r in shared)
            res[k] = {'share_any_example': anyr/len(tests), 'share_matching_example': match/len(tests), 'n': len(tests)}
        out['fewshot_shared_record'][b] = res
    man = json.loads((CC[1]/'manifest.json').read_text())['runs']
    for key, m in man.items():
        b, s = key.split('/')
        out['test_pair_duplicates_in_splits'].setdefault(b, {})[s] = m['leak_check']['test_pairs']
    for r, w in CC.items():
        hits = {}
        for key, m in json.loads((w/'manifest.json').read_text())['runs'].items():
            t = (w/m['rid']/'workspace/output/NOTES.md')
            if t.exists() and re.search(NOTES_PATTERN, t.read_text(errors='ignore'), re.I):
                hits[key] = sorted({x.lower() for x in re.findall(NOTES_PATTERN, t.read_text(errors='ignore'), re.I)})
        out['claudecode_notes_naming_benchmark'][f'run{r}'] = {'count': len(hits), 'of': 24, 'runs': hits}
    out['notes'] = {
        'fewshot_shared_record': 'Share of test pairs for which at least one retrieved example has the same left or the same '
                                 'right record id (any example / an example labelled match). The example pair itself is never '
                                 'a test pair (official test and validation pairs were removed before selection).',
        'test_pair_duplicates_in_splits': 'Labelled pairs (train + validation of the frozen split) whose two records are '
                                          'content-identical to a test pair (duplicate records with other ids in the source '
                                          'tables); Aaron 28 Oct: not filtered. Same counts for every method (shared splits).',
        'claudecode_notes_naming_benchmark': f'NOTES.md matching grep -i -E "{NOTES_PATTERN}" (as counted on 29 Oct); '
                                             'matchers contain only generic tokens.'}
    return out


def table(full):
    """vs_pplx_b0 -> method -> size -> {diff, ci95}: Ditto and Claude Code as the mean of their runs, single runs under per_seed / per_run."""
    pick = lambda name, s: {k: full[name][s]['mean'][k] for k in ('f1', 'diff', 'ci95')}
    out = {'lr': {s: pick('lr', s) for s in SIZES},
           'ditto': {s: {**pick('ditto_seed_mean', s), 'per_seed': {str(x): pick(f'ditto_seed{x}', s) for x in SEEDS}} for s in SIZES},
           'claudecode': {s: {**pick('claudecode_v3_mean', s), 'per_run': {str(r): pick(f'claudecode_v3_run{r}', s) for r in CC}} for s in SIZES}}
    for name in ('pplx_rules', 'jev_rules', 'pplx_fewshot', 'jev_fewshot'):
        out[name] = {s: pick(name, s) for s in full[name]}
    out['jev_b0'] = pick('jev_b0', 'all')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--astra-timing', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    lr = json.loads((LC/'lr/test_predictions.json').read_text())
    out = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(), 'replicates': REPS,
           'units': 'F1 as fraction; diff and ci95 in F1 points', 'reference': 'pplx_v12 B0 Run 1 (runs/prompt12_run01_20261019)'}
    out['bootstrap_vs_pplx_b0'] = full = bootstrap(lr)
    out['vs_pplx_b0'] = table(full)
    out['lr_gold_control'] = json.loads((LC/'lr/gold_control.json').read_text())
    out['label_sets'] = label_sets()
    out['method_facts'] = method_facts(args.astra_timing)
    out['inference_cost_test'] = inference_cost()
    out['leakage'] = leakage()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(out, f, indent=1)
    for name, sizes in out['bootstrap_vs_pplx_b0'].items():
        print(name, {s: (round(v['mean']['diff'], 2), [round(x, 2) for x in v['mean']['ci95']]) for s, v in sizes.items()}, flush=True)


if __name__ == '__main__':
    main()
