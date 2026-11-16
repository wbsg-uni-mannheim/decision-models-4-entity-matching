"""Facts on the 31 Oct labelled-chapter runs (Aaron's release "1-3 Experimente bitte einmal laufen lassen"). Read-only, no inference.

inference_cost_test: USD per 1M test pairs, method of labeled_chapter_facts.inference_cost (attempt costs kept in the test
journals, summed over the 8 benchmarks; archived failed attempts are only in the ledger). FS10_<n> from
labeledch_fewshot2_run01_20261031, RULESFS10_<n> from labeledch_rulesfs_run01_20261031, RULES_<n> of Astra rules runs 2/3.
astra_rules: rules per workspace from configs/labeled_chapter_rules_20261028{,_rep2,_rep3}.json, wall-clock minutes from the
status files of the Astra workspaces (launch.json started_at to completed.json finished_at); run 1 is checked against
labeled_chapter_facts_20261030.json. trained_matchers_vs_pplx_b0: LR and Ditto at 200/2000 copied unchanged from
labeled_chapter_facts_20261030.json (vs_pplx_b0).
fewshot_shared_record: method of labeled_chapter_facts.leakage (same left or same right record id as the test pair; any
example / an example labelled match) for the FS10_200 and FS10_2000 example pools; the old FS10 pool is recomputed and checked.
sol_vs_gold: method of labelsets_sol_eval_20261028.py (Sol labels vs official benchmark training labels on the overlapping
pairs, either orientation) for set200, set2000 (frozen sets, all pairs) and set4000; set200/set4000 are checked against
runs/labelsets_sol_b0_20261028/eval_vs_benchmark_train.json (= artifacts/results/labelsets_sol_b0_eval_20261028.json).

    cluster: PYTHONPATH=src:artifacts/operations .venv/bin/python src/labeled_facts_20261101.py cost --out <cost.json>
    cluster: PYTHONPATH=src:artifacts/operations .venv/bin/python src/labeled_facts_20261101.py extra --out <extra.json>
    local:   python3 src/labeled_facts_20261101.py merge --cost <cost.json> --extra <extra.json> --out artifacts/results/labeled_facts_20261101.json
"""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT/'artifacts/results/labeled_chapter_facts_20261030.json'
SIZES = ['200', '2000']
MODELS = {'jev': 'jev', 'pplx': 'pplx_v12'}
DIRS = {**{f'FS10_{s}': f'runs/labeledch_fewshot2_run01_20261031/{{m}}/FS10_{s}' for s in SIZES},
        **{f'RULESFS10_{s}': f'runs/labeledch_rulesfs_run01_20261031/{{m}}/RULESFS10_{s}' for s in SIZES},
        **{f'RULES_{s}_run{r}': f'runs/labeledch_rules_run0{r}_20261028/{{m}}/RULES_{s}' for r in (2, 3) for s in SIZES}}
WORKSPACES = {1: ROOT.parent/'rule-agent-labeled-20261028', 2: ROOT.parent/'rule-agent-labeled-20261028-rep2',
              3: ROOT.parent/'rule-agent-labeled-20261028-rep3'}
RULES = {1: 'configs/labeled_chapter_rules_20261028.json', 2: 'configs/labeled_chapter_rules_20261028_rep2.json',
         3: 'configs/labeled_chapter_rules_20261028_rep3.json'}


def cost():
    import protocol
    from labeled_chapter_facts import journal_cost
    out = {}
    for name, m in MODELS.items():
        for v, d in DIRS.items():
            c = [journal_cost(ROOT/d.format(m=m)/b/'test.jsonl') for b in protocol.BENCHMARKS]
            usd, pairs = sum(x[0] for x in c), sum(x[2] for x in c)
            out.setdefault(name, {})[v] = {'usd_journals': usd, 'pairs': pairs, 'usd_per_1m_pairs': usd/pairs*1e6,
                                           'unknown_cost_attempts': sum(x[1] for x in c)}
    led = json.loads((ROOT/'artifacts/budget.json').read_text())['inference_credits']
    ledger = {k: led[k]['charged'] for k in led
              if k.startswith(('labeledch-fewshot2-', 'labeledch-rulesfs-', 'labeledch-rules-run2-', 'labeledch-rules-run3-'))}
    return {'per_model_variant': out, 'ledger_charged_usd': ledger,
            'note': 'usd_journals = attempts kept in the test journals (sums over 8 benchmarks); the ledger also holds the '
                    'archived failed attempts (HTTP 429).'}


def shared_record(examples, train, tests):
    """Method of labeled_chapter_facts.leakage: share of test pairs with an example that has the same left or right record id."""
    anyr = match = 0
    for c in tests:
        ex = [train[p] for p in examples[c['pair_id']]]
        shared = [r for r in ex if str(r['id_left']) == str(c['entity_ids']['left']) or str(r['id_right']) == str(c['entity_ids']['right'])]
        anyr += bool(shared); match += any(r['label'] for r in shared)
    return {'share_any_example': anyr/len(tests), 'share_matching_example': match/len(tests), 'n': len(tests)}


def fewshot_overlap(old):
    import protocol
    import labeled_chapter
    labeled_chapter.load_fewshot(); labeled_chapter.load_fewshot2()
    out = {}
    for b in protocol.BENCHMARKS:
        tests = json.loads((ROOT/'data'/b/'test.json').read_text())
        assert shared_record(labeled_chapter.FEWSHOT[b]['k10'], labeled_chapter.TRAIN[b], tests) == old[b]['k10'], b
        out[b] = {f'FS10_{s}': shared_record(labeled_chapter.FEWSHOT2[b][s]['k10'], labeled_chapter.TRAIN2[b][s], tests) for s in SIZES}
    return out


def sol_vs_gold(old):
    """Method of artifacts/operations/labelsets_sol_eval_20261028.py, applied to the frozen 2000 sets (all 2000 pairs)."""
    import gzip
    import sys
    sys.argv = ['x']; sys.path.insert(0, str(ROOT/'artifacts/operations'))
    import labelsets_sol_20261028 as L
    import labeled_chapter_20261028 as lc
    L.lab_modules()
    import prepare
    train_file = {'abt-buy': 'abt-buy-train.json', 'wdc': 'wdcproducts80cc20rnd000un_train_large.json.gz'}
    out = {}
    for b in L.RUNS:
        p = L.LAB/'benchmarks'/b/train_file.get(b, f'{b}-train.json.gz')
        text = gzip.open(p, 'rt').read() if p.suffix == '.gz' else None
        if text is not None and text.lstrip().startswith('{') and '\n{' not in text.strip():
            data = json.loads(text); rows = [{k: v[i] for k, v in data.items()} for i in next(iter(data.values()))]
        else:
            rows = prepare.read_rows(p)
        gold = {}
        for r in rows:
            a, c, y = str(r['id_left']), str(r['id_right']), int(r['label'])
            gold[(a, c)] = y; gold.setdefault((c, a), y)
        res = {}
        for size in (200, 2000, 4000):
            items = lc.manifest_rows(b, size, 'all')
            ov = [(it['label'], gold[(it['id_left'], it['id_right'])]) for it in items if (it['id_left'], it['id_right']) in gold]
            tp = sum(p and y for p, y in ov); fp = sum(p and not y for p, y in ov); fn = sum(not p and y for p, y in ov)
            tn = len(ov)-tp-fp-fn
            pr = tp/(tp+fp) if tp+fp else None; rc = tp/(tp+fn) if tp+fn else None
            res[f'set{size}'] = {'n': len(items), 'sol_pos_share': round(sum(it['label'] for it in items)/len(items), 3),
                                 'overlap_with_train': len(ov),
                                 'overlap_gold_pos_share': round(sum(y for _, y in ov)/len(ov), 3) if ov else None,
                                 'sol_vs_gold': {'n': len(ov), 'precision': pr and round(pr, 3), 'recall': rc and round(rc, 3),
                                                 'f1': round(2*pr*rc/(pr+rc), 3) if pr and rc else None,
                                                 'accuracy': round((tp+tn)/len(ov), 3) if ov else None,
                                                 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}}
        for name in ('set200', 'set4000'):
            assert res[name]['sol_vs_gold'] == old[b][name]['sol_vs_gold'], (b, name)
        out[b] = res
    return out


def extra():
    facts = json.loads(FACTS.read_text())
    return {'fewshot_shared_record': fewshot_overlap(facts['leakage']['fewshot_shared_record']),
            'sol_vs_gold': sol_vs_gold(json.loads((ROOT/'runs/labelsets_sol_b0_20261028/eval_vs_benchmark_train.json').read_text()))}


def astra(old):
    per = {}
    for r, w in WORKSPACES.items():
        src = json.loads((ROOT/RULES[r]).read_text())['sources']
        for key in sorted(k for k in src if k.split('/')[1] in SIZES):
            b, s = key.split('/')
            status = w/b/f'status_set{s}'
            done = json.loads((status/'completed.json').read_text())
            minutes = (datetime.fromisoformat(done['finished_at']) -
                       datetime.fromisoformat(json.loads((status/'launch.json').read_text())['started_at'])).total_seconds()/60
            per.setdefault(str(r), {})[key] = {'n_rules': src[key]['n_rules'], 'minutes': minutes, 'exit_code': done['exit_code']}
            if r == 1:
                assert abs(minutes-old[key]['minutes']) < 0.01 and src[key]['n_rules'] == old[key]['n_rules'], key

    def summary(rows):
        n, t = [x['n_rules'] for x in rows], [x['minutes'] for x in rows]
        return {'workspaces': len(rows), 'min_rules': min(n), 'max_rules': max(n), 'mean_rules': statistics.mean(n),
                'mean_minutes': statistics.mean(t), 'min_minutes': min(t), 'max_minutes': max(t)}
    return {'per_size': {s: {'runs_1_3': summary([x for r in per for k, x in per[r].items() if k.endswith('/'+s)]),
                             **{f'run{r}': summary([x for k, x in per[r].items() if k.endswith('/'+s)]) for r in per}}
                         for s in SIZES},
            'per_run': per,
            'note': 'Minutes are wall-clock per agent (launch to exit). Each run used two sequential streams; runs 2 and 3 ran '
                    'at the same time (31 Oct), run 1 alone (28 Oct).'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['cost', 'extra', 'merge'])
    ap.add_argument('--cost', type=Path)
    ap.add_argument('--extra', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists(), args.out
    if args.mode == 'cost':
        res = cost()
    elif args.mode == 'extra':
        res = extra()
    else:
        facts = json.loads(FACTS.read_text())
        res = {'description': __doc__, 'created_at': datetime.now(timezone.utc).isoformat(),
               'inference_cost_test': json.loads(args.cost.read_text()),
               'astra_rules': astra(facts['method_facts']['astra_rules']['per_run']),
               'trained_matchers_vs_pplx_b0': {'units': facts['units'], 'reference': facts['reference'],
                                               **{m: {s: facts['vs_pplx_b0'][m][s] for s in SIZES} for m in ('lr', 'ditto')}},
               **json.loads(args.extra.read_text())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
