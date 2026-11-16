"""Scores the validation prompt study and selects one prompt per model and benchmark. Read-only.

Validation F1 (rules_train pairs excluded, as in the threshold tuning), precision,
recall, pairs and cost per model, prompt and benchmark; selection = highest
validation F1 (ties: promptval.PROMPTS order, B0 first); test F1 of the selected
prompt from the test prompt study; cells whose selection differs from F0.

    PYTHONPATH=src python3 src/promptval_eval.py --out artifacts/results/prompt_validation_20261027.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import inputs_study
import prompt12
import promptval
import protocol

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'promptval_run01_20261027'
TEST_NAMESPACE = 'prompt12_run01_20261019'


def journal(path):
    return {r['pair_id']: r for r in map(json.loads, Path(path).read_text().splitlines())}


def scores(rows, labels):
    tp = sum(labels[i] == 1 and r['prediction'] == 1 for i, r in rows.items())
    fp = sum(labels[i] == 0 and r['prediction'] == 1 for i, r in rows.items())
    fn = sum(labels[i] == 1 and r['prediction'] == 0 for i, r in rows.items())
    return {'f1': 2*tp/(2*tp+fp+fn) if tp else 0., 'precision': tp/(tp+fp) if tp+fp else 0.,
            'recall': tp/(tp+fn) if tp+fn else 0., 'pairs': len(rows)}


def cost(rows):
    attempts = [a for r in rows.values() for a in r['attempts']]
    return {'cost_usd': sum(a['cost'] or 0. for a in attempts), 'attempts': len(attempts),
            'unknown_cost_attempts': sum(a['cost'] is None for a in attempts)}


def validation(root):
    out = {}
    for b in protocol.BENCHMARKS:
        labels = {c['pair_id']: c['label'] for c in json.loads((root/'data'/b/'valid.json').read_text())}
        skip = {r['pair_id'] for r in json.loads((root/'data'/b/'rules_train.json').read_text())}
        for m in promptval.RUN_MODELS:
            for v in promptval.PROMPTS:
                rows = journal(root/'runs'/NAMESPACE/m/v/b/'valid.jsonl')
                assert rows.keys() == labels.keys(), ('incomplete', m, v, b)
                kept = {i: r for i, r in rows.items() if i not in skip}
                out.setdefault(m, {}).setdefault(v, {})[b] = {
                    **scores(kept, labels), 'excluded_rules_train': len(rows)-len(kept), **cost(rows)}
    return out


def repeat_check(root, val):
    """Same request run twice: valsel_run01_20261019 vs this run (validation F1, prediction agreement)."""
    out = {}
    for (m, v, b), path in sorted(promptval.reused(root).items()):
        labels = {c['pair_id']: c['label'] for c in json.loads((root/'data'/b/'valid.json').read_text())}
        skip = {r['pair_id'] for r in json.loads((root/'data'/b/'rules_train.json').read_text())}
        old = {i: r for i, r in journal(path).items() if i not in skip}
        new = journal(root/'runs'/NAMESPACE/m/v/b/'valid.jsonl')
        out['/'.join((m, v, b))] = {'valsel_f1': scores(old, labels)['f1'], 'promptval_f1': val[m][v][b]['f1'],
                                    'prediction_agreement': mean([old[i]['prediction'] == new[i]['prediction'] for i in old])}
    return out


def test_f1(root):
    """Test F1 per model, prompt (all twelve) and benchmark: B0 from the baseline table except pplx_v12."""
    out = {}
    for b in protocol.BENCHMARKS:
        labels = {c['pair_id']: c['label'] for c in json.loads((root/'data'/b/'test.json').read_text())}
        for m in promptval.RUN_MODELS:
            for v in prompt12.VARIANTS:
                own = v != 'B0' or m in prompt12.EXTRA_MODELS
                rows = journal(root/'runs'/(TEST_NAMESPACE if own else 'quality')/m/v/b/'test.jsonl')
                assert rows.keys() == labels.keys(), ('incomplete test', m, v, b)
                out.setdefault(m, {}).setdefault(v, {})[b] = scores(rows, labels)['f1']
    return out


def mean(xs):
    return sum(xs)/len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = ROOT
    promptval.register_models()
    val = validation(root)
    test = test_f1(root)
    f0 = inputs_study.load_selection(root)['selected']
    selected, changed, per_model = {}, [], {}
    for m in promptval.RUN_MODELS:
        selected[m] = {}
        for b in protocol.BENCHMARKS:
            v = max(promptval.PROMPTS, key=lambda p: (val[m][p][b]['f1'], -promptval.PROMPTS.index(p)))
            selected[m][b] = {'prompt': v, 'valid_f1': val[m][v][b]['f1'], 'test_f1': test[m][v][b],
                              'f0_prompt': f0[m][b], 'f0_test_f1': test[m][f0[m][b]][b], 'b0_test_f1': test[m]['B0'][b]}
            if v != f0[m][b]:
                changed.append({'model': m, 'benchmark': b, 'valid_selected': v, 'f0': f0[m][b]})
        per_model[m] = {k: mean([selected[m][b][k] for b in protocol.BENCHMARKS])
                        for k in ('test_f1', 'f0_test_f1', 'b0_test_f1')}
        best10 = {b: max(test[m][p][b] for p in promptval.PROMPTS) for b in protocol.BENCHMARKS}
        per_model[m]['test_oracle_10_prompts_f1'] = mean(list(best10.values()))
    total = {k: sum(c[k] for m in val.values() for p in m.values() for c in p.values())
             for k in ('cost_usd', 'attempts', 'unknown_cost_attempts')}
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'description': 'Prompt study on the official validation sets (one run, promptval_run01_20261027). '
                       'Validation F1 excludes the Astra training pairs (rules_train). Selection: highest validation F1 '
                       'per model and benchmark, ties by prompt order (B0 first). Test F1 from the test prompt study '
                       '(prompt12_run01_20261019; B0 from the baseline table, except pplx_v12). '
                       'F0 = prompt selected on test among twelve prompts (configs/inputs_prompt_selection_20261019.json).',
        'prompts': promptval.PROMPTS,
        'prompt_names': {v: n for v, n, _ in prompt12.PROMPTS},
        'models': promptval.RUN_MODELS,
        'benchmarks': list(protocol.BENCHMARKS),
        'validation': val,
        'selection': selected,
        'mean_test_f1_per_model': per_model,
        'changed_vs_f0': changed,
        'changed_vs_f0_count': len(changed),
        'cost_kept_journals': total,
        'repeat_check_vs_valsel': repeat_check(root, val),
        'test_f1_all_prompts': test,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write('\n')
    print(json.dumps({'changed_vs_f0_count': len(changed), 'mean_test_f1_per_model': per_model, 'cost_kept_journals': total}, indent=1))


if __name__ == '__main__':
    main()
