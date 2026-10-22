"""F1 of the reference LLMs on all test pairs: the three ensemble members (GPT-6.1 Sol, Claude Opus 5.5,
Gemini 4 Argon), their majority vote, and Qwen3.8-27B (generative B0, three runs).

Each ensemble member answered the escalated pairs in runs/escalation_run01_20261020 and the remaining
pairs in runs/reference_rest_run01_20261021 (Sol direct, Opus batch, Argon OpenRouter flex). The 1,000-pair
check of the Opus batch against normal requests is copied from opus55_live_sample1000/comparison.json.

    python3 src/reference_eval.py --root . --out artifacts/results/reference_eval_20261022.json
"""
import argparse
import json
from pathlib import Path
import statistics

import protocol

MEMBERS = ('sol61', 'opus55', 'argon4')
REST = 'runs/reference_rest_run01_20261021'
REST_FILES = {'sol61': 'sol61_direct/results*.jsonl', 'opus55': 'opus55_batch/results.jsonl',
              'argon4': 'argon4/E0/*/reference_rest.jsonl'}
QWEN = 'runs/qwen38_b0_run0{}_20261022/qwen38/B0/{}/test.jsonl'


def predictions(paths):
    out = {}
    for p in paths:
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get('prediction') is not None:
                out[(r['benchmark'], r['pair_id'])] = r['prediction']
    return out


def f1(pred, labels):
    tp = sum(pred.get(k) == 1 and y == 1 for k, y in labels.items())
    fp = sum(pred.get(k) == 1 and y == 0 for k, y in labels.items())
    fn = sum(pred.get(k) != 1 and y == 1 for k, y in labels.items())
    return 100*2*tp/(2*tp+fp+fn)


def evaluate(root):
    root = Path(root)
    labels = {b: {(b, c['pair_id']): c['label'] for c in json.loads((root/'data'/b/'test.json').read_text())}
              for b in protocol.BENCHMARKS}
    every = {k for d in labels.values() for k in d}
    pred = {m: predictions(sorted((root/REST).glob(REST_FILES[m])) +
                           sorted((root/'runs/escalation_run01_20261020'/m/'E0').glob('*/escalation.jsonl')))
            for m in MEMBERS}
    for m, d in pred.items():
        assert set(d) == every, (m, len(every - set(d)))
    pred['majority'] = {k: int(sum(pred[m][k] for m in MEMBERS) >= 2) for k in every}
    ensemble = {}
    for m, d in pred.items():
        ensemble[m] = {b: round(f1(d, labels[b]), 2) for b in protocol.BENCHMARKS}
        ensemble[m]['macro'] = round(statistics.mean(ensemble[m][b] for b in protocol.BENCHMARKS), 2)
    runs = {}
    for r in (1, 2, 3):
        runs[r] = {}
        for b in protocol.BENCHMARKS:
            d = predictions([root/QWEN.format(r, b)])
            assert set(d) == set(labels[b]), (r, b)
            runs[r][b] = f1(d, labels[b])
    qwen = {b: {'runs': [round(runs[r][b], 2) for r in runs], 'mean': round(statistics.mean(runs[r][b] for r in runs), 2),
                'sd': round(statistics.stdev(runs[r][b] for r in runs), 2)} for b in protocol.BENCHMARKS}
    macro = [statistics.mean(runs[r].values()) for r in runs]
    qwen['macro'] = {'runs': [round(x, 2) for x in macro], 'mean': round(statistics.mean(macro), 2),
                     'sd': round(statistics.stdev(macro), 2)}
    check = json.loads((root/REST/'opus55_live_sample1000/comparison.json').read_text())['overall']
    return {'pairs': len(every), 'ensemble_f1': ensemble, 'qwen38_b0_f1': qwen, 'opus_batch_vs_live_sample': check}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    result = evaluate(a.root.resolve())
    text = json.dumps(result, indent=1)+'\n'
    if a.out:
        (a.root/a.out).write_text(text)
    print(text)
