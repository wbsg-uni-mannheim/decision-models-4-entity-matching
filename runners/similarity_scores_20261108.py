"""Surface-form similarity per record pair for the similarity-deferral baseline of Table VII (Chris, relayed 8 Nov). Offline, free.

For every test pair and every validation pair without rules_train (the pool of the validation thresholds) of the eight benchmarks:
cosine: cosine between the left and the right record embedding (text-embedding-3-small, the existing embeddings of
  labeled_chapter_20261028.Records: labelling-run embeddings plus <b>_<side>.npz and <b>_<side>_valid.npz; no new embedding call, a
  missing vector stops the script).
jaccard: Jaccard of the token sets [a-z0-9]+ of the two records' attribute values (fewshot_labeled_20261028.tokens).
No labels and no model outputs are read. Output: artifacts/results/similarity_scores_20261108.json,
{benchmark: {split: {pair_id: [cosine, jaccard]}}}, write-once.

    /path/to/envs/ditto-modern/bin/python similarity_scores_20261108.py
"""
import json
import sys
from pathlib import Path

ROOT = Path('/path/to/decision-em')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
import fewshot_labeled_20261028 as v1  # noqa: E402
import labeled_valid_prep_20261101 as vp  # noqa: E402
import protocol  # noqa: E402

OUT = ROOT/'artifacts/results/similarity_scores_20261108.json'


def record_tokens(record):
    return set(v1.tokens(' '.join(str(v) for v in record.values() if v is not None)))


def scores(cases, rec, np):
    pairs = [(c['entity_ids']['left'], c['entity_ids']['right']) for c in cases]
    got = rec.keys(pairs)
    assert got is not None, 'embedding missing'
    keys, _, idx, emb = got
    v = {}
    for s, col in (('left', 0), ('right', 1)):
        x = emb[s][[idx[s][k[col]] for k in keys]]
        v[s] = x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    cos = (v['left']*v['right']).sum(axis=1)
    out = {}
    for c, x in zip(cases, cos):
        a, b = record_tokens(c['records']['left']), record_tokens(c['records']['right'])
        out[c['pair_id']] = [float(x), len(a & b)/max(len(a | b), 1)]
    return out


def main():
    import numpy as np
    assert not OUT.exists(), OUT
    res = {}
    for b in protocol.BENCHMARKS:
        rec = vp.Records(b)
        res[b] = {'test': scores(json.loads((ROOT/'data'/b/'test.json').read_text()), rec, np), 'valid': scores(vp.valid_cases(b), rec, np)}
        print(b, {k: len(v) for k, v in res[b].items()}, flush=True)
    OUT.write_text(json.dumps(res)+'\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
