"""10 retrieved examples for the validation split of the Sol-labelled 2,000 set (Aaron, 5 Nov, relayed: "Lauf freigeben").

Purpose: PPLX v1.2 with RULESFS10_2000 on the 3,200 validation-split pairs of the 2,000 set (400 per benchmark,
configs/labeled_chapter_sets_20261028.json), so that the cascade threshold of that configuration can be tuned on Sol
labels (src/solvalid_rulesfs.py). The validation split is disjoint from the train split that the examples and the
Astra rules come from (the rule agents saw it as their held-out check, input/valid.json).
retrieve: exactly as labeled_valid_prep_20261101.retrieve / fewshot_labeled_v2_20261031.py: BM25 + embedding cosine,
RRF k=60, at least 2 per class, the same duplicate filter, examples from the train split of the 2,000 set only. The query
case is the pair of data/<b>/solset.json (records of the Sol request); embeddings of the labelling run (all set records
have one, no embedding call). Writes configs/labeled_chapter_fewshot_solvalid_20261105.json (write-once, layout of the
v2 file: benchmarks.<b>.2000.k10.<pair_id> = [train pair ids]). Free.

    /path/to/envs/ditto-modern/bin/python labeled_solvalid_prep_20261105.py retrieve
"""
import json
import sys
from pathlib import Path

ROOT = Path('/path/to/decision-em')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
import labeled_chapter_20261028 as lc  # noqa: E402
import fewshot_labeled_20261028 as v1  # noqa: E402
import fewshot_labeled_v2_20261031 as v2  # noqa: E402

OUT = ROOT/'configs/labeled_chapter_fewshot_solvalid_20261105.json'
SIZE = 2000


def query_cases(b):
    """Validation split of the 2,000 set as solset cases, in split-file order."""
    split = lc.manifest_rows(b, SIZE, 'valid')
    cases = {c['pair_id']: c for c in json.loads((ROOT/'data'/b/'solset.json').read_text())}
    out = [cases[r['pair_id']] for r in split]
    assert all(c['label'] == r['label'] for c, r in zip(out, split)), b
    return out


def retrieve(benchmarks):
    import numpy as np
    import protocol
    assert not OUT.exists(), 'retrieval is write-once'
    out = {'k': v2.K, 'sizes': [SIZE], 'rrf_k': v1.RRF_K, 'min_class_share': v1.MIN_SHARE,
           'split': 'validation split of the Sol-labelled 2,000 set (queries); examples from its train split',
           'filter': 'drop candidates with the pair entity ids or content-identical records (either order)', 'benchmarks': {}}
    for b in benchmarks:
        rec = lc.Records(b)
        cases = query_cases(b)
        train = lc.manifest_rows(b, SIZE, 'train')
        assert not {c['pair_id'] for c in cases} & {r['pair_id'] for r in train}, b
        labels = np.array([r['label'] for r in train])
        bm = v1.BM25([v1.tokens(protocol.state(v1.case_of(r, b), 'B0')) for r in train])
        content = [(v2.canon(v1.case_of(r, b)['records']['left']), v2.canon(v1.case_of(r, b)['records']['right'])) for r in train]
        ids = [(str(r['id_left']), str(r['id_right'])) for r in train]

        def vectors(pairs):
            got = rec.keys(pairs)
            lc.require(got is not None, f'{b}: embedding missing')
            keys, _, idx, emb = got
            res = {}
            for s, col in (('left', 0), ('right', 1)):
                v = emb[s][[idx[s][k[col]] for k in keys]]
                res[s] = v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            return res
        tv = vectors([(r['id_left'], r['id_right']) for r in train])
        sel, report = {}, {'train_pairs': len(train), 'candidates_dropped': 0, 'pairs_with_drop': 0}
        for c in cases:
            s_tok = v1.ranks(bm.scores(v1.tokens(protocol.state(c, 'B0'))))
            qv = vectors([(c['entity_ids']['left'], c['entity_ids']['right'])])
            cos = (tv['left']@qv['left'][0]+tv['right']@qv['right'][0])/2
            fused = 1/(v1.RRF_K+s_tok)+1/(v1.RRF_K+v1.ranks(cos))
            tl, tr = v2.canon(c['records']['left']), v2.canon(c['records']['right'])
            ti = (str(c['entity_ids']['left']), str(c['entity_ids']['right']))
            drop = {i for i in range(len(train)) if ids[i] == ti or content[i] in ((tl, tr), (tr, tl))}
            keep = np.array([i for i in range(len(train)) if i not in drop])
            chosen = [int(keep[j]) for j in v1.select(fused[keep], labels[keep], v2.K)]
            sel[c['pair_id']] = [train[i]['pair_id'] for i in chosen]
            report['candidates_dropped'] += len(drop); report['pairs_with_drop'] += bool(drop)
            assert len(chosen) == v2.K and min((labels[chosen] == y).sum() for y in (0, 1)) >= 2, (b, c['pair_id'])
        out['benchmarks'][b] = {str(SIZE): {'k10': sel, 'pairs': len(cases), **report}}
        print(b, report, flush=True)
    OUT.write_text(json.dumps(out)+'\n')


if __name__ == '__main__':
    import protocol
    {'retrieve': retrieve}[sys.argv[1]](list(protocol.BENCHMARKS))
