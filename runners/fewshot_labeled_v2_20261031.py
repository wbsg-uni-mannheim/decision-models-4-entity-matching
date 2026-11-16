"""Labelled-data chapter, few-shot v2 (Aaron, 31 Oct: "1-3 Experimente bitte einmal laufen lassen"): 10 retrieved examples
from the train split of the 200 set (160 pairs) and of the 2000 set (1,600 pairs), with a duplicate filter.

Retrieval as fewshot_labeled_20261028.py (BM25 over the B0 serialization + mean left/right embedding cosine, RRF k=60,
at least 2 examples per class, ordered by fused rank). New: before selection, every candidate is dropped that is the test
pair itself (same left and right entity ids) or content-identical to it (both records equal as in the leak check of
claudecode_labeled_20261028.leak_check, in either order). Other offers of the same entity stay. Per benchmark and size
the file reports how many candidates the filter dropped, how many test pairs had one, and how many of them the unfiltered
retrieval would have placed among the 10 examples. Used by FS10_<n> and RULESFS10_<n> (src/labeled_chapter.py).

    python fewshot_labeled_v2_20261031.py   # writes configs/labeled_chapter_fewshot_v2_20261031.json (write-once)
"""
import json
import sys
from pathlib import Path

ROOT = Path('/path/to/decision-em')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
import fewshot_labeled_20261028 as v1  # noqa: E402

OUT = ROOT/'configs/labeled_chapter_fewshot_v2_20261031.json'
SIZES, K = [200, 2000], 10


def canon(rec):
    return json.dumps(rec, sort_keys=True, ensure_ascii=False)


def main():
    import numpy as np
    import protocol
    import labeled_chapter_20261028 as lc
    assert not OUT.exists(), 'retrieval is write-once'
    out = {'k': K, 'sizes': SIZES, 'rrf_k': v1.RRF_K, 'min_class_share': v1.MIN_SHARE,
           'filter': 'drop candidates with the test pair entity ids or content-identical records (either order)', 'benchmarks': {}}
    for b in protocol.BENCHMARKS:
        rec = lc.Records(b)
        tests = json.loads((ROOT/'data'/b/'test.json').read_text())
        out['benchmarks'][b] = {}
        for size in SIZES:
            train = lc.manifest_rows(b, size, 'train')
            labels = np.array([r['label'] for r in train])
            bm = v1.BM25([v1.tokens(protocol.state(v1.case_of(r, b), 'B0')) for r in train])
            content = [(canon(v1.case_of(r, b)['records']['left']), canon(v1.case_of(r, b)['records']['right'])) for r in train]
            ids = [(str(r['id_left']), str(r['id_right'])) for r in train]

            def vectors(pairs):
                keys, _, idx, emb = rec.keys(pairs)
                res = {}
                for s, col in (('left', 0), ('right', 1)):
                    v = emb[s][[idx[s][k[col]] for k in keys]]
                    res[s] = v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
                return res
            tv = vectors([(r['id_left'], r['id_right']) for r in train])
            sel, report = {}, {'train_pairs': len(train), 'candidates_dropped': 0, 'test_pairs_with_drop': 0,
                               'dropped_from_unfiltered_top10': 0, 'test_pairs_whose_top10_changed': 0}
            for c in tests:
                s_tok = v1.ranks(bm.scores(v1.tokens(protocol.state(c, 'B0'))))
                qv = vectors([(c['entity_ids']['left'], c['entity_ids']['right'])])
                cos = (tv['left']@qv['left'][0]+tv['right']@qv['right'][0])/2
                fused = 1/(v1.RRF_K+s_tok)+1/(v1.RRF_K+v1.ranks(cos))
                tl, tr = canon(c['records']['left']), canon(c['records']['right'])
                ti = (str(c['entity_ids']['left']), str(c['entity_ids']['right']))
                drop = {i for i in range(len(train)) if ids[i] == ti or content[i] in ((tl, tr), (tr, tl))}
                keep = np.array([i for i in range(len(train)) if i not in drop])
                chosen = [int(keep[j]) for j in v1.select(fused[keep], labels[keep], K)]
                unfiltered = v1.select(fused, labels, K)
                sel[c['pair_id']] = [train[i]['pair_id'] for i in chosen]
                report['candidates_dropped'] += len(drop); report['test_pairs_with_drop'] += bool(drop)
                report['dropped_from_unfiltered_top10'] += len(drop & set(unfiltered))
                report['test_pairs_whose_top10_changed'] += bool(drop & set(unfiltered))
                assert len(chosen) == K and min((labels[chosen] == y).sum() for y in (0, 1)) >= 2, (b, size, c['pair_id'])
            out['benchmarks'][b][str(size)] = {'k10': sel, 'test_pairs': len(tests), **report}
            print(b, size, report, flush=True)
    OUT.write_text(json.dumps(out)+'\n')


if __name__ == '__main__':
    main()
