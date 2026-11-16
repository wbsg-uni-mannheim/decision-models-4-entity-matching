"""Labelled-data chapter, part 3: few-shot example retrieval for Jev and PPLX (B0 + examples).

Per test pair, k in {10, 20} labelled examples from the train split of the 4000 set (3,200 pairs),
ranked by reciprocal-rank fusion (RRF, k_rrf=60) of
- BM25 over the serialized pair text (protocol.state, B0 serialization; tokens [a-z0-9]+), and
- embedding cosine: mean of the left-left and right-right cosines of the 256-d text-embedding-3-small
  record embeddings (labelling-run embeddings; new test-record embeddings from labeled_chapter embed).
Both classes are represented: at least k/5 examples of each class (swap in the best-ranked pairs of
the missing class for the lowest-ranked ones of the other). Examples are ordered by fused rank.

The Decisions API (OpenRouter /api/alpha/decisions) has no examples field; examples go into the
noul instructions after B0 (src/labeled_chapter.py, variants FS10 / FS20).

    python fewshot_labeled_20261028.py --action retrieve [--dry]   # --dry: BM25 only where an embedding is missing
    python fewshot_labeled_20261028.py --action estimate           # request sizes and cost from the B0 runs
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/path/to/decision-em')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
OUT = ROOT/'configs/labeled_chapter_fewshot_20261028.json'
DRY_OUT = ROOT/'runs/labeled_chapter_20261028/fewshot/retrieval_dry.json'
KS, RRF_K, MIN_SHARE = [10, 20], 60, .2
TOKEN = re.compile(r'[a-z0-9]+')


def tokens(text):
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=.75):
        self.docs = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]; self.avg = sum(self.len)/len(self.len)
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1+(n-f+.5)/(f+.5)) for t, f in df.items()}
        self.k1, self.b = k1, b
        self.post = {}
        for i, d in enumerate(self.docs):
            for t, f in d.items():
                self.post.setdefault(t, []).append((i, f))

    def scores(self, query):
        import numpy as np
        s = np.zeros(len(self.docs))
        for t in set(query):
            for i, f in self.post.get(t, ()):
                s[i] += self.idf[t]*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*self.len[i]/self.avg))
        return s


def case_of(row, b):
    return {'benchmark': b, 'records': {s: {k[:-len(s)-1]: v for k, v in sorted(row.items()) if k.endswith('_'+s) and k != 'id_'+s}
                                        for s in ('left', 'right')}}


def ranks(scores):
    import numpy as np
    order = np.argsort(-scores, kind='stable')
    r = np.empty(len(scores)); r[order] = np.arange(1, len(scores)+1)
    return r


def select(fused, labels, k):
    import numpy as np
    order = list(np.argsort(-fused, kind='stable'))
    top = order[:k]
    need = math.ceil(MIN_SHARE*k)
    for y in (0, 1):
        have = [i for i in top if labels[i] == y]
        if len(have) < need:
            extra = [i for i in order[k:] if labels[i] == y][:need-len(have)]
            drop = [i for i in reversed(top) if labels[i] != y][:len(extra)]
            top = [i for i in top if i not in drop]+extra
            top.sort(key=lambda i: -fused[i])
    return top


def retrieve(dry):
    import numpy as np
    import protocol
    import labeled_chapter_20261028 as lc
    sets = json.loads((ROOT/lc.MANIFEST).read_text())
    out = {'k': KS, 'rrf_k': RRF_K, 'min_class_share': MIN_SHARE, 'dry': dry, 'benchmarks': {}}
    for b in protocol.BENCHMARKS:
        train = lc.manifest_rows(b, 4000, 'train')
        rec = lc.Records(b)
        bm = BM25([tokens(protocol.state(case_of(r, b), 'B0')) for r in train])
        labels = [r['label'] for r in train]

        def vectors(pairs):
            got = rec.keys(pairs)
            if got is None:
                return None
            keys, _, idx, emb = got
            out = {}
            for s, col in (('left', 0), ('right', 1)):
                v = emb[s][[idx[s][k[col]] for k in keys]]
                out[s] = v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            return out
        tv = vectors([(r['id_left'], r['id_right']) for r in train])
        tests = json.loads((ROOT/'data'/b/'test.json').read_text())
        res = {f'k{k}': {} for k in KS}
        with_emb = 0
        for c in tests:
            s_tok = ranks(bm.scores(tokens(protocol.state(c, 'B0'))))
            qv = vectors([(c['entity_ids']['left'], c['entity_ids']['right'])])
            if qv is None:
                if not dry:
                    raise ValueError(f'{b}: test record without embedding; run labeled_chapter --action embed first')
                fused = 1/(RRF_K+s_tok)
            else:
                with_emb += 1
                cos = (tv['left']@qv['left'][0]+tv['right']@qv['right'][0])/2
                fused = 1/(RRF_K+s_tok)+1/(RRF_K+ranks(cos))
            for k in KS:
                res[f'k{k}'][c['pair_id']] = [train[i]['pair_id'] for i in select(fused, labels, k)]
        out['benchmarks'][b] = {**res, 'test_pairs': len(tests), 'with_embedding': with_emb}
        print(b, len(tests), 'with embedding', with_emb, flush=True)
    path = DRY_OUT if dry else OUT
    assert dry or not path.exists(), 'retrieval is write-once'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out)+'\n')


def estimate():
    """Instruction and state sizes per request for FS10/FS20; cost scaled from each model's B0 test cost per character."""
    import labeled_chapter as study
    import protocol
    study.register_models()
    study.load_fewshot(OUT if OUT.exists() else DRY_OUT)
    print('retrieval file:', OUT if OUT.exists() else DRY_OUT)
    B0_RUNS = {'jev': 'runs/quality', 'pplx_v12': 'runs/prompt12_run01_20261019'}
    total = {}
    for m in study.RUN_MODELS:
        for b in protocol.BENCHMARKS:
            tests = json.loads((ROOT/'data'/b/'test.json').read_text())
            cost = sum(sum(a['cost'] or 0 for a in json.loads(l)['attempts']) for l in open(ROOT/B0_RUNS[m]/m/'B0'/b/'test.jsonl'))
            base_chars = sum(len(protocol.BASE)+len(protocol.state(c, 'B0')) for c in tests)
            for k in KS:
                chars = sum(len(study.instruction_fewshot(c, f'FS{k}'))+len(protocol.state(c, 'B0')) for c in tests)
                usd = cost*chars/base_chars
                total[(m, k)] = total.get((m, k), 0)+usd
                print(m, b, f'k={k}', f'mean request chars {chars/len(tests):,.0f} (B0 {base_chars/len(tests):,.0f})', f'~{usd:.2f} USD', flush=True)
    for (m, k), v in sorted(total.items()):
        print('TOTAL', m, f'k={k}', round(v, 2), 'USD')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--action', required=True, choices=['retrieve', 'estimate'])
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()
    retrieve(args.dry) if args.action == 'retrieve' else estimate()


if __name__ == '__main__':
    main()
