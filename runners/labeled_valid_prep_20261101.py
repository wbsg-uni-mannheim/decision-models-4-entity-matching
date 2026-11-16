"""Labelled-data chapter, validation runs for per-benchmark selection (Aaron, 1 Nov, relayed: "Freiggabe erteill bitte
launchen es gibt doch um 11 usd richtig"; embeddings for the 5,744 validation records and PPLX v1.2 on validation).

Validation pool: the official validation sets without the rules_train pairs (data/<b>/valid.json minus
data/<b>/rules_train.json; 30,080 pairs), the pool of the B0 threshold tuning (cascade_b0_valsel_eval.validation).

plan:     embedding requests for the validation records without embedding (text format of labeled_chapter_20261028.embed_plan),
          runs/labeled_chapter_20261028/embeddings/requests_valid/<b>_<side>.jsonl.
embed:    paid (text-embedding-3-small, ~0.02 USD), approval_labeled_valid_embed_aaron_20261101.json; writes
          runs/labeled_chapter_20261028/embeddings/<b>_<side>_valid.npz (write-once; the test npz files stay untouched).
retrieve: 10 examples per validation pair from the train split of the 200 and 2000 sets, exactly as
          fewshot_labeled_v2_20261031.py on test (BM25 + embedding cosine, RRF k=60, at least 2 per class, same duplicate
          filter); writes configs/labeled_chapter_fewshot_valid_20261101.json (write-once, same format as the v2 file).

    /path/to/envs/ditto-modern/bin/python labeled_valid_prep_20261101.py plan|embed|retrieve
"""
import json
import sys
from pathlib import Path

ROOT = Path('/path/to/decision-em')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations')]
import labeled_chapter_20261028 as lc  # noqa: E402
import fewshot_labeled_20261028 as v1  # noqa: E402
import fewshot_labeled_v2_20261031 as v2  # noqa: E402

EMB = ROOT/lc.OUT/'embeddings'
REQUESTS = EMB/'requests_valid'
APPROVAL = ROOT/'artifacts/operations/approval_labeled_valid_embed_aaron_20261101.json'
OUT = ROOT/'configs/labeled_chapter_fewshot_valid_20261101.json'


class Records(lc.Records):
    """Test embeddings (<b>_<side>.npz) plus the validation embeddings (<b>_<side>_valid.npz)."""

    def load_new(self):
        new = super().load_new()
        for s in ('left', 'right'):
            p = EMB/f'{self.b}_{s}_valid.npz'
            if p.exists():
                z = self.np.load(p)
                new.setdefault(s, {}).update(zip(z['ids'].tolist(), z['vectors']))
        return new


def valid_cases(b):
    skip = {r['pair_id'] for r in json.loads((ROOT/'data'/b/'rules_train.json').read_text())}
    return [c for c in json.loads((ROOT/'data'/b/'valid.json').read_text()) if c['pair_id'] not in skip]


def plan(benchmarks):
    REQUESTS.mkdir(parents=True, exist_ok=True)
    total = 0
    for b in benchmarks:
        rec = Records(b)
        spec = lc.embed_spec(b, rec)
        miss = rec.missing([(c['entity_ids']['left'], c['entity_ids']['right']) for c in valid_cases(b)])
        info = {}
        for s in ('left', 'right'):
            rows = [{'custom_id': f'{s}-{i}', 'method': 'POST', 'url': '/v1/embeddings',
                     'body': {'model': lc.EMB_MODEL, 'dimensions': lc.EMB_DIM, 'input': lc.embed_text(spec, rec, s, i)}}
                    for i in sorted(miss[s]-set(rec.new.get(s, {})))]
            (REQUESTS/f'{b}_{s}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
            info[s] = [len(rows), sum(len(r['body']['input']) for r in rows)]
            total += len(rows)
        print(b, 'text spec', json.dumps(spec), 'missing (n, chars) L/R', info, flush=True)
    print('total records', total, flush=True)


def embed(benchmarks):
    import numpy as np
    from openai import OpenAI
    import astra_discovery_submit
    import reference_sol_direct_20261021 as ref
    a = json.loads(APPROVAL.read_text())
    lc.require(a.get('approved_by') == 'Aaron' and a.get('scope') == 'labeled_valid_embeddings' and a.get('quote'), 'Approval missing')
    budget = ref.Budget(ROOT, f"labeled-valid-embed-{int(__import__('time').time())}", float(a['run_cap_usd']), a['quote'], tier='default')
    client = OpenAI(api_key=astra_discovery_submit.credential(ROOT), timeout=120, max_retries=4)
    try:
        for b in benchmarks:
            for side in ('left', 'right'):
                out = EMB/f'{b}_{side}_valid.npz'
                reqs = [json.loads(x) for x in (REQUESTS/f'{b}_{side}.jsonl').read_text().splitlines()]
                if out.exists() or not reqs:
                    continue
                dims = lc.Records(b).emb[side].shape[1]  # wdc: 1536 (built without a dimensions parameter), others 256
                extra = {} if dims == 1536 else {'dimensions': dims}
                vecs = []
                for i in range(0, len(reqs), 64):
                    chunk = reqs[i:i+64]
                    lc.require(budget.admit(), 'Embedding run cap reached')
                    r = client.embeddings.create(model=lc.EMB_MODEL, input=[q['body']['input'] for q in chunk], **extra)
                    budget.settle(r.usage.total_tokens*0.02e-6)
                    vecs += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
                np.savez(out, ids=np.array([q['custom_id'].split('-', 1)[1] for q in reqs]), vectors=np.array(vecs, dtype=np.float32))
                print(b, side, len(vecs), flush=True)
    finally:
        budget.ledger('reconcile')
        print('embedding cost USD', round(budget.spent, 4), flush=True)


def retrieve(benchmarks):
    import numpy as np
    import protocol
    assert not OUT.exists(), 'retrieval is write-once'
    out = {'k': v2.K, 'sizes': v2.SIZES, 'rrf_k': v1.RRF_K, 'min_class_share': v1.MIN_SHARE, 'split': 'valid without rules_train',
           'filter': 'drop candidates with the pair entity ids or content-identical records (either order)', 'benchmarks': {}}
    for b in benchmarks:
        rec = Records(b)
        cases = valid_cases(b)
        out['benchmarks'][b] = {}
        for size in v2.SIZES:
            train = lc.manifest_rows(b, size, 'train')
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
                assert len(chosen) == v2.K and min((labels[chosen] == y).sum() for y in (0, 1)) >= 2, (b, size, c['pair_id'])
            out['benchmarks'][b][str(size)] = {'k10': sel, 'pairs': len(cases), **report}
            print(b, size, report, flush=True)
    OUT.write_text(json.dumps(out)+'\n')


if __name__ == '__main__':
    import protocol
    {'plan': plan, 'embed': embed, 'retrieve': retrieve}[sys.argv[1]](list(protocol.BENCHMARKS))
