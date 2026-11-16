"""Sol-labelled training sets for the labelled-data chapter: a 200 and a 4000 set per benchmark.

Method of the labelling paper (Steiner & Bizer; /path/to/Automatic-data-labeling), teacher
GPT-6.1 Sol for every label (the GPT-5.2 labels of the source runs only pick the pairs):
- 200 set: the similarity-search seed of the source run, cleaned and topped up to 100 pairs with
  the same per-query seed rule (_label_iteratively_per_query_seed, 30/70 target), plus one AL-ML
  round of 100 pairs (_run_active_learning_same_prompt: LR, RF, ET, HistGBDT, XGBoost, rule
  scorers; 20,000 pool candidates; top-5 committee variance).
- 4000 set: nested-profile selection (_subset_from_master logic) on the run's untrimmed label
  list in labelling order, seed first, positives at the share of the benchmark's small profile
  (dn7: medium, semi: large); fewer if not enough positives remain. Semi-HETER has only 1,880
  labelled pairs: they are continued with AL-ML rounds of 500 on the same pool up to 4,000.
- Every pair of the official test and validation sets (labelling-repo files and our data/<b>)
  is removed from seed, label lists and pool before any selection; pairs are deduplicated.
Teacher (v2, Aaron 28 Oct: "den gleichen Prompt ... wie in Table 1, also B0"): the Sol request of
the paper's reference runs, built by reference_rest.sol_direct_body (B0 system prompt, default
record serialization protocol.state, strict JSON schema {match: bool}, reasoning effort medium,
2,048 output tokens), standard tier instead of flex. Records come typed from the official
benchmark pair files (train first, then valid, test), as data/<b>/test.json was built, so a
request has exactly the form of a test request. Up to TRIES attempts per pair. The labelling
paper's own _label_pair (Listing 1) was used only by the first pilot (runs/labelsets_sol_20261028).
Every call goes through a cache (a pair is asked once) and a metered client (cost per call,
run cap, ledger commitment as reference_sol_direct_20261021).

Run with the labelling env (/path/to/envs/ditto-modern/bin/python).
Actions: select (offline), dry-run (all stages with an offline oracle instead of Sol, no API),
authorize-budget, label (stages known, topup, al, semi, export), close-stale.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/path/to/decision-em')
LAB = Path('/path/to/Automatic-data-labeling')
OUT = 'runs/labelsets_sol_b0_20261028'
DRY = 'runs/labelsets_sol_b0_dryrun_20261028'
MANIFEST = 'configs/labelsets_sol_b0_manifest_20261028.json'
APPROVAL = 'artifacts/operations/approval_labelsets_sol_b0_aaron_20261028.json'
TRIES = 6
MODEL = 'gpt-6.1-sol'
RUNS = {
    'abt-buy': 'output/by_method/active_learning_ditto/labelling_run/gpt/benchmark_abt-buy_20260323_202820',
    'dblp-acm': 'output/by_method/active_learning_ditto/labelling_run/gpt/benchmark_dblp-acm_20260323_202820',
    'dblp-scholar': 'output/by_method/active_learning_ditto/labelling_run/gpt/benchmark_dblp-scholar_20260323_202820',
    'walmart-amazon': 'output/by_method/active_learning_ditto/labelling_run/gpt/benchmark_walmart-amazon_20260323_202820',
    'wdc': 'output/three_phase_labeling_ditto_only_v2/benchmark_wdc_20260323_202820',
    'dn7-walmart-amazon': 'output/three_phase_labeling/benchmark_dn7-walmart-amazon_medium_embedding_20260902_013413',
    'semi-heter': 'output/three_phase_labeling/benchmark_semi-heter_large_embedding_20260902_013415',
    'billiger-de': 'output/three_phase_labeling/benchmark_billiger-de_medium_embedding_20260902_013436',
}
EXTEND = 'semi-heter'  # 4000 set by AL-ML continuation instead of selection
SHARE_PROFILE = {'dn7-walmart-amazon': 'medium', 'semi-heter': 'large'}
SEED_SIZE, SEED_POS, SEED_MAX_CALLS = 100, 30, 400
AL_ROUND, ACTIVE_CANDIDATES, TOP_MATCHERS = 100, 20000, 5
SET_SIZE, EXTEND_ROUND = 4000, 500
WORKERS = 16
RECORD_FILES = {'abt-buy': ['abt-buy-train.json', 'abt-buy-valid.csv', 'abt-buy-gs.json.gz'],
                'wdc': ['wdcproducts80cc20rnd000un_train_large.json.gz', 'wdcproducts80cc20rnd000un_valid_large.json.gz',
                        'wdcproducts80cc20rnd100un_gs.json.gz']}
STAGES = ['known', 'topup', 'al', 'semi', 'export']
PRICE = (2e-6, 1e-7, 1e-5)  # GPT-6.1 Sol standard: input, cached input, output per token
USAGE_KEYS = ['prompt_tokens', 'completion_tokens', 'total_tokens'] + [f'{s}_{k}' for s in ('seed', 'active')
                                                                     for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write_json(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(value, indent=1, sort_keys=True)+'\n'); tmp.replace(p)


def lab_modules():
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'artifacts/operations'), str(LAB/'scripts/labeling')]
    import active_learning_ml as alm
    import pandas as pd
    import yaml
    return alm, pd, yaml


# ------------------------------------------------------------------ data

def bad_pairs(b, pd, cfg):
    """Official test and validation pairs (source ids), from the labelling repo and data/<b>."""
    out = set()
    for key in ('test_path', 'valid_path'):
        p = LAB/cfg[key]
        if p.suffix == '.csv':
            df = pd.read_csv(p, dtype=str)
            out |= set(zip(df.id_left, df.id_right)) if 'id_left' in df else {tuple(x.split('#', 1)) for x in df.pair_id}
        else:
            df = pd.read_json(p, lines=True, compression='gzip', dtype=False)
            out |= set(zip(df.id_left.astype(str), df.id_right.astype(str)))
    for split in ('test', 'valid'):
        out |= {tuple(c['pair_id'].split('#', 1)) for c in read(ROOT/'data'/b/(split+'.json'))}
    return out


def typed_records(b):
    """{side: {record id: record}} with the fields, value types and key order of data/<b>; first file wins."""
    import prepare
    from protocol import BENCHMARKS
    fields = sorted(BENCHMARKS[b]['fields'])  # data/<b>/*.json store records with sorted keys
    out = {'left': {}, 'right': {}}
    for name in RECORD_FILES.get(b, [f'{b}-train.json.gz', f'{b}-valid.json.gz', f'{b}-gs.json.gz']):
        p = LAB/'benchmarks'/b/name
        if not p.exists():
            continue
        text = gzip.open(p, 'rt').read() if p.suffix == '.gz' else None
        if text is not None and text.lstrip().startswith('{') and '\n{' not in text.strip():
            data = json.loads(text)  # pandas orient=columns
            rows = [{k: v[i] for k, v in data.items()} for i in next(iter(data.values()))]
        else:
            rows = prepare.read_rows(p)
        if rows and set(rows[0]) == {'pair_id'}:
            continue
        for row in rows:
            if b == 'abt-buy':
                for s in ('left', 'right'):
                    if 'name_'+s not in row and 'title_'+s in row:
                        row['name_'+s] = row['title_'+s]
            for s in ('left', 'right'):
                rec = {f: prepare.clean(row.get(f+'_'+s)) for f in fields}
                if b == 'billiger-de':  # our billiger test records hold strings only ('' for missing, '156.31')
                    rec = {f: '' if v is None else repr(v) if isinstance(v, float) else v for f, v in rec.items()}
                out[s].setdefault(str(row['id_'+s]), rec)
    return out


class Data:
    """Records, embeddings and cleaned label lists of one source run (internal ids L:i / R:j)."""

    def __init__(self, b, alm, pd, cfg):
        import numpy as np
        self.b, self.run = b, LAB/RUNS[b]
        self.left = alm._load_df(self.run/'canonical/left.csv', 'L', {})
        self.right = alm._load_df(self.run/'canonical/right.csv', 'R', {})
        emb = LAB/cfg['embeddings']['dir']
        self.left_emb = np.load(emb/cfg['embeddings']['left_file']).astype(np.float32)
        self.right_emb = np.load(emb/cfg['embeddings']['right_file']).astype(np.float32)
        require(len(self.left) == len(self.left_emb) and len(self.right) == len(self.right_emb), b+': embedding rows')
        self.left_map = {r['__rid']: r for r in self.left.to_dict('records')}
        self.right_map = {r['__rid']: r for r in self.right.to_dict('records')}
        self.left_idx = {r: i for i, r in enumerate(self.left['__rid'])}
        self.right_idx = {r: i for i, r in enumerate(self.right['__rid'])}
        self.fields = [f for f in cfg['fields'] if f in self.left.columns and f in self.right.columns]
        self.bad = bad_pairs(b, pd, cfg)
        self.typed = typed_records(b)
        lid, rid = dict(zip(self.left['__rid'], self.left['id'])), dict(zip(self.right['__rid'], self.right['id']))

        def load(name):
            df = pd.read_csv(self.run/name, dtype={'id1': str, 'id2': str, 'rid1': str, 'rid2': str})
            require(all(lid[a] == x for a, x in zip(df.rid1, df.id1)) and all(rid[a] == x for a, x in zip(df.rid2, df.id2)),
                    f'{b}/{name}: internal ids do not match source ids')
            df['bad'] = [(x, y) in self.bad for x, y in zip(df.id1, df.id2)]
            df['gpt'] = df['label'].astype(str).str.upper().isin(['TRUE', '1']) if 'label' in df else None
            return df
        self.seed_raw = load('seed_labels.csv')
        self.master_raw = load('active_labels_latest.csv')
        self.pool_raw = load('faiss_candidates.csv')
        self.seed = self.seed_raw[~self.seed_raw.bad].drop_duplicates(['id1', 'id2']).reset_index(drop=True)
        self.master = self.master_raw[~self.master_raw.bad].drop_duplicates(['id1', 'id2']).reset_index(drop=True)
        self.pool = self.pool_raw[~self.pool_raw.bad].drop_duplicates(['id1', 'id2']).reset_index(drop=True)
        for name, df in (('seed', self.seed), ('labels', self.master), ('pool', self.pool)):
            require(set(df.id1) <= self.typed['left'].keys() and set(df.id2) <= self.typed['right'].keys(),
                    f'{b}/{name}: record without typed source')

    def body(self, left, right):
        """Sol request of the reference runs for one pair (standard tier)."""
        import reference_rest
        case = {'benchmark': self.b, 'pair_id': f"{left['id']}#{right['id']}",
                'records': {'left': self.typed['left'][str(left['id'])], 'right': self.typed['right'][str(right['id'])]}}
        return {**reference_rest.sol_direct_body(case), 'service_tier': 'default'}

    def internal(self, df):
        """Frame keyed by internal ids, as the labelling functions expect."""
        out = df[['rid1', 'rid2', 'similarity']].rename(columns={'rid1': 'id1', 'rid2': 'id2'}).copy()
        out['similarity'] = out['similarity'].astype(float)
        return out


def selection(d, cfg):
    """Seed (cleaned) and the 4000 base set; GPT labels only order and stratify."""
    seed = [[r.rid1, r.rid2, r.id1, r.id2, bool(r.gpt)] for r in d.seed.itertuples()]
    seed_keys = {(s[0], s[1]) for s in seed}
    order = list(d.seed.itertuples()) + [r for r in d.master.itertuples() if (r.rid1, r.rid2) not in seed_keys]
    prof = cfg['profiles'][SHARE_PROFILE.get(d.b, 'small')]
    share = prof['target_pos']/prof['target_total']
    if d.b == EXTEND:
        base = order
        target = {'share': share, 'pos': None, 'neg': None}
    else:
        pos_avail = sum(bool(r.gpt) for r in order)
        n_pos = min(round(SET_SIZE*share), pos_avail)
        n_neg = SET_SIZE-n_pos
        require(sum(not r.gpt for r in order) >= n_neg, d.b+': not enough negatives')
        pos = [r for r in order if r.gpt][:n_pos]; neg = [r for r in order if not r.gpt][:n_neg]
        keep = {(r.rid1, r.rid2) for r in pos+neg}
        base = [r for r in order if (r.rid1, r.rid2) in keep]
        target = {'share': share, 'pos': n_pos, 'neg': n_neg}
    base_keys = {(r.rid1, r.rid2) for r in base}
    return {'benchmark': d.b, 'run': RUNS[d.b], 'seed': seed,
            'base': [[r.rid1, r.rid2, r.id1, r.id2, bool(r.gpt)] for r in base], 'target': target,
            'counts': {'seed_raw': len(d.seed_raw), 'seed_bad': int(d.seed_raw.bad.sum()), 'seed_clean': len(d.seed),
                       'labels_raw': len(d.master_raw), 'labels_bad': int(d.master_raw.bad.sum()), 'labels_clean': len(d.master),
                       'pool_raw': len(d.pool_raw), 'pool_bad': int(d.pool_raw.bad.sum()), 'pool_clean': len(d.pool),
                       'base': len(base), 'base_gpt_pos': sum(bool(r.gpt) for r in base),
                       'seed_in_base': seed_keys <= base_keys},
            'feature_fields': d.fields}


# ------------------------------------------------------------- labelling

class Budget:
    """Run cap and ledger commitment (direct_openai_commitments), as reference_sol_direct_20261021."""

    def __init__(self, cap, quote, commitment):
        sys.path.insert(0, str(ROOT/'artifacts/operations'))
        import reference_sol_direct_20261021 as ref
        self.inner = ref.Budget(ROOT, commitment, cap, quote, tier='default')

    def admit(self):
        return self.inner.admit(hold=.005)

    def settle(self, amount):
        self.inner.settle(amount)

    def close(self):
        self.inner.ledger('reconcile')
        return self.inner.spent


class MeteredClient:
    """OpenAI client stand-in: one journal row and one cost per call; refuses beyond the run cap."""

    def __init__(self, key, budget, journal):
        from openai import OpenAI
        self.real = OpenAI(api_key=key, timeout=120, max_retries=2)
        self.budget, self.journal, self.lock = budget, journal, threading.Lock()
        self.chat = self; self.completions = self

    def create(self, **kwargs):
        require(kwargs.get('model') == MODEL, 'Unexpected model')
        if not self.budget.admit():
            raise RuntimeError('Run cap reached')
        amount = 0.
        try:
            resp = self.real.chat.completions.create(**kwargs)
            u = resp.usage
            cached = getattr(getattr(u, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
            amount = (u.prompt_tokens-cached)*PRICE[0]+cached*PRICE[1]+u.completion_tokens*PRICE[2]
            row = {'utc': now(), 'model': resp.model, 'service_tier': getattr(resp, 'service_tier', None),
                   'usage': u.model_dump(), 'cost_usd': amount,
                   'content': resp.choices[0].message.content if resp.choices else None}
            return resp
        except Exception as exc:
            row = {'utc': now(), 'error': type(exc).__name__, 'message': str(exc)[:300], 'cost_usd': 0.}
            raise
        finally:
            self.budget.settle(amount)
            with self.lock, self.journal.open('a') as f:
                f.write(json.dumps(row)+'\n')


class Labeler:
    """Patched _label_pair: every pair (internal ids) is asked once; answers go to labels.jsonl."""

    def __init__(self, alm, d, directory, client=None, oracle=None):
        self.alm, self.original, self.d = alm, alm._label_pair, d
        self.path, self.client, self.oracle = directory/'labels.jsonl', client, oracle
        self.lock, self.stage, self.cache = threading.Lock(), None, {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line); self.cache[(r['rid1'], r['rid2'])] = r
        alm._label_pair = self

    def __call__(self, client, model, left, right):
        key = (left['__rid'], right['__rid'])
        if key in self.cache:
            return self.cache[key]['label'], {k: 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        body = self.d.body(left, right)
        if self.oracle is not None:
            label, usage = self.oracle(key), {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        else:
            label, usage = self.teacher(body)
        row = {'rid1': key[0], 'rid2': key[1], 'id1': str(left['id']), 'id2': str(right['id']), 'label': label,
               'stage': self.stage, 'usage': usage, 'utc': now()}
        with self.lock:
            self.cache[key] = row
            with self.path.open('a') as f:
                f.write(json.dumps(row)+'\n')
        return label, usage

    def teacher(self, body):
        import reference_rest
        for attempt in range(TRIES):
            try:
                resp = self.client.chat.completions.create(**body)
                choice, = resp.choices
                require(resp.model.startswith(MODEL), 'Unexpected model '+resp.model)
                require(choice.finish_reason == 'stop' and not getattr(choice.message, 'refusal', None), 'Incomplete or refused answer')
                vote = reference_rest.vote(choice.message.content)
                u = resp.usage
                return ('TRUE' if vote else 'FALSE'), {'prompt_tokens': u.prompt_tokens, 'completion_tokens': u.completion_tokens,
                                                       'total_tokens': u.total_tokens}
            except Exception as exc:
                if attempt == TRIES-1 or 'Run cap' in str(exc):
                    raise
                time.sleep(min(2**attempt, 30))

    def label_all(self, d, pairs):
        todo = [p for p in pairs if (p[0], p[1]) not in self.cache]
        failed = []
        with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(self, None, MODEL, d.left_map[a], d.right_map[b]): (a, b) for a, b in todo}
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as exc:
                    failed.append([*futures[f], type(exc).__name__])
        return {'asked': len(todo), 'failed': failed}

    def frame(self, pd, d, keys):
        rows = [self.cache[k] for k in keys if k in self.cache]
        sim = dict(zip(zip(d.pool_raw.rid1, d.pool_raw.rid2), d.pool_raw.similarity.astype(float)))
        return pd.DataFrame([{'id1': r['rid1'], 'id2': r['rid2'], 'label': r['label'],
                              'similarity': sim.get((r['rid1'], r['rid2']), 0.)} for r in rows],
                            columns=['id1', 'id2', 'label', 'similarity'])


def oracle_for(d, sel):
    """Dry run only: GPT-5.2 label where the source run has one, else similarity >= 0.9."""
    known = {(r.rid1, r.rid2): bool(r.gpt) for r in d.master_raw.itertuples()}
    known.update({(r.rid1, r.rid2): bool(r.gpt) for r in d.seed_raw.itertuples()})
    sim = dict(zip(zip(d.pool_raw.rid1, d.pool_raw.rid2), d.pool_raw.similarity.astype(float)))
    return lambda k: 'TRUE' if known.get(k, sim.get(k, 0.) >= .9) else 'FALSE'


# ----------------------------------------------------------------- stages

def stage_known(lab, d, sel, info):
    pairs = {(s[0], s[1]) for s in sel['seed']} | {(s[0], s[1]) for s in sel['base']}
    lab.stage = 'known'
    info['known'] = lab.label_all(d, sorted(pairs))


def seed100(lab, sel, directory, pd):
    seed = [(s[0], s[1]) for s in sel['seed'] if (s[0], s[1]) in lab.cache]
    top = pd.read_csv(directory/'topup.csv', dtype=str) if (directory/'topup.csv').exists() else None
    return seed + ([] if top is None else list(zip(top.id1, top.id2)))


def stage_topup(lab, d, sel, info, directory, alm, pd):
    seed = [(s[0], s[1]) for s in sel['seed'] if (s[0], s[1]) in lab.cache]
    pos = sum(lab.cache[k]['label'] == 'TRUE' for k in seed)
    need_total, need_pos = SEED_SIZE-len(seed), max(SEED_POS-pos, 0)
    info['topup'] = {'seed_sol': len(seed), 'seed_sol_pos': pos, 'need_total': need_total, 'need_pos': need_pos}
    if need_total <= 0 or (directory/'topup.csv').exists():
        return
    used = {k[0] for k in seed} | {s[0] for s in sel['seed']}
    neighbors = d.internal(d.pool)
    neighbors = neighbors[~neighbors.id1.isin(used)].rename(columns={'id1': 'query_id', 'id2': 'neighbor_id'})
    lab.stage = 'topup'
    before = len(lab.cache)
    out = alm._label_iteratively_per_query_seed(
        None, MODEL, neighbors, d.left_map, d.right_map, target_positives_per_query=1, target_negatives_per_query=4,
        total_target_positives=need_pos, total_target_size=need_total, max_calls=SEED_MAX_CALLS, batch_size=5,
        query_order='random', bottom_k=2, state_path=directory/'topup_state.json', out_csv=directory/'topup.csv',
        usage_stats={k: 0 for k in USAGE_KEYS})
    out[['id1', 'id2', 'label', 'similarity']].to_csv(directory/'topup.csv', index=False)
    info['topup'].update(added=len(out), added_pos=int((out.label == 'TRUE').sum()), calls=len(lab.cache)-before)


def stage_al(lab, d, sel, info, directory, alm, pd):
    if (directory/'al100.csv').exists():
        return
    seed = seed100(lab, sel, directory, pd)
    labeled = lab.frame(pd, d, seed)
    lab.stage = 'al'
    before = len(lab.cache)
    out, summary = alm._run_active_learning_same_prompt(
        client=None, model=MODEL, labeled=labeled, candidates=d.internal(d.pool), left_map=d.left_map,
        right_map=d.right_map, left_idx=d.left_idx, right_idx=d.right_idx, left_emb=d.left_emb, right_emb=d.right_emb,
        feature_fields=d.fields, target_pos=10**9, target_neg=10**9, labels_per_iteration=AL_ROUND,
        active_candidates=ACTIVE_CANDIDATES, active_top_matchers=TOP_MATCHERS, max_iterations=1, llm_concurrency=WORKERS,
        label_stage='al_ml_round', max_total_labels_override=AL_ROUND, usage_stats={k: 0 for k in USAGE_KEYS})
    new = out[~out.set_index(['id1', 'id2']).index.isin(seed)]
    new[['id1', 'id2', 'label', 'similarity', 'al_source']].to_csv(directory/'al100.csv', index=False)
    info['al'] = {'added': len(new), 'added_pos': int((new.label == 'TRUE').sum()), 'calls': len(lab.cache)-before,
                  'rounds': summary['rounds']}


def stage_semi(lab, d, sel, info, directory, alm, pd):
    if d.b != EXTEND:
        return
    base = [(s[0], s[1]) for s in sel['base'] if (s[0], s[1]) in lab.cache]
    ext = [k for k, r in lab.cache.items() if r['stage'] == 'semi']
    labeled = lab.frame(pd, d, base+ext)
    missing = SET_SIZE-len(labeled)
    info['semi'] = {'base_sol': len(base), 'extension_before': len(ext), 'missing': missing}
    if missing <= 0:
        return
    lab.stage = 'semi'
    out, summary = alm._run_active_learning_same_prompt(
        client=None, model=MODEL, labeled=labeled, candidates=d.internal(d.pool), left_map=d.left_map,
        right_map=d.right_map, left_idx=d.left_idx, right_idx=d.right_idx, left_emb=d.left_emb, right_emb=d.right_emb,
        feature_fields=d.fields, target_pos=10**9, target_neg=10**9, labels_per_iteration=EXTEND_ROUND,
        active_candidates=ACTIVE_CANDIDATES, active_top_matchers=TOP_MATCHERS, max_iterations=20, llm_concurrency=WORKERS,
        label_stage='al_ml_extension', max_total_labels_override=missing, usage_stats={k: 0 for k in USAGE_KEYS})
    info['semi'].update(total_after=len(out), rounds=summary['rounds'])


def ditto_row(d, k, label):
    """Official pair format with the typed records of the request: id_<side>, <field>_<side>, label, pair_id."""
    ids = {'left': str(d.left_map[k[0]]['id']), 'right': str(d.right_map[k[1]]['id'])}
    row = {}
    for side, i in ids.items():
        row['id_'+side] = i
        row.update({f'{f}_{side}': v for f, v in d.typed[side][i].items()})
    row.update(label=int(label == 'TRUE'), pair_id=f"{ids['left']}#{ids['right']}")
    return row


def stage_export(lab, d, sel, info, directory, pd):
    set200 = seed100(lab, sel, directory, pd)
    if (directory/'al100.csv').exists():
        al = pd.read_csv(directory/'al100.csv', dtype=str); set200 += list(zip(al.id1, al.id2))
    if d.b == EXTEND:
        set4000 = [(s[0], s[1]) for s in sel['base']] + [k for k, r in lab.cache.items() if r['stage'] == 'semi']
    else:
        set4000 = [(s[0], s[1]) for s in sel['base']]
    gpt = {(s[0], s[1]): s[4] for s in sel['seed']+sel['base']}
    out = {}
    for name, keys in (('set200', set200), ('set4000', set4000)):
        keys = list(dict.fromkeys(keys))
        ok = [k for k in keys if k in lab.cache]
        src = [(lab.cache[k]['id1'], lab.cache[k]['id2']) for k in ok]
        require(not set(src) & d.bad, f'{d.b}/{name}: test or validation pair in set')
        require(len(set(src)) == len(src), f'{d.b}/{name}: duplicate source pair')
        rows = [ditto_row(d, k, lab.cache[k]['label']) for k in ok]
        with gzip.open(directory/f'{d.b}_{name}_train.json.gz', 'wt') as f:
            f.writelines(json.dumps(r, ensure_ascii=False, default=str)+'\n' for r in rows)
        both = [k for k in ok if k in gpt]
        out[name] = {'n': len(ok), 'unlabelled': len(keys)-len(ok), 'sol_pos': sum(r['label'] for r in rows),
                     'sol_pos_share': sum(r['label'] for r in rows)/max(len(rows), 1),
                     'gpt_agreement': sum((lab.cache[k]['label'] == 'TRUE') == gpt[k] for k in both)/max(len(both), 1),
                     'gpt_compared': len(both)}
    out['set200_in_set4000'] = len(set(set200) & set(set4000))
    info['export'] = out


# ------------------------------------------------------------------ driver

def load_all(benchmarks):
    alm, pd, yaml = lab_modules()
    cfg = yaml.safe_load((LAB/'configs/labeling/benchmarks_active.yaml').read_text())['benchmarks']
    return alm, pd, {b: (Data(b, alm, pd, cfg[b]), cfg[b]) for b in benchmarks}


def select(benchmarks):
    alm, pd, data = load_all(benchmarks)
    require(not (ROOT/MANIFEST).exists(), 'Manifest exists; selections are write-once')
    files = {}
    for b, (d, cfg) in data.items():
        sel = selection(d, cfg)
        path = ROOT/OUT/'selection'/f'{b}.json'
        require(not path.exists(), 'Selection exists: '+b)
        write_json(path, sel)
        files[b] = {'selection': str(path.relative_to(ROOT)), 'sha256': sha(path), 'counts': sel['counts'],
                    'target': sel['target'], 'source_sha256': {n: sha(LAB/RUNS[b]/n) for n in
                                                               ('seed_labels.csv', 'active_labels_latest.csv', 'faiss_candidates.csv')}}
        print(b, json.dumps(sel['counts']), json.dumps(sel['target']), flush=True)
    write_json(ROOT/MANIFEST, {'created_at': now(), 'model': MODEL, 'script_sha256': sha(__file__),
                               'labeling_module_sha256': sha(LAB/'scripts/labeling/active_learning_ml.py'),
                               'labeling_config_sha256': sha(LAB/'configs/labeling/benchmarks_active.yaml'),
                               'parameters': {'seed': [SEED_SIZE, SEED_POS, SEED_MAX_CALLS], 'al_round': AL_ROUND,
                                              'active_candidates': ACTIVE_CANDIDATES, 'top_matchers': TOP_MATCHERS,
                                              'set_size': SET_SIZE, 'extend_round': EXTEND_ROUND, 'workers': WORKERS},
                               'benchmarks': files})


def manifest_check():
    m = read(ROOT/MANIFEST)
    require(m['script_sha256'] == sha(__file__), 'Script changed after the manifest; re-select')
    require(m['labeling_module_sha256'] == sha(LAB/'scripts/labeling/active_learning_ml.py'), 'Labelling module drift')
    for b, f in m['benchmarks'].items():
        require(sha(ROOT/f['selection']) == f['sha256'], 'Selection drift: '+b)
    return m


def run(benchmarks, stages, dry):
    manifest_check()
    alm, pd, data = load_all(benchmarks)
    out = ROOT/(DRY if dry else OUT)
    report = {}
    budget = client = None
    if not dry:
        a = read(ROOT/APPROVAL)
        require(a.get('approved_by') == 'Aaron' and a.get('scope') == 'labelsets_sol' and a.get('quote'), 'Approval missing')
        spent = sum(r.get('actual_usd') or 0 for k, r in read(ROOT/'artifacts/budget.json').get('direct_openai_commitments', {}).items()
                    if k.startswith('labelsets-sol-'))
        sys.path.insert(0, str(ROOT/'artifacts/operations'))
        import astra_discovery_submit
        budget = Budget(a['run_cap_usd']-spent, a['quote'], f"labelsets-sol-{os.environ.get('SLURM_JOB_ID', 'local')}-{int(time.time())}")
        (out/'journal').mkdir(parents=True, exist_ok=True)
        client = MeteredClient(astra_discovery_submit.credential(ROOT), budget, out/'journal'/'calls.jsonl')
    try:
        for b, (d, cfg) in data.items():
            sel = read(ROOT/OUT/'selection'/f'{b}.json')
            directory = out/b; directory.mkdir(parents=True, exist_ok=True)
            lab = Labeler(alm, d, directory, client, oracle_for(d, sel) if dry else None)
            info = report.setdefault(b, {})
            t0 = time.time()
            if 'known' in stages: stage_known(lab, d, sel, info)
            if 'topup' in stages: stage_topup(lab, d, sel, info, directory, alm, pd)
            if 'al' in stages: stage_al(lab, d, sel, info, directory, alm, pd)
            if 'semi' in stages: stage_semi(lab, d, sel, info, directory, alm, pd)
            if 'export' in stages: stage_export(lab, d, sel, info, directory, pd)
            info['calls_total'] = len(lab.cache); info['seconds'] = round(time.time()-t0, 1)
            alm._label_pair = lab.original
            print(b, json.dumps(info, default=str), flush=True)
    finally:
        if budget is not None:
            report['_cost_usd'] = budget.close()
        write_json(out/f"report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", report)
    return report


def authorize_budget():
    """Raises the project cap to the approved value only (never lowers it)."""
    a = read(ROOT/APPROVAL)
    require(a.get('approved_by') == 'Aaron' and a.get('scope') == 'labelsets_sol', 'Approval missing')
    sys.path.insert(0, str(ROOT/'artifacts/operations'))
    import leased_budget
    with leased_budget.ledger(ROOT) as obj:
        require(a['project_cap_usd'] >= obj['project_cap_usd'], 'Cap may only increase')
        obj.setdefault('budget_authority_history', []).append(
            {'utc': now(), 'previous_cap_usd': obj['project_cap_usd'], 'previous_authority_sha256': obj.get('budget_authority_sha256'),
             'new_cap_usd': a['project_cap_usd'], 'approval': APPROVAL, 'approval_sha256': sha(ROOT/APPROVAL)})
        obj['project_cap_usd'] = a['project_cap_usd']; obj['budget_authority_sha256'] = sha(ROOT/APPROVAL)


def close_stale(commitment):
    """Reconcile a killed job's commitment from the call journal (lower bound: in-flight calls are lost)."""
    rows = [json.loads(x) for x in (ROOT/OUT/'journal'/'calls.jsonl').read_text().splitlines()]
    sys.path.insert(0, str(ROOT/'artifacts/operations'))
    import leased_budget
    with leased_budget.ledger(ROOT) as obj:
        row = obj['direct_openai_commitments'][commitment]
        require(row['status'] == 'reserved', 'Commitment not open')
        done = sum(r['actual_usd'] or 0 for k, r in obj['direct_openai_commitments'].items()
                   if k.startswith('labelsets-sol-') and r['status'] != 'reserved')
        spent = sum(r['cost_usd'] for r in rows)-done
        obj['reserved'] -= row['reserved_usd']; obj['spent'] += spent
        row.update(status='reconciled_stale', actual_usd=spent, closed_at=now())
    return spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--action', required=True, choices=['select', 'dry-run', 'authorize-budget', 'label', 'close-stale'])
    ap.add_argument('--benchmarks', nargs='+', default=list(RUNS))
    ap.add_argument('--stages', nargs='+', default=STAGES, choices=STAGES)
    ap.add_argument('--commitment')
    args = ap.parse_args()
    if args.action == 'select':
        select(args.benchmarks)
    elif args.action == 'dry-run':
        run(args.benchmarks, args.stages, dry=True)
    elif args.action == 'label':
        require(os.environ.get('SLURM_JOB_ID'), 'Paid labelling only inside a Slurm job')
        run(args.benchmarks, args.stages, dry=False)
    elif args.action == 'authorize-budget':
        authorize_budget()
    else:
        print(close_stale(args.commitment))


if __name__ == '__main__':
    main()
