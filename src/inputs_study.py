"""Input serialization (F1-F4) and robustness (R1-R6) on the full core test sets.

Each model and benchmark keeps the prompt with the highest F1 in the 12-prompt
study (ties: order of prompt12.PROMPTS, B0 first), as Peeters et al. Only the record text
(decision `state`, generative user message) changes. F0 is the selected prompt
with the B0 serialization and already exists in the prompt study.

Perturbations follow Ditto's augmentation operators (Li et al., VLDB 2020)
where one exists. They are deterministic: every random choice is seeded by the
pair ID, the record side and the variant.
"""
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import expanded_protocol
import prompt12
import protocol

PROJECT = Path(__file__).resolve().parents[1]
SELECTION = 'configs/inputs_prompt_selection_20261019.json'

VARIANTS = [
    ('F1_values', 'Values only, concatenated (Peeters et al.)', 'serialization'),
    ('F2_json', 'JSON object', 'serialization'),
    ('F3_aligned', 'Aligned attributes', 'serialization'),
    ('F4_no_missing', 'Attribute lines without missing values', 'serialization'),
    ('R1_swap', 'Records swapped (entry_swap)', 'robustness'),
    ('R2_attr_reverse', 'Attribute order reversed (attr_shuffle)', 'robustness'),
    ('R3_lowercase', 'Lowercased values', 'robustness'),
    ('R4_attr_del', 'One non-title attribute deleted per record (attr_del)', 'robustness'),
    ('R5_span_del', '1-2 title tokens deleted per record (span_del)', 'robustness'),
    ('R6_typos', 'Adjacent-character swap in 10% of title tokens, at least one', 'robustness'),
]
VARIANT_IDS = [v for v, _, _ in VARIANTS]
RUN_MODELS = prompt12.RUN_MODELS
TITLE = {'abt-buy': 'name', 'billiger-de': 'name'}  # all other benchmarks use 'title'
SELECTED = {}


def register_models():
    prompt12.register_models()


# Abt-Buy, DBLP-Scholar, Walmart-Amazon and DBLP-ACM are lowercase already: R3 would repeat F0 there.
LOWERCASE_ALREADY = ('abt-buy', 'dblp-scholar', 'walmart-amazon', 'dblp-acm')


def run_variants(alias):
    return list(VARIANT_IDS)


def benchmarks(variant):
    return [b for b in protocol.BENCHMARKS if not (variant == 'R3_lowercase' and b in LOWERCASE_ALREADY)]


def rng(case, side, variant):
    seed = hashlib.sha256(f"{case['pair_id']}|{side}|{variant}".encode()).hexdigest()
    return random.Random(int(seed[:16], 16))


def title_field(benchmark):
    return TITLE.get(benchmark, 'title')


def empty(value):
    return value is None or value == '' or value == []


def text(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value)) if abs(value) < 1e15 else str(value)
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def span_del(value, r):
    tokens = value.split()
    if len(tokens) < 4:
        return value
    width = r.choice([1, 2])
    start = r.randrange(0, len(tokens)-width+1)
    return ' '.join(tokens[:start]+tokens[start+width:])


def typos(value, r, rate=.1):
    """Swap two adjacent inner characters in max(1, round(rate*n)) of the n eligible title tokens."""
    tokens = value.split()
    eligible = [i for i, t in enumerate(tokens) if len(t) >= 4 and t.isalpha() and len(set(t[1:-1])) > 1]
    for i in r.sample(eligible, min(len(eligible), max(1, round(rate*len(eligible))))):
        t = tokens[i]
        j = r.choice([k for k in range(1, len(t)-2) if t[k] != t[k+1]] or [1])
        tokens[i] = t[:j]+t[j+1]+t[j]+t[j+2:]
    return ' '.join(tokens)


def perturb(case, variant):
    """Records after a robustness perturbation (None: serialization variant)."""
    records = {s: dict(case['records'][s]) for s in ('left', 'right')}
    field = title_field(case['benchmark'])
    for side, record in records.items():
        r = rng(case, side, variant)
        if variant == 'R3_lowercase':
            records[side] = {k: v.lower() if isinstance(v, str) else v for k, v in record.items()}
        elif variant == 'R4_attr_del':
            options = sorted(k for k, v in record.items() if k != field and not empty(v))
            if options:
                record.pop(r.choice(options))
        elif variant in ('R5_span_del', 'R6_typos') and isinstance(record.get(field), str):
            record[field] = span_del(record[field], r) if variant == 'R5_span_del' else typos(record[field], r)
    return records


def state(case, variant):
    assert variant in VARIANT_IDS, variant
    if variant == 'F1_values':
        return '\n'.join(f"Record {label}: '" + ' '.join(text(v) for v in case['records'][side].values() if not empty(v)) + "'"
                         for label, side in (('A', 'left'), ('B', 'right')))
    legacy = {'F2_json': 'X_R_json', 'F3_aligned': 'X_R_aligned', 'F4_no_missing': 'X_R_missing',
              'R1_swap': 'X_R_swap', 'R2_attr_reverse': 'X_R_order'}
    if variant in legacy:
        return expanded_protocol.representation(case, legacy[variant])
    return protocol.state({**case, 'records': perturb(case, variant)}, 'B0')


def load_selection(root=PROJECT):
    obj = json.loads((Path(root)/SELECTION).read_text())
    assert set(obj['selected']) == set(RUN_MODELS)
    assert all(set(per) == set(protocol.BENCHMARKS) and all(v in prompt12.VARIANTS for v in per.values())
               for per in obj['selected'].values())
    SELECTED.clear(); SELECTED.update(obj['selected'])
    return obj


def payload(case, alias, variant):
    if not SELECTED:
        load_selection()
    body = prompt12.payload(case, alias, SELECTED[alias][case['benchmark']])
    value = state(case, variant)
    if 'questions' in body:
        body['state'] = value
    else:
        body['messages'][1]['content'] = value
    return body


# --------------------------------------------------------------- prompt selection

def f1(rows, labels):
    tp = fp = fn = 0
    for pid, r in rows.items():
        y, p = labels[pid], r['prediction']
        tp += y == p == 1; fp += y == 0 and p == 1; fn += y == 1 and p == 0
    return 2*tp/(2*tp+fp+fn) if tp else 0.


def journal(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        rows[r['pair_id']] = r
    return rows


def prompt_scores(root, namespace='prompt12_run01_20261019'):
    """Test F1 of all twelve prompts per model and benchmark. B0 comes from the baseline
    table (Run 1) except for pplx_v12, which ran its own B0 in the prompt study."""
    root = Path(root)
    scores = {}
    for m in RUN_MODELS:
        for v in prompt12.VARIANTS:
            for b in protocol.BENCHMARKS:
                labels = {c['pair_id']: c['label'] for c in json.loads((root/'data'/b/'test.json').read_text())}
                own = v != 'B0' or m in prompt12.EXTRA_MODELS
                path = root/'runs'/(namespace if own else 'quality')/m/v/b/'test.jsonl'
                rows = journal(path)
                assert rows.keys() == labels.keys(), ('incomplete', m, v, b)
                scores.setdefault(m, {}).setdefault(v, {})[b] = f1(rows, labels)
    return scores


def select(scores):
    """Best prompt per model and benchmark (ties: order of prompt12.PROMPTS, B0 first)."""
    out = {'selected': {}, 'f1': scores, 'margin_over_b0': {}}
    for m, by_variant in scores.items():
        out['selected'][m] = {}
        out['margin_over_b0'][m] = {}
        for b in protocol.BENCHMARKS:
            best = max(prompt12.VARIANTS, key=lambda v: (by_variant[v][b], -prompt12.VARIANTS.index(v)))
            out['selected'][m][b] = best
            out['margin_over_b0'][m][b] = by_variant[best][b]-by_variant['B0'][b]
    return out
