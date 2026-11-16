"""Labelled-data chapter: B0 plus learned rules or retrieved examples for the decision models Jev and PPLX, full test sets.

RULES_<n>: B0 instruction + the Astra rules learned from the train split of the Sol-B0 set of size n
(configs/labeled_chapter_rules_20261028.json), in the format of the old Astra-rules prompt
(expanded_rules.payload: BASE + header line + one "- rule" line per rule).
FS<k>: B0 instruction + k labelled example pairs from the train split of the 4000 set, retrieved per
test pair (artifacts/operations/fewshot_labeled_20261028.py, configs/labeled_chapter_fewshot_20261028.json),
each serialized like the test pair (protocol.state, B0) with its Sol-B0 label. The Decisions API has
no examples field, so the examples are part of the noul instructions.
Serialization of the test pair, answer schema and model versions are those of B0.
"""
import json
import threading
from pathlib import Path

import prompt12
import protocol

PROJECT = Path(__file__).resolve().parents[1]
RULES_PATH = PROJECT/'configs/labeled_chapter_rules_20261028.json'
FEWSHOT_PATH = PROJECT/'configs/labeled_chapter_fewshot_20261028.json'
SETS_PATH = PROJECT/'configs/labeled_chapter_sets_20261028.json'
KS = [10, 20]
RUN_MODELS = ['jev', 'pplx_v12']
SIZES = [200, 2000, 4000]
VARIANTS = ([(f'RULES_{n}', f'B0 + Astra rules from the {n} set', 'rules') for n in SIZES]
            + [(f'FS{k}', f'B0 + {k} retrieved examples from the 4000 set', 'fewshot') for k in KS])
VARIANT_IDS = [v for v, _, _ in VARIANTS]
HEADER = '\nGeneral rules derived from a separate labeled training sample:\n'
EXAMPLES = '\nLabeled example pairs from a separate training sample (the records to decide are given separately):'
RULES, FEWSHOT, TRAIN = {}, {}, {}
FEWSHOT_LOCK = threading.Lock()


def register_models():
    prompt12.register_models()


def run_variants(alias, group='rules'):
    """Variants of a runner group: 'rules' (RULES_200/2000/4000) or 'fewshot' (FS10/FS20)."""
    return [v for v in VARIANT_IDS if v.startswith('RULES_' if group == 'rules' else 'FS')]


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def reused(root=PROJECT):
    return {}


def load_rules(path=RULES_PATH):
    RULES.clear(); RULES.update(json.loads(Path(path).read_text())['rules'])


def load_fewshot(path=FEWSHOT_PATH):
    """Fills TRAIN before FEWSHOT: a non-empty FEWSHOT means both are complete (runner threads check it)."""
    import gzip
    sets = json.loads(SETS_PATH.read_text())['benchmarks']
    train = {}
    for b, info in sets.items():
        with gzip.open(PROJECT/info['4000']['files']['train']['path'], 'rt') as f:
            train[b] = {r['pair_id']: r for r in map(json.loads, f)}
    FEWSHOT.clear(); TRAIN.clear(); TRAIN.update(train)
    FEWSHOT.update(json.loads(Path(path).read_text())['benchmarks'])


def example_case(row, benchmark):
    return {'benchmark': benchmark, 'records': {s: {k[:-len(s)-1]: v for k, v in sorted(row.items())
                                                    if k.endswith('_'+s) and k != 'id_'+s} for s in ('left', 'right')}}


def instruction_fewshot(case, variant):
    if not FEWSHOT:
        with FEWSHOT_LOCK:  # runner threads: load once, the others wait
            if not FEWSHOT:
                load_fewshot()
    b = case['benchmark']
    if case.get('pair_id') == 'template':  # runner's request-template digest: structure only, examples come from the frozen retrieval file
        return protocol.BASE + EXAMPLES
    ids = FEWSHOT[b]['k'+variant[2:]][case['pair_id']]
    parts = [protocol.BASE + EXAMPLES]
    for i, pid in enumerate(ids, 1):
        row = TRAIN[b][pid]
        parts.append(f"\nExample {i} ({'match' if row['label'] else 'no match'}):\n" + protocol.state(example_case(row, b), 'B0'))
    return '\n'.join(parts)


def instruction(benchmark, variant):
    if not RULES:
        load_rules()
    size = variant.split('_')[1]
    return protocol.BASE + HEADER + '\n'.join('- '+r for r in RULES[benchmark][size])


def payload(case, alias, variant):
    assert variant in VARIANT_IDS, variant
    body = protocol.payload(case, alias, 'B0')
    text = instruction_fewshot(case, variant) if variant.startswith('FS') else instruction(case['benchmark'], variant)
    if 'questions' in body:
        body['questions']['match']['instructions'] = text
    else:
        body['messages'][0]['content'] = text
    return body
