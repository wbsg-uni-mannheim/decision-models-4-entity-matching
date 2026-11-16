"""Labelled-data chapter: B0 plus learned rules or retrieved examples for the decision models Jev and PPLX, full test sets.

RULES_<n>: B0 instruction + the Astra rules learned from the train split of the Sol-B0 set of size n
(configs/labeled_chapter_rules_20261028.json), in the format of the old Astra-rules prompt
(expanded_rules.payload: BASE + header line + one "- rule" line per rule).
FS<k>: B0 instruction + k labelled example pairs from the train split of the 4000 set, retrieved per
test pair (artifacts/operations/fewshot_labeled_20261028.py, configs/labeled_chapter_fewshot_20261028.json),
each serialized like the test pair (protocol.state, B0) with its Sol-B0 label. The Decisions API has
no examples field, so the examples are part of the noul instructions.
Serialization of the test pair, answer schema and model versions are those of B0.
31 Oct (Aaron: "1-3 Experimente bitte einmal laufen lassen"):
- rules runs 2 and 3 (RULES_200 / RULES_2000) use the rules of their own Astra repetition
  (configs/labeled_chapter_rules_20261028_rep<n>.json), selected with use_rules_run(n);
- FS10_<n>: B0 + 10 examples from the train split of the n set (n = 200, 2000), retrieved with the duplicate filter
  (artifacts/operations/fewshot_labeled_v2_20261031.py, configs/labeled_chapter_fewshot_v2_20261031.json);
- RULESFS10_<n>: B0 + the run-1 Astra rules of the n set + the same 10 examples as FS10_<n>.
The variants of 28 Oct build byte-identical requests.
1 Nov (validation selection): group 'valid' runs RULES_<n>, FS10_<n> and RULESFS10_<n> (n = 200, 2000; run-1 rules) on the
official validation sets; use_split('valid') makes FS10_<n> / RULESFS10_<n> read the examples retrieved per validation
pair (artifacts/operations/labeled_valid_prep_20261101.py, configs/labeled_chapter_fewshot_valid_20261101.json).
Instruction text and request layout are those of the test runs; only the pair and its examples change.
4 Nov (Table 13 over three runs): runs 2 and 3 of FS10_<n> and RULESFS10_<n> (run-1 rules), and group 'rulesfsrep':
RULESFS10_<n> with the rules of Astra repetition 2 or 3 (use_rulesfs_rules(run); run 1 of that series is RULESFS10 run 1).
"""
import json
import threading
from pathlib import Path

import prompt12
import protocol

PROJECT = Path(__file__).resolve().parents[1]
RULES_PATH = PROJECT/'configs/labeled_chapter_rules_20261028.json'
FEWSHOT_PATH = PROJECT/'configs/labeled_chapter_fewshot_20261028.json'
FEWSHOT2_PATH = PROJECT/'configs/labeled_chapter_fewshot_v2_20261031.json'
FEWSHOT_VALID_PATH = PROJECT/'configs/labeled_chapter_fewshot_valid_20261101.json'
RULES_REP_PATH = 'configs/labeled_chapter_rules_20261028_rep{}.json'
SIZES2 = [200, 2000]
SETS_PATH = PROJECT/'configs/labeled_chapter_sets_20261028.json'
KS = [10, 20]
RUN_MODELS = ['jev', 'pplx_v12']
SIZES = [200, 2000, 4000]
VARIANTS = ([(f'RULES_{n}', f'B0 + Astra rules from the {n} set', 'rules') for n in SIZES]
            + [(f'FS{k}', f'B0 + {k} retrieved examples from the 4000 set', 'fewshot') for k in KS]
            + [(f'FS10_{n}', f'B0 + 10 retrieved examples from the {n} set, duplicate filter', 'fewshot2') for n in SIZES2]
            + [(f'RULESFS10_{n}', f'B0 + Astra rules (run 1) + 10 retrieved examples from the {n} set, duplicate filter',
                'rulesfs') for n in SIZES2])
VARIANT_IDS = [v for v, _, _ in VARIANTS]
HEADER = '\nGeneral rules derived from a separate labeled training sample:\n'
EXAMPLES = '\nLabeled example pairs from a separate training sample (the records to decide are given separately):'
RULES, FEWSHOT, TRAIN = {}, {}, {}
RULES1, FEWSHOT2, TRAIN2 = {}, {}, {}
RULES_RUN = [1]
SPLIT = ['test']
RULESFS_RULES = [1]
FEWSHOT_LOCK = threading.Lock()


def register_models():
    prompt12.register_models()


def run_variants(alias, group='rules'):
    """Variants of a runner group: 'rules' (RULES_200/2000/4000; runs 2 and 3 only 200/2000), 'fewshot' (FS10/FS20),
    'fewshot2' (FS10_200/2000), 'rulesfs' (RULESFS10_200/2000) or 'valid' (all six n = 200/2000 variants)."""
    if group == 'rules' and RULES_RUN[0] > 1:
        return [f'RULES_{n}' for n in SIZES2]
    if group == 'rulesfsrep':
        return [f'RULESFS10_{n}' for n in SIZES2]
    if group == 'valid':
        return [f'{p}_{n}' for p in ('RULES', 'FS10', 'RULESFS10') for n in SIZES2]
    return [v for v, _, g in VARIANTS if g == group]


def use_rules_run(run):
    """RULES_<n> of rules run 2 or 3 read the rules of that Astra repetition."""
    RULES_RUN[0] = run
    RULES.clear()


def use_rulesfs_rules(run):
    """RULESFS10_<n> with the rules of Astra repetition run (1: configs/labeled_chapter_rules_20261028.json)."""
    RULESFS_RULES[0] = run
    FEWSHOT2.clear()


def use_split(split):
    """'valid': FS10_<n> / RULESFS10_<n> read the examples retrieved for the validation pairs."""
    assert split in ('test', 'valid'), split
    SPLIT[0] = split
    FEWSHOT2.clear()


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def reused(root=PROJECT):
    return {}


def load_rules(path=None):
    path = path or (RULES_PATH if RULES_RUN[0] == 1 else PROJECT/RULES_REP_PATH.format(RULES_RUN[0]))
    RULES.clear(); RULES.update(json.loads(Path(path).read_text())['rules'])


def load_fewshot2():
    """Fills TRAIN2 and RULES1 before FEWSHOT2 (same thread contract as load_fewshot)."""
    import gzip
    sets = json.loads(SETS_PATH.read_text())['benchmarks']
    train = {}
    for b, info in sets.items():
        for n in SIZES2:
            with gzip.open(PROJECT/info[str(n)]['files']['train']['path'], 'rt') as f:
                train.setdefault(b, {})[str(n)] = {r['pair_id']: r for r in map(json.loads, f)}
    FEWSHOT2.clear(); TRAIN2.clear(); TRAIN2.update(train)
    path = RULES_PATH if RULESFS_RULES[0] == 1 else PROJECT/RULES_REP_PATH.format(RULESFS_RULES[0])
    RULES1.clear(); RULES1.update(json.loads(path.read_text())['rules'])
    FEWSHOT2.update(json.loads((FEWSHOT_VALID_PATH if SPLIT[0] == 'valid' else FEWSHOT2_PATH).read_text())['benchmarks'])


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


def examples_v2(case, variant):
    """FS10_<n> / RULESFS10_<n>: the examples block from the filtered retrieval of the n set."""
    if not FEWSHOT2:
        with FEWSHOT_LOCK:
            if not FEWSHOT2:
                load_fewshot2()
    b, n = case['benchmark'], variant.rsplit('_', 1)[1]
    rules = HEADER + '\n'.join('- '+r for r in RULES1[b][n]) if variant.startswith('RULESFS') else ''
    if case.get('pair_id') == 'template':
        return protocol.BASE + rules + EXAMPLES
    parts = [protocol.BASE + rules + EXAMPLES]
    for i, pid in enumerate(FEWSHOT2[b][n]['k10'][case['pair_id']], 1):
        row = TRAIN2[b][n][pid]
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
    if variant.startswith('RULES_'):
        text = instruction(case['benchmark'], variant)
    elif variant in ('FS10', 'FS20'):
        text = instruction_fewshot(case, variant)
    else:
        text = examples_v2(case, variant)
    if 'questions' in body:
        body['questions']['match']['instructions'] = text
    else:
        body['messages'][0]['content'] = text
    return body
