"""Selected prompt (F0) on the full validation sets, for the selective-prediction thresholds.

Six decision models (Luna generative returns no probability). Each model and
benchmark keeps the prompt selected in the 12-prompt study with the B0
serialization, the same request as F0 of the inputs study.

The validation set is the full benchmark validation split, prepared exactly as
in src/prepare.py: pairs whose input also occurs in the test set and input
groups with conflicting labels are removed. Its size matches
manifest['benchmarks'][b]['valid_unique_disjoint_inputs']. The prompt and route
samples and rules_train (the Astra training pairs) are subsets of it; the
analysis excludes rules_train from threshold tuning.
"""
import json
from pathlib import Path

import inputs_study
import prompt12
import protocol

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = 'valid'
RUN_MODELS = [m for m in prompt12.RUN_MODELS if m != 'luna']
SELECTION = inputs_study.SELECTION
SELECTED = inputs_study.SELECTED


def register_models():
    prompt12.register_models()


def run_variants(alias):
    return ['F0']


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def load_selection(root=PROJECT):
    return inputs_study.load_selection(root)


def payload(case, alias, variant='F0'):
    assert variant == 'F0', variant
    if not SELECTED:
        load_selection()
    return prompt12.payload(case, alias, SELECTED[alias][case['benchmark']])


def eligible_valid(root, b):
    """Full validation split of benchmark b, filtered as in prepare.main."""
    import prepare
    directory = Path(root)/'data'/'source'/b
    valid_path = directory/('wdcproducts80cc20rnd000un_valid_large.json.gz' if b == 'wdc' else b+'-valid.csv')
    valid_rows = prepare.read_rows(valid_path)
    if set(valid_rows[0]) == {'pair_id'}:
        source = directory/(b+'-valid.json.gz')
        if not source.exists():
            source = directory/(b+'-train.json')
        indexed = {str(r['pair_id']): r for r in prepare.read_rows(source)}
        valid_rows = [indexed[str(r['pair_id'])] for r in valid_rows]
    valid = [prepare.case(r, b) for r in valid_rows]
    tests = json.loads((Path(root)/'data'/b/'test.json').read_text())
    test_inputs = {r['input_hash'] for r in tests}
    unique, ambiguous = {}, set()
    for r in valid:
        if r['input_hash'] not in test_inputs:
            if r['input_hash'] in unique and r['label'] != unique[r['input_hash']]['label']:
                ambiguous.add(r['input_hash'])
            unique.setdefault(r['input_hash'], r)
    return [r for h, r in unique.items() if h not in ambiguous]


def export_valid(root=PROJECT):
    """Write data/<b>/valid.json once; checks size and subset relations against the frozen splits."""
    import prepare
    root = Path(root)
    manifest = json.loads((root/'configs/manifest.json').read_text())['benchmarks']
    report = {}
    for b in protocol.BENCHMARKS:
        rows = eligible_valid(root, b)
        ids = {r['pair_id'] for r in rows}
        assert len(rows) == len(ids) == manifest[b]['valid_unique_disjoint_inputs'], b
        assert not ids & {r['pair_id'] for r in json.loads((root/'data'/b/'test.json').read_text())}, b
        for name in ('prompt', 'route'):
            assert {r['pair_id'] for r in json.loads((root/'data'/b/(name+'.json')).read_text())} <= ids, (b, name)
        rules = {r['pair_id'] for r in json.loads((root/'data'/b/'rules_train.json').read_text())}
        prepare.write_frozen(root/'data'/b/(SPLIT+'.json'), rows)
        report[b] = {'n': len(rows), 'positive': sum(r['label'] for r in rows), 'sha256': protocol.digest(rows),
                     'rules_train_in_valid': len(rules & ids), 'rules_train': len(rules)}
    return report
