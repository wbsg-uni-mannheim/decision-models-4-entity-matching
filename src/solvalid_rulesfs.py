"""PPLX v1.2 with RULESFS10_2000 on the validation split of the Sol-labelled 2,000 set (cascade threshold on Sol labels).

Aaron (5 Nov, relayed: "Lauf freigeben"): the RULESFS10_2000 cascade of Tables X/XI gets its t from Sol labels of the
same configuration, as the B0 cascades (src/solset_b0.py). Cases: data/<b>/solvalid2000.json, the 400 validation-split
pairs per benchmark of the 2,000 set (3,200 pairs), taken unchanged from data/<b>/solset.json; label = Sol-B0 label.
Requests: labeled_chapter.payload RULESFS10_2000 (B0 + Astra rules of run 1 + 10 examples from the train split of the
2,000 set), with the examples retrieved for these pairs (configs/labeled_chapter_fewshot_solvalid_20261105.json,
artifacts/operations/labeled_solvalid_prep_20261105.py); the request layout is that of the test and validation runs.
labeled_chapter.py stays unchanged: this module points its validation-examples path to the new file and selects the
'valid' split once at import.
Caveat: the Astra rule agents saw this validation split as their held-out check of the rules (input/valid.json).
"""
import gzip
import json
from pathlib import Path

import labeled_chapter as lc
import protocol

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = 'solvalid2000'
RUN_MODELS = ['pplx_v12']
PROMPTS = ['RULESFS10_2000']
FEWSHOT = 'configs/labeled_chapter_fewshot_solvalid_20261105.json'
CASES = 'configs/solvalid2000_cases_20261105.json'
SETS = 'configs/labeled_chapter_sets_20261028.json'

lc.FEWSHOT_VALID_PATH = PROJECT/FEWSHOT
lc.use_split('valid')


def register_models():
    lc.register_models()


def run_variants(alias):
    return list(PROMPTS)


def benchmarks(variant):
    return list(protocol.BENCHMARKS)


def payload(case, alias, variant):
    assert variant in PROMPTS and lc.SPLIT[0] == 'valid' and lc.FEWSHOT_VALID_PATH == PROJECT/FEWSHOT, variant
    return lc.payload(case, alias, variant)


def export_cases(root=PROJECT):
    """Write data/<b>/solvalid2000.json once (subset of solset.json in split-file order) and the cases config."""
    import prepare
    root = Path(root)
    sets = json.loads((root/SETS).read_text())['benchmarks']
    report = {'description': __doc__, 'benchmarks': {}}
    for b in protocol.BENCHMARKS:
        info = sets[b]['2000']['files']['valid']
        with gzip.open(root/info['path'], 'rt') as f:
            split = [json.loads(x) for x in f]
        assert len(split) == info['n'] and sum(r['label'] for r in split) == info['pos'], b
        cases = {c['pair_id']: c for c in json.loads((root/'data'/b/'solset.json').read_text())}
        rows = [cases[r['pair_id']] for r in split]
        assert all(c['label'] == r['label'] for c, r in zip(rows, split)), b
        test_inputs = {c['input_hash'] for c in json.loads((root/'data'/b/'test.json').read_text())}
        prepare.write_frozen(root/'data'/b/(SPLIT+'.json'), rows)
        report['benchmarks'][b] = {'n': len(rows), 'positive': sum(c['label'] for c in rows), 'sha256': protocol.digest(rows),
                                   'records_equal_a_test_pair': sum(c['input_hash'] in test_inputs for c in rows)}
    prepare.write_frozen(root/CASES, report)
    return report


if __name__ == '__main__':
    print(json.dumps(export_cases()['benchmarks'], indent=1))
