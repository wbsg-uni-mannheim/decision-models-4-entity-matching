"""After the 1 Nov validation edit of src/labeled_chapter.py: every test journal row of the labelled-chapter runs must keep
its request hash (protocol.digest of study.payload). Read-only.

    PYTHONPATH=src .venv/bin/python artifacts/operations/labeledch_valid_20261101/verify_test_hashes.py
"""
import json
from pathlib import Path

import protocol
import labeled_chapter as study

ROOT = Path(__file__).resolve().parents[3]
RUNS = [('labeledch_rules_run01_20261028', 1), ('labeledch_rules_run02_20261028', 2), ('labeledch_rules_run03_20261028', 3),
        ('labeledch_fewshot_run01_20261028', 1), ('labeledch_fewshot2_run01_20261031', 1), ('labeledch_rulesfs_run01_20261031', 1)]

study.register_models()
for ns, run in RUNS:
    study.use_rules_run(run); study.use_split('test')
    rows = bad = 0
    for path in sorted((ROOT/'runs'/ns).glob('*/*/*/test.jsonl')):
        m, v, b = path.parts[-4:-1]
        cases = {c['pair_id']: c for c in json.loads((ROOT/'data'/b/'test.json').read_text())}
        for line in path.read_text().splitlines():
            r = json.loads(line); rows += 1
            bad += r['request_hash'] != protocol.digest(study.payload(cases[r['pair_id']], m, v))
    print(ns, {'rows': rows, 'mismatches': bad}, flush=True)
    assert rows and not bad, ns
