"""Pre-launch estimate for the validation group: per variant and benchmark, PPLX test-journal cost scaled by the ratio of
request characters (validation pool / test set). Also checks that every request shares the test instruction prefix. Free."""
import json
from pathlib import Path
import protocol
import labeled_chapter as study

ROOT = Path(__file__).resolve().parents[3]
NS = {'RULES': 'labeledch_rules_run01_20261028', 'FS10': 'labeledch_fewshot2_run01_20261031', 'RULESFS10': 'labeledch_rulesfs_run01_20261031'}
study.register_models(); study.use_rules_run(1)


def chars(cases, v):
    return sum(len(protocol.canonical(study.payload(c, 'pplx_v12', v))) for c in cases)


total, out = 0., {}
for v in study.run_variants('pplx_v12', 'valid'):
    for b in protocol.BENCHMARKS:
        test = json.loads((ROOT/'data'/b/'test.json').read_text())
        skip = {r['pair_id'] for r in json.loads((ROOT/'data'/b/'rules_train.json').read_text())}
        valid = [c for c in json.loads((ROOT/'data'/b/'valid.json').read_text()) if c['pair_id'] not in skip]
        study.use_split('test'); ct = chars(test, v)
        prefix = study.payload({**test[0]}, 'pplx_v12', v)
        study.use_split('valid'); cv = chars(valid, v)
        usd = sum(a['cost'] or 0. for line in (ROOT/'runs'/NS[v.rsplit('_', 1)[0]]/'pplx_v12'/v/b/'test.jsonl').read_text().splitlines()
                  for a in json.loads(line)['attempts'])
        est = usd*cv/ct
        out[f'{v}/{b}'] = {'test_usd': usd, 'chars_ratio': cv/ct, 'estimate_usd': est, 'valid_pairs': len(valid)}
        total += est
    print(v, round(sum(x['estimate_usd'] for k, x in out.items() if k.startswith(v+'/')), 3), flush=True)
print('total estimate USD', round(total, 2))
Path(ROOT/'runs/labeled_chapter_20261028/fewshot/estimate_valid_20261101.json').write_text(json.dumps({'total_usd': total, 'cells': out}, indent=1))
