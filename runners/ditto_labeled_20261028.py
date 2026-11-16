"""Labelled-data chapter, part 4: Ditto with the labelling paper's training script, 3 sizes x 8 benchmarks x 3 seeds.

Uses /path/to/Automatic-data-labeling/scripts/training/train_ditto.py unchanged, with one config per
run in the format of the paper's fixed-pair controls (roberta-base, batch 64, max_len 256, 50 epochs,
lr 5e-5, early stopping 10, da=del, fp16; fields and aliases from configs/ditto/benchmarks_training.yaml):
train = train split, valid = validation split of the frozen Sol-B0 set (configs/labeled_chapter_sets_20261028.json),
test = our full test set (data/<b>/test.json written in the official pair format), exclude_valid_from_train false.
Seeds 42, 52, 62 as in the paper. GPU (gpu-vram-48gb), Slurm array, no API calls.

    python ditto_labeled_20261028.py --action prepare     # data files, 72 configs, sbatch, manifest
    python ditto_labeled_20261028.py --action dry-run     # train_ditto.py --dry-run for every config
    python ditto_labeled_20261028.py --action collect     # test F1 per run -> results.json
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/path/to/decision-em')
LAB = Path('/path/to/Automatic-data-labeling')
PY = '/path/to/envs/ditto-modern/bin/python'
OUT = ROOT/'runs/labeled_chapter_20261028/ditto'
SETS = ROOT/'configs/labeled_chapter_sets_20261028.json'
BENCHMARKS = ['abt-buy', 'wdc', 'dblp-scholar', 'walmart-amazon', 'dn7-walmart-amazon', 'billiger-de', 'dblp-acm', 'semi-heter']
SIZES, SEEDS = [200, 2000, 4000], [42, 52, 62]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_name(b, size, seed):
    return f'{b}_set{size}_seed{seed}'


def prepare():
    import yaml
    sets = json.loads(SETS.read_text())['benchmarks']
    paper = yaml.safe_load((LAB/'configs/ditto/benchmarks_training.yaml').read_text())
    defaults = {k: v for k, v in paper['defaults'].items() if k not in ('output_root', 'active_profiles')}
    defaults['exclude_valid_from_train'] = False
    (OUT/'data').mkdir(parents=True, exist_ok=True); (OUT/'configs').mkdir(exist_ok=True)
    manifest = {'sets_manifest_sha256': sha(SETS), 'train_script_sha256': sha(LAB/'scripts/training/train_ditto.py'),
                'paper_config_sha256': sha(LAB/'configs/ditto/benchmarks_training.yaml'), 'runs': []}
    for b in BENCHMARKS:
        test = OUT/'data'/f'{b}_test.json.gz'
        with gzip.open(test, 'wt') as f:
            for c in json.loads((ROOT/'data'/b/'test.json').read_text()):
                row = {'pair_id': c['pair_id'], 'label': c['label']}
                for s in ('left', 'right'):
                    row['id_'+s] = c['entity_ids'][s]
                    row.update({f'{k}_{s}': v for k, v in c['records'][s].items()})
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        bcfg = paper['benchmarks'][b]
        for size in SIZES:
            files = sets[b][str(size)]['files']
            for seed in SEEDS:
                cfg = {'defaults': {**defaults, 'seed': seed},
                       'benchmarks': {b: {'train': str(ROOT/files['train']['path']), 'valid': str(ROOT/files['valid']['path']),
                                          'test': str(test), 'fields': bcfg['fields'],
                                          **({'field_aliases': bcfg['field_aliases']} if 'field_aliases' in bcfg else {})}}}
                p = OUT/'configs'/f'{run_name(b, size, seed)}.yaml'
                p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                manifest['runs'].append({'name': run_name(b, size, seed), 'config': str(p), 'config_sha256': sha(p)})
        manifest[f'test_{b}_sha256'] = sha(test)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n')
    names = [r['name'] for r in manifest['runs']]
    (OUT/'runs.txt').write_text('\n'.join(names)+'\n')
    (OUT/'ditto_labeled_20261028.sbatch').write_text(f'''#!/bin/bash
# Ditto, labelled-data chapter: one array task per (benchmark, set size, seed); GPU only, no API calls.
#SBATCH --job-name=em-ditto-labeled
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=1-{len(names)}%6
#SBATCH --output={OUT}/logs/%A_%a.log
set -euo pipefail
cd {LAB}
NAME=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {OUT}/runs.txt)
{PY} scripts/training/train_ditto.py --config {OUT}/configs/$NAME.yaml --output-root {OUT}/output --run-name $NAME
''')
    (OUT/'logs').mkdir(exist_ok=True)
    print(len(names), 'configs;', 'sbatch:', OUT/'ditto_labeled_20261028.sbatch')


def dry_run():
    m = json.loads((OUT/'manifest.json').read_text())
    bad = []
    for r in m['runs']:
        res = subprocess.run([PY, 'scripts/training/train_ditto.py', '--config', r['config'], '--output-root', str(OUT/'dryrun'),
                              '--run-name', r['name'], '--dry-run'], cwd=LAB, capture_output=True, text=True)
        if res.returncode:
            bad.append((r['name'], res.stderr[-400:]))
        print(r['name'], 'ok' if not res.returncode else 'FAILED', flush=True)
    print('failed', len(bad)); [print(x) for x in bad[:3]]


def collect():
    out = {}
    for r in json.loads((OUT/'manifest.json').read_text())['runs']:
        reports = list((OUT/'output'/r['name']).rglob('benchmark_report.json'))
        out[r['name']] = json.loads(reports[0].read_text()).get('metrics') if reports else None
    (OUT/'results.json').write_text(json.dumps(out, indent=1)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--action', required=True, choices=['prepare', 'dry-run', 'collect'])
    {'prepare': prepare, 'dry-run': dry_run, 'collect': collect}[ap.parse_args().action]()


if __name__ == '__main__':
    main()
