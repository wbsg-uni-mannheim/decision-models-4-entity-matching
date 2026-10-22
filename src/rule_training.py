"""Freeze separate training pools and a training-only packet for the authorized Astra agent."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare import case, read_rows, write_frozen
from protocol import BENCHMARKS, canonical, digest

PROJECT = Path(__file__).resolve().parents[1]
SPLITS = ['test', 'prompt', 'route', 'expanded_screen', 'calibration_pool', 'expanded_repeat', 'cascade_live', 'schema_pilot']

def read(path):
    return json.loads(path.read_text())


def family(b):
    return 'billiger-de' if b.startswith('billiger-') else ('wdc' if b.startswith('wdc') else b)


def unordered_input(c):
    return digest(sorted(canonical(r) for r in c['records'].values()))


def training_rows(path):
    rows = read_rows(path)
    if len(rows) == 1 and isinstance(rows[0].get('label'), dict):
        columns = rows[0]
        keys = list(columns['label'])
        assert all(isinstance(v, dict) and set(v) == set(keys) for v in columns.values()), 'Malformed column-oriented training source'
        rows = [{k: v[i] for k, v in columns.items()} for i in keys]
    return rows


def select_training(rows, b, excluded_ids, excluded_inputs):
    unique, ambiguous = {}, set()
    for c in rows:
        h = unordered_input(c)
        if c['pair_id'] in excluded_ids or h in excluded_inputs:
            continue
        if h in unique and unique[h]['label'] != c['label']:
            ambiguous.add(h)
        unique.setdefault(h, c)
    eligible = [c for h, c in unique.items() if h not in ambiguous]
    groups = {y: sorted([c for c in eligible if c['label'] == y], key=lambda c: digest(['rules-train-v1', b, c['pair_id']])) for y in [0, 1]}
    info = {'eligible': len(eligible), 'positive_available': len(groups[1]), 'negative_available': len(groups[0]),
            'ambiguous_input_groups_removed': len(ambiguous)}
    if len(eligible) < 100 or not all(groups.values()):
        return [], {**info, 'available': False, 'reason': 'Require 100 distinct separate training inputs and both labels'}
    npos = max(100-len(groups[0]), min(50, len(groups[1])))
    selected = sorted(groups[1][:npos]+groups[0][:100-npos], key=lambda c: digest(['rules-order-v1', b, c['pair_id']]))
    assert len(selected) == 100 and len({c['pair_id'] for c in selected}) == 100
    return selected, {**info, 'available': True, 'positive_selected': npos, 'negative_selected': 100-npos}



def prepare(source):
    excluded_ids = {b: set() for b in BENCHMARKS}
    excluded_inputs, split_hashes = set(), {}
    for directory in sorted((PROJECT/'data').iterdir()):
        if not directory.is_dir() or family(directory.name) not in BENCHMARKS:
            continue
        for split in SPLITS:
            p = directory/(split+'.json')
            if p.exists():
                rows = read(p)
                split_hashes[str(p.relative_to(PROJECT))] = digest(rows)
                excluded_inputs.update(unordered_input(c) for c in rows)
                excluded_ids[family(directory.name)].update(c['pair_id'] for c in rows)
    spec = {'registration': '2026-10-13 user authorized Codex CLI gpt-6-astra agent to derive matching rules; original Luna single-call plan superseded before execution',
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'generator': {'model': 'gpt-6-astra', 'interface': 'Codex CLI', 'workflow': 'Iterative agent inspection and critique on training data only; no held-out performance access'},
            'excluded_split_hashes': split_hashes, 'benchmarks': {},
            'sampling': '100 unique input pairs; target 50/50 labels, fill from available class when fewer than 50; both classes required; deterministic hash order; not natural prevalence',
            'separation': 'Exclude frozen evaluation/development pair IDs within family, and direction-invariant exact descriptive inputs globally. Entity overlap may remain; reported separately.',
            'costs': 'Codex usage tracked separately from OpenRouter ledger; no claim of zero monetary cost. No direct API calls or purchases authorized for the rule agent.',
            'evaluation': 'Same seven evaluation models; shared USD100 cap unchanged; agent rules must be frozen before model quality evaluation.'}
    for b in BENCHMARKS:
        name = 'wdcproducts80cc20rnd000un_train_large.json.gz' if b == 'wdc' else b+'-train.json'+('' if b == 'abt-buy' else '.gz')
        path = source/b/name
        if not path.exists():
            spec['benchmarks'][b] = {'available': False, 'reason': 'Training source missing', 'source': str(path)}
            continue
        raw = path.read_bytes()
        dest = PROJECT/'data/source/rules'/b/name;dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == raw
        else:
            dest.write_bytes(raw)
        rows = [case(r, b) for r in training_rows(dest)]
        selected, info = select_training(rows, b, excluded_ids[b], excluded_inputs)
        info.update(source=str(path), source_copy=str(dest.relative_to(PROJECT)), source_sha256=hashlib.sha256(raw).hexdigest())
        spec['benchmarks'][b] = info
        if not info['available']:
            continue
        write_frozen(PROJECT/'data'/b/'rules_train.json', selected)
        info['training_sha256'] = digest(selected)
        packet = {'benchmark': b, 'sampling': spec['sampling'], 'examples': [
            {'training_index': i+1, 'record_a': c['records']['left'], 'record_b': c['records']['right'], 'match': bool(c['label'])}
            for i, c in enumerate(selected)]}
        info['agent_packet_sha256'] = digest(packet)
        write_frozen(PROJECT/'artifacts/rule_agent_input'/(b+'.json'), packet)
        tests = read(PROJECT/'data'/b/'test.json')
        info['training_entity_overlap_core_test'] = {side: len({c['entity_ids'][side] for c in selected} & {c['entity_ids'][side] for c in tests}) for side in ['left', 'right']}
    write_frozen(PROJECT/'configs/rule_agent_training_manifest.json', spec)
    print(canonical({b: {k: v for k, v in i.items() if k in ['available', 'reason', 'eligible', 'positive_selected', 'negative_selected', 'training_entity_overlap_core_test']} for b, i in spec['benchmarks'].items()}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser();ap.add_argument('--source', type=Path, required=True);args = ap.parse_args()
    prepare(args.source)
