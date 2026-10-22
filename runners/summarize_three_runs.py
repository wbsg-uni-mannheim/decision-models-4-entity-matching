"""Summarize normalized, independently verified result rows across real run IDs.

Input rows: run_id, model, condition (policy ID, not selected prompt), benchmark,
metric, value. Unavailable values are null. Source exporters must verify caches,
full cell counts and frozen run-specific selections before creating this input.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from repeated_runs import three_run_summary

CORE = {'abt-buy', 'wdc', 'dblp-scholar', 'walmart-amazon',
        'dn7-walmart-amazon', 'billiger-de', 'dblp-acm', 'semi-heter'}


def summarize(rows):
    by_cell = defaultdict(dict)
    by_run = defaultdict(dict)
    seen = set()
    for row in rows:
        rid = row['run_id']
        if type(rid) is not int or rid not in (1, 2, 3):
            raise ValueError('Three explicit run IDs required')
        group = (row['model'], row['condition'], row['metric'])
        key = (*group, row['benchmark'])
        if (rid, key) in seen:
            raise ValueError('Duplicate run/cell; never count a cache twice')
        seen.add((rid, key))
        by_cell[key]  # Retain all-null planned cells.
        value = row['value']
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Nonfinite metric')
            by_cell[key][rid] = value
        if row['benchmark'] in CORE:
            by_run[(group, rid)][row['benchmark']] = value
    output = []
    for (model, condition, metric, benchmark), values in sorted(by_cell.items()):
        output.append({'model': model, 'condition': condition, 'metric': metric,
                       'benchmark': benchmark, 'run_values': values,
                       **three_run_summary(values)})
    macros = defaultdict(dict)
    for (group, rid), benchmarks in by_run.items():
        macros[group]
        if set(benchmarks) == CORE and all(v is not None for v in benchmarks.values()):
            macros[group][rid] = statistics.mean(benchmarks.values())
    for (model, condition, metric), values in sorted(macros.items()):
        output.append({'model': model, 'condition': condition, 'metric': metric,
                       'benchmark': 'macro_8_core_families', 'run_values': values,
                       **three_run_summary(values)})
    return {'uncertainty': 'SD of three fresh inference runs, not a confidence interval over entities',
            'macro_policy': 'Equal eight-family mean within each complete run; billiger50 excluded',
            'rows': output}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path); p.add_argument('output', type=Path)
    args = p.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
