"""Reviewed continuation preserving the two pre-existing development reports.

The original USD200 launcher, prepared wrappers and frozen sources stay intact.
Only the confirmation report-directory guard is narrowed to two hash-pinned
development artifacts; every other existing report file remains a stop.
"""
import ast
import csv
import io
import json
from pathlib import Path
import types

import authorized_recovery_200 as budget

SELF = 'artifacts/operations/authorized_recovery_200_reviewed.py'
DEVELOPMENT = {
 'development_metrics.json': 'de72909e8e38045fd7f31c250e6f5517ac0b69e2966425eb78214f10d0c3f227',
 'development_metrics.csv': '1c77be36a5960a86a28e049956eb6ecc94dc01d98b0608fdd60cf231191f33e6'}


def unexpected_report_files(directory):
    if not directory.exists():
        return False
    observed={p.name:budget.sha(p) if p.is_file() else None for p in directory.iterdir()}
    return bool(observed) and observed != DEVELOPMENT


def review_development(root):
    import sys
    sys.path.insert(0,str(root/'src'))
    import expanded_confirmation as ec
    import analyze
    directory=root/'artifacts/results/expanded_confirmation'
    if unexpected_report_files(directory):raise ValueError('Unreviewed existing report')
    load=types.FunctionType(analyze.load.__code__,{**analyze.load.__globals__,'payload':ec.payload},argdefs=analyze.load.__defaults__)
    rows=[]
    for m in ec.MODELS:
        for b in ec.BENCHMARKS:
            for v in ec.PROMPT_VARIANTS+ec.SERIALIZATIONS:
                d=load(m,v,b,'prompt')
                rows.append({'model':m,'benchmark':b,'variant':v,**analyze.metric(d['y'],d['pred']),
                             'known_cost_usd':float(d['costs'].sum()),'unknown_cost_attempts':d['unknown_cost_attempts']})
    if (directory/'development_metrics.json').read_bytes()!=(ec.canonical(rows)+'\n').encode():
        raise ValueError('Development JSON does not reproduce from complete sources')
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if (directory/'development_metrics.csv').read_bytes()!=stream.getvalue().encode():
        raise ValueError('Development CSV does not reproduce from complete sources')
    return {'rows':len(rows),'sha256':DEVELOPMENT,'recomputed_from_all_development_cells':True}


def make_wrapper(root,stage):
    wrapper=budget.make_wrapper(root,stage)
    wrapper.REQUIRED.add(SELF)
    if stage=='confirmation':
        source=(root/'artifacts/operations/resume_confirmation_drained.py').read_text()
        functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        fragments=[]
        for name,old,new in [('_execute_bound','report_dir.exists() and any(report_dir.iterdir())','_unexpected_report_files(report_dir)'),
                             ('finish_report','out.exists() and any(out.iterdir())','_unexpected_report_files(out)')]:
            fragment=functions[name]
            if fragment.count(old)!=1:raise ValueError('Report guard source drift')
            fragments.append(fragment.replace(old,new))
        wrapper.__dict__['_unexpected_report_files']=unexpected_report_files
        exec(compile('\n\n'.join(fragments),'<reviewed development report guard>','exec'),wrapper.__dict__)
    return wrapper


def preflight(root,stage,run_id,spec,wrapper):
    development=review_development(root) if stage=='confirmation' else None
    result=budget.preflight(root,stage,run_id,spec,wrapper)
    return {**result,'preserved_development_reports':development}


main=types.FunctionType(budget.main.__code__,{**budget.main.__globals__,'SELF':SELF,
    'make_wrapper':make_wrapper,'preflight':preflight},argdefs=budget.main.__defaults__)


if __name__=='__main__':raise SystemExit(main())
