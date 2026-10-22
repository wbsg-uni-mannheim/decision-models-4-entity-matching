"""Exercise the complete offline report with synthetic data, never API calls."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import analyze
from protocol import MODELS, BENCHMARKS, VARIANTS, ABLATION_MODELS, digest, payload, canonical

def main():
    with tempfile.TemporaryDirectory(prefix='decision-em-analysis-test-') as tmp:
        project=Path(tmp)
        (project/'configs').mkdir()
        (project/'artifacts').mkdir()
        (project/'manuscript').mkdir()
        (project/'configs/manifest.json').write_text(json.dumps({'benchmarks':{b:{} for b in BENCHMARKS}}))
        (project/'configs/selected_variants.json').write_text(json.dumps({m:{'variant':'P1'} for m in ABLATION_MODELS}))
        (project/'artifacts/budget.json').write_text(json.dumps({'spent':1.,'reserved':0.,'unknown_cost_attempts':0}))
        for b,spec in BENCHMARKS.items():
            (project/'data'/b).mkdir(parents=True)
            for split in ['test','prompt','route']:
                cases=[]
                for i in range(24):
                    records={side:{f:str(i) for f in spec['fields']} for side in ['left','right']}
                    cases.append({'benchmark':b,'pair_id':split+'-'+str(i),'label':int(i%3==0),
                                  'records':records,'input_hash':digest(records),
                                  'entity_ids':{'left':str(i),'right':str(i)}})
                (project/'data'/b/(split+'.json')).write_text(json.dumps(cases))
                for m in MODELS:
                    variants=VARIANTS if m in ABLATION_MODELS else ['B0']
                    for variant in variants:
                        for namespace in ['quality']+(['timing_serial','timing_concurrent8'] if split=='prompt' and variant=='B0' else []):
                            directory=project/'runs'/namespace/m/variant/b
                            directory.mkdir(parents=True,exist_ok=True)
                            selected=cases[:20] if namespace.startswith('timing_') else cases
                            rows=[]
                            for i,c in enumerate(selected):
                                p=.8 if c['label'] else .2
                                if i%7==0:p=1-p
                                rows.append({'pair_id':c['pair_id'],'request_hash':digest(payload(c,m,variant)),
                                             'prediction':int(p>=.5),'probability_match':None if m=='luna' else p,
                                             'attempts':[{'cost':.0001,'seconds':.1}], 'input_hash':c['input_hash']})
                            (directory/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows))
                            if namespace.startswith('timing_'):
                                (directory/(split+'.executions.jsonl')).write_text(json.dumps({'new_pairs':20,'wall_seconds':2})+'\n')
        analyze.PROJECT=project
        analyze.OUT=project/'artifacts/results'
        original=analyze.bootstrap_delta
        analyze.bootstrap_delta=lambda *args,**kwargs:original(*args,**kwargs,repeats=20)
        analyze.main()
        summary=json.loads((analyze.OUT/'summary.json').read_text())
        assert summary['cells']==56
        assert summary['baseline_pairs']==56*24
        assert len(json.loads((analyze.OUT/'cascades.json').read_text()))==144
        assert len(list((analyze.OUT/'figures').glob('*.pdf')))==5
        assert (project/'manuscript/draft.md').exists()
        assert 'independent' in (project/'manuscript/draft.md').read_text().lower()
        print('SYNTHETIC INTEGRATION PASSED: 56 cells, 144 cascades, 5 figures, manuscript and audit packets')

if __name__=='__main__':
    main()
