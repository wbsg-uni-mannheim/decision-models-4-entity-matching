"""Publication-quality figures and a result-grounded first manuscript draft."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from protocol import MODELS, BENCHMARKS

LABELS={'jev':'Jev 1.13','clef':'Clef','clef_flash':'Clef-flash','luna_decisions':'Luna Decisions',
        'pplx':'Perplexity Decider','kev':'Kev-4B FP8','luna':'Luna generative'}
REFERENCES='''1. Peeters, Steiner, and Bizer. Entity Matching using Large Language Models. EDBT 2025. https://www.openproceedings.org/2025/conf/edbt/paper-81.pdf
2. Steiner and Bizer. Labeling Training Data for Entity Matching Using Large Language Models. 2026. https://arxiv.org/abs/2606.28823
3. Eltokhy. jlink. Public software and benchmark report. https://github.com/keltokhy/jlink
4. Simon. GPT-4o-mini tied Jev on entity matching, at about the same price. Author report, October 2026. https://beyondthetraverse.substack.com/p/gpt-4o-mini-tied-jev-on-entity-matching
5. Construct. Clef vs Jev Benchmark. Public benchmark report. https://construct.computer/blog/clef-vs-jev-benchmark/
6. Improving Typed Entity Decisions with Jev. Public manuscript. https://github.com/CompleteDotTech/paper-package/blob/main/manuscript/paper.md
7. Kostakos. Jev OpenSanctions benchmark. Public software. https://github.com/panios/jev-opensanctions-benchmark
8. Zhang et al. Beyond Scale and Generation: Understanding Language Model-based Entity Matching. 2026. https://arxiv.org/abs/2607.24688
9. Zhang et al. AnyMatch: Efficient Zero-Shot Entity Matching with a Small Language Model. https://arxiv.org/abs/2409.04073
10. Kamsteeg et al. Confidence Calibration in Large Language Model-Based Entity Matching. https://arxiv.org/abs/2509.19557
11. CaRL-EM: Cost-Aware Reinforcement Learning for Entity Matching with LLMs. https://arxiv.org/abs/2609.01195
12. KEN: An Execution Engine for Unstructured Database Systems. https://www.vldb.org/pvldb/vol19/p902-kossmann.pdf
13. LOTUS Semantic Operators and Their Optimization. https://www.vldb.org/pvldb/vol18/p4171-patel.pdf
14. Zeighami, Shankar, and Parameswaran. Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees. https://arxiv.org/abs/2509.02896
15. Chuang et al. Learning to Route LLMs with Confidence Tokens. ICML 2025. https://proceedings.mlr.press/v267/chuang25b.html
16. FrugalGPT. https://arxiv.org/abs/2305.05176
17. RouteLLM. https://arxiv.org/abs/2406.18665
18. Jellyfish. https://arxiv.org/abs/2312.01678
19. OpenSanctions Pairs. https://arxiv.org/abs/2603.11051
20. Ditto: Deep Entity Matching with Pre-Trained Language Models. https://arxiv.org/abs/2004.00584
'''

def build(out):
    project=out.parents[1]
    rows=json.loads((out/'main_metrics.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    reliability=json.loads((out/'reliability.json').read_text())
    cascades=json.loads((out/'cascades.json').read_text())
    manifest=json.loads((project/'configs/manifest.json').read_text())
    selection=json.loads((project/'configs/selected_variants.json').read_text())
    with (out/'ablations.csv').open() as f:
        ablations=list(csv.DictReader(f))
    figures=out/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=dict(zip(MODELS,plt.get_cmap('tab10').colors[:7]))
    def savefig(fig,name):
        for ext in ['pdf','svg','png']:
            fig.savefig(figures/(name+'.'+ext),bbox_inches='tight',dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for i,m in enumerate(MODELS):
        x=float(np.mean([r['cost_per_1000'] for r in rows if r['model']==m]))
        y=summary['macro_f1'][m]*100
        ax.scatter(x,y,color=colors[m],s=50)
        ax.annotate(LABELS[m],(x,y),xytext=(5,5+(i%2)*7),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Mean known API-reported USD per 1,000 pairs (equal benchmark weights)')
    ax.set_ylabel('Mean match F1 across eight benchmarks (%)')
    ax.grid(alpha=.2)
    savefig(fig,'quality_cost')
    for name in ['reliability','risk_coverage','match_precision_coverage']:
        fig,axes=plt.subplots(2,4,figsize=(14,6.8),layout='constrained')
        for ax,b in zip(axes.flat,BENCHMARKS):
            for r in [r for r in reliability if r['benchmark']==b]:
                if name=='reliability':
                    points=[(v['mean_p'],v['positive_rate']) for v in r['bins'] if v['n']]
                elif name=='risk_coverage':
                    points=[(v['coverage'],v['risk']) for v in r['risk_coverage']]
                else:
                    points=[(v['match_prediction_coverage'],v['accepted_match_precision']) for v in r['risk_coverage'] if v['accepted_match_precision'] is not None]
                if points:
                    x,y=zip(*points)
                    ax.plot(x,y,marker='.',markersize=3,linewidth=1,label=LABELS[r['model']],color=colors[r['model']])
            ax.set_title(b)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.grid(alpha=.2)
            if name=='reliability':
                ax.plot([0,1],[0,1],color='gray',linestyle='--',linewidth=.7)
                ax.set_xlabel('Mean predicted match probability');ax.set_ylabel('Observed match fraction')
            elif name=='risk_coverage':
                ax.set_xlabel('Accepted pair fraction');ax.set_ylabel('Error fraction among accepted pairs')
            else:
                ax.set_xlabel('Accepted fraction of predicted matches');ax.set_ylabel('Precision of accepted matches')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=6,fontsize=8)
        savefig(fig,name)
    fig,axes=plt.subplots(2,4,figsize=(14,6.8),layout='constrained')
    for ax,b in zip(axes.flat,BENCHMARKS):
        for m in MODELS:
            base=next(r for r in rows if r['benchmark']==b and r['model']==m)
            points=[(base['known_cost_usd'],base['f1']*100)]
            points += [(r['known_cost_usd'],r['f1']*100) for r in cascades if r['benchmark']==b and r['model']==m]
            points.sort()
            ax.plot(*zip(*points),marker='o',linewidth=.8,markersize=3,color=colors[m],label=LABELS[m])
        ax.set_title(b);ax.set_xlabel('Known API-reported USD on full test set');ax.set_ylabel('Match F1 (%)');ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=8)
    savefig(fig,'cascade_quality_cost')
    table=['| Benchmark | '+' | '.join(LABELS[m] for m in MODELS)+' |', '| --- | '+' | '.join(['---:']*len(MODELS))+' |']
    for b in BENCHMARKS:
        table.append('| '+b+' | '+' | '.join(f"{next(r['f1'] for r in rows if r['model']==m and r['benchmark']==b)*100:.2f}" for m in MODELS)+' |')
    table.append('| Macro average | '+' | '.join(f"{summary['macro_f1'][m]*100:.2f}" for m in MODELS)+' |')
    cost_table=['| Model | Mean known USD / 1,000 pairs | Known B0 test cost USD | Attempts with unknown charge |','| --- | ---: | ---: | ---: |']
    for m in MODELS:
        subset=[r for r in rows if r['model']==m]
        cost_table.append(f"| {LABELS[m]} | {np.mean([r['cost_per_1000'] for r in subset]):.5f} | {sum(r['known_cost_usd'] for r in subset):.5f} | {sum(r['unknown_cost_attempts'] for r in subset)} |")
    best=max(summary['macro_f1'],key=summary['macro_f1'].get)
    deltas={m:float(np.mean([float(r['delta_f1']) for r in ablations if r['model']==m and r['split']=='test'])) for m in selection}
    candidate=[r for r in cascades if r['unknown_cost_attempts']==0 and r['luna_control_unknown_cost_attempts']==0 and r['cost_relative_to_luna']<=.5 and r['paired_f1_difference_vs_luna']['difference']>=-.01]
    confirmed=[r for r in candidate if r['paired_f1_difference_vs_luna']['entity_component_bootstrap']['ci95'] is not None and r['paired_f1_difference_vs_luna']['entity_component_bootstrap']['ci95'][0]>=-.01]
    body='''# Decision Models for Entity Matching

## Abstract

We evaluate six off-the-shelf instruction-conditioned decision models and one generative control on eight established entity-matching benchmarks. All models receive the same descriptive attributes, matching instructions, and test pairs, without target-specific weight training. The study separates fixed zero-shot inference from development-selected prompts and uncertainty routing. It measures match-class precision, recall and F1, probability reliability, input sensitivity, API latency, and actual billed inference costs. Public benchmark contamination, provider drift, and the single generative control limit the scope of conclusions. This is a first manuscript draft; independent human error annotation and author review remain outstanding.

## Motivation and research questions

Entity matching requires deciding whether two descriptions identify the same entity. Typed decision interfaces make it possible to obtain probabilities directly, but their usefulness depends on the matching distribution, the instructions, and the serving configuration. Cost savings and well-calibrated confidence are empirical questions. We ask how the selected models compare on identical pairs, how instruction and serialization changes affect predictions, and when uncertainty-based delegation to a fixed generative model improves quality and cost.

## Related work and scope of contribution

LLM-based entity matching already includes extensive work on prompts, rules, demonstrations, and fine-tuning [1]. LLM-generated training labels support specialized student matchers [2]; these incur a different mix of labeling, training, and inference cost. AnyMatch [9], Jellyfish [18], and Ditto [20] further establish small or task-trained models as relevant alternatives. Beyond Scale and Generation [8] studies matching architectures under controlled training conditions. Our comparison of differently trained commercial and open-weight model endpoints cannot isolate a causal architectural effect.

Direct predecessors also evaluate decision models. jlink [3] reports Jev on twelve classical test sets and additional local approaches. Simon [4] compares Jev with GPT-4o-mini on WDC, including confidence and costs. Construct [5] compares Jev, Clef, and Clef-flash on a small synthetic EM subset. A public Jev manuscript [6] studies instructions, output schemas, and calibration on DBLP–ACM. Another repository [7] evaluates Jev on OpenSanctions, complementing the benchmark paper [19]. These software reports and manuscripts preclude claims of the first application or first broad Jev evaluation, even though they are not all peer reviewed.

Calibration in EM [10], confidence-based routing [15], general model cascades [16,17], adaptive EM operators [11], and data-system execution and quality guarantees [12–14] are established topics. Our contribution is a controlled empirical comparison across the chosen decision-model families, with separate measurements of input sensitivity, probability reliability, class-specific selective prediction, and actual endpoint economics. We propose no new training or routing architecture and make no universal claim against generative LLMs.

## Experimental protocol

The main matrix contains seven configurations and eight benchmarks: Abt–Buy, WDC Products, DBLP–Scholar, Walmart–Amazon, Dn7 Walmart–Amazon, billiger.de, DBLP–ACM, and Semi-HETER. It contains 30,225 original test pairs per model, including 3,073 matches. The billiger.de 100un file is identified by checksum rather than by a potentially ambiguous filename. Original test rows and labels are retained, including duplicated descriptive inputs. Pair IDs, source identifiers, labels, clusters, and explanations are excluded from model requests through explicit descriptive-field whitelists. Abt–Buy name/title aliases are reconciled. Full fields are retained.

The fixed baseline uses attribute-labeled text and the common identity instruction. Decision probabilities are thresholded at 0.5. Luna generative uses a Boolean JSON schema, a fixed non-reasoning setting, and an output limit of 64 tokens. Providers are pinned, fallback is disabled, and actual provider and model names are validated. Kev is explicitly the SiliconFlow FP8 endpoint. Quality runs use 16 concurrent requests per model; development smoke-test responses from concurrency eight are reused. Dedicated latency measurements use 20 development pairs per benchmark under serial and concurrency-eight conditions. These are hosted endpoint measurements, not GPU or architectural speed comparisons.

Two disjoint development samples of 200 pairs per benchmark are fixed before inference, preserving the available class distribution approximately. Exact duplicate descriptive inputs and test-input overlaps are removed from the development sampling pool only; conflicting duplicate development groups are excluded and recorded. Entity overlap across splits is reported, not assumed absent. Dn7 has only four positive examples in each development sample, limiting conclusions about positive-class calibration and prompt selection.

For Jev, Clef-flash, Luna Decisions and Luna generative, we compare the baseline with domain-specific instructions, conservative instructions, JSON serialization, and reversed record order. These factors are changed separately. One of the first three alternatives is selected per model by the mean D_prompt match F1 across all eight benchmarks, with a fixed tie order, and confirmed on every full test set. Reversed order remains a development diagnostic. No test-label-driven prompt selection is performed.

Probability diagnostics include Brier score, clipped log loss, ten-bin equal-width calibration error, reliability plots, and risk–coverage curves. Predicted matches receive a separate precision-versus-coverage analysis. Routing cutoffs are selected on D_route to target 10%, 25%, or 50% escalation; deterministic hash tie-breaking is frozen. The realized test rate may differ. Cached Luna answers yield offline cascade predictions. Comparators are decision-only, Luna-only, and twenty random-routing controls with the same realized number of escalated pairs. All first-stage costs and routed second-stage costs are counted. Cached analyses do not measure live cascade latency.

Match F1 is the primary metric. Dataset macro averages weight each benchmark equally. Paired F1 intervals use 1,000 bootstrap replicates, with connected components of shared entity IDs as clusters when enough effective clusters exist; otherwise the cluster interval is withheld. Pair-level bootstrap intervals are a labeled sensitivity analysis. The intervals are descriptive and do not establish familywise statistical significance across the full comparison matrix.

## Results

'''
    body+=f"The highest observed baseline macro F1 is {summary['macro_f1'][best]*100:.2f}% for {LABELS[best]}. This is a point-estimate ranking within the selected matrix. Per-benchmark precision, recall, confusion counts, paired uncertainty intervals, raw responses, and costs accompany the table.\n\n"
    body+='\n'.join(table)+'\n\n### Measured cost\n\n'+'\n'.join(cost_table)+'\n\n'
    body+='Cost tables and figures sum known API-reported charges, including recorded failed attempts. Transport failures may have an unknown charge; their counts are explicit and their costs are not assumed to be zero. Settings with unknown charges are excluded from the cost-target screening below. The operational budget ledger conservatively accounts for such attempts and is not an invoice total.\n\n'
    body+='### Input sensitivity\n\n'
    for m,v in selection.items():
        body+=f"For {LABELS[m]}, development selection chose {v['variant']}; its mean confirmed test F1 change relative to B0 is {100*deltas[m]:+.2f} percentage points.\n\n"
    body+='### Selective prediction and fallback\n\n'
    body+=f"Of {len(cascades)} benchmark/model/routing settings, {len(candidate)} meet the predeclared cost and quality target at the point-estimate level (at most half the Luna-only cost and at most one F1 point loss). Of these, {len(confirmed)} also have a usable component-bootstrap lower bound above the one-point margin. These are descriptive screening counts, not multiplicity-adjusted equivalence claims. The full cascade table reports corrections, newly introduced errors, realized escalation rates, and matched-rate random controls.\n\n"
    body+='''## Error analysis and limits

The reproducible audit packet contains 80 randomly sampled cases (ten per benchmark) and 80 separately identified diagnostic cases, plus a blinded 40-case second-reviewer packet. Annotation fields remain empty until reviewed. No completed independent human audit, adjudicated error taxonomy, or corrected gold standard is claimed in this draft.

The evaluated configurations do not cover the full decision-model market. One generative control does not represent all generative LLMs. Training data, model size, architecture and serving infrastructure vary jointly. Public benchmark exposure and prior exploratory use of WDC can bias apparent generalization. A stable endpoint name does not ensure stable internal weights. Hosted prices, precision and load can change, and repeated-run stability is not established by a single main pass. Confidence transfer here means development-to-test transfer within a benchmark, not calibration on unseen domains. No theoretical risk guarantee follows from the empirical curves.

## Reproducibility and completion status

The release candidate includes source checksums, fixed descriptive fields and prompts, development selections, raw pair responses, request hashes, actual costs, selected variants, metrics, plots, and this generated draft. Scripts regenerate numerical results without new inference. Credentials are excluded. Benchmark redistribution and raw-output release should follow the original licenses. Independent audit, scientific interpretation by the authors, bibliography formatting, and submission remain outstanding. Historical Ditto scores are not inserted into the main table without a verified split and training provenance match.

## References

'''+REFERENCES
    (project/'manuscript/draft.md').write_text(body)
    (out/'main_table.md').write_text('\n'.join(table)+'\n')
    (out/'data_provenance.json').write_text(json.dumps(manifest['benchmarks'],indent=2)+'\n')
    print('REPORT: figures and manuscript/draft.md generated; independent audit pending',flush=True)
