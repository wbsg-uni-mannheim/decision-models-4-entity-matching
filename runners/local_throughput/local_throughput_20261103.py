"""Local inference throughput on one H100 (Aaron, 3 Nov 2026; Chris asked how fast the models run locally).

Three models on the same GPU, B0 prompt and default serialization as in the paper, no quantization:
  ditto  RoBERTa-base Ditto checkpoints of the labelled-data chapter (set 4000, seed 42, one per benchmark), fp16
  pplx   perplexity-ai/pplx-decider-v1.2 with Perplexity's own pplx_decider package (bf16, one forward pass);
         --path vllm: the same checkpoint in vLLM via the plugin in pplx_vllm_plugin/ (prompts from pplx_decider)
  qwen   Qwen/Qwen3.8-27B generative with vLLM (bf16, B0 chat request, thinking off, JSON schema, max_tokens 64);
         speed only: the greedy runs keep no decisions (Aaron, 3 Nov). The decision check uses --sampling api, the
         sampling the API runs had (no temperature set, so the model defaults)
  qwen-medium  Qwen/Qwen3.8-27B generative with thinking on, reasoning_effort='medium', no output-token limit
         (up to the 32k context), JSON schema only for the answer after </think>, the model's sampling defaults;
         (a) on a stratified 200-pair subset
  glm    zai-org/GLM-5.3-Flash generative with vLLM on four H100 (tensor parallel), native FP8 weights; thinking
         cannot be switched off, so the lowest level reasoning_effort='low'; (a) on a stratified 1,000-pair subset
Two settings per model: (a) one request after the other (batch size 1) and (b) the fastest supported batched path.

Sample: up to 1,250 test pairs per benchmark (semi-heter has 414), fixed seed, split round-robin into three
blocks; 200 further test pairs as an uncounted warm-up. Times are wall-clock seconds from record text to decision
(tokenization included, model loading and compilation excluded). Per block and benchmark: seconds per 1,000 pairs;
per block: unweighted mean over the eight benchmarks, as Table 7. No API calls.

    python local_throughput_20261103.py sample
    python local_throughput_20261103.py ditto
    python local_throughput_20261103.py pplx --path ref|fast|fast32k|vllm
    python local_throughput_20261103.py qwen [--sampling api]
    python local_throughput_20261103.py qwen-medium
    python local_throughput_20261103.py glm
    python local_throughput_20261103.py report
"""
import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/path/to/decision-em')
SUBSET_OUT = ROOT/'runs/local_throughput_20261103'
OUT = Path(os.environ.get('LT_OUT', SUBSET_OUT))  # full test sets (7 Nov): runs/local_throughput_full_20261107
FULL = os.environ.get('LT_FULL') == '1'  # all 30,225 test pairs instead of the 9,164-pair sample (Aaron, 7 Nov)
DITTO_SET = os.environ.get('LT_DITTO_SET', 'set4000_seed42')  # full sets: set2000_seed{42,52,62} (Table IX)
HUB = Path(os.environ.get('HF_HUB_CACHE', '/path/to/hf_cache/hub'))
DITTO = ROOT/'runs/labeled_chapter_20261028/ditto/output'
DITTO_LIB = '/path/to/Automatic-data-labeling/third_party'
SEED, PER_BENCHMARK, WARMUP, BLOCKS = 20261103, 1250, 200, 3
sys.path.insert(0, str(ROOT/'src'))
import protocol  # noqa: E402

BENCHMARKS = list(protocol.BENCHMARKS)
RUN = os.environ.get('LT_RUN', '')  # r2, r3: repeat only the batched run of the table, under its own name
# The batched run behind each row of the local table and its two repetitions (Aaron, 6 Nov)
REPEATS = {'ditto_b1024': ['ditto_b1024', 'ditto_b1024_r2', 'ditto_b1024_r3'],
           'pplx_vllm_b': ['pplx_vllm_b', 'pplx_vllm_r2_b', 'pplx_vllm_r3_b'],
           'pplx_vllm_tok_b': ['pplx_vllm_tok_b', 'pplx_vllm_tok_r2_b', 'pplx_vllm_tok_r3_b'],
           'qwen_b_apisampling': ['qwen_b_apisampling', 'qwen_b_apisampling_r2', 'qwen_b_apisampling_r3'],
           'glm_b': ['glm_b', 'glm_b_r2', 'glm_b_r3']}


def run_name(base):
    return f'{base}_{RUN}' if RUN else base


def snapshot(repo):
    base = HUB/('models--' + repo.replace('/', '--'))
    if (base/'refs/main').exists():
        return base/'snapshots'/(base/'refs/main').read_text().strip()
    only, = (base/'snapshots').iterdir()  # downloaded by revision (GLM, 17 Oct): no refs/main, one snapshot
    return only


def load_cases(b):
    return {c['pair_id']: c for c in json.loads((ROOT/'data'/b/'test.json').read_text())}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n')


def environment():
    import torch
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0), 'host': platform.node(),
           'slurm_job': os.environ.get('SLURM_JOB_ID')}
    try:
        env['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,power.limit',
                                            '--format=csv,noheader', '-i', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]],
                                           capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    for name in ('transformers', 'vllm', 'fla', 'flashinfer'):
        try:
            env[name] = __import__(name).__version__
        except ImportError:
            pass
    return env


def sample():
    pairs, pool = [], []
    for b in BENCHMARKS:
        ids = sorted(load_cases(b))
        rng = random.Random(f'{SEED}:{b}')
        rng.shuffle(ids)
        chosen = ids[:PER_BENCHMARK]
        pairs += [{'benchmark': b, 'pair_id': p, 'block': i % BLOCKS} for i, p in enumerate(chosen)]
        pool += [(b, p) for p in ids[PER_BENCHMARK:]]
    random.Random(SEED).shuffle(pool)
    write_json(OUT/'sample.json', {'seed': SEED, 'per_benchmark': PER_BENCHMARK, 'blocks': BLOCKS,
                                   'pairs': pairs, 'warmup': [list(x) for x in pool[:WARMUP]]})
    print(len(pairs), {b: sum(p['benchmark'] == b for p in pairs) for b in BENCHMARKS})


def load_sample():
    cases = {b: load_cases(b) for b in BENCHMARKS}
    if FULL:  # every test pair, three round-robin blocks in the sample's shuffled order; warm-up pairs of the sample run
        pairs = []
        for b in BENCHMARKS:
            ids = sorted(cases[b])
            random.Random(f'{SEED}:{b}').shuffle(ids)
            pairs += [{'benchmark': b, 'pair_id': p, 'block': i % BLOCKS, 'case': cases[b][p]} for i, p in enumerate(ids)]
        s = json.loads((SUBSET_OUT/'sample.json').read_text())
        return pairs, [cases[b][p] for b, p in s['warmup']]
    s = json.loads((OUT/'sample.json').read_text())
    pairs = [dict(p, case=cases[p['benchmark']][p['pair_id']]) for p in s['pairs']]
    warmup = [cases[b][p] for b, p in s['warmup']]
    return pairs, warmup


def groups(pairs):
    """(block, benchmark, pairs) in a fixed order; benchmark order rotates per block."""
    for k in range(BLOCKS):
        for b in BENCHMARKS[k:] + BENCHMARKS[:k]:
            yield k, b, [p for p in pairs if p['block'] == k and p['benchmark'] == b]


class Recorder:
    def __init__(self, name, settings):
        self.name, self.settings, self.rows, self.timing = name, settings, [], []

    def add(self, k, b, items, seconds, results, latencies=None):
        self.timing.append({'block': k, 'benchmark': b, 'n': len(items), 'seconds': seconds})
        for i, (p, r) in enumerate(zip(items, results, strict=True)):
            row = {'benchmark': b, 'pair_id': p['pair_id'], 'block': k, 'label': p['case']['label'], **r}
            if latencies:
                row['latency'] = latencies[i]
            self.rows.append(row)
        print(self.name, k, b, len(items), f'{seconds:.2f}s', f'{1000 * seconds / len(items):.1f}s/1k', flush=True)

    def save(self, env):
        with open(OUT/f'{self.name}.jsonl', 'w') as f:
            for r in self.rows:
                f.write(json.dumps(r) + '\n')
        write_json(OUT/f'{self.name}.timing.json', {'name': self.name, 'settings': self.settings,
                                                    'environment': env, 'groups': self.timing})


def measure_unbatched(rec, pairs, one):
    for k, b, items in groups(pairs):
        results, lat = [], []
        for p in items:
            t = time.perf_counter()
            results.append(one(p['case']))
            lat.append(time.perf_counter() - t)
        rec.add(k, b, items, sum(lat), results, lat)


def measure_batched(rec, pairs, many, before=None):
    for k, b, items in groups(pairs):
        if before:
            before()
        t = time.perf_counter()
        results = many([p['case'] for p in items])
        rec.add(k, b, items, time.perf_counter() - t, results)


# ---------------------------------------------------------------- Ditto

def ditto_inputs(b):
    """Ditto's own serialized test lines (COL/VAL) from the training run, keyed by pair_id."""
    run = Path((DITTO/f'{b}_{DITTO_SET}'/b/'training_output/LATEST_RUN').read_text().strip())
    lines = (run/'data/test.txt').read_text().splitlines()
    import csv
    with open(run/'predictions.csv') as f:
        preds = list(csv.DictReader(f))
    assert len(lines) == len(preds)
    metrics = json.loads((DITTO/f'{b}_{DITTO_SET}'/b/'metrics.json').read_text())
    texts = {r['pair_id']: tuple(line.split('\t')[:2]) for line, r in zip(lines, preds, strict=True)}
    probs = {r['pair_id']: float(r['prob']) for r in preds}
    return run/'checkpoints/best', texts, probs, metrics['best_threshold']


def ditto():
    import torch
    sys.path.insert(0, DITTO_LIB)
    from ditto_modern.model import load_model, load_tokenizer
    pairs, _ = load_sample()
    dev = torch.device('cuda')
    models, data = {}, {}
    for b in BENCHMARKS:
        ckpt, texts, probs, th = ditto_inputs(b)
        m = load_model(str(ckpt)).half().eval().to(dev)
        models[b] = (m, load_tokenizer(str(ckpt)), th)
        data[b] = (texts, probs)

    def forward(b, cases):
        m, tok, th = models[b]
        texts = data[b][0]
        enc = [tok.encode(text=texts[c['pair_id']][0], text_pair=texts[c['pair_id']][1], max_length=256, truncation=True)
               for c in cases]
        return enc

    @torch.inference_mode()
    def run(b, enc_batch):
        m = models[b][0]
        width = max(len(x) for x in enc_batch)
        ids = torch.tensor([x + [0] * (width - len(x)) for x in enc_batch], device=dev)
        att = (ids != 0).long()  # Ditto's collate: padding and the attention mask both use id 0
        return torch.softmax(m(input_ids=ids, attention_mask=att).float(), dim=-1)[:, 1].cpu().tolist()

    def result(b, c, p, enc):
        return {'prob': p, 'pred': int(p > models[b][2]), 'input_tokens': len(enc),
                'reference_prob_fp32': data[b][1][c['pair_id']]}

    def one(c):
        b = c['benchmark']
        enc = forward(b, [c])
        return result(b, c, run(b, enc)[0], enc[0])

    def batched(size):
        def many(cases):
            b = cases[0]['benchmark']
            enc = forward(b, cases)
            order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
            probs = [None] * len(enc)
            for s in range(0, len(order), size):
                idx = order[s:s + size]
                for i, p in zip(idx, run(b, [enc[i] for i in idx])):
                    probs[i] = p
            return [result(b, c, p, e) for c, p, e in zip(cases, probs, enc)]
        return many

    for b in BENCHMARKS:  # warm-up: 200 sample pairs per model one by one, then batched, uncounted
        cases = [p['case'] for p in pairs if p['benchmark'] == b][:200]
        for c in cases:
            one(c)
        for size in (64, 256, 1024):
            batched(size)(cases)
    torch.cuda.synchronize()
    env = environment()
    common = {'framework': 'PyTorch eager + ditto_modern DittoModel (RoBERTa-base, linear head)',
              'checkpoints': f'labelled-data chapter, {DITTO_SET.replace("_", ", ")}, one per benchmark; threshold from its validation',
              'precision': 'fp16 (model.half())', 'max_len': 256, 'tokenizer': 'fast (Rust) RoBERTa tokenizer, per pair'}
    if not RUN or os.environ.get('LT_DITTO_A') == '1':
        rec = Recorder(run_name('ditto_a'), {**common, 'batch_size': 1})
        measure_unbatched(rec, pairs, one)
        rec.save(env)
    for size in (1024,) if RUN else (64, 256, 1024):
        rec = Recorder(run_name(f'ditto_b{size}'), {**common, 'batch_size': size, 'batching': 'length-sorted batches per benchmark block'})
        measure_batched(rec, pairs, batched(size))
        rec.save(env)


# ---------------------------------------------------------------- PPLX

def pplx(path):
    if path == 'vllm':
        return pplx_vllm()
    import torch
    ckpt = snapshot('perplexity-ai/pplx-decider-v1.2')
    sys.path.insert(0, str(ckpt/'source/src'))
    from pplx_decider import Decider, FastOptions
    import prompt12
    prompt12.register_models()
    pairs, warmup = load_sample()
    api = api_rows('runs/prompt12_run01_20261019/pplx_v12/B0')

    def request(c):
        body = prompt12.payload(c, 'pplx_v12', 'B0')
        hit = api.get((c['benchmark'], c['pair_id']))
        if hit is not None:
            assert hit['request_hash'] == protocol.digest(body), 'request differs from the API run'
        return {'state': body['state'], 'questions': body['questions']}

    def result(r):
        return {'prob': r['answers']['match']['noul'], 'pred': int(r['answers']['match']['noul'] >= .5),
                'input_tokens': r['usage']['input_tokens'], 'sequences': r['usage']['sequences']}

    fast = {'ref': False, 'fast': True, 'fast32k': FastOptions(buckets=(32768,))}[path]
    decider = Decider.from_pretrained(ckpt, device='cuda', fast=fast)
    decider.warmup()
    decider.decide([request(c) for c in warmup])
    for c in warmup[:50]:
        decider.decide(request(c))
    torch.cuda.synchronize()
    env = environment()
    common = {'framework': 'pplx_decider 0.3.0 (model card source), transformers ' + env.get('transformers', ''),
              'precision': 'bf16', 'temperature': decider.temperature,
              'path': {'ref': 'reference path (fast=False): unmodified Transformers Qwen3_5Model, SDPA, left-padded batches',
                       'fast': 'fast=True default FastOptions(): packed varlen FlashAttention, whole-layer torch.compile, CUDA graphs, buckets 2k-32k, last-layer pruning',
                       'fast32k': 'fast=FastOptions(buckets=(32768,)): as fast, one static 32k pack (model card: offline throughput)'}[path]}
    many = lambda cases: [result(r) for r in decider.decide([request(c) for c in cases])]  # noqa: E731
    if path in ('ref', 'fast'):
        rec = Recorder(f'pplx_{path}_a', {**common, 'batch_size': 1})
        measure_unbatched(rec, pairs, lambda c: result(decider.decide(request(c))))
        rec.save(env)
    batch = 'batch_tokens=32768 (default)' if path == 'ref' else 'packs of up to 32,768 tokens'
    rec = Recorder(f'pplx_{path}_b', {**common, 'batching': 'one decide() call per benchmark block, ' + batch})
    measure_batched(rec, pairs, many)
    rec.save(env)


def pplx_vllm():
    from transformers import AutoProcessor
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    ckpt = snapshot('perplexity-ai/pplx-decider-v1.2')
    sys.path.insert(0, str(ckpt/'source/src'))
    from pplx_decider.protocol import MARKER, Encoder
    import prompt12
    prompt12.register_models()
    pairs, warmup = load_sample()
    api = api_rows('runs/prompt12_run01_20261019/pplx_v12/B0')
    decision = json.loads((ckpt/'decision_config.json').read_text())
    temperature = float(decision['temperature'])
    processor = AutoProcessor.from_pretrained(str(ckpt))
    encoder = Encoder(processor, decision['codes'])

    batch_tokenize = os.environ.get('LT_PPLX_TOKENIZE', 'encoder') == 'batch'

    def request(c):
        body = prompt12.payload(c, 'pplx_v12', 'B0')
        hit = api.get((c['benchmark'], c['pair_id']))
        if hit is not None:
            assert hit['request_hash'] == protocol.digest(body), 'request differs from the API run'
        req = {'state': body['state'], 'questions': body['questions']}
        encoder.validate(req)
        return req

    def tokenize(reqs):
        if not batch_tokenize:
            return [encoder.encode(r).input_ids.tolist() for r in reqs]
        # The Encoder's own template text, tokenized in one call of the fast tokenizer: the same ids as
        # Encoder.encode (checked on 417 abt-buy pairs), about 5 times faster than one processor call per request.
        ids = processor.tokenizer([encoder.text(r) for r in reqs], add_special_tokens=False)['input_ids']
        assert all(x.count(marker_id) == 1 for x in ids), 'one readout marker per request'
        return ids

    def result(o, n):
        logits = o.outputs.data[0, :2].float()  # one noul question: codes A (false), B (true)
        p = (logits / temperature).softmax(-1)[1].item()
        return {'prob': p, 'pred': int(p >= .5), 'input_tokens': n, 'sequences': 1}

    def many(cases):
        ids = tokenize([request(c) for c in cases])
        outs = llm.encode([TokensPrompt(prompt_token_ids=x) for x in ids], pooling_task='token_classify', use_tqdm=False)
        return [result(o, len(x)) for o, x in zip(outs, ids, strict=True)]

    marker_id = processor.tokenizer.convert_tokens_to_ids(MARKER)
    llm = LLM(model=str(ckpt), runner='pooling', dtype='bfloat16', max_model_len=8192,
              max_num_batched_tokens=int(os.environ.get('LT_PPLX_BATCHED_TOKENS', 32768)),
              max_num_seqs=int(os.environ.get('LT_PPLX_MAX_SEQS', 128)),  # ~94 pairs fit in 32k tokens; the
              # per-sequence Gated DeltaNet state cache grows with max_num_seqs (the default did not fit)
              gpu_memory_utilization=0.90, enable_prefix_caching=False, seed=0,
              limit_mm_per_prompt={'image': 0, 'video': 0},
              hf_overrides={'architectures': ['PplxJointDecider'], 'pplx_marker_id': marker_id})
    many(warmup)
    for c in warmup[:50]:
        many([c])
    env = environment()
    cfg = llm.llm_engine.vllm_config
    common = {'framework': 'vLLM ' + env.get('vllm', '') + ' offline LLM.encode (token_classify) with the pplx_vllm plugin '
                           '(D3ForSequenceClassification backbone, readout at each marker)',
              'prompts': 'pplx_decider 0.3.0 Encoder template (token ids, same as the reference path); tokenized ' +
                         ('in one batched fast-tokenizer call per LLM.encode call' if batch_tokenize else 'by Encoder.encode per request'),
              'precision': 'bf16 backbone, readout in vLLM head dtype ' + str(cfg.model_config.head_dtype),
              'temperature': temperature, 'attention': 'noncausal full attention, causal Gated DeltaNet (is_causal=False)',
              'max_num_seqs': cfg.scheduler_config.max_num_seqs,
              'max_num_batched_tokens': cfg.scheduler_config.max_num_batched_tokens,
              'chunked_prefill': bool(cfg.scheduler_config.enable_chunked_prefill),
              'prefix_caching': bool(cfg.cache_config.enable_prefix_caching),
              'cuda_graphs': not cfg.model_config.enforce_eager}
    name = os.environ.get('LT_PPLX_NAME', 'pplx_vllm')  # tuning runs get their own names
    if os.environ.get('LT_PPLX_SKIP_A') != '1':
        rec = Recorder(name + '_a', {**common, 'batch_size': 1, 'note': 'one LLM.encode call per pair'})
        measure_unbatched(rec, pairs, lambda c: many([c])[0])
        rec.save(env)
    rec = Recorder(name + '_b', {**common, 'batching': 'one LLM.encode call per benchmark block'})
    measure_batched(rec, pairs, many)
    rec.save(env)


def api_rows(rel):
    rows = {}
    for b in BENCHMARKS:
        with open(ROOT/rel/b/'test.jsonl') as f:
            for line in f:
                r = json.loads(line)
                rows[b, r['pair_id']] = r
    return rows


# ---------------------------------------------------------------- Qwen3.8-27B (vLLM)

def qwen(sampling='greedy'):
    from vllm import LLM, SamplingParams
    import qwen38_baseline
    qwen38_baseline.register_models()
    pairs, warmup = load_sample()
    api = api_rows('runs/qwen38_b0_run01_20261022/qwen38/B0')
    body0 = qwen38_baseline.payload(pairs[0]['case'])
    schema = body0['response_format']['json_schema']['schema']
    # 'api': the API requests set no temperature, so the provider used the model defaults (generation_config.json)
    decode = ({'temperature': 0} if sampling == 'greedy' else
              {k: v for k, v in json.loads((snapshot('Qwen/Qwen3.8-27B')/'generation_config.json').read_text()).items()
               if k in ('temperature', 'top_p', 'top_k')})
    try:
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(**decode, max_tokens=body0['max_tokens'], structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(**decode, max_tokens=body0['max_tokens'], guided_decoding=GuidedDecodingParams(json=schema))

    def messages(c):
        body = qwen38_baseline.payload(c)
        hit = api.get((c['benchmark'], c['pair_id']))
        if hit is not None:
            assert hit['request_hash'] == protocol.digest(body), 'request differs from the API run'
        return body['messages']

    def result(o):
        out = o.outputs[0]
        tokens = {'input_tokens': len(o.prompt_token_ids), 'output_tokens': len(out.token_ids)}
        if sampling == 'greedy':
            return tokens
        try:
            ans = json.loads(out.text)
            pred = int(ans['match']) if set(ans) == {'match'} and type(ans['match']) is bool else None
        except (json.JSONDecodeError, TypeError):
            pred = None
        return {'pred': pred, 'text': out.text, 'finish_reason': out.finish_reason, **tokens}

    gdn = os.environ.get('LT_GDN_PREFILL_BACKEND')  # fallback if the default kernel cannot load
    llm = LLM(model=str(snapshot('Qwen/Qwen3.8-27B')), dtype='bfloat16', max_model_len=16384,
              gpu_memory_utilization=0.90, max_num_seqs=512, limit_mm_per_prompt={'image': 0, 'video': 0},
              seed=int(os.environ.get('LT_SEED', 0)),  # repetitions of the sampled run use their own seed
              **({'gdn_prefill_backend': gdn} if gdn else {}))
    kw = {'chat_template_kwargs': {'enable_thinking': False}, 'use_tqdm': False}
    chat = lambda cases: [result(o) for o in llm.chat([messages(c) for c in cases], sp, **kw)]  # noqa: E731
    chat(warmup)
    for c in warmup[:50]:
        chat([c])
    env = environment()
    cfg = llm.llm_engine.vllm_config
    common = {'framework': 'vLLM ' + env.get('vllm', '') + ' offline LLM.chat', 'precision': 'bf16 (no quantization)',
              'max_tokens': body0['max_tokens'], 'decoding': decode, 'thinking': 'enable_thinking=False',
              'structured_output': 'JSON schema {match: boolean} (as the API request)',
              'max_model_len': 16384, 'gpu_memory_utilization': 0.90,
              'max_num_seqs': cfg.scheduler_config.max_num_seqs,
              'max_num_batched_tokens': cfg.scheduler_config.max_num_batched_tokens,
              'prefix_caching': bool(cfg.cache_config.enable_prefix_caching),
              'cuda_graphs': not cfg.model_config.enforce_eager,
              'gdn_prefill_backend': gdn or 'default (flashinfer, precompiled jit-cache)'}
    if sampling == 'api':  # decision check only: batched run with the API's sampling defaults
        rec = Recorder(run_name('qwen_b_apisampling'), {**common, 'batching': 'as qwen_b', 'seed': int(os.environ.get('LT_SEED', 0))})
        measure_batched(rec, pairs, chat, before=llm.reset_prefix_cache)
        rec.save(env)
        return
    rec = Recorder('qwen_a', {**common, 'batch_size': 1, 'note': 'one LLM.chat call per pair'})
    measure_unbatched(rec, pairs, lambda c: chat([c])[0])
    rec.save(env)
    rec = Recorder('qwen_b', {**common, 'batching': 'one LLM.chat call per benchmark block, continuous batching; prefix cache reset before each call'})
    measure_batched(rec, pairs, chat, before=llm.reset_prefix_cache)
    rec.save(env)


QWEN_MEDIUM_SUBSET = 25  # (a): first 25 sample pairs per benchmark, 200 pairs in all


def qwen_medium():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    import qwen38_baseline
    qwen38_baseline.register_models()
    pairs, warmup = load_sample()
    ckpt = snapshot('Qwen/Qwen3.8-27B')
    body0 = qwen38_baseline.payload(pairs[0]['case'])
    schema = body0['response_format']['json_schema']['schema']
    # Thinking with greedy decoding and no length limit can loop, so the model's own sampling defaults
    decode = {k: v for k, v in json.loads((ckpt/'generation_config.json').read_text()).items()
              if k in ('temperature', 'top_p', 'top_k')}
    sp = SamplingParams(**decode, max_tokens=None, structured_outputs=StructuredOutputsParams(json=schema))
    llm = LLM(model=str(ckpt), dtype='bfloat16', max_model_len=32768, gpu_memory_utilization=0.90, max_num_seqs=512,
              limit_mm_per_prompt={'image': 0, 'video': 0}, seed=0, reasoning_parser='qwen3')
    end_think = llm.get_tokenizer().convert_tokens_to_ids('</think>')
    kw = {'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}, 'use_tqdm': False}

    def result(o):
        out = o.outputs[0]
        ids = list(out.token_ids)
        reasoning = ids.index(end_think) if end_think in ids else len(ids)
        answer = out.text.split('</think>')[-1].strip()
        try:
            ans = json.loads(answer)
            pred = int(ans['match']) if set(ans) == {'match'} and type(ans['match']) is bool else None
        except (json.JSONDecodeError, TypeError):
            pred = None
        return {'pred': pred, 'answer': answer[:200], 'finish_reason': out.finish_reason,
                'input_tokens': len(o.prompt_token_ids), 'output_tokens': len(ids), 'reasoning_tokens': reasoning}

    chat = lambda cases: [result(o) for o in llm.chat([qwen38_baseline.payload(c)['messages'] for c in cases], sp, **kw)]  # noqa: E731
    chat(warmup)
    for c in warmup[:10]:
        chat([c])
    env = environment()
    cfg = llm.llm_engine.vllm_config
    common = {'framework': 'vLLM ' + env.get('vllm', '') + ' offline LLM.chat', 'precision': 'bf16 (no quantization)',
              'thinking': "enable_thinking=True, reasoning_effort='medium'", 'max_tokens': None,
              'max_model_len': 32768, 'decoding': decode,
              'structured_output': 'JSON schema {match: boolean} for the answer after </think> (reasoning_parser qwen3)',
              'max_num_seqs': cfg.scheduler_config.max_num_seqs,
              'max_num_batched_tokens': cfg.scheduler_config.max_num_batched_tokens,
              'prefix_caching': bool(cfg.cache_config.enable_prefix_caching), 'cuda_graphs': not cfg.model_config.enforce_eager}
    rec = Recorder('qwen_medium_b', {**common, 'batching': 'one LLM.chat call per benchmark block, continuous batching; prefix cache reset before each call'})
    measure_batched(rec, pairs, chat, before=llm.reset_prefix_cache)
    rec.save(env)
    subset, seen = [], {}
    for p in pairs:
        seen[p['benchmark']] = seen.get(p['benchmark'], 0) + 1
        if seen[p['benchmark']] <= QWEN_MEDIUM_SUBSET:
            subset.append(p)
    rec = Recorder('qwen_medium_a', {**common, 'batch_size': 1, 'pairs': f'{QWEN_MEDIUM_SUBSET} per benchmark, {len(subset)} in all'})
    measure_unbatched(rec, subset, lambda c: chat([c])[0])
    rec.save(env)


# ---------------------------------------------------------------- GLM-5.3-Flash (vLLM, 4 GPUs)

GLM_SUBSET = 125  # (a): first 125 sample pairs per benchmark (round-robin blocks), 1,000 pairs in all


def glm():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    import qwen38_baseline
    qwen38_baseline.register_models()
    pairs, warmup = load_sample()
    body0 = qwen38_baseline.payload(pairs[0]['case'])
    schema = body0['response_format']['json_schema']['schema']
    max_tokens = int(os.environ.get('LT_GLM_MAX_TOKENS', 1024))  # room for the 'low' reasoning before the JSON
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, structured_outputs=StructuredOutputsParams(json=schema))
    eager = os.environ.get('LT_EAGER') == '1'
    llm = LLM(model=str(snapshot('zai-org/GLM-5.3-Flash')), tensor_parallel_size=4, max_model_len=8192,
              gpu_memory_utilization=0.92, max_num_seqs=int(os.environ.get('LT_GLM_MAX_SEQS', 64)),
              disable_custom_all_reduce=True, enforce_eager=eager, reasoning_parser='glm47',
              limit_mm_per_prompt={'image': 0, 'video': 0}, seed=0)
    kw = {'chat_template_kwargs': {'reasoning_effort': 'low', 'clear_thinking': True}, 'use_tqdm': False}

    def result(o):
        out = o.outputs[0]
        text = out.text.split('</think>')[-1].strip()
        try:
            ans = json.loads(text)
            pred = int(ans['match']) if set(ans) == {'match'} and type(ans['match']) is bool else None
        except (json.JSONDecodeError, TypeError):
            pred = None
        return {'pred': pred, 'text': out.text[-400:], 'finish_reason': out.finish_reason,
                'input_tokens': len(o.prompt_token_ids), 'output_tokens': len(out.token_ids),
                'reasoning_chars': len(out.text.split('</think>')[0]) if '</think>' in out.text else 0}

    chat = lambda cases: [result(o) for o in llm.chat([qwen38_baseline.payload(c)['messages'] for c in cases], sp, **kw)]  # noqa: E731
    chat(warmup)
    for c in warmup[:20]:
        chat([c])
    env = environment()
    env['gpus'] = __import__('torch').cuda.device_count()
    cfg = llm.llm_engine.vllm_config
    common = {'framework': 'vLLM ' + env.get('vllm', '') + ' offline LLM.chat, tensor parallel 4 (NVLink pairs, PCIe between pairs)',
              'precision': 'native FP8 block-scaled weights (zai-org release), bf16 activations; no own quantization',
              'max_tokens': max_tokens, 'decoding': {'temperature': 0},
              'thinking': "not switchable; reasoning_effort='low' (lowest), clear_thinking=True",
              'structured_output': 'JSON schema {match: boolean} after the reasoning (reasoning_parser glm47)',
              'max_model_len': 8192, 'gpu_memory_utilization': 0.92,
              'max_num_seqs': cfg.scheduler_config.max_num_seqs,
              'max_num_batched_tokens': cfg.scheduler_config.max_num_batched_tokens,
              'prefix_caching': bool(cfg.cache_config.enable_prefix_caching), 'cuda_graphs': not eager,
              'workarounds': {k: os.environ.get(k) for k in ('VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER', 'VLLM_ALLREDUCE_USE_FLASHINFER',
                                                              'VLLM_ALLREDUCE_USE_SYMM_MEM')} | {'disable_custom_all_reduce': True}}
    subset, seen = [], {}
    for p in pairs:
        seen[p['benchmark']] = seen.get(p['benchmark'], 0) + 1
        if seen[p['benchmark']] <= GLM_SUBSET:
            subset.append(p)
    if not RUN:
        rec = Recorder('glm_a', {**common, 'batch_size': 1, 'pairs': f'{GLM_SUBSET} per benchmark, {len(subset)} in all'})
        measure_unbatched(rec, subset, lambda c: chat([c])[0])
        rec.save(env)
    rec = Recorder(run_name('glm_b'), {**common, 'batching': 'one LLM.chat call per benchmark block, continuous batching; prefix cache reset before each call'})
    measure_batched(rec, pairs, chat, before=llm.reset_prefix_cache)
    rec.save(env)


# ---------------------------------------------------------------- report

def f1(rows):
    tp = sum(r['pred'] == 1 and r['label'] == 1 for r in rows)
    fp = sum(r['pred'] == 1 and r['label'] == 0 for r in rows)
    fn = sum(r['pred'] != 1 and r['label'] == 1 for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def summarize(name):
    t = json.loads((OUT/f'{name}.timing.json').read_text())
    rows = [json.loads(line) for line in open(OUT/f'{name}.jsonl')]
    per_block = []
    for k in range(BLOCKS):
        g = [x for x in t['groups'] if x['block'] == k]
        per_block.append(statistics.mean(1000 * x['seconds'] / x['n'] for x in g))
    per_bench = {b: 1000 * sum(x['seconds'] for x in t['groups'] if x['benchmark'] == b) /
                 sum(x['n'] for x in t['groups'] if x['benchmark'] == b) for b in BENCHMARKS}
    s1k = statistics.mean(per_bench.values())
    seconds, n = sum(x['seconds'] for x in t['groups']), sum(x['n'] for x in t['groups'])
    out = {'settings': t['settings'], 'environment': t['environment'], 'pairs': n, 'wall_seconds': seconds,
           'seconds_per_1k': s1k, 'pairs_per_second': n / seconds,
           'pairs_per_second_from_seconds_per_1k': 1000 / s1k,
           'hours_per_1m': s1k * 1000 / 3600,
           'blocks_seconds_per_1k': per_block, 'blocks_sd': statistics.stdev(per_block),
           'blocks_cv': statistics.stdev(per_block) / statistics.mean(per_block),
           'per_benchmark_seconds_per_1k': per_bench,
           'input_tokens_mean': statistics.mean(r['input_tokens'] for r in rows),
           'input_tokens_mean_over_benchmarks': statistics.mean(
               statistics.mean(r['input_tokens'] for r in rows if r['benchmark'] == b) for b in BENCHMARKS)}
    if 'pred' in rows[0]:
        out['f1_macro'] = statistics.mean(f1([r for r in rows if r['benchmark'] == b]) for b in BENCHMARKS)
        out['f1_per_benchmark'] = {b: f1([r for r in rows if r['benchmark'] == b]) for b in BENCHMARKS}
    if rows and 'latency' in rows[0]:
        lat = sorted(r['latency'] for r in rows)
        out['latency_p50'], out['latency_p95'] = lat[len(lat) // 2], lat[int(.95 * len(lat))]
    if name.startswith(('qwen', 'glm')):
        if 'pred' in rows[0]:
            out['unparsed'] = sum(r['pred'] is None for r in rows)
        out['output_tokens_mean'] = statistics.mean(r['output_tokens'] for r in rows)
    if rows and 'reasoning_tokens' in rows[0]:
        tokens = sorted(r['reasoning_tokens'] for r in rows)
        out['reasoning_tokens'] = {'mean': statistics.mean(tokens), 'median': tokens[len(tokens) // 2],
                                   'p95': tokens[int(.95 * len(tokens))], 'max': tokens[-1]}
        out['finish_reasons'] = {k: sum(r['finish_reason'] == k for r in rows) for k in {r['finish_reason'] for r in rows}}
    return out, rows


def agreement(rows, api, prob=True):
    pairs = [(r, api[r['benchmark'], r['pair_id']]) for r in rows]
    out = {'pairs': len(pairs), 'decision_agreement': sum(r['pred'] == a['prediction'] for r, a in pairs) / len(pairs),
           'flips': sum(r['pred'] != a['prediction'] for r, a in pairs)}
    if prob:
        d = [abs(r['prob'] - a['probability_match']) for r, a in pairs]
        out.update(mean_abs_dprob=statistics.mean(d), max_abs_dprob=max(d))
    return out


def report():
    names = [p.name[:-len('.timing.json')] for p in sorted(OUT.glob('*.timing.json'))]
    result = {'description': __doc__, 'date': '2026-11-03', 'sample': json.loads((OUT/'sample.json').read_text())['seed'],
              'runs': {}, 'checks': {}}
    pplx_api = api_rows('runs/prompt12_run01_20261019/pplx_v12/B0')
    qwen_api = api_rows('runs/qwen38_b0_run01_20261022/qwen38/B0')
    for name in names:
        s, rows = summarize(name)
        result['runs'][name] = s
        if name.startswith('pplx'):
            result['checks'][name + '_vs_api_b0_run1'] = agreement(rows, pplx_api)
            result['checks'][name + '_vs_api_b0_run1']['api_input_tokens_mean'] = statistics.mean(
                pplx_api[r['benchmark'], r['pair_id']]['attempts'][-1]['response']['usage']['input_tokens'] for r in rows)
            result['checks'][name + '_api_f1_macro_same_pairs'] = statistics.mean(
                f1([{'pred': pplx_api[b, r['pair_id']]['prediction'], 'label': r['label']} for r in rows if r['benchmark'] == b])
                for b in BENCHMARKS)
        if name.startswith('qwen') and 'pred' in rows[0] and 'medium' not in name:
            result['checks'][name + '_vs_api_b0_run1'] = agreement(rows, qwen_api, prob=False)
            result['checks'][name + '_vs_api_b0_run1']['api_prompt_tokens_mean'] = statistics.mean(
                qwen_api[r['benchmark'], r['pair_id']]['attempts'][-1]['response']['usage']['prompt_tokens'] for r in rows)
            result['checks'][name + '_vs_api_b0_run1']['api_completion_tokens_mean'] = statistics.mean(
                qwen_api[r['benchmark'], r['pair_id']]['attempts'][-1]['response']['usage']['completion_tokens'] for r in rows)
            result['checks'][name + '_vs_api_b0_run1']['per_benchmark_agreement'] = {
                b: statistics.mean(r['pred'] == qwen_api[b, r['pair_id']]['prediction'] for r in rows if r['benchmark'] == b)
                for b in BENCHMARKS}
    if 'qwen_b_apisampling' in names:
        # The API requests set no temperature, so the provider sampled with the model defaults (generation_config:
        # temperature 1.0, top_p 0.95, top_k 20); qwen_b_apisampling samples the same way locally.
        runs = {k: api_rows(f'runs/qwen38_b0_run0{k}_20261022/qwen38/B0') for k in (1, 2, 3)}
        rows = summarize('qwen_b_apisampling')[1]
        keys = [(r['benchmark'], r['pair_id']) for r in rows]
        result['checks']['qwen_api_same_pairs'] = {
            'sampling': 'API request without temperature: provider default sampling (model generation_config temperature 1.0, top_p 0.95, top_k 20)',
            'run_to_run_agreement': {f'{i}-{j}': statistics.mean(runs[i][k]['prediction'] == runs[j][k]['prediction'] for k in keys)
                                     for i, j in ((1, 2), (1, 3), (2, 3))},
            'f1_macro': {f'run{i}': statistics.mean(f1([{'pred': runs[i][b, r['pair_id']]['prediction'], 'label': r['label']}
                                                        for r in rows if r['benchmark'] == b]) for b in BENCHMARKS) for i in runs},
            'f1_per_benchmark_run1': {b: f1([{'pred': runs[1][b, r['pair_id']]['prediction'], 'label': r['label']}
                                             for r in rows if r['benchmark'] == b]) for b in BENCHMARKS}}
        if name.startswith('ditto'):
            d = [abs(r['prob'] - r['reference_prob_fp32']) for r in rows]
            result['checks'][name + '_vs_fp32_eval'] = {'mean_abs_dprob': statistics.mean(d), 'max_abs_dprob': max(d)}
    result['three_runs'] = {}
    for base, names_ in REPEATS.items():
        if not all(n in result['runs'] for n in names_):
            continue
        runs = [result['runs'][n] for n in names_]
        rows = [{(r['benchmark'], r['pair_id']): r['pred'] for r in summarize(n)[1]} for n in names_]
        agree = {f'{i + 1}-{j + 1}': statistics.mean(rows[i][k] == rows[j][k] for k in rows[i])
                 for i, j in ((0, 1), (0, 2), (1, 2))}
        table = {'seconds_per_1k': [r['seconds_per_1k'] for r in runs],
                 'pairs_per_second': [r['pairs_per_second_from_seconds_per_1k'] for r in runs],
                 'hours_per_1m': [r['hours_per_1m'] for r in runs], 'f1_macro': [r['f1_macro'] for r in runs]}
        result['three_runs'][base] = {
            'runs': names_, 'pairs': runs[0]['pairs'], 'decision_agreement_between_runs': agree,
            **{k: {'runs': v, 'mean': statistics.mean(v), 'sd': statistics.stdev(v)} for k, v in table.items()}}
    result['large_llm_feasibility'] = {
        'labeling_paper_models': 'GPT-5.2 and Kimi K2.6 via API (Automatic-data-labeling/configs/labeling); neither GLM-5.3-Flash nor DeepSeek-V4',
        'cluster': '2 GPU nodes with 4 x H100 NVL 94 GB each (two nodes); NVLink pairs 0-1 and 2-3, PCIe between pairs; 4 GPUs per job allowed, walltime up to 5 days',
        'GLM-5.3-Flash': {'params_total': 321323031390, 'params_active': '18B (model card)', 'format': 'native FP8 e4m3, 128x128 blocks',
                          'weight_bytes': 328326771576, 'fits_one_node': True, 'job': 389867, 'runs': ['glm_a', 'glm_b'],
                          'earlier_attempt': '/path/to/glm53-dws12-20261017 (17-18 Oct): ran with vLLM 0.31, TP4, eager, max_num_seqs 4, '
                                             '~8 output tokens/s per request after fixes (CUDA_HOME, ninja, FlashInfer FP8 block GEMM and all-reduce off); '
                                             'hard part was getting a free 4-GPU node and the kernel fixes, not memory'},
        'DeepSeek-V4-Pro': {'params_total': 1598839674782, 'format': 'native FP8 + packed FP4 experts', 'repo_storage_bytes': 864800000000,
                            'fits_one_node': False,
                            'note': 'more than both H100 nodes together (8 x 94 GB = 752 GB); only possible with own quantization or more hardware'}}
    path = ROOT/'artifacts/results/local_throughput_20261103.json'
    write_json(path, result)
    for name, s in result['runs'].items():
        print(f"{name:18s} {s['pairs_per_second']:8.1f} pairs/s {s['seconds_per_1k']:8.2f} s/1k "
              f"{s['hours_per_1m']:7.2f} h/1M  blocks {['%.2f' % x for x in s['blocks_seconds_per_1k']]} "
              f"cv {100 * s['blocks_cv']:.1f}%  F1 {s.get('f1_macro', float('nan')):.3f}  tok {s['input_tokens_mean']:.0f}")
    print(json.dumps(result['checks'], indent=1))
    print(json.dumps(result['three_runs'], indent=1))


# ---------------------------------------------------------------- report on the full test sets (7 Nov)

FULL_RUNS = {'ditto': {'batch1': 'ditto_a_{r}', 'batched': 'ditto_b1024_{r}'},
             'pplx_v12': {'batch1': 'pplx_vllm_{r}_a', 'batched': 'pplx_vllm_{r}_b'},
             'qwen38': {'batched': 'qwen_b_apisampling_{r}'},
             'glm53flash': {'batched': 'glm_b_{r}'}}


def report_full():
    """Throughput on all 30,225 test pairs, three runs per row, in the layout of paper sources/throughput_local.json."""
    assert FULL and OUT != SUBSET_OUT, 'set LT_FULL=1 and LT_OUT'
    pplx_api = api_rows('runs/prompt12_run01_20261019/pplx_v12/B0')
    qwen_api = api_rows('runs/qwen38_b0_run01_20261022/qwen38/B0')
    table9 = json.loads((DITTO.parent/'summary_seed_mean.json').read_text())
    result = {'description': (
        'Local throughput on the full test sets of the eight benchmarks (30,225 pairs, the pairs of Table II), three runs '
        'per row (Aaron, 7 Nov 2026), written by local_throughput_20261103.py report_full (LT_FULL=1). Same settings as the '
        'batched rows of sources/throughput_local.json: one H100 NVL per model (GLM-5.3-Flash: four, tensor parallel 4), '
        '200 uncounted warm-up pairs, every benchmark split into three round-robin blocks, seconds per 1,000 pairs as the '
        'mean over the eight benchmarks. Ditto uses the checkpoints of the 2,000-pair Sol-labelled set (Table IX), run 1/2/3 '
        'with seed 42/52/62, so its F1 mean is the Table IX setting; Qwen3.8-27B samples as the API runs (seeds 0/1/2). '
        'Per row: mean, sample SD (ddof=1) and the three run values. Batch-1 rows exist for Ditto and PPLX only.'),
        'gpu': 'NVIDIA H100 NVL', 'pairs': None, 'models': {}, 'checks': {}}
    for model, kinds in FULL_RUNS.items():
        result['models'][model] = {}
        for kind, pattern in kinds.items():
            names = [pattern.format(r=r) for r in ('r1', 'r2', 'r3')]
            if not all((OUT/f'{n}.timing.json').exists() for n in names):
                continue
            runs = [summarize(n) for n in names]
            stats_ = [x[0] for x in runs]
            preds = [{(r['benchmark'], r['pair_id']): r['pred'] for r in x[1]} for x in runs]
            ms = lambda key: {'mean': statistics.mean(x[key] for x in stats_), 'sd': statistics.stdev(x[key] for x in stats_),  # noqa: E731
                              'runs': [x[key] for x in stats_]}
            row = {'runs': names, 'pairs': stats_[0]['pairs'], 'seconds_per_1k': ms('seconds_per_1k'),
                   'pairs_per_second': {'mean': statistics.mean(1000 / x['seconds_per_1k'] for x in stats_),
                                        'sd': statistics.stdev(1000 / x['seconds_per_1k'] for x in stats_),
                                        'runs': [1000 / x['seconds_per_1k'] for x in stats_]},
                   'hours_per_1m': ms('hours_per_1m'), 'f1_macro': ms('f1_macro'),
                   'f1_per_benchmark': {b: statistics.mean(x['f1_per_benchmark'][b] for x in stats_) for b in BENCHMARKS},
                   'blocks_cv': [x['blocks_cv'] for x in stats_], 'input_tokens_mean': stats_[0]['input_tokens_mean'],
                   'wall_seconds': [x['wall_seconds'] for x in stats_],
                   'decision_agreement_between_runs': {f'{i + 1}-{j + 1}': statistics.mean(preds[i][k] == preds[j][k] for k in preds[i])
                                                       for i, j in ((0, 1), (0, 2), (1, 2))},
                   'settings': [x['settings'] for x in stats_], 'environment': stats_[0]['environment']}
            if 'unparsed' in stats_[0]:
                row['unparsed'] = [x['unparsed'] for x in stats_]
                row['output_tokens_mean'] = [x['output_tokens_mean'] for x in stats_]
            result['models'][model][kind] = row
            result['pairs'] = result['pairs'] or row['pairs']
            if model == 'pplx_v12':
                result['checks'][f'pplx_{kind}_vs_api_b0_run1'] = [agreement(x[1], pplx_api) for x in runs]
            if model == 'qwen38':
                result['checks']['qwen_vs_api_b0_run1'] = [agreement(x[1], qwen_api, prob=False) for x in runs]
            if model == 'ditto' and kind == 'batched':
                result['checks']['ditto_vs_table9_set2000'] = {
                    'f1_per_benchmark_table9_mean': {b: table9[f'{b}/2000']['mean'] for b in BENCHMARKS},
                    'f1_macro_table9_mean': statistics.mean(table9[f'{b}/2000']['mean'] for b in BENCHMARKS),
                    'f1_macro_here_mean': row['f1_macro']['mean'],
                    'vs_fp32_eval': [{'mean_abs_dprob': statistics.mean(abs(r['prob'] - r['reference_prob_fp32']) for r in x[1]),
                                      'max_abs_dprob': max(abs(r['prob'] - r['reference_prob_fp32']) for r in x[1])} for x in runs]}
    path = ROOT/'artifacts/results/throughput_local_full_20261107.json'
    write_json(path, result)
    for model, kinds in result['models'].items():
        for kind, row in kinds.items():
            print(f"{model:11s} {kind:8s} n={row['pairs']} s/1k {row['seconds_per_1k']['mean']:8.2f} +- {row['seconds_per_1k']['sd']:.2f}  "
                  f"pairs/s {row['pairs_per_second']['mean']:8.1f} +- {row['pairs_per_second']['sd']:.1f}  "
                  f"F1 {row['f1_macro']['mean']:.4f} +- {row['f1_macro']['sd']:.4f}  agree {row['decision_agreement_between_runs']}")
    print(json.dumps(result['checks'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['sample', 'ditto', 'pplx', 'qwen', 'qwen-medium', 'glm', 'report', 'report_full'])
    ap.add_argument('--path', choices=['ref', 'fast', 'fast32k', 'vllm'], default='fast')
    ap.add_argument('--sampling', choices=['greedy', 'api'], default='greedy')
    a = ap.parse_args()
    if a.action == 'pplx':
        pplx(a.path)
    elif a.action == 'qwen':
        qwen(a.sampling)
    elif a.action == 'qwen-medium':
        qwen_medium()
    else:
        globals()[a.action]()
