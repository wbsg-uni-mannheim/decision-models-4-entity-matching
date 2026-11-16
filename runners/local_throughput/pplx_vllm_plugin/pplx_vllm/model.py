"""pplx-decider-v1.2 in vLLM: D3's noncausal Qwen3.5 backbone with the readout applied at every marker token.

Load with
    LLM(model=<checkpoint>, runner='pooling', hf_overrides={'architectures': ['PplxJointDecider'],
                                                            'pplx_marker_id': <id of <|fim_suffix|>>})
and call llm.encode(TokensPrompt(prompt_token_ids=...), pooling_task='token_classify'). Prompts are built by the
checkpoint's own pplx_decider Encoder. The output per request is a [questions, 255] tensor of raw readout logits,
one row per marker in prompt order; the caller divides by the saved temperature and takes the softmax over the
question's valid codes, as pplx_decider does.
"""

import json
from collections.abc import Set
from pathlib import Path

import torch

from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig
from vllm.model_executor.models.d3 import D3ForSequenceClassification, D3ProcessingInfo
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata


class PplxJointDeciderConfig(Qwen3_5ForConditionalGenerationConfig):
    """The attention contract and label space of a joint (format_version 2) decision checkpoint."""

    @staticmethod
    def verify_and_update_model_config(model_config) -> None:
        decision = json.loads((Path(model_config.model)/'decision_config.json').read_text())
        joint = json.loads((Path(model_config.model)/'joint_config.json').read_text())
        if decision.get('format_version') != 2 or joint.get('template') != 'joint-question-inline-readout-v1':
            raise ValueError('PplxJointDecider expects a joint inline-readout checkpoint (decision_config v2)')
        if decision.get('attention_mode') != 'noncausal_full_attention':
            raise ValueError(f"unsupported attention_mode {decision.get('attention_mode')!r}")
        if not isinstance(getattr(model_config.hf_config, 'pplx_marker_id', None), int):
            raise ValueError("pass hf_overrides={'pplx_marker_id': <token id of the readout marker>}")
        # Full attention layers read the whole request in both directions; the Gated DeltaNet layers stay causal.
        for c in {id(c): c for c in (model_config.hf_config, model_config.hf_text_config)}.values():
            c.is_causal = False
        model_config.hf_config.id2label = dict(enumerate(decision['codes']))
        if model_config.pooler_config.use_activation is None:
            model_config.pooler_config.use_activation = False


class MarkerReadoutPooler(Pooler):
    """Applies the readout at each marker of a finished prompt. Never raises (it runs in the engine core)."""

    def __init__(self, readout: torch.nn.Module, marker_id: int) -> None:
        super().__init__()
        self.pooling = AllPool()
        self.readout = readout
        self.marker_id = marker_id

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {'token_classify'}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata) -> PoolerOutput:
        outputs = []
        for hidden, token_ids in zip(self.pooling(hidden_states, pooling_metadata),
                                     pooling_metadata.get_prompt_token_ids_cpu()):
            if hidden is None:
                outputs.append(None)
                continue
            markers = (token_ids[:hidden.shape[0]] == self.marker_id).nonzero().flatten()
            if markers.numel() == 0:  # e.g. the profiling dummy
                outputs.append(hidden.new_empty(0, dtype=torch.float32))
                continue
            rows = hidden[markers.to(hidden.device)]
            outputs.append(self.readout(rows.to(self.readout.weight.dtype)).float())
        return outputs


@default_pooling_type(tok_pooling_type='ALL')
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=D3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class PplxJointDecider(D3ForSequenceClassification):
    def _init_pooler(self, vllm_config, prefix: str = '') -> Pooler:
        super()._init_pooler(vllm_config, prefix)  # builds self.score, the [255, hidden] readout
        return MarkerReadoutPooler(self.score, vllm_config.model_config.hf_config.pplx_marker_id)
