"""vLLM plugin: pplx-decider-v1.2 (joint inline-readout protocol) as a token_classify pooling model.

vLLM 0.33 serves Decision 3.0 checkpoints (a bare Qwen3_5Model, readout.safetensors, decision_config.json) as
D3ForSequenceClassification, which reads the codes at the last prompt token (decision_config format_version 1).
pplx-decider-v1.2 (format_version 2) reads them at each <|fim_suffix|> marker instead. This plugin keeps D3's
backbone, attention contract and weight loading and only swaps the pooler.
"""


def register():
    from vllm import ModelRegistry
    from vllm.model_executor.models import config

    from .model import PplxJointDeciderConfig

    ModelRegistry.register_model("PplxJointDecider", "pplx_vllm.model:PplxJointDecider")
    config.MODELS_CONFIG_MAP["PplxJointDecider"] = PplxJointDeciderConfig
