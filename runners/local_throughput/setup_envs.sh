#!/bin/bash
set -e
export UV_CACHE_DIR=/path/to/lt_envs/uv-cache UV_LINK_MODE=copy
cd /path/to/lt_envs
uv venv -p 3.12 pplx
VIRTUAL_ENV=/path/to/lt_envs/pplx uv pip install torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 safetensors==0.8.0 pillow==12.3.0 huggingface-hub==1.31.0 flash-linear-attention==0.5.2 numpy pandas pyyaml
echo PPLX_DONE
uv venv -p 3.12 vllm
VIRTUAL_ENV=/path/to/lt_envs/vllm uv pip install vllm==0.33.0
rm -rf /path/to/lt_envs/uv-cache
echo SETUP_DONE
