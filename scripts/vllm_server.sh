#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for testing and development.
#
# Usage:
#   bash scripts/vllm_server.sh                       
#   VLLM_TEST_MODEL=Qwen/Qwen2-0.5B-Instruct bash scripts/vllm_server.sh
#   VLLM_PORT=9000 bash scripts/vllm_server.sh
set -euo pipefail

MODEL="${VLLM_TEST_MODEL:-Qwen/Qwen2-0.5B-Instruct}"
PORT="${VLLM_PORT:-7775}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.3}"

echo "Starting vLLM server: model=$MODEL port=$PORT gpu_util=$GPU_UTIL"
exec uv run --active python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL"
