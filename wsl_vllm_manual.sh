#!/bin/bash
# Reproduce the node's exact vLLM launch to capture the real crash output.
source "$HOME/vllm-venv/bin/activate"
export VLLM_SERVER_DEV_MODE=1
echo "=== launching vllm serve (node's args) ==="
timeout 90 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8199 \
  --max-model-len 4096 --max-num-seqs 16 \
  --gpu-memory-utilization 0.24 \
  --dtype auto \
  --enable-prefix-caching \
  --enable-sleep-mode 2>&1 | tail -45
echo "=== exit: ${PIPESTATUS[0]} (124=timeout-still-running=GOOD) ==="
