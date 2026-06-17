#!/bin/bash
source "$HOME/vllm-venv/bin/activate"
export VLLM_SERVER_DEV_MODE=1
LOG=/tmp/vllm_manual.log
timeout 90 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8199 \
  --max-model-len 4096 --max-num-seqs 16 \
  --gpu-memory-utilization 0.24 \
  --dtype auto --enable-prefix-caching --enable-sleep-mode > "$LOG" 2>&1
echo "=== ROOT CAUSE lines (Error/CUDA/raise/Exception) ==="
grep -nE "Error|error|CUDA|raise |Exception|Traceback|EngineCore|failed|assert|RuntimeError|ValueError|ImportError|sleep" "$LOG" | grep -viE "no error|stderr" | head -40
echo ""
echo "=== first 5 lines mentioning EngineCore context ==="
grep -n "EngineCore" "$LOG" | head -5
