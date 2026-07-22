#!/bin/bash
source "$HOME/vllm-venv/bin/activate"
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=/tmp/vllm_test2.log
echo "=== vllm serve with FlashInfer sampler disabled ==="
timeout 180 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8198 \
  --max-model-len 4096 --max-num-seqs 16 \
  --gpu-memory-utilization 0.24 \
  --dtype auto --enable-prefix-caching --enable-sleep-mode > "$LOG" 2>&1 &
VPID=$!
# Poll health up to 170s
ok=0
for i in $(seq 1 170); do
  if curl -s --max-time 2 http://127.0.0.1:8198/health >/dev/null 2>&1; then ok=1; echo "HEALTHY after ${i}s"; break; fi
  if ! kill -0 $VPID 2>/dev/null; then echo "process died at ${i}s"; break; fi
  sleep 1
done
if [ "$ok" = "1" ]; then
  echo "=== real generation ==="
  curl -s --max-time 30 http://127.0.0.1:8198/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hi in 3 words"}],"max_tokens":16,"temperature":0}'
  echo ""
  echo "=== /metrics sample ==="
  curl -s --max-time 5 http://127.0.0.1:8198/metrics | grep -E "num_requests_running|num_requests_waiting" | grep -v "^#" | head
fi
echo "=== last errors if any ==="
grep -nE "Error|raise |Exception|failed|SM 12" "$LOG" | grep -viE "no error" | tail -8
kill $VPID 2>/dev/null; pkill -f "port 8198" 2>/dev/null
echo "=== DONE ==="
