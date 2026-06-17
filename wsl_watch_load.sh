#!/bin/bash
KEY="testkey123"; NODE="http://127.0.0.1:7070"
# Fire a load in the background (don't wait).
curl -s --max-time 320 -H "Authorization: Bearer $KEY" -X POST "$NODE/api/node/load-model" \
  -H "Content-Type: application/json" -d '{
    "model_path":"Qwen/Qwen2.5-0.5B-Instruct","inference_backend":"vllm","agent_id":"agent-w",
    "vllm_settings":{"max_model_len":4096,"gpu_memory_utilization":0.24,"max_num_seqs":16,"enable_sleep_mode":true}
  }' > /tmp/watch_load_resp.txt 2>&1 &
echo "load fired; sampling for 75s..."
for t in 5 10 15 25 40 55 75; do
  sleep_to=$t
  while [ "$(cut -d. -f1 < /proc/uptime)" -lt 0 ]; do :; done
  sleep 1
  : # spacing handled below
done >/dev/null 2>&1
# Simpler: sample every 10s up to 80s
for i in 1 2 3 4 5 6 7 8; do
  sleep 10
  echo "=== t=$((i*10))s ==="
  echo "  GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB"
  echo "  vllm procs:"
  ps -eo pid,ppid,etimes,comm,args | grep -iE "vllm|EngineCore" | grep -v grep | grep -v "load-model" | sed 's/^/    /' | cut -c1-160 | head -6
  P=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8100/health 2>/dev/null)
  echo "  engine :8100/health -> $P"
done
echo "=== load response (if returned) ==="
cat /tmp/watch_load_resp.txt 2>/dev/null
