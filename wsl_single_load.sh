#!/bin/bash
KEY="testkey123"; NODE="http://127.0.0.1:7070"
PY="$HOME/vllm-venv/bin/python"
echo "VRAM before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB"
echo "loading one engine via node (signal fix in place)..."
RESP=$(curl -s --max-time 320 -H "Authorization: Bearer $KEY" -X POST "$NODE/api/node/load-model" \
  -H "Content-Type: application/json" -d '{
    "model_path":"Qwen/Qwen2.5-0.5B-Instruct","inference_backend":"vllm","agent_id":"agent-a",
    "vllm_settings":{"max_model_len":4096,"gpu_memory_utilization":0.24,"max_num_seqs":16,"enable_sleep_mode":true,"enable_prefix_caching":true}
  }')
echo "load response: $RESP"
echo "VRAM after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB"
echo "node slots:"
curl -s -H "Authorization: Bearer $KEY" "$NODE/api/node/status" | "$PY" -c "import sys,json;d=json.load(sys.stdin);[print('  slot',s['id'][:8],'state=',s.get('state'),'port=',s.get('port'),'agents=',s.get('agent_ids'),'metrics_valid=',s.get('engine_metrics_valid')) for s in d.get('slots',[])];print('  used_fraction=',d.get('vllm_gpu_fraction_used'))"
