#!/bin/bash
for i in $(seq 1 25); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 -H "Authorization: Bearer testkey123" http://127.0.0.1:7070/api/node/status)
  if [ "$code" = "200" ]; then
    echo "NODE UP after ${i}s (status 200 with key)"
    curl -s -H "Authorization: Bearer testkey123" http://127.0.0.1:7070/api/node/status \
      | "$HOME/vllm-venv/bin/python" -c "import sys,json;d=json.load(sys.stdin);print('node_id=%s budget=%s used=%s max_slots=%s vllm_path=%s'%(d.get('node_id','')[:8],d.get('vllm_gpu_budget'),d.get('vllm_gpu_fraction_used'),d.get('max_slots'),d.get('vllm_server_path')))"
    break
  fi
  sleep 1
done
echo "--- node console tail ---"
tail -12 "$HOME/mm-run/node/logs/console.log" 2>/dev/null
