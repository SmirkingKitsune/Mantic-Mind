#!/bin/bash
echo "node proc: $(pgrep -af './mantic-mind --mode cli' | grep -v control | head -1)"
code=$(curl -s -o /tmp/ns.json -w "%{http_code}" --max-time 5 -H "Authorization: Bearer testkey123" http://127.0.0.1:7070/api/node/status)
echo "status http: $code"
if [ "$code" = "200" ]; then
  "$HOME/vllm-venv/bin/python" -c "import json;d=json.load(open('/tmp/ns.json'));print('budget',d.get('vllm_gpu_budget'),'used',d.get('vllm_gpu_fraction_used'),'slots',len(d.get('slots',[])))"
fi
echo "control http: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:9090/v1/nodes)"
