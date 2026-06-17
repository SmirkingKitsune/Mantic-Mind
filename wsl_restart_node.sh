#!/bin/bash
# Kill any running node, relaunch with the rebuilt binary, wait until up.
pkill -9 -f "./mantic-mind --mode cli" 2>/dev/null
pkill -f "wsl_launch_node" 2>/dev/null
sleep 2
cd "$HOME/mm-run/node"
export MM_API_KEY=testkey123
export MM_LOAD_TIMEOUT_S=300
export VLLM_USE_FLASHINFER_SAMPLER=0
nohup bash -c 'sleep 1000000 | ./mantic-mind --mode cli > logs/console.log 2>&1' >/dev/null 2>&1 &
disown
for i in $(seq 1 20); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 -H "Authorization: Bearer testkey123" http://127.0.0.1:7070/api/node/status)
  if [ "$code" = "200" ]; then echo "NODE UP after ${i}s"; break; fi
  sleep 1
done
echo "node pid: $(pgrep -f './mantic-mind --mode cli' | head -1)"
