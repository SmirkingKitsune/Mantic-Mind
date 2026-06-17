#!/bin/bash
# Reproduce vLLM startup under the NODE's exact environment.
NPID=$(pgrep -f "./mantic-mind --mode cli" | head -1)
echo "node pid: $NPID"
echo "=== node's full env ==="
tr '\0' '\n' < "/proc/$NPID/environ" 2>/dev/null | sort
echo ""
echo "=== launch vllm with node's exact env (env -i + node environ) + dev mode ==="
# Build env args from node environ, add VLLM_SERVER_DEV_MODE=1 (node sets this for sleep)
mapfile -t NENV < <(tr '\0' '\n' < "/proc/$NPID/environ" 2>/dev/null)
LOG=/tmp/vllm_envrepro.log
env -i "${NENV[@]}" VLLM_SERVER_DEV_MODE=1 \
  /home/ryanm/vllm-venv/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8197 \
  --max-model-len 4096 --max-num-seqs 16 --gpu-memory-utilization 0.24 \
  --dtype auto --enable-prefix-caching --enable-sleep-mode > "$LOG" 2>&1 &
VPID=$!
ok=0
for i in $(seq 1 150); do
  if curl -s --max-time 2 http://127.0.0.1:8197/health >/dev/null 2>&1; then ok=1; echo "HEALTHY after ${i}s under node env"; break; fi
  if ! kill -0 $VPID 2>/dev/null; then echo "process DIED at ${i}s under node env"; break; fi
  sleep 1
done
if [ "$ok" != "1" ]; then
  echo "=== NOT healthy; last 25 log lines ==="
  tail -25 "$LOG"
fi
kill $VPID 2>/dev/null; pkill -f "port 8197" 2>/dev/null
echo "=== DONE ==="
