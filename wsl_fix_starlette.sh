#!/bin/bash
source "$HOME/vllm-venv/bin/activate"
echo "=== downgrade fastapi+starlette to pre-_IncludedRouter combo ==="
pip install "fastapi==0.115.6" "starlette==0.41.3" 2>&1 | tail -6
echo "=== verify vLLM still imports ==="
python -c "import vllm, fastapi, starlette; print('vllm', vllm.__version__, 'fastapi', fastapi.__version__, 'starlette', starlette.__version__)" 2>&1 | tail -3
echo "=== launch engine + check /health STATUS CODE (not just exit) ==="
export VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=/tmp/vllm_fix.log
timeout 160 vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8196 \
  --max-model-len 4096 --max-num-seqs 16 --gpu-memory-utilization 0.24 \
  --dtype auto --enable-prefix-caching --enable-sleep-mode > "$LOG" 2>&1 &
VPID=$!
ok=0
for i in $(seq 1 150); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8196/health 2>/dev/null)
  if [ "$code" = "200" ]; then ok=1; echo "HEALTH 200 after ${i}s"; break; fi
  if ! kill -0 $VPID 2>/dev/null; then echo "process died at ${i}s"; break; fi
  sleep 1
done
if [ "$ok" = "1" ]; then
  echo "=== real generation ==="
  curl -s --max-time 30 http://127.0.0.1:8196/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Reply with exactly: MANTIC MIND LIVE"}],"max_tokens":16,"temperature":0}'
  echo ""
else
  echo "=== still failing; health-related log ==="
  grep -nE "IncludedRouter|health|Error|Exception" "$LOG" | tail -10
fi
kill $VPID 2>/dev/null; pkill -f "port 8196" 2>/dev/null
echo "=== DONE ==="
