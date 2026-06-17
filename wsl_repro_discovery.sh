#!/bin/bash
# Clean relaunch of ONE control + ONE node, then inspect discovered vs registered.
PY="$HOME/vllm-venv/bin/python"

echo "=== ensure nothing running ==="
pkill -9 -f "mantic-mind-control --mode cli" 2>/dev/null
pkill -9 -f "./mantic-mind --mode cli" 2>/dev/null
pkill -9 -f "sleep 1000000" 2>/dev/null
sleep 2
echo "node procs: $(pgrep -fc './mantic-mind --mode cli')  control procs: $(pgrep -fc 'mantic-mind-control --mode cli')"

echo "=== launch control ==="
cd "$HOME/mm-run/control"
nohup bash -c 'sleep 1000000 | ./mantic-mind-control --mode cli > logs/console.log 2>&1' >/dev/null 2>&1 &
disown
echo "=== launch node ==="
cd "$HOME/mm-run/node"
export MM_API_KEY=testkey123 MM_LOAD_TIMEOUT_S=300 VLLM_USE_FLASHINFER_SAMPLER=0
nohup bash -c 'sleep 1000000 | ./mantic-mind --mode cli > logs/console.log 2>&1' >/dev/null 2>&1 &
disown

echo "=== wait for both up ==="
for i in $(seq 1 25); do
  c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:9090/v1/nodes)
  n=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 -H "Authorization: Bearer testkey123" http://127.0.0.1:7070/api/node/status)
  if [ "$c" = "200" ] && [ "$n" = "200" ]; then echo "both up after ${i}s"; break; fi
  sleep 1
done

echo "=== discovered BEFORE registering the node ==="
sleep 6   # let a beacon or two arrive
curl -s http://127.0.0.1:9090/v1/nodes/discovered | "$PY" -c "import sys,json;d=json.load(sys.stdin);ns=d if isinstance(d,list) else d.get('nodes',d.get('discovered',[]));print('count=',len(ns));[print('  ',n.get('node_id','?')[:24],n.get('url')) for n in ns]"

echo "=== register the real node ==="
curl -s -X POST http://127.0.0.1:9090/v1/nodes -H "Content-Type: application/json" -d '{"url":"http://127.0.0.1:7070","api_key":"testkey123"}'; echo

echo "=== discovered AFTER registering (should drop the real node by URL) ==="
sleep 6
curl -s http://127.0.0.1:9090/v1/nodes/discovered | "$PY" -c "import sys,json;d=json.load(sys.stdin);ns=d if isinstance(d,list) else d.get('nodes',d.get('discovered',[]));print('count=',len(ns));[print('  ',n.get('node_id','?')[:24],n.get('url')) for n in ns]"
echo "=== registered nodes ==="
curl -s http://127.0.0.1:9090/v1/nodes | "$PY" -c "import sys,json;d=json.load(sys.stdin);print('count=',len(d));[print('  ',n.get('id','?')[:24],n.get('url'),'connected=',n.get('connected')) for n in d]"
echo "=== how many node processes / broadcasters actually running ==="
pgrep -af './mantic-mind --mode cli' | grep -v control
echo "DONE"
