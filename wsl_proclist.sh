#!/bin/bash
echo "=== all mantic-mind node processes ==="
ps -eo pid,ppid,etimes,args | grep -E "mantic-mind( |$)|./mantic-mind" | grep -v control | grep -v grep
echo ""
echo "=== all mantic-mind-control processes ==="
ps -eo pid,ppid,etimes,args | grep "mantic-mind-control" | grep -v grep
echo ""
echo "=== node_ids each is using (from their logs) ==="
grep -h "Local node identity" "$HOME/mm-run/node/logs/node.log" 2>/dev/null | tail -5
echo ""
echo "=== control's discovered-nodes view ==="
curl -s --max-time 5 http://127.0.0.1:9090/v1/nodes/discovered 2>/dev/null | "$HOME/vllm-venv/bin/python" -c "import sys,json;d=json.load(sys.stdin);[print('  discovered:',n.get('node_id','?')[:20],n.get('url')) for n in (d if isinstance(d,list) else d.get('nodes',d.get('discovered',[])))]" 2>/dev/null || echo "(could not parse discovered)"
echo ""
echo "=== listeners on discovery port 7072 ==="
ss -lunp 2>/dev/null | grep 7072 || echo "(none visible)"
