#!/bin/bash
echo "=== node console.log (last 60) ==="
tail -60 "$HOME/mm-run/node/logs/console.log" 2>/dev/null
echo ""
echo "=== node.log (last 40) ==="
tail -40 "$HOME/mm-run/node/logs/node.log" 2>/dev/null
echo ""
echo "=== any vllm processes still alive? ==="
pgrep -af "vllm" | head
