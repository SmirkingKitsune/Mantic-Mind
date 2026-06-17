#!/bin/bash
NPID=$(pgrep -f "./mantic-mind --mode cli" | head -1)
echo "node pid: $NPID"
echo "=== node env (flashinfer / dev mode / api key) ==="
tr '\0' '\n' < "/proc/$NPID/environ" 2>/dev/null | grep -E "FLASHINFER|VLLM|MM_API|MM_LOAD" | sort
echo ""
echo "=== latest node.log vLLM errors ==="
tail -25 "$HOME/mm-run/node/logs/node.log" 2>/dev/null
