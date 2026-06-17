#!/bin/bash
echo "=== all vllm-related processes ==="
ps -eo pid,ppid,etimes,rss,comm,args | grep -iE "vllm|EngineCore|VLLM" | grep -v grep | head -30
echo ""
echo "=== GPU memory / processes ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head
echo ""
echo "=== port 8100 listener? ==="
ss -ltnp 2>/dev/null | grep -E ":810[0-9]" || echo "(no listener on 8100-8109)"
