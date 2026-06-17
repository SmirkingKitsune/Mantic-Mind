#!/bin/bash
for i in $(seq 1 20); do
  if curl -s --max-time 3 http://127.0.0.1:9090/v1/nodes >/dev/null 2>&1; then
    echo "CONTROL UP after ${i}s"
    echo "nodes: $(curl -s http://127.0.0.1:9090/v1/nodes)"
    break
  fi
  sleep 1
done
echo "--- control console tail ---"
tail -8 "$HOME/mm-run/control/logs/console.log" 2>/dev/null
echo "--- control log tail ---"
tail -8 "$HOME/mm-run/control/logs/control.log" 2>/dev/null
