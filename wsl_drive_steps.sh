#!/bin/bash
# Drive a live node+control+vLLM deployment to confirm steps 1-4.
# Assumes node (7070) and control (9090) are already running.
NODE="http://127.0.0.1:7070"
CTRL="http://127.0.0.1:9090"
KEY="testkey123"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PY="$HOME/vllm-venv/bin/python"
AUTH=(-H "Authorization: Bearer $KEY")
hr(){ echo "------------------------------------------------------------"; }
jq_(){ "$PY" -c "import sys,json;d=json.load(sys.stdin);print($1)" 2>/dev/null; }

# load_engine <agent_id> <max_model_len> <util>  -> prints slot_id (or empty)
load_engine(){
  local aid=$1 mml=$2 util=$3
  curl -s --max-time 320 "${AUTH[@]}" -X POST "$NODE/api/node/load-model" \
    -H "Content-Type: application/json" -d "{
      \"model_path\":\"$MODEL\",
      \"inference_backend\":\"vllm\",
      \"agent_id\":\"$aid\",
      \"vllm_settings\":{\"max_model_len\":$mml,\"gpu_memory_utilization\":$util,
        \"max_num_seqs\":16,\"enable_sleep_mode\":true,\"enable_prefix_caching\":true}
    }"
}
vram(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
nstatus(){ curl -s --max-time 10 "${AUTH[@]}" "$NODE/api/node/status"; }

echo "######## PHASE 0: health + control registration ########"
for i in $(seq 1 30); do
  if curl -s --max-time 5 "${AUTH[@]}" "$NODE/api/node/status" >/dev/null 2>&1; then break; fi
  sleep 1
done
echo "node status reachable. budget fields:"
nstatus | jq_ "'vllm_gpu_budget=%s vllm_gpu_fraction_used=%s max_slots=%s'%(d.get('vllm_gpu_budget'),d.get('vllm_gpu_fraction_used'),d.get('max_slots'))"
echo "register node with control:"
curl -s --max-time 10 -X POST "$CTRL/v1/nodes" -H "Content-Type: application/json" \
  -d "{\"url\":\"$NODE\",\"api_key\":\"$KEY\",\"remember\":false}"; echo
sleep 6
echo "control sees node (with new fields):"
curl -s --max-time 10 "$CTRL/v1/nodes" | jq_ "'; '.join('node %s budget=%s used=%s slots=%d'%(n.get('id','?')[:8],n.get('vllm_gpu_budget'),n.get('vllm_gpu_fraction_used'),len(n.get('slots',[]))) for n in (d if isinstance(d,list) else d.get('nodes',[])))"

hr; echo "######## STEP 1: GPU budget (0.50) ########"
echo "VRAM before: $(vram) MiB"
SA=$(load_engine agent-a 4096 0.24 | jq_ "d.get('slot_id','')"); echo "load A (mml4096,0.24) -> slot=$SA"
SB=$(load_engine agent-b 2048 0.24 | jq_ "d.get('slot_id','')"); echo "load B (mml2048,0.24) -> slot=$SB"
echo "used now: $(nstatus | jq_ "d.get('vllm_gpu_fraction_used')")  VRAM: $(vram) MiB"
echo "load C (mml1024,0.24) -> expect REJECTION:"
RC=$(load_engine agent-c 1024 0.24); echo "  response: $(echo "$RC" | jq_ "d.get('error','(no error)')+' | detail='+str(d.get('detail',''))")"
echo "  slot_id present? $(echo "$RC" | jq_ "bool(d.get('slot_id'))")  (expect False)"

hr; echo "######## STEP 4: engine /metrics scraped into status ########"
sleep 4
echo "per-slot engine metrics from NODE status:"
nstatus | jq_ "chr(10).join('  slot %s port=%s agents=%s metrics_valid=%s running=%s waiting=%s kv=%.3f'%(s['id'][:8],s.get('port'),s.get('agent_ids'),s.get('engine_metrics_valid'),s.get('num_requests_running'),s.get('num_requests_waiting'),s.get('kv_cache_usage',0)) for s in d.get('slots',[]))"
echo "same metrics propagated to CONTROL (/v1/nodes):"
curl -s --max-time 10 "$CTRL/v1/nodes" | jq_ "chr(10).join('  node %s budget=%s used=%s'%(n.get('id','?')[:8],n.get('vllm_gpu_budget'),n.get('vllm_gpu_fraction_used'))+chr(10)+chr(10).join('    slot %s metrics_valid=%s running=%s waiting=%s kv=%.3f'%(s['id'][:8],s.get('engine_metrics_valid'),s.get('num_requests_running'),s.get('num_requests_waiting'),s.get('kv_cache_usage',0)) for s in n.get('slots',[])) for n in (d if isinstance(d,list) else []))"

hr; echo "######## STEP 2: shared engine (attach) ########"
echo "slots before share: $(nstatus | jq_ "len(d.get('slots',[]))")"
SS=$(load_engine agent-share 4096 0.24 | jq_ "d.get('slot_id','')")
echo "load agent-share with A's EXACT settings -> slot=$SS  (A was $SA)"
echo "  same slot as A? $([ "$SS" = "$SA" ] && echo YES || echo NO)"
echo "slots after share: $(nstatus | jq_ "len(d.get('slots',[]))")  (expect unchanged - no new engine)"
echo "A's slot agents now: $(nstatus | jq_ "[s.get('agent_ids') for s in d.get('slots',[]) if s['id']=='$SA']")"

hr; echo "######## STEP 3: sleep / wake ########"
echo "VRAM before sleep: $(vram) MiB"
echo "suspend A's slot ($SA):"
curl -s --max-time 60 "${AUTH[@]}" -X POST "$NODE/api/node/suspend-slot" -H "Content-Type: application/json" -d "{\"slot_id\":\"$SA\"}" | jq_ "d";
sleep 4
echo "A slot state after suspend: $(nstatus | jq_ "[(s.get('state'),s.get('sleeping')) for s in d.get('slots',[]) if s['id']=='$SA']")"
echo "VRAM after sleep: $(vram) MiB  (expect lower - weights offloaded)"
echo "restore agent-a (wake):"
RW=$(curl -s --max-time 320 "${AUTH[@]}" -X POST "$NODE/api/node/restore-slot" -H "Content-Type: application/json" \
  -d "{\"model_path\":\"$MODEL\",\"inference_backend\":\"vllm\",\"agent_id\":\"agent-a\",\"kv_cache_path\":\"\",
       \"vllm_settings\":{\"max_model_len\":4096,\"gpu_memory_utilization\":0.24,\"max_num_seqs\":16,\"enable_sleep_mode\":true,\"enable_prefix_caching\":true}}")
echo "  restore -> slot=$(echo "$RW" | jq_ "d.get('slot_id','')")"
sleep 3
echo "A slot state after wake: $(nstatus | jq_ "[(s.get('state'),s.get('sleeping')) for s in d.get('slots',[]) if s['id']=='$SA']")"
echo "VRAM after wake: $(vram) MiB  (expect back up)"

hr; echo "######## BONUS: real generation from engine A ########"
APORT=$(nstatus | jq_ "[s.get('port') for s in d.get('slots',[]) if s['id']=='$SA'][0]")
echo "engine A on port $APORT; asking it to complete a prompt:"
curl -s --max-time 60 "http://127.0.0.1:$APORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: MANTIC MIND LIVE\"}],\"max_tokens\":16,\"temperature\":0}" \
  | jq_ "'  completion: '+d['choices'][0]['message']['content']"

hr; echo "######## DONE ########"
