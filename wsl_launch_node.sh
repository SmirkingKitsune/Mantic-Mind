#!/bin/bash
cd "$HOME/mm-run/node"
export MM_API_KEY=testkey123
export MM_LOAD_TIMEOUT_S=300
# RTX 5090 (Blackwell sm_120): FlashInfer's prebuilt sampling kernels reject
# SM 12.x under this CUDA; use vLLM's native PyTorch sampler instead. The node
# spawns vLLM as a child, so it inherits this env. Likely also needed on the
# DGX Sparks (also Blackwell).
export VLLM_USE_FLASHINFER_SAMPLER=0
# Keep stdin open (sleep) so --mode cli does not EOF-exit; server runs until killed.
sleep 1000000 | ./mantic-mind --mode cli > logs/console.log 2>&1
