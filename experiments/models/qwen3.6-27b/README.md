# Qwen3.6-27B

This model is the dense, pure-GPU comparison for the CPU/GPU MoE serving
experiments. Weights live at `/data/qinchong/models/Qwen3.6-27B`; scripts and
runtime artifacts stay in this repository.

The initial launch uses TP2, 128K total tokens, eight running requests, Triton
attention, Qwen reasoning parsing, and the Qwen3 Coder tool-call parser. MTP is
left disabled for the baseline run.

Mooncake capacity and Poisson experiments write under
`experiments/artifacts/mooncake-toolagent/results/qwen3.6-27b`. The Poisson
curve explicitly disables thinking so its request behavior matches the prior
non-thinking serving tests.

```bash
cd ~/workspace/code/ktransformers
./experiments/models/qwen3.6-27b/verify_qwen36_27b.py
./experiments/models/qwen3.6-27b/start_qwen36_27b.sh
./experiments/models/qwen3.6-27b/stop_qwen36_27b.sh
```
