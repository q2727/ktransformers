# MiniMax-M2.5

Official FP8 checkpoint: `/data/qinchong/models/MiniMax-M2.5`.

The launch defaults take the official KT MiniMax-M2.5 example as the baseline
and map it once to this target's 2×48 GB RTX 4090 and dual-socket EPYC topology:
TP=2, two NUMA-local CPU pools, 120 physical CPU threads, 30 GPU experts per
layer, max-running=8, and a 150K token pool. Perform only a startup/128K
feasibility check before the formal run; do not add a parameter sweep.

```bash
experiments/models/minimax-m2.5/start_minimax_m25.sh
experiments/models/minimax-m2.5/stop_minimax_m25.sh
```
