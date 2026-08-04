# Qwen3-Coder-Next-FP8

Default model path: `/data/qinchong/models/Qwen3-Coder-Next-FP8`

```bash
cd ~/workspace/code/ktransformers

# Verify downloaded shards.
./experiments/models/qwen3-coder-next-fp8/verify_qwen3_coder_next_fp8.py

# Start or stop the OpenAI-compatible server on 127.0.0.1:30005.
./experiments/models/qwen3-coder-next-fp8/start_qwen3_coder_next_fp8.sh
./experiments/models/qwen3-coder-next-fp8/stop_qwen3_coder_next_fp8.sh
```

Runtime state is written to:

```text
~/workspace/code/ktransformers/experiments/artifacts/services/qwen3-coder-next-fp8/server.log
~/workspace/code/ktransformers/experiments/artifacts/services/qwen3-coder-next-fp8/server.pid
```

The launch defaults reproduce the tested FP8, TP2, 100-GPU-experts-per-layer
configuration. Environment variables in `launch_qwen3_coder_next_fp8.sh` can
override the serving parameters.
