# Qwen3.5-122B-A10B on gpu08

## Current state

- Host: `qinchong@100.64.0.3`
- Code: `~/workspace/code/ktransformers`
- Runtime: Python 3.12, Torch 2.9.1+cu128, CuDNN 9.16.0, `kt-kernel==0.6.3.post1`, `sglang-kt==0.6.3.post1`
- CPU backend: AVX512_BF16; two NUMA nodes
- GPU topology: two 48 GB RTX 4090 cards on different CPU sockets, without NVLink
- GPU experts: 48 routed experts per MoE layer by default (`GPU_EXPERTS` can override it)
- Model: `/data/qinchong/models/Qwen3.5-122B-A10B` (official BF16 weights, verified)
- Service: running on `127.0.0.1:30005` after the first successful inference check

`kt doctor` passes. The root filesystem has insufficient space for the model, so weights are stored on `/data`.

## One-time storage setup

The writable model directory was created with:

```bash
sudo install -d -o qinchong -g qinchong /data/qinchong
mkdir -p /data/qinchong/models
```

The official BF16 weights were downloaded and verified with:

```bash
cd ~/workspace/code/ktransformers
source .venv/bin/activate
modelscope download Qwen/Qwen3.5-122B-A10B \
  --local-dir /data/qinchong/models/Qwen3.5-122B-A10B \
  --max-workers 8
MODEL=/data/qinchong/models/Qwen3.5-122B-A10B \
  ./experiments/models/qwen3.5-122b-a10b/verify_qwen35_122b.py
```

The verified copy on the previous host contained 39 shards, 1,949 indexed tensors, and 250,173,007,840 bytes of tensor data.

## Start only during an exclusive GPU window

The launcher refuses to start unless both selected GPUs have at least 46,000 MiB free.

```bash
cd ~/workspace/code/ktransformers
nvidia-smi
./experiments/models/qwen3.5-122b-a10b/start_qwen35_122b.sh
tail -f experiments/artifacts/services/qwen3.5-122b-a10b/server.log
```

The default endpoint binds only to `127.0.0.1:30005`. Override `HOST=0.0.0.0` only when network exposure is intended.

The current defaults are the already-validated hardware-adjusted formal
configuration: TP=2, two NUMA-local CPU pools, 120 physical CPU threads, 48 GPU
experts per layer, max-running=8, a 262144-token pool, and 160 Mamba-state
slots. This covers the 128K prefill and 8×8K decode tests without another
parameter sweep.

## Verify inference

```bash
curl -s http://127.0.0.1:30005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-122B-A10B",
    "messages": [{"role": "user", "content": "只回答一个词：中国的首都是哪里？"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

Confirm the log contains `BF16SafeTensorLoader`, `backend=AVX512-BF16`, both `TP0` and `TP1`, and an HTTP 200 response.

## GPU expert benchmark

The service was warmed up before each run. Prefix cache was flushed between cases;
concurrency was 1 and each request generated 500 tokens.

| Nominal prompt | Actual prompt | GPU experts 1: prefill/decode tok/s | GPU experts 32: prefill/decode tok/s |
| ---: | ---: | ---: | ---: |
| 1024 | 1028 | 966.97 / 27.39 | 1112.57 / 28.25 |
| 2048 | 2041 | 1079.10 / 27.41 | 1032.62 / 27.97 |
| 4096 | 4067 | 1271.22 / 27.24 | 1367.72 / 27.67 |

Average decode throughput increased from 27.34 to 27.96 tok/s (about 2.3%).
At 32 experts per layer, model weights use about 19.8 GB per GPU, the Mamba
cache automatically shrinks from 233 to 147 slots, and steady total usage is
about 32-33 GB per GPU.

## Stop

```bash
cd ~/workspace/code/ktransformers
./experiments/models/qwen3.5-122b-a10b/stop_qwen35_122b.sh
```

Then verify that port 30005 and this user's GPU processes are gone:

```bash
ss -ltn 'sport = :30005'
nvidia-smi
```
