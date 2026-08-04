# DeepSeek-V4-Flash

Official checkpoint: `/data/qinchong/models/DeepSeek-V4-Flash`. Its routed
experts are native FP4/MXFP4. Do not convert them before establishing the native
MXFP4 baseline on this AMD EPYC host.

This model needs a separate runtime because it requires Transformers 4.57.1,
FlashInfer 0.6.9, TileLang 0.1.10, and `apache-tvm-ffi<0.1.12`, while the main KT
environment currently serves the other models with Transformers 5.x and
FlashInfer 0.6.3. Create the isolated copy under `/data` with:

```bash
experiments/models/deepseek-v4-flash/setup_dsv4_env.sh
```

The launch defaults map the official single-GPU KT command once to this host:
TP=2, two NUMA-local CPU pools, 20 GPU experts, max-running=8, 196K context, and
a 150K token pool. Perform only a startup/128K feasibility check before the
formal run; do not add a parameter sweep.

The launch script also installs the checkpoint's documented chat-mode framing
from `chat_template.jinja`, because the official tokenizer metadata does not
ship a Hugging Face `chat_template` field.

```bash
experiments/models/deepseek-v4-flash/start_deepseek_v4_flash.sh
experiments/models/deepseek-v4-flash/stop_deepseek_v4_flash.sh
```
