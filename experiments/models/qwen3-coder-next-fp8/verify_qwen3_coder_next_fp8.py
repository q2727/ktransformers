#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


model = Path(
    os.environ.get(
        "MODEL", "/data/qinchong/models/Qwen3-Coder-Next-FP8"
    )
)
config_path = model / "config.json"
index_path = model / "model.safetensors.index.json"

if not config_path.is_file():
    raise SystemExit(f"Missing model config: {config_path}")
if not index_path.is_file():
    raise SystemExit(f"Missing model index: {index_path}")

config = json.loads(config_path.read_text())
if config.get("model_type") != "qwen3_next":
    raise SystemExit(f"Unexpected model_type: {config.get('model_type')!r}")
if config.get("quantization_config", {}).get("quant_method") != "fp8":
    raise SystemExit("Model is not the expected FP8 checkpoint")

index = json.loads(index_path.read_text())
by_file: dict[str, set[str]] = defaultdict(set)
for tensor_name, shard_name in index["weight_map"].items():
    by_file[shard_name].add(tensor_name)

actual = {path.name for path in model.glob("*.safetensors")}
expected = set(by_file)
if expected != actual:
    raise SystemExit(
        f"Shard mismatch: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}"
    )

for shard_name, expected_keys in sorted(by_file.items()):
    with safe_open(model / shard_name, framework="pt", device="cpu") as shard:
        actual_keys = set(shard.keys())
    if actual_keys != expected_keys:
        raise SystemExit(f"Tensor index mismatch in {shard_name}")

file_bytes = sum((model / name).stat().st_size for name in expected)
tensor_bytes = int(index["metadata"]["total_size"])
if file_bytes < tensor_bytes:
    raise SystemExit("Shard files are smaller than the indexed tensor payload")

print(f"OK: {len(expected)} shards, {len(index['weight_map'])} tensors")
print(f"Tensor bytes: {tensor_bytes}; file bytes: {file_bytes}")
print(
    "Architecture: "
    f"layers={config.get('num_hidden_layers')} "
    f"experts={config.get('num_experts')} "
    f"active={config.get('num_experts_per_tok')} "
    f"context={config.get('max_position_embeddings')}"
)
