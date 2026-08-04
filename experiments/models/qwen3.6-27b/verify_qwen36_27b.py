#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


model = Path(os.environ.get("MODEL", "/data/qinchong/models/Qwen3.6-27B"))
config_path = model / "config.json"
index_path = model / "model.safetensors.index.json"

if not config_path.is_file():
    raise SystemExit(f"Missing model config: {config_path}")
if not index_path.is_file():
    raise SystemExit(f"Missing model index: {index_path}")

config = json.loads(config_path.read_text())
text_config = config.get("text_config", config)
if config.get("model_type") != "qwen3_5":
    raise SystemExit(f"Unexpected model_type: {config.get('model_type')!r}")
if int(text_config.get("num_hidden_layers", 0)) != 64:
    raise SystemExit("Unexpected Qwen3.6-27B layer count")

index = json.loads(index_path.read_text())
by_file: dict[str, set[str]] = defaultdict(set)
for tensor_name, shard_name in index["weight_map"].items():
    by_file[shard_name].add(tensor_name)

expected = set(by_file)
actual = {path.name for path in model.glob("*.safetensors")}
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
    f"layers={text_config.get('num_hidden_layers')} "
    f"hidden={text_config.get('hidden_size')} "
    f"context={text_config.get('max_position_embeddings')}"
)
