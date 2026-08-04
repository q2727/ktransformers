#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


model = Path(os.environ.get("MODEL", "/data/qinchong/models/Qwen3.5-122B-A10B"))
index_path = model / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit(f"Missing model index: {index_path}")

index = json.loads(index_path.read_text())
by_file: dict[str, set[str]] = defaultdict(set)
for tensor_name, shard_name in index["weight_map"].items():
    by_file[shard_name].add(tensor_name)

actual = {path.name for path in model.glob("model.safetensors-*.safetensors")}
if set(by_file) != actual:
    raise SystemExit(
        f"Shard mismatch: missing={sorted(set(by_file) - actual)}, "
        f"extra={sorted(actual - set(by_file))}"
    )

for shard_name, expected_keys in sorted(by_file.items()):
    with safe_open(model / shard_name, framework="pt", device="cpu") as shard:
        actual_keys = set(shard.keys())
    if actual_keys != expected_keys:
        raise SystemExit(f"Tensor index mismatch in {shard_name}")

file_bytes = sum((model / name).stat().st_size for name in by_file)
tensor_bytes = index["metadata"]["total_size"]
if file_bytes < tensor_bytes:
    raise SystemExit("Shard files are smaller than the indexed tensor payload")

print(f"OK: {len(by_file)} shards, {len(index['weight_map'])} tensors")
print(f"Tensor bytes: {tensor_bytes}; file bytes: {file_bytes}")
