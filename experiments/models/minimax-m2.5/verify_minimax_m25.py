#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


model = Path(os.environ.get("MODEL", "/data/qinchong/models/MiniMax-M2.5"))
config_path = model / "config.json"
index_path = model / "model.safetensors.index.json"
if not config_path.is_file() or not index_path.is_file():
    raise SystemExit(f"Incomplete model directory: {model}")

config = json.loads(config_path.read_text(encoding="utf-8"))
index = json.loads(index_path.read_text(encoding="utf-8"))
quant = config.get("quantization_config") or {}
weight_files = sorted(set(index.get("weight_map", {}).values()))
missing_weight_files = [name for name in weight_files if not (model / name).is_file()]
actual_tensor_bytes = sum((model / name).stat().st_size for name in weight_files if (model / name).is_file())
checks = {
    "model_type": config.get("model_type") == "minimax_m2",
    "architecture": "MiniMaxM2ForCausalLM" in config.get("architectures", []),
    "fp8_checkpoint": quant.get("quant_method") == "fp8",
    "context_at_least_128k": int(config.get("max_position_embeddings", 0)) >= 131072,
    "weight_map_present": bool(index.get("weight_map")),
    "all_weight_files_present": not missing_weight_files,
    "all_weight_files_nonempty": all((model / name).stat().st_size > 0 for name in weight_files if (model / name).is_file()),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Model verification failed: {', '.join(failed)}")

print(
    json.dumps(
        {
            "model": str(model),
            "checks": checks,
            "layers": config.get("num_hidden_layers"),
            "experts_per_layer": config.get("num_local_experts"),
            "active_experts_per_token": config.get("num_experts_per_tok"),
            "context_length": config.get("max_position_embeddings"),
            "weight_files": len(weight_files),
            "tensor_bytes": actual_tensor_bytes,
        },
        indent=2,
    )
)
