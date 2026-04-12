from __future__ import annotations

import json
from pathlib import Path

from .bootstrap import RuntimeContext
from .io import sha256_file


def sidecar_path_for(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.rigmeta.json"


def build_sidecar(output_path: Path, input_path: Path, seed: int, context: RuntimeContext) -> dict:
    return {
        "metadata_version": 1,
        "extension_id": context.extension_id,
        "node_id": "rig-mesh",
        "source_mesh": input_path.name,
        "output_mesh": output_path.name,
        "output_sha256": sha256_file(output_path),
        "seed": int(seed),
        "runtime": {
            "mode": context.runtime_mode,
            "source_ref": context.source_ref,
            "python_version": context.python_version,
        },
        "pipeline": {
            "stages": ["prepare", "skeleton", "skin", "merge"],
            "deterministic_output_name": output_path.name,
        },
    }


def write_sidecar(output_path: Path, input_path: Path, seed: int, context: RuntimeContext) -> Path:
    payload = build_sidecar(output_path=output_path, input_path=input_path, seed=seed, context=context)
    destination = sidecar_path_for(output_path)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination
