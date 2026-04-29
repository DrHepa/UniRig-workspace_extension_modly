from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from .bootstrap import RuntimeContext
from .io import sha256_file
from .humanoid_contract import HumanoidContractError, build_contract_from_declared_data
from .humanoid_source import HumanoidResolutionFailure, resolve_humanoid_source
from .metadata_mode import MetadataMode, normalize_metadata_mode


def sidecar_path_for(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.rigmeta.json"


def build_sidecar(
    output_path: Path,
    input_path: Path,
    seed: int,
    context: RuntimeContext,
    humanoid_source: dict[str, Any] | None = None,
    metadata_mode: str = "auto",
) -> dict:
    mode = normalize_metadata_mode({"metadata_mode": metadata_mode})
    source_sha256 = sha256_file(input_path)
    output_sha256 = sha256_file(output_path)
    payload = {
        "metadata_version": 1,
        "extension_id": context.extension_id,
        "node_id": "rig-mesh",
        "source_mesh": input_path.name,
        "source_sha256": source_sha256,
        "output_mesh": output_path.name,
        "output_sha256": output_sha256,
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
    if mode != "legacy":
        _apply_humanoid_metadata(
            payload,
            mode=mode,
            output_path=output_path,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
            humanoid_source=humanoid_source,
        )
    payload["sidecar_payload_sha256"] = _payload_hash_without_self_reference(payload)
    return payload


def write_sidecar(
    output_path: Path,
    input_path: Path,
    seed: int,
    context: RuntimeContext,
    humanoid_source: dict[str, Any] | None = None,
    metadata_mode: str = "auto",
) -> Path:
    payload = build_sidecar(
        output_path=output_path,
        input_path=input_path,
        seed=seed,
        context=context,
        humanoid_source=humanoid_source,
        metadata_mode=metadata_mode,
    )
    destination = sidecar_path_for(output_path)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def _read_optional_humanoid_source(output_path: Path) -> dict[str, Any] | None:
    companion = output_path.with_name(f"{output_path.stem}.humanoid.json")
    if not companion.exists():
        return None
    payload = json.loads(companion.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Humanoid metadata companion must be a JSON object: {companion}")
    return payload


def _apply_humanoid_metadata(
    payload: dict,
    *,
    mode: MetadataMode,
    output_path: Path,
    source_sha256: str,
    output_sha256: str,
    humanoid_source: dict[str, Any] | None,
) -> None:
    payload["metadata_mode"] = mode
    try:
        if humanoid_source is None:
            resolved = resolve_humanoid_source(output_path)
        else:
            resolved = _manual_resolved_source(humanoid_source)
        contract = build_contract_from_declared_data(
            resolved.payload,
            source_hash=source_sha256,
            output_hash=output_sha256,
        )
    except (HumanoidResolutionFailure, HumanoidContractError, ValueError) as exc:
        if mode == "humanoid":
            raise HumanoidResolutionFailure(
                "metadata_mode=humanoid requires a valid humanoid contract source before publishing done. "
                f"Resolution/validation failed: {exc}. Remediation: provide a valid adjacent "
                "<output_stem>.humanoid.json, retained GLB extras.unirig_humanoid, or GLB skin joints/rest transforms "
                "that the semantic resolver can classify with sufficient confidence."
            ) from exc
        payload["humanoid_source_kind"] = "fallback"
        payload["humanoid_provenance"] = {
            "source_kind": "fallback",
            "reason": "no_valid_humanoid_source",
            "mode": mode,
        }
        payload["humanoid_warnings"] = [
            {
                "code": "humanoid_metadata_unavailable",
                "message": f"No valid humanoid metadata source was available; wrote legacy-compatible sidecar in metadata_mode={mode}.",
                "severity": "warning",
            }
        ]
        return

    payload["humanoid_contract"] = contract
    payload["humanoid_source_kind"] = resolved.kind
    payload["humanoid_provenance"] = dict(resolved.provenance)
    payload["humanoid_warnings"] = sorted(
        list(resolved.warnings) + list(contract.get("warnings", [])),
        key=lambda item: (str(item.get("code", "")), str(item.get("message", ""))),
    )


def _manual_resolved_source(humanoid_source: dict[str, Any]):
    from types import SimpleNamespace

    return SimpleNamespace(
        kind="provided",
        payload=humanoid_source,
        provenance={"source_kind": "provided", "method": "explicit-argument"},
        warnings=[],
    )


def _payload_hash_without_self_reference(payload: dict) -> str:
    canonical_payload = dict(payload)
    canonical_payload.pop("sidecar_payload_sha256", None)
    canonical = json.dumps(canonical_payload, indent=2, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
