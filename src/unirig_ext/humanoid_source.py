from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .topology_profiles import TopologyProfileError, build_declared_data_from_known_profile


class HumanoidResolutionFailure(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedHumanoidSource:
    kind: str
    payload: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[dict[str, str]]


def resolve_humanoid_source(output_path: Path) -> ResolvedHumanoidSource:
    companion = output_path.with_name(f"{output_path.stem}.humanoid.json")
    if companion.exists():
        payload = _read_json_file(companion, source_kind="companion")
        return ResolvedHumanoidSource(
            kind="companion",
            payload=payload,
            provenance={
                "source_kind": "companion",
                "path": str(companion),
                "payload_sha256": _hash_payload(payload),
            },
            warnings=[],
        )

    glb_json = _read_glb_json(output_path)
    extras = glb_json.get("extras") if isinstance(glb_json, dict) else None
    declared = extras.get("unirig_humanoid") if isinstance(extras, dict) else None
    if declared is not None:
        if not isinstance(declared, dict):
            raise HumanoidResolutionFailure(
                "GLB extras source extras.unirig_humanoid must be a JSON object. "
                "Provide a valid companion .humanoid.json or repair the retained extras metadata."
            )
        return ResolvedHumanoidSource(
            kind="glb_extras",
            payload=declared,
            provenance={
                "source_kind": "glb_extras",
                "path": str(output_path),
                "json_pointer": "/extras/unirig_humanoid",
                "payload_sha256": _hash_payload(declared),
            },
            warnings=[],
        )

    try:
        payload = build_declared_data_from_known_profile(glb_json)
    except TopologyProfileError as exc:
        raise HumanoidResolutionFailure(str(exc)) from exc
    return ResolvedHumanoidSource(
        kind="topology_profile",
        payload=payload,
        provenance={
            "source_kind": "topology_profile",
            "path": str(output_path),
            "profile_id": payload.get("provenance", {}).get("profile_id", ""),
            "payload_sha256": _hash_payload(payload),
        },
        warnings=[
            {
                "code": "humanoid_source_from_bounded_topology_profile",
                "message": "Humanoid metadata was generated from an exact known UniRig topology profile.",
                "severity": "warning",
            }
        ],
    )


def _read_json_file(path: Path, *, source_kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HumanoidResolutionFailure(f"{source_kind} humanoid source is not valid JSON: {path}; {exc}") from exc
    if not isinstance(payload, dict):
        raise HumanoidResolutionFailure(f"{source_kind} humanoid source must be a JSON object: {path}")
    return payload


def _read_glb_json(output_path: Path) -> dict[str, Any]:
    data = output_path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        return {}
    json_length = struct.unpack_from("<I", data, 12)[0]
    chunk_type = data[16:20]
    if chunk_type != b"JSON":
        return {}
    raw_json = data[20 : 20 + json_length].rstrip(b" \t\r\n\x00")
    payload = json.loads(raw_json.decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
