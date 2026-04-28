from __future__ import annotations

import hashlib
import json
from typing import Any


IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

KNOWN_MINIMAL_NODE_NAMES = (
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
)


class TopologyProfileError(ValueError):
    pass


def build_declared_data_from_known_profile(glb_json: dict[str, Any]) -> dict[str, Any]:
    fingerprint = _fingerprint(glb_json)
    if fingerprint != _known_minimal_fingerprint():
        raise TopologyProfileError(
            "Unknown or ambiguous UniRig topology; refusing to infer humanoid metadata from arbitrary nodes. "
            "Provide a valid companion .humanoid.json or retained GLB extras.unirig_humanoid."
        )

    nodes = _nodes_with_parents(glb_json)
    declared_nodes = []
    for index, node in enumerate(nodes):
        name = str(node.get("name") or "")
        declared_nodes.append(
            {
                "id": name,
                "name": name,
                "index": index,
                "parent": node.get("parent"),
                "rest_local": IDENTITY_4X4,
                "rest_world": IDENTITY_4X4,
                "inverse_bind": "available" if _has_inverse_bind(glb_json) else "unavailable",
            }
        )

    roles = {name: name for name in KNOWN_MINIMAL_NODE_NAMES}
    role_confidence = {name: 1.0 for name in roles}
    return {
        "roles": roles,
        "nodes": declared_nodes,
        "basis": {"up": "Y", "forward": "Z", "handedness": "right", "status": "inferred"},
        "confidence": {"roles": role_confidence},
        "provenance": {
            "source": "known-unirig-topology-profile",
            "method": "exact-bounded-fingerprint",
            "profile_id": "unirig-minimal-humanoid-17",
            "topology_fingerprint_sha256": fingerprint,
        },
    }


def _known_minimal_fingerprint() -> str:
    payload = {
        "node_names": list(KNOWN_MINIMAL_NODE_NAMES),
        "parents": [None, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 0, 14, 15],
        "skin_joints": [list(range(len(KNOWN_MINIMAL_NODE_NAMES)))],
        "inverse_bind": True,
    }
    return _hash_json(payload)


def _fingerprint(glb_json: dict[str, Any]) -> str:
    nodes = _nodes_with_parents(glb_json)
    skins = glb_json.get("skins") if isinstance(glb_json.get("skins"), list) else []
    skin_joints = []
    for skin in skins:
        if isinstance(skin, dict) and isinstance(skin.get("joints"), list):
            skin_joints.append([int(item) for item in skin["joints"] if isinstance(item, int)])
    payload = {
        "node_names": [str(node.get("name") or "") for node in nodes],
        "parents": [node.get("parent_index") for node in nodes],
        "skin_joints": skin_joints,
        "inverse_bind": _has_inverse_bind(glb_json),
    }
    return _hash_json(payload)


def _nodes_with_parents(glb_json: dict[str, Any]) -> list[dict[str, Any]]:
    raw_nodes = glb_json.get("nodes")
    if not isinstance(raw_nodes, list):
        return []
    nodes = [dict(node) if isinstance(node, dict) else {} for node in raw_nodes]
    parent_by_index: dict[int, int | None] = {index: None for index in range(len(nodes))}
    for parent_index, node in enumerate(nodes):
        children = node.get("children") if isinstance(node.get("children"), list) else []
        for child_index in children:
            if isinstance(child_index, int) and 0 <= child_index < len(nodes):
                if parent_by_index[child_index] is not None:
                    parent_by_index[child_index] = -1
                else:
                    parent_by_index[child_index] = parent_index
    for index, node in enumerate(nodes):
        parent_index = parent_by_index[index]
        node["parent_index"] = parent_index
        node["parent"] = None if parent_index is None or parent_index == -1 else str(nodes[parent_index].get("name") or "")
    return nodes


def _has_inverse_bind(glb_json: dict[str, Any]) -> bool:
    skins = glb_json.get("skins") if isinstance(glb_json.get("skins"), list) else []
    return any(isinstance(skin, dict) and skin.get("inverseBindMatrices") is not None for skin in skins)


def _hash_json(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
