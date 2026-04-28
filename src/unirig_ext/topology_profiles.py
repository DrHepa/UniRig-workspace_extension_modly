from __future__ import annotations

import hashlib
import json
import math
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

REAL_UNIRIG_52_ROLE_MAP = {
    "hips": "bone_0",
    "spine": "bone_1",
    "chest": "bone_3",
    "neck": "bone_4",
    "head": "bone_5",
    "left_upper_arm": "bone_6",
    "left_lower_arm": "bone_7",
    "left_hand": "bone_9",
    "right_upper_arm": "bone_25",
    "right_lower_arm": "bone_26",
    "right_hand": "bone_28",
    "left_upper_leg": "bone_44",
    "left_lower_leg": "bone_45",
    "left_foot": "bone_47",
    "right_upper_leg": "bone_48",
    "right_lower_leg": "bone_49",
    "right_foot": "bone_51",
}

REAL_UNIRIG_52_EDGES = (
    ("bone_0", "bone_1"),
    ("bone_1", "bone_2"),
    ("bone_2", "bone_3"),
    ("bone_3", "bone_4"),
    ("bone_4", "bone_5"),
    ("bone_3", "bone_6"),
    ("bone_6", "bone_7"),
    ("bone_7", "bone_8"),
    ("bone_8", "bone_9"),
    ("bone_9", "bone_10"),
    ("bone_10", "bone_11"),
    ("bone_11", "bone_12"),
    ("bone_9", "bone_13"),
    ("bone_13", "bone_14"),
    ("bone_14", "bone_15"),
    ("bone_9", "bone_16"),
    ("bone_16", "bone_17"),
    ("bone_17", "bone_18"),
    ("bone_9", "bone_19"),
    ("bone_19", "bone_20"),
    ("bone_20", "bone_21"),
    ("bone_9", "bone_22"),
    ("bone_22", "bone_23"),
    ("bone_23", "bone_24"),
    ("bone_3", "bone_25"),
    ("bone_25", "bone_26"),
    ("bone_26", "bone_27"),
    ("bone_27", "bone_28"),
    ("bone_28", "bone_29"),
    ("bone_29", "bone_30"),
    ("bone_30", "bone_31"),
    ("bone_28", "bone_32"),
    ("bone_32", "bone_33"),
    ("bone_33", "bone_34"),
    ("bone_28", "bone_35"),
    ("bone_35", "bone_36"),
    ("bone_36", "bone_37"),
    ("bone_28", "bone_38"),
    ("bone_38", "bone_39"),
    ("bone_39", "bone_40"),
    ("bone_28", "bone_41"),
    ("bone_41", "bone_42"),
    ("bone_42", "bone_43"),
    ("bone_0", "bone_44"),
    ("bone_44", "bone_45"),
    ("bone_45", "bone_46"),
    ("bone_46", "bone_47"),
    ("bone_0", "bone_48"),
    ("bone_48", "bone_49"),
    ("bone_49", "bone_50"),
    ("bone_50", "bone_51"),
)


class TopologyProfileError(ValueError):
    pass


def build_declared_data_from_known_profile(glb_json: dict[str, Any]) -> dict[str, Any]:
    nodes = _nodes_with_parents(glb_json)
    if _matches_real_unirig_52_profile(glb_json, nodes):
        return _build_real_unirig_52_declared_data(glb_json, nodes)

    fingerprint = _fingerprint(glb_json)
    if fingerprint != _known_minimal_fingerprint():
        raise TopologyProfileError(
            "Unknown or ambiguous UniRig topology; refusing to infer humanoid metadata from arbitrary nodes. "
            "Provide a valid companion .humanoid.json or retained GLB extras.unirig_humanoid."
        )

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


def _matches_real_unirig_52_profile(glb_json: dict[str, Any], nodes: list[dict[str, Any]]) -> bool:
    skins = glb_json.get("skins") if isinstance(glb_json.get("skins"), list) else []
    if len(skins) != 1 or not isinstance(skins[0], dict):
        return False
    joints = skins[0].get("joints") if isinstance(skins[0].get("joints"), list) else []
    if len(joints) != 52:
        return False
    if any(not isinstance(index, int) or index < 0 or index >= len(nodes) for index in joints):
        return False
    expected_names = [f"bone_{number}" for number in range(52)]
    if [str(nodes[index].get("name") or "") for index in joints] != expected_names:
        return False

    index_by_name = {str(node.get("name") or ""): index for index, node in enumerate(nodes)}
    if any(index_by_name.get(name) is None for name in expected_names):
        return False
    for parent, child in REAL_UNIRIG_52_EDGES:
        child_index = index_by_name[child]
        parent_index = index_by_name[parent]
        if nodes[child_index].get("parent_index") != parent_index:
            return False
    return True


def _build_real_unirig_52_declared_data(glb_json: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, Any]:
    local_by_index = [_local_matrix(node) for node in nodes]
    world_by_index: dict[int, list[list[float]]] = {}
    for index in range(len(nodes)):
        _world_matrix(index, nodes, local_by_index, world_by_index)

    declared_nodes = []
    for index, node in enumerate(nodes):
        name = str(node.get("name") or "")
        declared_nodes.append(
            {
                "id": name,
                "name": name,
                "index": index,
                "parent": node.get("parent"),
                "rest_local": local_by_index[index],
                "rest_world": world_by_index[index],
                "inverse_bind": "available" if _has_inverse_bind(glb_json) else "unavailable",
            }
        )
    fingerprint = _fingerprint(glb_json)
    role_confidence = {role: 1.0 for role in REAL_UNIRIG_52_ROLE_MAP}
    return {
        "roles": dict(REAL_UNIRIG_52_ROLE_MAP),
        "nodes": declared_nodes,
        "basis": {"up": "Y", "forward": "Z", "handedness": "right", "status": "inferred"},
        "confidence": {"roles": role_confidence},
        "provenance": {
            "source": "known-unirig-topology-profile",
            "method": "exact-bounded-bone-name-topology",
            "profile_id": "unirig-anonymous-bone-52",
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


def _local_matrix(node: dict[str, Any]) -> list[list[float]]:
    raw_matrix = node.get("matrix")
    if isinstance(raw_matrix, list) and len(raw_matrix) == 16 and all(isinstance(value, (int, float)) for value in raw_matrix):
        return [[float(raw_matrix[row * 4 + column]) for column in range(4)] for row in range(4)]

    translation = _numeric_triplet(node.get("translation"), default=(0.0, 0.0, 0.0))
    scale = _numeric_triplet(node.get("scale"), default=(1.0, 1.0, 1.0))
    rotation = _numeric_quaternion(node.get("rotation"), default=(0.0, 0.0, 0.0, 1.0))
    return _compose_trs(translation, rotation, scale)


def _numeric_triplet(value: Any, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, list) and len(value) == 3 and all(isinstance(item, (int, float)) for item in value):
        return (float(value[0]), float(value[1]), float(value[2]))
    return default


def _numeric_quaternion(value: Any, *, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if isinstance(value, list) and len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return default


def _compose_trs(
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    scale: tuple[float, float, float],
) -> list[list[float]]:
    x, y, z, w = rotation
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length == 0.0:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / length, y / length, z / length, w / length
    sx, sy, sz = scale
    tx, ty, tz = translation
    return [
        [(1.0 - 2.0 * (y * y + z * z)) * sx, (2.0 * (x * y - z * w)) * sy, (2.0 * (x * z + y * w)) * sz, tx],
        [(2.0 * (x * y + z * w)) * sx, (1.0 - 2.0 * (x * x + z * z)) * sy, (2.0 * (y * z - x * w)) * sz, ty],
        [(2.0 * (x * z - y * w)) * sx, (2.0 * (y * z + x * w)) * sy, (1.0 - 2.0 * (x * x + y * y)) * sz, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _world_matrix(
    index: int,
    nodes: list[dict[str, Any]],
    local_by_index: list[list[list[float]]],
    world_by_index: dict[int, list[list[float]]],
) -> list[list[float]]:
    if index in world_by_index:
        return world_by_index[index]
    parent_index = nodes[index].get("parent_index")
    if isinstance(parent_index, int) and parent_index >= 0:
        world_by_index[index] = _multiply_matrices(_world_matrix(parent_index, nodes, local_by_index, world_by_index), local_by_index[index])
    else:
        world_by_index[index] = local_by_index[index]
    return world_by_index[index]


def _multiply_matrices(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][inner] * right[inner][column] for inner in range(4)) for column in range(4)]
        for row in range(4)
    ]
