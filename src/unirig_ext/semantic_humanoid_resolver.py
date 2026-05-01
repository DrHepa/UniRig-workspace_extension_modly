from __future__ import annotations

"""Semantic UniRig humanoid resolver.

This module infers humanoid roles from skin-joint graph structure, composed rest
transforms, and lateral symmetry. It intentionally avoids exact ``bone_N`` count
dispatch; known 40/52 topologies remain tests/oracles, not runtime strategy.
"""

import hashlib
import json
import math
from itertools import combinations
from dataclasses import dataclass
from typing import Any


IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

REQUIRED_ROLES = (
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

MINIMUM_TRUNK_LENGTH = 5


class SemanticHumanoidResolutionError(ValueError):
    def __init__(self, message: str, diagnostics: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or []


@dataclass(frozen=True)
class JointNode:
    id: str
    name: str
    index: int
    parent: str | None
    children: list[str]
    depth: int
    rest_local: list[list[float]]
    rest_world: list[list[float]]


@dataclass(frozen=True)
class JointGraph:
    nodes: dict[str, JointNode]
    roots: list[str]
    leaves: list[str]
    joints: list[str]
    has_inverse_bind: bool


@dataclass(frozen=True)
class ChestCandidateScore:
    index: int
    pair: tuple[str, str]
    axis: int
    min_chain_length: int
    center_separation: float
    center_balance: float
    straddle_margin: float
    downward_score: float
    trunk_position_score: float
    total: float


def extract_joint_graph(glb_json: dict[str, Any]) -> JointGraph:
    raw_nodes = glb_json.get("nodes") if isinstance(glb_json, dict) else None
    if not isinstance(raw_nodes, list) or not raw_nodes:
        _fail("semantic_nodes_missing", "GLB JSON does not contain a non-empty nodes array.")

    skins = glb_json.get("skins") if isinstance(glb_json.get("skins"), list) else []
    skin = skins[0] if len(skins) == 1 and isinstance(skins[0], dict) else None
    joints = skin.get("joints") if isinstance(skin, dict) and isinstance(skin.get("joints"), list) else None
    if not joints:
        _fail("semantic_skin_missing", "A semantic humanoid resolver requires exactly one skin with a non-empty joints list.")
    if any(not isinstance(index, int) or index < 0 or index >= len(raw_nodes) for index in joints):
        _fail("semantic_skin_malformed", "Skin joints must reference valid node indices.")

    parent_by_index: dict[int, int | None] = {index: None for index in range(len(raw_nodes))}
    for parent_index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            _fail("semantic_node_malformed", f"Node at index {parent_index} must be an object.")
        children = node.get("children") if isinstance(node.get("children"), list) else []
        for child_index in children:
            if isinstance(child_index, int) and 0 <= child_index < len(raw_nodes):
                if parent_by_index[child_index] is not None:
                    _fail("semantic_graph_ambiguous", f"Node index {child_index} has multiple parents.")
                parent_by_index[child_index] = parent_index

    joint_indices = list(joints)
    joint_index_set = set(joint_indices)
    id_by_index = {_index: _node_id(raw_nodes[_index], _index) for _index in joint_indices}
    local_by_index = {_index: _local_matrix(raw_nodes[_index], _index) for _index in joint_indices}
    world_cache: dict[int, list[list[float]]] = {}

    graph_nodes: dict[str, JointNode] = {}
    for index in joint_indices:
        parent_index = parent_by_index[index]
        parent_id = id_by_index[parent_index] if parent_index in joint_index_set else None
        child_ids = [id_by_index[child] for child in _children(raw_nodes[index]) if child in joint_index_set]
        graph_nodes[id_by_index[index]] = JointNode(
            id=id_by_index[index],
            name=str(raw_nodes[index].get("name") or id_by_index[index]),
            index=index,
            parent=parent_id,
            children=child_ids,
            depth=_depth(index, parent_by_index, joint_index_set),
            rest_local=local_by_index[index],
            rest_world=_world_matrix(index, raw_nodes, parent_by_index, local_by_index, world_cache),
        )

    roots = [node_id for node_id in id_by_index.values() if graph_nodes[node_id].parent is None]
    leaves = [node_id for node_id in id_by_index.values() if not graph_nodes[node_id].children]
    return JointGraph(
        nodes=graph_nodes,
        roots=roots,
        leaves=leaves,
        joints=[id_by_index[index] for index in joint_indices],
        has_inverse_bind=bool(skin and skin.get("inverseBindMatrices") is not None),
    )


def resolve_humanoid(
    glb_json: dict[str, Any],
    *,
    min_role: float = 0.75,
    min_total: float = 0.80,
    semantic_report: Any | None = None,
) -> dict[str, Any]:
    if semantic_report is not None:
        return _resolve_from_semantic_report(glb_json, semantic_report, min_role=min_role, min_total=min_total)
    graph = extract_joint_graph(glb_json)
    roles: dict[str, str] = {}
    confidence: dict[str, float] = {}
    diagnostics: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    hips = _resolve_single_root(graph)
    roles["hips"] = hips
    confidence["hips"] = 0.95

    trunk_path = _highest_path_from(graph, hips)
    if len(trunk_path) < MINIMUM_TRUNK_LENGTH:
        _fail(
            "semantic_spine_missing",
            "Unable to find a long enough hips-to-head trunk chain.",
            joint_count=len(graph.joints),
            root_count=len(graph.roots),
            roots=list(graph.roots),
            highest_path=list(trunk_path),
            highest_path_length=len(trunk_path),
            minimum_trunk_length=MINIMUM_TRUNK_LENGTH,
        )
    chest_index = _find_chest_index(graph, trunk_path)
    if chest_index is None or chest_index < 2 or chest_index + 2 >= len(trunk_path):
        _fail("semantic_chest_missing", "Unable to identify a chest branch with symmetric arm evidence.")
    roles.update({"spine": trunk_path[1], "chest": trunk_path[chest_index], "neck": trunk_path[-2], "head": trunk_path[-1]})
    confidence.update({role: 0.9 for role in ("spine", "chest", "neck", "head")})

    chest = roles["chest"]
    arm_roots = _arm_roots_for_chest_candidate(graph, trunk_path, chest_index)
    left_arm_root, right_arm_root = _symmetric_pair(graph, arm_roots, label="arms")
    _assign_arm_roles(graph, roles, confidence, warnings, "left", left_arm_root)
    _assign_arm_roles(graph, roles, confidence, warnings, "right", right_arm_root)

    left_leg_root, right_leg_root = _leg_roots_for_hips(graph, trunk_path)
    _assign_leg_roles(graph, roles, confidence, "left", left_leg_root)
    _assign_leg_roles(graph, roles, confidence, "right", right_leg_root)

    missing = [role for role in REQUIRED_ROLES if role not in roles]
    if missing:
        diagnostics.append({"code": "semantic_required_roles_missing", "message": f"Missing required roles: {', '.join(missing)}.", "roles": missing})

    low = [role for role in REQUIRED_ROLES if confidence.get(role, 0.0) < min_role]
    overall = min((confidence.get(role, 0.0) for role in REQUIRED_ROLES), default=0.0)
    if low or overall < min_total:
        diagnostics.append({"code": "semantic_confidence_below_threshold", "message": "Required role confidence is below semantic resolver thresholds.", "roles": low, "overall": overall})
    if diagnostics:
        raise SemanticHumanoidResolutionError("; ".join(item["code"] for item in diagnostics), diagnostics)

    return {
        "roles": roles,
        "nodes": _declared_nodes(graph),
        "basis": {"up": "Y", "forward": "Z", "handedness": "right", "status": "inferred"},
        "confidence": {"roles": dict(sorted(confidence.items())), "overall": overall},
        "provenance": {
            "source": "semantic_humanoid_resolver",
            "method": "semantic-graph-rest-symmetry",
            "topology_fingerprint_sha256": _fingerprint(glb_json),
        },
        "diagnostics": warnings,
    }


def _resolve_from_semantic_report(glb_json: dict[str, Any], semantic_report: Any, *, min_role: float, min_total: float) -> dict[str, Any]:
    diagnostic = semantic_report.as_diagnostic() if hasattr(semantic_report, "as_diagnostic") else {}
    if not getattr(semantic_report, "publishable", False):
        _fail(
            "semantic_body_graph_not_publishable",
            "Semantic body graph evidence is not publishable for humanoid contract derivation.",
            semantic_body_graph=diagnostic,
        )
    roles = dict(getattr(semantic_report, "core_roles", {}) or {})
    missing = [role for role in REQUIRED_ROLES if role not in roles]
    if missing:
        _fail(
            "semantic_required_roles_missing",
            "Semantic body graph did not provide all required humanoid roles.",
            roles=missing,
            semantic_body_graph=diagnostic,
        )
    nodes_by_id = getattr(semantic_report, "nodes", {}) or {}
    role_confidence = {
        role: round(float(getattr(nodes_by_id.get(joint), "confidence", 0.0)), 3)
        for role, joint in roles.items()
    }
    low = [role for role in REQUIRED_ROLES if role_confidence.get(role, 0.0) < min_role]
    overall = min(float(getattr(semantic_report, "contract_core_confidence", 0.0)), min((role_confidence.get(role, 0.0) for role in REQUIRED_ROLES), default=0.0))
    if low or overall < min_total:
        _fail(
            "semantic_confidence_below_threshold",
            "Semantic body graph role confidence is below resolver thresholds.",
            roles=low,
            overall=overall,
            semantic_body_graph=diagnostic,
        )
    return {
        "roles": roles,
        "nodes": _declared_nodes_from_report(nodes_by_id),
        "basis": {"up": "Y", "forward": "Z", "handedness": "right", "status": "inferred"},
        "confidence": {"roles": dict(sorted(role_confidence.items())), "overall": round(overall, 3)},
        "provenance": {
            "source": "semantic_humanoid_resolver",
            "method": "semantic-body-report",
            "topology_fingerprint_sha256": _fingerprint(glb_json),
        },
        "diagnostics": [
            {
                "code": "semantic_body_report_consumed",
                "message": "Humanoid roles were derived from precomputed SemanticBodyReport core role evidence.",
            }
        ],
    }


def _declared_nodes_from_report(nodes_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    declared = []
    for index, node_id in enumerate(sorted(nodes_by_id)):
        node = nodes_by_id[node_id]
        declared.append(
            {
                "id": node_id,
                "name": node_id,
                "index": index,
                "parent": getattr(node, "parent", None),
                "rest_local": IDENTITY_4X4,
                "rest_world": [list(row) for row in getattr(node, "rest_world", IDENTITY_4X4)],
                "inverse_bind": "unavailable",
            }
        )
    return declared


def _resolve_single_root(graph: JointGraph) -> str:
    if len(graph.roots) != 1:
        _fail("semantic_graph_disconnected", f"Expected exactly one joint root; found {len(graph.roots)}.", roots=graph.roots)
    return graph.roots[0]


def _find_chest_index(graph: JointGraph, trunk_path: list[str]) -> int | None:
    candidates: list[ChestCandidateScore] = []
    for index in range(2, len(trunk_path) - 2):
        score = _best_chest_score_for_candidate(graph, trunk_path, index)
        if score is not None:
            candidates.append(score)
    if not candidates:
        return None
    candidates.sort(key=_chest_score_sort_key, reverse=True)
    if len(candidates) > 1 and _chest_scores_too_close(candidates[0], candidates[1]):
        return None
    return candidates[0].index


def _arm_roots_for_chest_candidate(graph: JointGraph, trunk_path: list[str], index: int) -> list[str]:
    score = _best_chest_score_for_candidate(graph, trunk_path, index)
    if score is None:
        return []
    return list(score.pair)


def _best_chest_score_for_candidate(graph: JointGraph, trunk_path: list[str], index: int) -> ChestCandidateScore | None:
    next_trunk = trunk_path[index + 1]
    side_children = [child for child in graph.nodes[trunk_path[index]].children if child != next_trunk]
    plausible = [child for child in side_children if _arm_branch_chain_length(graph, child) >= 2]
    scored_pairs: list[ChestCandidateScore] = []
    for first, second in combinations(plausible, 2):
        score = _score_arm_pair_for_chest(graph, trunk_path, index, first, second)
        if score is not None:
            scored_pairs.append(score)
    if not scored_pairs:
        return None
    scored_pairs.sort(key=_chest_score_sort_key, reverse=True)
    if len(scored_pairs) > 1 and _chest_scores_too_close(scored_pairs[0], scored_pairs[1]):
        return None
    return scored_pairs[0]


def _score_arm_pair_for_chest(graph: JointGraph, trunk_path: list[str], index: int, first: str, second: str) -> ChestCandidateScore | None:
    first_length = _arm_branch_chain_length(graph, first)
    second_length = _arm_branch_chain_length(graph, second)
    min_chain_length = min(first_length, second_length)
    if max(first_length, second_length) < 3:
        return None
    axis = _best_chest_lateral_axis(graph, trunk_path[index], first, second)
    if axis is None:
        return None

    candidate_center = _axis_value(graph, trunk_path[index], axis)
    first_offset = _arm_branch_center(graph, first, axis) - candidate_center
    second_offset = _arm_branch_center(graph, second, axis) - candidate_center
    if first_offset == 0.0 or second_offset == 0.0 or first_offset * second_offset >= 0.0:
        return None

    first_magnitude = abs(first_offset)
    second_magnitude = abs(second_offset)
    straddle_margin = min(first_magnitude, second_magnitude)
    if straddle_margin < 0.08:
        return None
    average_magnitude = (first_magnitude + second_magnitude) / 2.0
    if average_magnitude <= 0.0:
        return None
    balance_delta = abs(first_magnitude - second_magnitude) / average_magnitude
    if balance_delta > 0.65:
        return None

    downward = min(_arm_branch_downward_drop(graph, first), _arm_branch_downward_drop(graph, second))
    if downward < 0.04:
        return None

    center_separation = first_magnitude + second_magnitude
    center_balance = max(0.0, 1.0 - balance_delta)
    downward_score = min(downward / 0.30, 1.0)
    trunk_position_score = _chest_trunk_position_score(trunk_path, index)
    total = (
        min_chain_length * 0.60
        + center_separation * 0.80
        + center_balance * 1.40
        + straddle_margin * 0.70
        + downward_score * 1.50
        + trunk_position_score * 0.80
    )
    pair = tuple(sorted((first, second), key=lambda node_id: (_arm_branch_center(graph, node_id, axis), graph.nodes[node_id].index)))
    return ChestCandidateScore(
        index=index,
        pair=pair,
        axis=axis,
        min_chain_length=min_chain_length,
        center_separation=center_separation,
        center_balance=center_balance,
        straddle_margin=straddle_margin,
        downward_score=downward_score,
        trunk_position_score=trunk_position_score,
        total=total,
    )


def _chest_score_sort_key(score: ChestCandidateScore) -> tuple[float, int, float, float, float, float, int]:
    return (
        score.total,
        score.min_chain_length,
        score.center_balance,
        score.downward_score,
        score.straddle_margin,
        score.trunk_position_score,
        -score.index,
    )


def _chest_scores_too_close(best: ChestCandidateScore, next_best: ChestCandidateScore) -> bool:
    if best.min_chain_length != next_best.min_chain_length:
        return False
    return math.isclose(best.total, next_best.total, rel_tol=0.04, abs_tol=0.20)


def _best_chest_lateral_axis(graph: JointGraph, chest: str, first: str, second: str) -> int | None:
    scored_axes: list[tuple[float, int]] = []
    for axis in (0, 2):
        center = _axis_value(graph, chest, axis)
        first_offset = _arm_branch_center(graph, first, axis) - center
        second_offset = _arm_branch_center(graph, second, axis) - center
        if first_offset * second_offset >= 0.0:
            continue
        separation = abs(first_offset) + abs(second_offset)
        if separation >= 0.16:
            scored_axes.append((separation, axis))
    if not scored_axes:
        return None
    scored_axes.sort(reverse=True)
    if len(scored_axes) > 1 and math.isclose(scored_axes[0][0], scored_axes[1][0], rel_tol=0.05, abs_tol=0.01):
        return None
    return scored_axes[0][1]


def _arm_branch_center(graph: JointGraph, root: str, axis: int) -> float:
    chain = _single_child_chain(graph, root, max_nodes=4)
    return sum(_axis_value(graph, node_id, axis) for node_id in chain) / len(chain)


def _arm_branch_downward_drop(graph: JointGraph, root: str) -> float:
    chain = _single_child_chain(graph, root, max_nodes=4)
    return _y(graph, root) - min(_y(graph, node_id) for node_id in chain)


def _chest_trunk_position_score(trunk_path: list[str], index: int) -> float:
    available = max(1, len(trunk_path) - 4)
    nodes_above = max(0, len(trunk_path) - index - 2)
    return min(1.0, nodes_above / available)


def _arm_branch_chain_length(graph: JointGraph, root: str) -> int:
    return len(_single_child_chain(graph, root, max_nodes=4))


def _assign_arm_roles(
    graph: JointGraph,
    roles: dict[str, str],
    confidence: dict[str, float],
    diagnostics: list[dict[str, Any]],
    side: str,
    arm_root: str,
) -> None:
    chain = _single_child_chain(graph, arm_root, max_nodes=4)
    if len(chain) < 3:
        _fail("semantic_arm_chain_missing", f"{side} arm chain does not include upper arm, lower arm, and hand.", node=arm_root)
    if len(chain) >= 4:
        roles[f"{side}_shoulder"] = chain[0]
        roles[f"{side}_upper_arm"] = chain[1]
        roles[f"{side}_lower_arm"] = chain[2]
        roles[f"{side}_hand"] = chain[3]
        confidence[f"{side}_shoulder"] = 0.85
    else:
        roles[f"{side}_upper_arm"] = chain[0]
        roles[f"{side}_lower_arm"] = chain[1]
        roles[f"{side}_hand"] = chain[2]
        diagnostics.append(
            {
                "code": "optional_shoulder_unavailable",
                "message": f"{side} shoulder evidence is unavailable; resolved arm from upper_arm -> lower_arm -> hand chain.",
                "side": side,
                "node": arm_root,
            }
        )
    confidence[f"{side}_upper_arm"] = 0.9
    confidence[f"{side}_lower_arm"] = 0.9
    confidence[f"{side}_hand"] = 0.9


def _assign_leg_roles(graph: JointGraph, roles: dict[str, str], confidence: dict[str, float], side: str, upper_leg: str) -> None:
    chain = _single_child_chain(graph, upper_leg, max_nodes=4)
    if len(chain) < 3:
        _fail("semantic_leg_chain_missing", f"{side} leg chain does not include upper leg, lower leg, and foot.", node=upper_leg)
    roles[f"{side}_upper_leg"] = chain[0]
    roles[f"{side}_lower_leg"] = chain[1]
    roles[f"{side}_foot"] = chain[-1]
    confidence[f"{side}_upper_leg"] = 0.9
    confidence[f"{side}_lower_leg"] = 0.9
    confidence[f"{side}_foot"] = 0.9


def _leg_roots_for_hips(graph: JointGraph, trunk_path: list[str]) -> list[str]:
    scanned_candidates: list[str] = []
    for index, parent in enumerate(trunk_path[:3]):
        next_trunk = trunk_path[index + 1] if index + 1 < len(trunk_path) else None
        candidates = [child for child in graph.nodes[parent].children if child != next_trunk]
        scanned_candidates.extend(candidates)
        leg_roots = _select_leg_roots_from_candidates(graph, candidates)
        if leg_roots is not None:
            return leg_roots
    _fail(
        "semantic_leg_symmetry_ambiguous",
        "Unable to find one clear symmetric leg pair with sufficient chain length, downward evidence, and lateral separation.",
        candidates=sorted(scanned_candidates),
    )


def _select_leg_roots_from_candidates(graph: JointGraph, candidates: list[str]) -> list[str] | None:
    plausible = [child for child in candidates if _leg_branch_chain_length(graph, child) >= 3 and _leg_branch_downward_drop(graph, child) >= 0.75]
    scored_pairs: list[tuple[tuple[float, float, float], tuple[str, str], int]] = []
    for first, second in combinations(plausible, 2):
        axis = _best_leg_lateral_axis(graph, [first, second])
        if axis is None:
            continue
        separation = abs(_leg_lateral_center(graph, first, axis) - _leg_lateral_center(graph, second, axis))
        downward = min(_leg_branch_downward_drop(graph, first), _leg_branch_downward_drop(graph, second))
        chain_length = min(_leg_branch_chain_length(graph, first), _leg_branch_chain_length(graph, second))
        balance_penalty = abs(_leg_branch_downward_drop(graph, first) - _leg_branch_downward_drop(graph, second))
        scored_pairs.append(((chain_length, downward - balance_penalty, separation), (first, second), axis))
    if not scored_pairs:
        return None
    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    if len(scored_pairs) > 1 and _leg_pair_scores_too_close(scored_pairs[0][0], scored_pairs[1][0]):
        _fail(
            "semantic_leg_symmetry_ambiguous",
            "Unable to choose between multiple plausible leg pairs without a clear score margin.",
            candidates=sorted(candidates),
            plausible=sorted(plausible),
            best_pair=sorted(scored_pairs[0][1]),
            next_pair=sorted(scored_pairs[1][1]),
            best_score=scored_pairs[0][0],
            next_score=scored_pairs[1][0],
        )
    first, second = scored_pairs[0][1]
    axis = scored_pairs[0][2]
    return sorted([first, second], key=lambda node_id: (_leg_lateral_center(graph, node_id, axis), graph.nodes[node_id].index))


def _leg_pair_scores_too_close(best: tuple[float, float, float], next_best: tuple[float, float, float]) -> bool:
    return best[0] == next_best[0] and math.isclose(best[1], next_best[1], rel_tol=0.05, abs_tol=0.05) and math.isclose(best[2], next_best[2], rel_tol=0.08, abs_tol=0.03)


def _leg_branch_chain_length(graph: JointGraph, root: str) -> int:
    return len(_single_child_chain(graph, root, max_nodes=4))


def _leg_branch_downward_drop(graph: JointGraph, root: str) -> float:
    chain = _single_child_chain(graph, root, max_nodes=4)
    return _y(graph, root) - min(_y(graph, node_id) for node_id in chain)


def _best_leg_lateral_axis(graph: JointGraph, candidates: list[str]) -> int | None:
    if len(candidates) != 2:
        return None
    scored_axes: list[tuple[float, int]] = []
    for axis in (0, 2):
        first, second = candidates
        separation = abs(_leg_lateral_center(graph, first, axis) - _leg_lateral_center(graph, second, axis))
        if separation >= 0.05:
            scored_axes.append((separation, axis))
    if not scored_axes:
        return None
    scored_axes.sort(reverse=True)
    if len(scored_axes) > 1 and math.isclose(scored_axes[0][0], scored_axes[1][0], rel_tol=0.05, abs_tol=0.01):
        return None
    return scored_axes[0][1]


def _leg_lateral_center(graph: JointGraph, root: str, axis: int) -> float:
    chain = _single_child_chain(graph, root, max_nodes=4)
    return sum(_axis_value(graph, node_id, axis) for node_id in chain) / len(chain)


def _symmetric_pair(graph: JointGraph, candidates: list[str], *, label: str) -> tuple[str, str]:
    if len(candidates) > 2:
        candidates = _best_symmetric_pair_from_many(graph, candidates)
    axis = _best_lateral_axis(graph, candidates)
    if axis is None:
        _fail(
            "semantic_symmetry_ambiguous",
            f"Unable to find one clear negative and one clear positive symmetric {label} branch on an inferred lateral axis.",
            candidates=sorted(candidates),
        )
    first, second = sorted(candidates, key=lambda node_id: (_axis_value(graph, node_id, axis), graph.nodes[node_id].index))
    return first, second


def _best_symmetric_pair_from_many(graph: JointGraph, candidates: list[str]) -> list[str]:
    scored_pairs: list[tuple[float, int, tuple[str, str]]] = []
    for first, second in combinations(candidates, 2):
        pair = [first, second]
        axis = _best_lateral_axis(graph, pair)
        if axis is None:
            continue
        separation = abs(_axis_value(graph, first, axis) - _axis_value(graph, second, axis))
        scored_pairs.append((separation, -max(graph.nodes[first].index, graph.nodes[second].index), (first, second)))
    if not scored_pairs:
        return []
    scored_pairs.sort(reverse=True)
    if len(scored_pairs) > 1 and math.isclose(scored_pairs[0][0], scored_pairs[1][0], rel_tol=0.05, abs_tol=0.01):
        return []
    return list(scored_pairs[0][2])


def _best_lateral_axis(graph: JointGraph, candidates: list[str]) -> int | None:
    if len(candidates) != 2:
        return None
    scored_axes: list[tuple[float, int]] = []
    for axis in (0, 2):
        first, second = candidates
        separation = abs(_axis_value(graph, first, axis) - _axis_value(graph, second, axis))
        if separation >= 0.05:
            scored_axes.append((separation, axis))
    if not scored_axes:
        return None
    scored_axes.sort(reverse=True)
    if len(scored_axes) > 1 and math.isclose(scored_axes[0][0], scored_axes[1][0], rel_tol=0.05, abs_tol=0.01):
        return None
    return scored_axes[0][1]


def _axis_value(graph: JointGraph, node_id: str, axis: int) -> float:
    return graph.nodes[node_id].rest_world[axis][3]


def _highest_path_from(graph: JointGraph, root: str) -> list[str]:
    path = [root]
    current = root
    while graph.nodes[current].children:
        current = max(graph.nodes[current].children, key=lambda child: (_descendant_max_y(graph, child), -abs(_x(graph, child)), -graph.nodes[child].index))
        path.append(current)
    return path


def _descendant_max_y(graph: JointGraph, node_id: str) -> float:
    return max([_y(graph, node_id), *(_descendant_max_y(graph, child) for child in graph.nodes[node_id].children)])


def _single_child_chain(graph: JointGraph, root: str, *, max_nodes: int) -> list[str]:
    chain = [root]
    current = root
    while len(chain) < max_nodes and graph.nodes[current].children:
        current = max(graph.nodes[current].children, key=lambda child: (abs(_x(graph, child) - _x(graph, root)), -graph.nodes[child].index))
        chain.append(current)
    return chain


def _has_negative_and_positive(graph: JointGraph, nodes: list[str]) -> bool:
    return any(_x(graph, node_id) < -0.05 for node_id in nodes) and any(_x(graph, node_id) > 0.05 for node_id in nodes)


def _declared_nodes(graph: JointGraph) -> list[dict[str, Any]]:
    return [
        {
            "id": node.id,
            "name": node.name,
            "index": node.index,
            "parent": node.parent,
            "rest_local": node.rest_local,
            "rest_world": node.rest_world,
            "inverse_bind": "available" if graph.has_inverse_bind else "unavailable",
        }
        for node in sorted(graph.nodes.values(), key=lambda item: item.index)
    ]


def _x(graph: JointGraph, node_id: str) -> float:
    return graph.nodes[node_id].rest_world[0][3]


def _y(graph: JointGraph, node_id: str) -> float:
    return graph.nodes[node_id].rest_world[1][3]


def _node_id(node: Any, index: int) -> str:
    name = str(node.get("name") or "").strip() if isinstance(node, dict) else ""
    return name or f"node_{index}"


def _children(node: Any) -> list[int]:
    children = node.get("children") if isinstance(node, dict) and isinstance(node.get("children"), list) else []
    return [child for child in children if isinstance(child, int)]


def _depth(index: int, parent_by_index: dict[int, int | None], joint_indices: set[int]) -> int:
    depth = 0
    current = parent_by_index[index]
    seen: set[int] = set()
    while current in joint_indices and current not in seen:
        seen.add(current)
        depth += 1
        current = parent_by_index[current]
    return depth


def _local_matrix(node: dict[str, Any], index: int) -> list[list[float]]:
    raw_matrix = node.get("matrix")
    if raw_matrix is not None:
        if isinstance(raw_matrix, list) and len(raw_matrix) == 16 and all(isinstance(value, (int, float)) for value in raw_matrix):
            return [[float(raw_matrix[row * 4 + column]) for column in range(4)] for row in range(4)]
        _fail("semantic_transform_malformed", f"Node index {index} matrix must contain 16 numeric values.")
    translation = _numeric_triplet(node.get("translation"), default=(0.0, 0.0, 0.0), label=f"Node index {index} translation")
    scale = _numeric_triplet(node.get("scale"), default=(1.0, 1.0, 1.0), label=f"Node index {index} scale")
    rotation = _numeric_quaternion(node.get("rotation"), default=(0.0, 0.0, 0.0, 1.0), label=f"Node index {index} rotation")
    return _compose_trs(translation, rotation, scale)


def _numeric_triplet(value: Any, *, default: tuple[float, float, float], label: str) -> tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, list) and len(value) == 3 and all(isinstance(item, (int, float)) for item in value):
        return (float(value[0]), float(value[1]), float(value[2]))
    _fail("semantic_transform_malformed", f"{label} must contain 3 numeric values.")


def _numeric_quaternion(value: Any, *, default: tuple[float, float, float, float], label: str) -> tuple[float, float, float, float]:
    if value is None:
        return default
    if isinstance(value, list) and len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    _fail("semantic_transform_malformed", f"{label} must contain 4 numeric values.")


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
    raw_nodes: list[Any],
    parent_by_index: dict[int, int | None],
    local_by_index: dict[int, list[list[float]]],
    world_cache: dict[int, list[list[float]]],
) -> list[list[float]]:
    if index in world_cache:
        return world_cache[index]
    parent_index = parent_by_index[index]
    if parent_index in local_by_index:
        world_cache[index] = _multiply_matrices(_world_matrix(parent_index, raw_nodes, parent_by_index, local_by_index, world_cache), local_by_index[index])
    else:
        world_cache[index] = local_by_index[index]
    return world_cache[index]


def _multiply_matrices(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [round(sum(left[row][inner] * right[inner][column] for inner in range(4)), 10) for column in range(4)]
        for row in range(4)
    ]


def _fingerprint(glb_json: dict[str, Any]) -> str:
    canonical = json.dumps(glb_json, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _fail(code: str, message: str, **details: Any) -> None:
    diagnostic = {"code": code, "message": message}
    diagnostic.update(details)
    raise SemanticHumanoidResolutionError(f"{code}: {message}", [diagnostic])
