from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gltf_skin_analysis import GlbContainer, JointWeightSummary, summarize_joint_weights
from .semantic_humanoid_resolver import JointGraph, extract_joint_graph


REQUIRED_CORE_ROLES = (
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

OPTIONAL_SHOULDER_CONNECTOR_ROLES = {
    "left_shoulder": "left",
    "right_shoulder": "right",
    "left_clavicle": "left",
    "right_clavicle": "right",
}

HIGH_REGION_THRESHOLD_NORMALIZED = 0.80
UPPER_ARM_HIGH_REGION_WARNING_CUTOFF_NORMALIZED = 0.85
UPPER_ARM_HIGH_REGION_WARNING_ROLES = {"chest", "left_upper_arm", "right_upper_arm"}
UPPER_ARM_ROLES = {"left_upper_arm", "right_upper_arm"}


@dataclass(frozen=True)
class SemanticBodyNode:
    id: str
    parent: str | None
    children: tuple[str, ...]
    rest_world: tuple[tuple[float, ...], ...]
    weight_bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    classes: tuple[str, ...]
    capabilities: tuple[str, ...]
    confidence: float
    reasons: tuple[str, ...]
    is_contract_candidate: bool

    def as_diagnostic(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "children": list(self.children),
            "classes": list(self.classes),
            "capabilities": list(self.capabilities),
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "is_contract_candidate": self.is_contract_candidate,
            "weight_bbox": self.weight_bbox,
        }


@dataclass(frozen=True)
class SemanticBodyReport:
    nodes: dict[str, SemanticBodyNode]
    predicates: dict[str, bool | float]
    publishable: bool
    core_roles: dict[str, str]
    contract_core_confidence: float
    diagnostic: dict[str, Any]

    def as_diagnostic(self) -> dict[str, Any]:
        return dict(self.diagnostic)


def build_semantic_body_report(
    container: GlbContainer,
    declared: dict[str, Any],
    *,
    weight_analysis: tuple[dict[str, JointWeightSummary], dict[str, Any]] | None = None,
) -> SemanticBodyReport:
    # Guardrail: this post-GLB evidence layer classifies and gates contract roles;
    # it does not mutate generated skin weights or replace upstream pre-rig segmentation.
    graph = extract_joint_graph(container.json)
    roles = declared.get("roles") if isinstance(declared.get("roles"), dict) else {}
    summaries, weight_summary = weight_analysis if weight_analysis is not None else summarize_joint_weights(container)
    role_by_joint = {str(joint): str(role) for role, joint in roles.items()}
    passive_roots = _passive_branch_roots(graph, roles)
    high_region = _high_region_reasons(roles, summaries, weight_summary)
    unknown_required = [role for role in REQUIRED_CORE_ROLES if not _role_is_candidate(role, roles, graph, passive_roots)]
    nodes = {
        node_id: _semantic_node(graph, node_id, role_by_joint, passive_roots, summaries, roles)
        for node_id in graph.joints
    }
    core_roles = {
        role: str(roles[role])
        for role in REQUIRED_CORE_ROLES
        if _role_is_candidate(role, roles, graph, passive_roots)
    }
    passive_nodes = sorted(node_id for node_id, node in nodes.items() if "passive" in node.classes)
    passive_reasons = _passive_reasons(graph, passive_roots, nodes)
    blocking_high_region, high_region_warnings = _split_high_region_reasons(
        high_region,
        core_roles=core_roles,
        unknown_required=unknown_required,
        passive_nodes=passive_nodes,
    )
    core_confidence = _core_confidence(nodes, core_roles, blocking_high_region, unknown_required)
    predicates: dict[str, bool | float] = {
        "has_clear_spine": all(role in core_roles for role in ("hips", "spine", "chest", "neck", "head")),
        "has_left_right_arm_pair": all(role in core_roles for role in ("left_upper_arm", "left_lower_arm", "left_hand", "right_upper_arm", "right_lower_arm", "right_hand")),
        "has_leg_pair": all(role in core_roles for role in ("left_upper_leg", "left_lower_leg", "left_foot", "right_upper_leg", "right_lower_leg", "right_foot")),
        "has_passive_noncontract_subtrees": bool(passive_nodes),
        "has_high_region_contamination": bool(blocking_high_region),
        "has_high_region_warning": bool(high_region_warnings),
        "unknown_near_required_roles": bool(unknown_required),
        "contract_core_confidence": core_confidence,
    }
    reasons: list[dict[str, Any]] = []
    reasons.extend(passive_reasons)
    reasons.extend(blocking_high_region)
    if unknown_required:
        reasons.append(
            {
                "code": "required_role_semantic_evidence_missing",
                "message": "One or more required humanoid roles lacks confident anatomical core evidence.",
                "roles": unknown_required,
            }
        )
    if core_confidence < 0.85:
        reasons.append(
            {
                "code": "semantic_contract_core_confidence_low",
                "message": "Semantic body graph confidence is too low to publish a humanoid contract safely.",
                "contract_core_confidence": core_confidence,
            }
        )
    publishable = (
        bool(predicates["has_clear_spine"])
        and bool(predicates["has_left_right_arm_pair"])
        and bool(predicates["has_leg_pair"])
        and not bool(predicates["unknown_near_required_roles"])
        and not passive_nodes
        and core_confidence >= 0.85
    )
    diagnostic = {
        "code": "semantic_body_graph",
        "status": "publishable" if publishable else "not_publishable",
        "publishable": publishable,
        "predicates": dict(predicates),
        "contract_core_confidence": core_confidence,
        "core_roles": dict(sorted(core_roles.items())),
        "unknown_required_roles": unknown_required,
        "nodes": {node_id: nodes[node_id].as_diagnostic() for node_id in sorted(nodes)},
        "weighted_joints": {joint: summaries[joint].as_diagnostic() for joint in sorted(summaries)},
        "weight_summary": weight_summary,
        "reasons": reasons,
        "warnings": high_region_warnings,
    }
    return SemanticBodyReport(
        nodes=nodes,
        predicates=predicates,
        publishable=publishable,
        core_roles=dict(sorted(core_roles.items())),
        contract_core_confidence=core_confidence,
        diagnostic=diagnostic,
    )


def _semantic_node(
    graph: JointGraph,
    node_id: str,
    role_by_joint: dict[str, str],
    passive_roots: set[str],
    summaries: dict[str, JointWeightSummary],
    roles: dict[str, Any],
) -> SemanticBodyNode:
    passive_descendants = set().union(*(_descendants(graph, root) for root in passive_roots)) if passive_roots else set()
    classes: list[str] = []
    capabilities: list[str] = []
    reasons: list[str] = []
    role = role_by_joint.get(node_id)
    if node_id in passive_descendants:
        passive_class = _classify_passive_node(graph, node_id, roles, summaries)
        classes.extend([passive_class, "passive"])
        capabilities.append("noncontract")
        reasons.append("passive_subtree_not_contract_candidate")
        confidence = 0.82
        is_candidate = False
    elif role:
        is_optional_connector = _is_safe_optional_connector(role, node_id, graph, roles)
        classes.extend(_classes_for_role(role))
        capabilities.append("humanoid_contract")
        reasons.append("optional_connector_matches_anatomical_path" if is_optional_connector else "declared_role_matches_anatomical_core")
        confidence = 0.94
        is_candidate = role in REQUIRED_CORE_ROLES or is_optional_connector
    else:
        classes.append("unknown")
        reasons.append("not_selected_as_required_core_role")
        confidence = 0.25
        is_candidate = False
    summary = summaries.get(node_id)
    bbox = None
    if summary and summary.count:
        bbox = ((summary.min_x, summary.min_y, summary.min_z), (summary.max_x, summary.max_y, summary.max_z))
    return SemanticBodyNode(
        id=node_id,
        parent=graph.nodes[node_id].parent,
        children=tuple(graph.nodes[node_id].children),
        rest_world=tuple(tuple(row) for row in graph.nodes[node_id].rest_world),
        weight_bbox=bbox,
        classes=tuple(classes),
        capabilities=tuple(capabilities),
        confidence=round(confidence, 3),
        reasons=tuple(reasons),
        is_contract_candidate=is_candidate,
    )


def _classes_for_role(role: str) -> list[str]:
    if role == "hips":
        return ["root", "body_core"]
    if role in {"spine", "chest", "neck", "head"}:
        return ["spine", "body_core"] if role != "head" else ["body_core"]
    if "hand" in role:
        return ["hand", "limb"]
    if "foot" in role:
        return ["foot", "leg", "limb"]
    if "leg" in role:
        return ["leg", "limb"]
    if "arm" in role or "shoulder" in role:
        return ["limb"]
    if "clavicle" in role:
        return ["limb"]
    return ["unknown"]


def _is_safe_optional_connector(role: str, joint: str, graph: JointGraph, roles: dict[str, Any]) -> bool:
    side = OPTIONAL_SHOULDER_CONNECTOR_ROLES.get(role)
    if side is None or joint not in graph.nodes:
        return False
    upper_arm = str(roles.get(f"{side}_upper_arm", ""))
    if not upper_arm or upper_arm not in graph.nodes:
        return False
    parent = graph.nodes[joint].parent
    expected_parents = {str(roles.get("chest", "")), str(roles.get("spine", ""))} - {""}
    return parent in expected_parents and graph.nodes[upper_arm].parent == joint


def _role_is_candidate(role: str, roles: dict[str, Any], graph: JointGraph, passive_roots: set[str]) -> bool:
    joint = str(roles.get(role, ""))
    if not joint or joint not in graph.nodes:
        return False
    return not any(joint in _descendants(graph, root) for root in passive_roots)


def _passive_branch_roots(graph: JointGraph, roles: dict[str, Any]) -> set[str]:
    roots: set[str] = set()
    for side in ("left", "right"):
        arm_chain = [str(roles.get(f"{side}_{name}", "")) for name in ("upper_arm", "lower_arm", "hand") if roles.get(f"{side}_{name}")]
        selected = set(arm_chain)
        for node in arm_chain[:-1]:
            if node not in graph.nodes:
                continue
            roots.update(child for child in graph.nodes[node].children if child not in selected)
    return roots


def _classify_passive_node(graph: JointGraph, node_id: str, roles: dict[str, Any], summaries: dict[str, JointWeightSummary]) -> str:
    head = str(roles.get("head", ""))
    y = graph.nodes[node_id].rest_world[1][3]
    if head and head in graph.nodes and y >= graph.nodes[head].rest_world[1][3] - 0.15:
        return "hair"
    summary = summaries.get(node_id)
    if summary and summary.count and summary.max_y < y + 0.2:
        return "clothing"
    if summary and summary.count and max(summary.max_x - summary.min_x, summary.max_z - summary.min_z) < 0.18:
        return "accessory"
    return "clothing"


def _passive_reasons(graph: JointGraph, passive_roots: set[str], nodes: dict[str, SemanticBodyNode]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for root in sorted(passive_roots):
        descendants = _descendants(graph, root)
        classes = sorted({klass for node in descendants for klass in nodes[node].classes if klass != "passive"})
        reasons.append(
            {
                "code": "semantic_passive_noncontract_subtree",
                "message": "A separable passive subtree is excluded from humanoid contract roles.",
                "branch_root": root,
                "descendants": descendants,
                "classes": classes,
            }
        )
        if "clothing" in classes:
            reasons.append(
                {
                    "code": "sleeve_branch_under_arm",
                    "message": "An arm-adjacent clothing subtree is separable but unsafe for humanoid contract roles.",
                    "branch_root": root,
                    "descendants": descendants,
                    "classes": classes,
                }
            )
        if "accessory" in classes:
            reasons.append(
                {
                    "code": "non_anatomical_leaf_under_arm",
                    "message": "An arm-adjacent accessory subtree is separable but unsafe for humanoid contract roles.",
                    "branch_root": root,
                    "descendants": descendants,
                    "classes": classes,
                }
            )
    return reasons


def _high_region_reasons(roles: dict[str, Any], summaries: dict[str, JointWeightSummary], weight_summary: dict[str, Any]) -> list[dict[str, Any]]:
    height_min = float(weight_summary.get("height_min", 0.0))
    height = max(float(weight_summary.get("height", 0.0)), 0.000001)
    high_threshold = height_min + height * HIGH_REGION_THRESHOLD_NORMALIZED
    suspect_roles = [role for role in roles if role in {"hips", "spine", "chest"} or "arm" in role or "hand" in role]
    high_joints = []
    for role in suspect_roles:
        joint = str(roles[role])
        summary = summaries.get(joint)
        if summary and summary.count and summary.max_y >= high_threshold:
            normalized_max_y = (float(summary.max_y) - height_min) / height
            high_joints.append(
                {
                    "role": role,
                    "joint": joint,
                    "max_y": round(summary.max_y, 6),
                    "normalized_max_y": round(normalized_max_y, 6),
                    "threshold_normalized": HIGH_REGION_THRESHOLD_NORMALIZED,
                }
            )
    if not high_joints:
        return []
    return [
        {
            "code": "high_region_weighted_by_torso_or_arm",
            "message": "Torso/arm role weights reach head/top model region.",
            "threshold_normalized": HIGH_REGION_THRESHOLD_NORMALIZED,
            "joints": high_joints,
        }
    ]


def _split_high_region_reasons(
    high_region: list[dict[str, Any]],
    *,
    core_roles: dict[str, str],
    unknown_required: list[str],
    passive_nodes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not high_region:
        return [], []
    has_required_core = all(role in core_roles for role in REQUIRED_CORE_ROLES)
    high_joints = [joint for reason in high_region for joint in reason.get("joints", [])]
    high_roles = {str(joint.get("role", "")) for joint in high_joints}
    is_safe_shoulder_band = bool(high_joints) and high_roles.issubset(UPPER_ARM_HIGH_REGION_WARNING_ROLES)
    upper_arm_joints = [joint for joint in high_joints if joint.get("role") in UPPER_ARM_ROLES]
    upper_arms_below_cutoff = all(
        float(joint.get("normalized_max_y", 1.0)) < UPPER_ARM_HIGH_REGION_WARNING_CUTOFF_NORMALIZED
        for joint in upper_arm_joints
    )
    if is_safe_shoulder_band and upper_arms_below_cutoff and has_required_core and not unknown_required and not passive_nodes:
        warnings = []
        for reason in high_region:
            warning = dict(reason)
            warning["severity"] = "warning"
            warning["upper_arm_warning_cutoff_normalized"] = UPPER_ARM_HIGH_REGION_WARNING_CUTOFF_NORMALIZED
            warning["message"] = "Chest or moderate upper-arm role weights reach the high model region; retained as non-blocking semantic evidence."
            warnings.append(warning)
        return [], warnings
    return high_region, []


def _core_confidence(nodes: dict[str, SemanticBodyNode], core_roles: dict[str, str], high_region: list[dict[str, Any]], unknown_required: list[str]) -> float:
    if unknown_required:
        return 0.35
    if not core_roles:
        return 0.0
    base = min(nodes[joint].confidence for joint in core_roles.values())
    if high_region:
        base = min(base, 0.72)
    return round(base, 3)


def _descendants(graph: JointGraph, root: str) -> list[str]:
    found = [root]
    for child in graph.nodes.get(root).children if root in graph.nodes else []:
        found.extend(_descendants(graph, child))
    return found
