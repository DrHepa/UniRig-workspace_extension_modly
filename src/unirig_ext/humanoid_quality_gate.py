from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gltf_skin_analysis import GlbContainer, GltfSkinAnalysisError, has_skinned_mesh_primitives, iter_weighted_vertices
from .semantic_humanoid_resolver import JointGraph, extract_joint_graph


WEIGHT_EPSILON = 0.01


class HumanoidQualityGateError(ValueError):
    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__(_format_failure(diagnostic))
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class HumanoidQualityReport:
    status: str
    diagnostic: dict[str, Any]


@dataclass
class JointWeightSummary:
    count: int = 0
    total_weight: float = 0.0
    min_x: float = float("inf")
    min_y: float = float("inf")
    min_z: float = float("inf")
    max_x: float = float("-inf")
    max_y: float = float("-inf")
    max_z: float = float("-inf")

    def add(self, position: tuple[float, float, float], weight: float) -> None:
        self.count += 1
        self.total_weight += weight
        x, y, z = position
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.min_z = min(self.min_z, z)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)
        self.max_z = max(self.max_z, z)

    def as_diagnostic(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "total_weight": 0.0}
        bbox = [[self.min_x, self.min_y, self.min_z], [self.max_x, self.max_y, self.max_z]]
        center = [(bbox[0][axis] + bbox[1][axis]) / 2.0 for axis in range(3)]
        spread = [bbox[1][axis] - bbox[0][axis] for axis in range(3)]
        return {
            "count": self.count,
            "total_weight": round(self.total_weight, 6),
            "bbox": [[round(value, 6) for value in point] for point in bbox],
            "center": [round(value, 6) for value in center],
            "spread": [round(value, 6) for value in spread],
        }


def run_humanoid_quality_gate(container: GlbContainer, declared: dict[str, Any]) -> HumanoidQualityReport:
    if not has_skinned_mesh_primitives(container.json):
        return HumanoidQualityReport(
            status="not_applicable",
            diagnostic={
                "code": "humanoid_rig_quality_gate",
                "status": "not_applicable",
                "reason": "skin_weight_data_not_present",
            },
        )
    graph = extract_joint_graph(container.json)
    roles = declared.get("roles") if isinstance(declared.get("roles"), dict) else {}
    summaries, weight_summary = _summarize_weights(container)
    joint_classes = _classify_joints(graph, roles, summaries)
    reasons: list[dict[str, Any]] = []
    reasons.extend(_arm_branch_reasons(graph, roles, joint_classes, summaries))
    reasons.extend(_non_local_weight_reasons(graph, roles, summaries, weight_summary))
    reasons.extend(_high_region_reasons(roles, summaries, weight_summary))
    unknown_selected = sorted(role for role, joint in roles.items() if joint_classes.get(str(joint)) in {"unknown", "clothing", "hair", "accessory"})
    if unknown_selected:
        reasons.append(
            {
                "code": "anatomical_role_not_separable",
                "message": "One or more selected humanoid roles overlaps non-body or unknown joint evidence.",
                "roles": unknown_selected,
            }
        )
    diagnostic = {
        "code": "humanoid_rig_quality_gate",
        "status": "passed" if not reasons else "failed",
        "joint_classes": dict(sorted(joint_classes.items())),
        "weight_summary": weight_summary,
        "weighted_joints": {joint: summaries[joint].as_diagnostic() for joint in sorted(summaries)},
        "checks": ["skin_weight_data", "arm_branch_separation", "non_local_weight_spread", "high_region_influences"],
    }
    if reasons:
        failure = {
            "code": "unsafe_for_humanoid_retarget",
            "severity": "error",
            "reasons": reasons,
            "joint_classes": diagnostic["joint_classes"],
            "weight_summary": diagnostic["weight_summary"],
            "weighted_joints": diagnostic["weighted_joints"],
            "remediation": "Provide explicit humanoid source without garment/accessory skin joints or separate clothing/hair before humanoid metadata export.",
        }
        raise HumanoidQualityGateError(failure)
    return HumanoidQualityReport(status="passed", diagnostic=diagnostic)


def _summarize_weights(container: GlbContainer) -> tuple[dict[str, JointWeightSummary], dict[str, Any]]:
    summaries: dict[str, JointWeightSummary] = {}
    vertex_count = 0
    influenced_vertices = 0
    min_y = float("inf")
    max_y = float("-inf")
    try:
        vertices = list(iter_weighted_vertices(container))
    except GltfSkinAnalysisError as exc:
        failure = {
            "code": "unsafe_for_humanoid_retarget",
            "severity": "error",
            "reasons": [{"code": "skin_weight_data_unavailable", "message": str(exc)}],
            "joint_classes": {},
            "weight_summary": {"vertex_count": 0},
            "remediation": "Export an embedded GLB with POSITION, JOINTS_0, WEIGHTS_0, and one skin, or avoid metadata_mode=humanoid.",
        }
        raise HumanoidQualityGateError(failure) from exc
    for vertex in vertices:
        vertex_count += 1
        min_y = min(min_y, vertex.position[1])
        max_y = max(max_y, vertex.position[1])
        if vertex.influences:
            influenced_vertices += 1
        for joint, weight in vertex.influences:
            if weight < WEIGHT_EPSILON:
                continue
            summaries.setdefault(joint, JointWeightSummary()).add(vertex.position, weight)
    if vertex_count == 0:
        raise HumanoidQualityGateError({"code": "unsafe_for_humanoid_retarget", "severity": "error", "reasons": [{"code": "skin_weight_data_unavailable", "message": "No weighted vertices were available."}], "joint_classes": {}, "weight_summary": {"vertex_count": 0}})
    return summaries, {
        "vertex_count": vertex_count,
        "influenced_vertices": influenced_vertices,
        "height_min": round(min_y, 6),
        "height_max": round(max_y, 6),
        "height": round(max_y - min_y, 6),
    }


def _classify_joints(graph: JointGraph, roles: dict[str, Any], summaries: dict[str, JointWeightSummary]) -> dict[str, str]:
    role_joints = {str(value) for value in roles.values()}
    classes = {joint: ("body" if joint in role_joints else "unknown") for joint in graph.joints}
    for side in ("left", "right"):
        arm_chain = _role_chain(roles, side, ("upper_arm", "lower_arm", "hand"))
        body_chain = set(arm_chain)
        for branch in _branches_under_chain(graph, arm_chain):
            branch_class = _classify_non_body_branch(graph, branch, roles, summaries)
            for descendant in _descendants(graph, branch):
                if descendant not in body_chain:
                    classes[descendant] = branch_class
    for joint in graph.joints:
        if joint in role_joints:
            classes[joint] = "body"
    return classes


def _classify_non_body_branch(graph: JointGraph, root: str, roles: dict[str, Any], summaries: dict[str, JointWeightSummary]) -> str:
    y = graph.nodes[root].rest_world[1][3]
    head = str(roles.get("head", ""))
    if head and y >= graph.nodes[head].rest_world[1][3] - 0.15:
        return "hair"
    summary = summaries.get(root)
    if summary and summary.count and summary.max_y < graph.nodes[root].rest_world[1][3] + 0.2:
        return "clothing"
    if summary and summary.count and max(summary.max_x - summary.min_x, summary.max_z - summary.min_z) < 0.18:
        return "accessory"
    return "clothing" if y < graph.nodes[root].rest_world[1][3] + 0.01 else "unknown"


def _arm_branch_reasons(graph: JointGraph, roles: dict[str, Any], joint_classes: dict[str, str], summaries: dict[str, JointWeightSummary]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for side in ("left", "right"):
        arm_chain = _role_chain(roles, side, ("upper_arm", "lower_arm", "hand"))
        for branch in _branches_under_chain(graph, arm_chain):
            descendants = _descendants(graph, branch)
            branch_classes = sorted({joint_classes.get(joint, "unknown") for joint in descendants})
            reason_code = "sleeve_branch_under_arm" if "clothing" in branch_classes else "non_anatomical_leaf_under_arm"
            reasons.append(
                {
                    "code": reason_code,
                    "message": f"{side} arm has a non-selected descendant branch that competes with anatomical hand selection.",
                    "side": side,
                    "branch_root": branch,
                    "descendants": descendants,
                    "classes": branch_classes,
                    "weighted": branch in summaries,
                }
            )
    return reasons


def _non_local_weight_reasons(graph: JointGraph, roles: dict[str, Any], summaries: dict[str, JointWeightSummary], weight_summary: dict[str, Any]) -> list[dict[str, Any]]:
    height = max(float(weight_summary.get("height", 0.0)), 0.000001)
    reasons: list[dict[str, Any]] = []
    for role, joint in sorted((str(role), str(joint)) for role, joint in roles.items()):
        summary = summaries.get(joint)
        if not summary or summary.count == 0 or joint not in graph.nodes:
            continue
        spread_y = summary.max_y - summary.min_y
        if ("arm" in role or "hand" in role or "leg" in role or "foot" in role) and spread_y > height * 0.55:
            reasons.append({"code": "non_local_weight_spread", "role": role, "joint": joint, "spread_y": round(spread_y, 6), "model_height": round(height, 6)})
    return reasons


def _high_region_reasons(roles: dict[str, Any], summaries: dict[str, JointWeightSummary], weight_summary: dict[str, Any]) -> list[dict[str, Any]]:
    high_threshold = float(weight_summary.get("height_min", 0.0)) + float(weight_summary.get("height", 0.0)) * 0.80
    suspect_roles = [role for role in roles if role in {"hips", "spine", "chest"} or "arm" in role or "hand" in role]
    high_joints = []
    for role in suspect_roles:
        joint = str(roles[role])
        summary = summaries.get(joint)
        if summary and summary.count and summary.max_y >= high_threshold:
            high_joints.append({"role": role, "joint": joint, "max_y": round(summary.max_y, 6)})
    if not high_joints:
        return []
    return [{"code": "high_region_weighted_by_torso_or_arm", "message": "Torso/arm role weights reach head/top model region.", "joints": high_joints}]


def _role_chain(roles: dict[str, Any], side: str, names: tuple[str, ...]) -> list[str]:
    return [str(roles.get(f"{side}_{name}", "")) for name in names if roles.get(f"{side}_{name}")]


def _branches_under_chain(graph: JointGraph, chain: list[str]) -> list[str]:
    if len(chain) < 2:
        return []
    selected = set(chain)
    branches: list[str] = []
    for node in chain[:-1]:
        if node not in graph.nodes:
            continue
        for child in graph.nodes[node].children:
            if child not in selected:
                branches.append(child)
    return branches


def _descendants(graph: JointGraph, root: str) -> list[str]:
    found = [root]
    for child in graph.nodes.get(root).children if root in graph.nodes else []:
        found.extend(_descendants(graph, child))
    return found


def _format_failure(diagnostic: dict[str, Any]) -> str:
    codes = ", ".join(str(reason.get("code", "unknown")) for reason in diagnostic.get("reasons", []))
    return f"unsafe_for_humanoid_retarget: {codes}. {diagnostic.get('remediation', '')}"
