from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from fixtures.unirig_real_topology import REAL_UNIRIG_40_EDGES, REAL_UNIRIG_52_EDGES, real_unirig_40_payload, real_unirig_52_payload
from test_humanoid_quality_gate import _declared_roles, write_embedded_skin_glb
from unirig_ext.gltf_skin_analysis import read_glb_container
from unirig_ext.humanoid_contract import build_contract_from_declared_data
from unirig_ext.semantic_body_graph import build_semantic_body_report
from unirig_ext.semantic_humanoid_resolver import SemanticHumanoidResolutionError, extract_joint_graph, resolve_humanoid
from unirig_ext.topology_profiles import REAL_UNIRIG_40_ROLE_MAP, REAL_UNIRIG_52_ROLE_MAP


def minimal_semantic_payload(*, prefix: str = "joint") -> dict:
    names = [
        "hips",
        "spine",
        "spine_mid",
        "chest",
        "neck",
        "head",
        "left_shoulder",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_shoulder",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_upper_leg",
        "left_lower_leg",
        "left_foot",
        "right_upper_leg",
        "right_lower_leg",
        "right_foot",
    ]
    edges = (
        ("hips", "spine"),
        ("spine", "spine_mid"),
        ("spine_mid", "chest"),
        ("chest", "neck"),
        ("neck", "head"),
        ("chest", "left_shoulder"),
        ("left_shoulder", "left_upper_arm"),
        ("left_upper_arm", "left_lower_arm"),
        ("left_lower_arm", "left_hand"),
        ("chest", "right_shoulder"),
        ("right_shoulder", "right_upper_arm"),
        ("right_upper_arm", "right_lower_arm"),
        ("right_lower_arm", "right_hand"),
        ("hips", "left_upper_leg"),
        ("left_upper_leg", "left_lower_leg"),
        ("left_lower_leg", "left_foot"),
        ("hips", "right_upper_leg"),
        ("right_upper_leg", "right_lower_leg"),
        ("right_lower_leg", "right_foot"),
    )
    translations = {
        "hips": [0.0, 1.0, 0.0],
        "spine": [0.0, 0.7, 0.0],
        "spine_mid": [0.0, 0.5, 0.0],
        "chest": [0.0, 0.5, 0.0],
        "neck": [0.0, 0.4, 0.0],
        "head": [0.0, 0.35, 0.0],
        "left_shoulder": [-0.25, 0.05, 0.0],
        "left_upper_arm": [-0.45, -0.05, 0.0],
        "left_lower_arm": [-0.45, -0.05, 0.0],
        "left_hand": [-0.25, -0.05, 0.0],
        "right_shoulder": [0.25, 0.05, 0.0],
        "right_upper_arm": [0.45, -0.05, 0.0],
        "right_lower_arm": [0.45, -0.05, 0.0],
        "right_hand": [0.25, -0.05, 0.0],
        "left_upper_leg": [-0.18, -0.8, 0.0],
        "left_lower_leg": [0.0, -0.8, 0.0],
        "left_foot": [0.0, -0.25, 0.35],
        "right_upper_leg": [0.18, -0.8, 0.0],
        "right_lower_leg": [0.0, -0.8, 0.0],
        "right_foot": [0.0, -0.25, 0.35],
    }
    nodes = [{"name": f"{prefix}_{name}", "translation": translations[name]} for name in names]
    index_by_name = {name: index for index, name in enumerate(names)}
    for parent, child in edges:
        nodes[index_by_name[parent]].setdefault("children", []).append(index_by_name[child])
    return {"asset": {"version": "2.0"}, "nodes": nodes, "skins": [{"joints": list(range(len(nodes))), "inverseBindMatrices": 0}]}


def shoulderless_semantic_payload(*, prefix: str = "joint") -> dict:
    names = [
        "hips",
        "spine",
        "spine_mid",
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
    ]
    edges = (
        ("hips", "spine"),
        ("spine", "spine_mid"),
        ("spine_mid", "chest"),
        ("chest", "neck"),
        ("neck", "head"),
        ("chest", "left_upper_arm"),
        ("left_upper_arm", "left_lower_arm"),
        ("left_lower_arm", "left_hand"),
        ("chest", "right_upper_arm"),
        ("right_upper_arm", "right_lower_arm"),
        ("right_lower_arm", "right_hand"),
        ("hips", "left_upper_leg"),
        ("left_upper_leg", "left_lower_leg"),
        ("left_lower_leg", "left_foot"),
        ("hips", "right_upper_leg"),
        ("right_upper_leg", "right_lower_leg"),
        ("right_lower_leg", "right_foot"),
    )
    translations = {
        "hips": [0.0, 1.0, 0.0],
        "spine": [0.0, 0.7, 0.0],
        "spine_mid": [0.0, 0.5, 0.0],
        "chest": [0.0, 0.5, 0.0],
        "neck": [0.0, 0.4, 0.0],
        "head": [0.0, 0.35, 0.0],
        "left_upper_arm": [-0.45, 0.0, 0.0],
        "left_lower_arm": [-0.45, -0.05, 0.0],
        "left_hand": [-0.25, -0.05, 0.0],
        "right_upper_arm": [0.45, 0.0, 0.0],
        "right_lower_arm": [0.45, -0.05, 0.0],
        "right_hand": [0.25, -0.05, 0.0],
        "left_upper_leg": [-0.18, -0.8, 0.0],
        "left_lower_leg": [0.0, -0.8, 0.0],
        "left_foot": [0.0, -0.25, 0.35],
        "right_upper_leg": [0.18, -0.8, 0.0],
        "right_lower_leg": [0.0, -0.8, 0.0],
        "right_foot": [0.0, -0.25, 0.35],
    }
    nodes = [{"name": f"{prefix}_{name}", "translation": translations[name]} for name in names]
    index_by_name = {name: index for index, name in enumerate(names)}
    for parent, child in edges:
        nodes[index_by_name[parent]].setdefault("children", []).append(index_by_name[child])
    return {"asset": {"version": "2.0"}, "nodes": nodes, "skins": [{"joints": list(range(len(nodes))), "inverseBindMatrices": 0}]}


def near_center_arm_branch_payload(*, prefix: str = "joint") -> dict:
    payload = minimal_semantic_payload(prefix=prefix)
    replacements = {
        f"{prefix}_left_shoulder": [-0.02, 0.05, 0.0],
        f"{prefix}_left_upper_arm": [-0.18, -0.08, 0.0],
        f"{prefix}_left_lower_arm": [-0.17, -0.08, 0.0],
        f"{prefix}_left_hand": [0.0, -0.06, 0.0],
        f"{prefix}_right_shoulder": [0.02, 0.05, 0.0],
        f"{prefix}_right_upper_arm": [0.18, -0.08, 0.0],
        f"{prefix}_right_lower_arm": [0.17, -0.08, 0.0],
        f"{prefix}_right_hand": [0.0, -0.06, 0.0],
    }
    for node in payload["nodes"]:
        if node["name"] in replacements:
            node["translation"] = replacements[node["name"]]
    return payload


def short_trunk_output_payload(*, prefix: str = "out") -> dict:
    names = [
        "hips",
        "spine",
        "left_upper_leg",
        "left_lower_leg",
        "left_foot",
        "right_upper_leg",
        "right_lower_leg",
        "right_foot",
        "loose_prop",
    ]
    edges = (
        ("hips", "spine"),
        ("hips", "left_upper_leg"),
        ("left_upper_leg", "left_lower_leg"),
        ("left_lower_leg", "left_foot"),
        ("hips", "right_upper_leg"),
        ("right_upper_leg", "right_lower_leg"),
        ("right_lower_leg", "right_foot"),
    )
    translations = {
        "hips": [0.0, 1.0, 0.0],
        "spine": [0.0, 0.5, 0.0],
        "left_upper_leg": [-0.18, -0.8, 0.0],
        "left_lower_leg": [0.0, -0.8, 0.0],
        "left_foot": [0.0, -0.25, 0.35],
        "right_upper_leg": [0.18, -0.8, 0.0],
        "right_lower_leg": [0.0, -0.8, 0.0],
        "right_foot": [0.0, -0.25, 0.35],
        "loose_prop": [0.0, 0.0, 0.0],
    }
    nodes = [{"name": f"{prefix}_{name}", "translation": translations[name]} for name in names]
    index_by_name = {name: index for index, name in enumerate(names)}
    for parent, child in edges:
        nodes[index_by_name[parent]].setdefault("children", []).append(index_by_name[child])
    return {"asset": {"version": "2.0"}, "nodes": nodes, "skins": [{"joints": list(range(8)), "inverseBindMatrices": 0}]}


def delayed_chest_semantic_payload(*, prefix: str = "joint") -> dict:
    names = [
        "hips",
        "spine",
        "spine_mid",
        "upper_spine",
        "chest",
        "neck",
        "head",
        "left_noise",
        "left_noise_tip",
        "right_noise",
        "right_noise_tip",
        "left_leaf",
        "right_leaf",
        "left_shoulder",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_shoulder",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_upper_leg",
        "left_lower_leg",
        "left_foot",
        "right_upper_leg",
        "right_lower_leg",
        "right_foot",
    ]
    edges = (
        ("hips", "spine"),
        ("spine", "spine_mid"),
        ("spine_mid", "upper_spine"),
        ("upper_spine", "chest"),
        ("chest", "neck"),
        ("neck", "head"),
        ("spine", "left_noise"),
        ("left_noise", "left_noise_tip"),
        ("spine", "right_noise"),
        ("right_noise", "right_noise_tip"),
        ("spine_mid", "left_leaf"),
        ("spine_mid", "right_leaf"),
        ("chest", "left_shoulder"),
        ("left_shoulder", "left_upper_arm"),
        ("left_upper_arm", "left_lower_arm"),
        ("left_lower_arm", "left_hand"),
        ("chest", "right_shoulder"),
        ("right_shoulder", "right_upper_arm"),
        ("right_upper_arm", "right_lower_arm"),
        ("right_lower_arm", "right_hand"),
        ("hips", "left_upper_leg"),
        ("left_upper_leg", "left_lower_leg"),
        ("left_lower_leg", "left_foot"),
        ("hips", "right_upper_leg"),
        ("right_upper_leg", "right_lower_leg"),
        ("right_lower_leg", "right_foot"),
    )
    translations = {
        "hips": [0.0, 1.0, 0.0],
        "spine": [0.0, 0.55, 0.0],
        "spine_mid": [0.0, 0.4, 0.0],
        "upper_spine": [0.0, 0.35, 0.0],
        "chest": [0.0, 0.3, 0.0],
        "neck": [0.0, 0.3, 0.0],
        "head": [0.0, 0.3, 0.0],
        "left_noise": [-0.08, 0.0, 0.0],
        "left_noise_tip": [-0.08, -0.05, 0.0],
        "right_noise": [0.08, 0.0, 0.0],
        "right_noise_tip": [0.08, -0.05, 0.0],
        "left_leaf": [-0.3, 0.0, 0.0],
        "right_leaf": [0.3, 0.0, 0.0],
        "left_shoulder": [-0.25, 0.0, 0.0],
        "left_upper_arm": [-0.35, -0.05, 0.0],
        "left_lower_arm": [-0.35, -0.05, 0.0],
        "left_hand": [-0.2, -0.05, 0.0],
        "right_shoulder": [0.25, 0.0, 0.0],
        "right_upper_arm": [0.35, -0.05, 0.0],
        "right_lower_arm": [0.35, -0.05, 0.0],
        "right_hand": [0.2, -0.05, 0.0],
        "left_upper_leg": [-0.18, -0.8, 0.0],
        "left_lower_leg": [0.0, -0.8, 0.0],
        "left_foot": [0.0, -0.25, 0.35],
        "right_upper_leg": [0.18, -0.8, 0.0],
        "right_lower_leg": [0.0, -0.8, 0.0],
        "right_foot": [0.0, -0.25, 0.35],
    }
    nodes = [{"name": f"{prefix}_{name}", "translation": translations[name]} for name in names]
    index_by_name = {name: index for index, name in enumerate(names)}
    for parent, child in edges:
        nodes[index_by_name[parent]].setdefault("children", []).append(index_by_name[child])
    return {"asset": {"version": "2.0"}, "nodes": nodes, "skins": [{"joints": list(range(len(nodes))), "inverseBindMatrices": 0}]}


def long_trunk_competing_chest_payload(*, prefix: str = "joint", false_pair: str = "head_side") -> dict:
    names = [
        "hips",
        "spine_low",
        "spine_mid",
        "true_chest",
        "upper_trunk",
        "neck",
        "head",
        "true_left_shoulder",
        "true_left_upper_arm",
        "true_left_lower_arm",
        "true_left_hand",
        "true_right_shoulder",
        "true_right_upper_arm",
        "true_right_lower_arm",
        "true_right_hand",
        "false_left_root",
        "false_left_mid",
        "false_left_tip",
        "false_left_end",
        "false_right_root",
        "false_right_mid",
        "false_right_tip",
        "false_right_end",
        "left_upper_leg",
        "left_lower_leg",
        "left_foot",
        "right_upper_leg",
        "right_lower_leg",
        "right_foot",
    ]
    edges = (
        ("hips", "spine_low"),
        ("spine_low", "spine_mid"),
        ("spine_mid", "true_chest"),
        ("true_chest", "upper_trunk"),
        ("upper_trunk", "neck"),
        ("neck", "head"),
        ("true_chest", "true_left_shoulder"),
        ("true_left_shoulder", "true_left_upper_arm"),
        ("true_left_upper_arm", "true_left_lower_arm"),
        ("true_left_lower_arm", "true_left_hand"),
        ("true_chest", "true_right_shoulder"),
        ("true_right_shoulder", "true_right_upper_arm"),
        ("true_right_upper_arm", "true_right_lower_arm"),
        ("true_right_lower_arm", "true_right_hand"),
        ("upper_trunk", "false_left_root"),
        ("false_left_root", "false_left_mid"),
        ("false_left_mid", "false_left_tip"),
        ("false_left_tip", "false_left_end"),
        ("upper_trunk", "false_right_root"),
        ("false_right_root", "false_right_mid"),
        ("false_right_mid", "false_right_tip"),
        ("false_right_tip", "false_right_end"),
        ("hips", "left_upper_leg"),
        ("left_upper_leg", "left_lower_leg"),
        ("left_lower_leg", "left_foot"),
        ("hips", "right_upper_leg"),
        ("right_upper_leg", "right_lower_leg"),
        ("right_lower_leg", "right_foot"),
    )
    translations = {
        "hips": [0.0, 1.0, 0.0],
        "spine_low": [0.0, 0.45, 0.0],
        "spine_mid": [0.0, 0.35, 0.0],
        "true_chest": [0.0, 0.3, 0.0],
        "upper_trunk": [0.0, 0.28, 0.0],
        "neck": [0.0, 0.25, 0.0],
        "head": [0.0, 0.35, 0.0],
        "true_left_shoulder": [-0.22, -0.02, 0.0],
        "true_left_upper_arm": [-0.28, -0.16, 0.0],
        "true_left_lower_arm": [-0.24, -0.14, 0.0],
        "true_left_hand": [-0.12, -0.08, 0.0],
        "true_right_shoulder": [0.22, -0.02, 0.0],
        "true_right_upper_arm": [0.28, -0.16, 0.0],
        "true_right_lower_arm": [0.24, -0.14, 0.0],
        "true_right_hand": [0.12, -0.08, 0.0],
        "left_upper_leg": [-0.18, -0.8, 0.0],
        "left_lower_leg": [0.0, -0.8, 0.0],
        "left_foot": [0.0, -0.25, 0.35],
        "right_upper_leg": [0.18, -0.8, 0.0],
        "right_lower_leg": [0.0, -0.8, 0.0],
        "right_foot": [0.0, -0.25, 0.35],
    }
    false_translations = {
        "head_side": {
            "false_left_root": [-0.34, 0.08, 0.0],
            "false_left_mid": [-0.06, 0.05, 0.0],
            "false_left_tip": [-0.04, 0.04, 0.0],
            "false_left_end": [-0.03, 0.02, 0.0],
            "false_right_root": [0.34, 0.08, 0.0],
            "false_right_mid": [0.06, 0.05, 0.0],
            "false_right_tip": [0.04, 0.04, 0.0],
            "false_right_end": [0.03, 0.02, 0.0],
        },
        "same_side": {
            "false_left_root": [0.18, -0.02, 0.0],
            "false_left_mid": [0.16, -0.12, 0.0],
            "false_left_tip": [0.14, -0.12, 0.0],
            "false_left_end": [0.08, -0.08, 0.0],
            "false_right_root": [0.46, -0.02, 0.0],
            "false_right_mid": [0.18, -0.12, 0.0],
            "false_right_tip": [0.12, -0.12, 0.0],
            "false_right_end": [0.06, -0.08, 0.0],
        },
        "centered": {
            "false_left_root": [0.01, -0.02, 0.0],
            "false_left_mid": [0.02, -0.12, 0.0],
            "false_left_tip": [0.01, -0.12, 0.0],
            "false_left_end": [0.0, -0.08, 0.0],
            "false_right_root": [0.38, -0.02, 0.0],
            "false_right_mid": [0.18, -0.12, 0.0],
            "false_right_tip": [0.12, -0.12, 0.0],
            "false_right_end": [0.06, -0.08, 0.0],
        },
        "ambiguous": {
            "false_left_root": [-0.23, -0.02, 0.0],
            "false_left_mid": [-0.27, -0.16, 0.0],
            "false_left_tip": [-0.23, -0.14, 0.0],
            "false_left_end": [-0.12, -0.08, 0.0],
            "false_right_root": [0.23, -0.02, 0.0],
            "false_right_mid": [0.27, -0.16, 0.0],
            "false_right_tip": [0.23, -0.14, 0.0],
            "false_right_end": [0.12, -0.08, 0.0],
        },
    }
    if false_pair not in false_translations:
        raise ValueError(f"unknown false_pair: {false_pair}")
    translations.update(false_translations[false_pair])
    nodes = [{"name": f"{prefix}_{name}", "translation": translations[name]} for name in names]
    index_by_name = {name: index for index, name in enumerate(names)}
    for parent, child in edges:
        nodes[index_by_name[parent]].setdefault("children", []).append(index_by_name[child])
    return {"asset": {"version": "2.0"}, "nodes": nodes, "skins": [{"joints": list(range(len(nodes))), "inverseBindMatrices": 0}]}


def long_trunk_without_true_arms_payload(*, prefix: str = "joint", false_pair: str) -> dict:
    payload = long_trunk_competing_chest_payload(prefix=prefix, false_pair=false_pair)
    index_by_name = {node["name"]: index for index, node in enumerate(payload["nodes"])}
    true_chest = payload["nodes"][index_by_name[f"{prefix}_true_chest"]]
    true_chest["children"] = [index_by_name[f"{prefix}_upper_trunk"]]
    detached = {
        index_by_name[f"{prefix}_{name}"]
        for name in (
            "true_left_shoulder",
            "true_left_upper_arm",
            "true_left_lower_arm",
            "true_left_hand",
            "true_right_shoulder",
            "true_right_upper_arm",
            "true_right_lower_arm",
            "true_right_hand",
        )
    }
    payload["skins"][0]["joints"] = [index for index in payload["skins"][0]["joints"] if index not in detached]
    return payload


def renamed_payload(source: dict, *, prefix: str) -> dict:
    renamed = copy.deepcopy(source)
    for index, node in enumerate(renamed["nodes"]):
        node["name"] = f"{prefix}_{index:02d}"
    return renamed


def shifted_real_payload(source: dict, edges: tuple[tuple[str, str], ...], *, offset: int) -> dict:
    shifted = {f"bone_{number}": f"joint_{number + offset}" for number in range(len(source["skins"][0]["joints"]))}
    renamed = copy.deepcopy(source)
    for node in renamed["nodes"]:
        name = node.get("name")
        if name in shifted:
            node["name"] = shifted[name]
    index_by_name = {node["name"]: index for index, node in enumerate(renamed["nodes"])}
    for node in renamed["nodes"]:
        node.pop("children", None)
    for parent, child in edges:
        renamed["nodes"][index_by_name[shifted[parent]]].setdefault("children", []).append(index_by_name[shifted[child]])
    renamed["skins"][0]["joints"] = [index_by_name[shifted[f"bone_{number}"]] for number in range(len(source["skins"][0]["joints"]))]
    return renamed


class SemanticHumanoidResolverTests(unittest.TestCase):
    def test_extract_joint_graph_composes_rest_world_and_reports_structure(self) -> None:
        graph = extract_joint_graph(minimal_semantic_payload())

        self.assertEqual(graph.roots, ["joint_hips"])
        self.assertEqual(graph.nodes["joint_chest"].parent, "joint_spine_mid")
        self.assertEqual(graph.nodes["joint_hips"].children, ["joint_spine", "joint_left_upper_leg", "joint_right_upper_leg"])
        self.assertEqual(graph.nodes["joint_left_foot"].depth, 3)
        self.assertEqual(graph.nodes["joint_left_hand"].rest_world[0][3], -1.4)
        self.assertEqual(graph.nodes["joint_head"].rest_world[1][3], 3.45)
        self.assertIn("joint_head", graph.leaves)

    def test_minimal_semantic_humanoid_resolves_required_roles_without_name_dependency(self) -> None:
        declared = resolve_humanoid(minimal_semantic_payload(prefix="anon"))
        contract = build_contract_from_declared_data(declared, source_hash="0" * 64, output_hash="1" * 64)

        self.assertEqual(contract["required_roles"]["hips"], "anon_hips")
        self.assertEqual(contract["required_roles"]["left_hand"], "anon_left_hand")
        self.assertEqual(contract["required_roles"]["right_foot"], "anon_right_foot")
        self.assertEqual(contract["optional_roles"]["left_shoulder"], "anon_left_shoulder")
        self.assertGreaterEqual(contract["confidence"]["overall"], 0.8)

    def test_shoulderless_humanoid_resolves_arm_chain_with_optional_shoulder_warning(self) -> None:
        declared = resolve_humanoid(shoulderless_semantic_payload(prefix="anon"))
        contract = build_contract_from_declared_data(declared, source_hash="0" * 64, output_hash="1" * 64)

        self.assertEqual(contract["required_roles"]["left_upper_arm"], "anon_left_upper_arm")
        self.assertEqual(contract["required_roles"]["left_lower_arm"], "anon_left_lower_arm")
        self.assertEqual(contract["required_roles"]["left_hand"], "anon_left_hand")
        self.assertEqual(contract["required_roles"]["right_upper_arm"], "anon_right_upper_arm")
        self.assertNotIn("left_shoulder", contract["optional_roles"])
        self.assertNotIn("right_shoulder", contract["optional_roles"])
        self.assertEqual(
            [(item["code"], item["side"]) for item in declared["diagnostics"]],
            [("optional_shoulder_unavailable", "left"), ("optional_shoulder_unavailable", "right")],
        )

    def test_chest_selection_scans_past_non_arm_side_branches_to_real_arm_evidence(self) -> None:
        declared = resolve_humanoid(delayed_chest_semantic_payload(prefix="anon"))

        self.assertEqual(declared["roles"]["chest"], "anon_chest")
        self.assertEqual(declared["roles"]["left_upper_arm"], "anon_left_upper_arm")
        self.assertEqual(declared["roles"]["left_lower_arm"], "anon_left_lower_arm")
        self.assertEqual(declared["roles"]["left_hand"], "anon_left_hand")
        self.assertEqual(declared["roles"]["right_upper_arm"], "anon_right_upper_arm")
        self.assertEqual(declared["roles"]["right_lower_arm"], "anon_right_lower_arm")
        self.assertEqual(declared["roles"]["right_hand"], "anon_right_hand")

    def test_near_center_arm_roots_resolve_from_mirrored_branch_centers(self) -> None:
        payload = renamed_payload(near_center_arm_branch_payload(prefix="source"), prefix="anon")
        graph = extract_joint_graph(payload)
        root_separation = abs(graph.nodes["anon_06"].rest_world[0][3] - graph.nodes["anon_10"].rest_world[0][3])
        self.assertLess(root_separation, 0.05)

        declared = resolve_humanoid(payload)

        self.assertEqual(declared["roles"]["left_upper_arm"], "anon_07")
        self.assertEqual(declared["roles"]["left_lower_arm"], "anon_08")
        self.assertEqual(declared["roles"]["left_hand"], "anon_09")
        self.assertEqual(declared["roles"]["right_upper_arm"], "anon_11")
        self.assertEqual(declared["roles"]["right_lower_arm"], "anon_12")
        self.assertEqual(declared["roles"]["right_hand"], "anon_13")

    def test_near_center_arm_roots_with_same_side_branch_centers_fail_closed(self) -> None:
        payload = near_center_arm_branch_payload(prefix="anon")
        replacements = {
            "anon_left_upper_arm": [0.18, -0.08, 0.0],
            "anon_left_lower_arm": [0.17, -0.08, 0.0],
            "anon_right_upper_arm": [0.28, -0.08, 0.0],
            "anon_right_lower_arm": [0.17, -0.08, 0.0],
        }
        for node in payload["nodes"]:
            if node["name"] in replacements:
                node["translation"] = replacements[node["name"]]

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_chest_missing") as raised:
            resolve_humanoid(payload)

        self.assertIn("symmetric arm evidence", raised.exception.diagnostics[0]["message"])

    def test_near_center_arm_roots_without_branch_center_separation_fail_closed(self) -> None:
        payload = near_center_arm_branch_payload(prefix="anon")
        replacements = {
            "anon_left_upper_arm": [-0.01, -0.08, 0.0],
            "anon_left_lower_arm": [0.01, -0.08, 0.0],
            "anon_right_upper_arm": [0.01, -0.08, 0.0],
            "anon_right_lower_arm": [-0.01, -0.08, 0.0],
        }
        for node in payload["nodes"]:
            if node["name"] in replacements:
                node["translation"] = replacements[node["name"]]

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_chest_missing") as raised:
            resolve_humanoid(payload)

        self.assertIn("symmetric arm evidence", raised.exception.diagnostics[0]["message"])

    def test_long_trunk_chest_selection_prefers_lower_balanced_arm_evidence_over_head_side_competitor(self) -> None:
        declared = resolve_humanoid(long_trunk_competing_chest_payload(prefix="anon"))

        self.assertEqual(declared["roles"]["chest"], "anon_true_chest")
        self.assertEqual(declared["roles"]["left_upper_arm"], "anon_true_left_upper_arm")
        self.assertEqual(declared["roles"]["left_lower_arm"], "anon_true_left_lower_arm")
        self.assertEqual(declared["roles"]["left_hand"], "anon_true_left_hand")
        self.assertEqual(declared["roles"]["right_upper_arm"], "anon_true_right_upper_arm")
        self.assertEqual(declared["roles"]["right_lower_arm"], "anon_true_right_lower_arm")
        self.assertEqual(declared["roles"]["right_hand"], "anon_true_right_hand")

    def test_long_trunk_chest_selection_is_independent_of_exact_joint_names(self) -> None:
        renamed = renamed_payload(long_trunk_competing_chest_payload(prefix="source"), prefix="randomized")
        declared = resolve_humanoid(renamed)

        self.assertEqual(declared["roles"]["chest"], "randomized_03")
        self.assertEqual(declared["roles"]["left_upper_arm"], "randomized_08")
        self.assertEqual(declared["roles"]["right_upper_arm"], "randomized_12")

    def test_same_side_branch_pairs_do_not_count_as_chest_arm_evidence(self) -> None:
        payload = long_trunk_without_true_arms_payload(prefix="anon", false_pair="same_side")

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_chest_missing") as raised:
            resolve_humanoid(payload)

        self.assertIn("symmetric arm evidence", raised.exception.diagnostics[0]["message"])

    def test_centered_branch_pairs_do_not_count_as_chest_arm_evidence(self) -> None:
        payload = long_trunk_without_true_arms_payload(prefix="anon", false_pair="centered")

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_chest_missing") as raised:
            resolve_humanoid(payload)

        self.assertIn("symmetric arm evidence", raised.exception.diagnostics[0]["message"])

    def test_genuine_long_trunk_chest_ambiguity_fails_closed(self) -> None:
        payload = long_trunk_competing_chest_payload(prefix="anon", false_pair="ambiguous")

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_chest_") as raised:
            resolve_humanoid(payload)

        self.assertIn(raised.exception.diagnostics[0]["code"], {"semantic_chest_missing", "semantic_chest_ambiguous"})

    def test_leg_selection_ignores_extra_centered_hip_branches_when_clear_lateral_pair_exists(self) -> None:
        payload = minimal_semantic_payload(prefix="anon")
        hips_index = next(index for index, node in enumerate(payload["nodes"]) if node["name"] == "anon_hips")
        for name, translation in (
            ("anon_center_tail_a", [0.0, -0.25, 0.02]),
            ("anon_center_tail_a_mid", [0.0, -0.25, 0.02]),
            ("anon_center_tail_a_tip", [0.0, -0.2, 0.02]),
            ("anon_center_tail_b", [0.0, -0.25, -0.02]),
            ("anon_center_tail_b_mid", [0.0, -0.25, -0.02]),
            ("anon_center_tail_b_tip", [0.0, -0.2, -0.02]),
        ):
            payload["nodes"].append({"name": name, "translation": translation})
        index_by_name = {node["name"]: index for index, node in enumerate(payload["nodes"])}
        payload["nodes"][hips_index].setdefault("children", []).extend([index_by_name["anon_center_tail_a"], index_by_name["anon_center_tail_b"]])
        payload["nodes"][index_by_name["anon_center_tail_a"]]["children"] = [index_by_name["anon_center_tail_a_mid"]]
        payload["nodes"][index_by_name["anon_center_tail_a_mid"]]["children"] = [index_by_name["anon_center_tail_a_tip"]]
        payload["nodes"][index_by_name["anon_center_tail_b"]]["children"] = [index_by_name["anon_center_tail_b_mid"]]
        payload["nodes"][index_by_name["anon_center_tail_b_mid"]]["children"] = [index_by_name["anon_center_tail_b_tip"]]
        payload["skins"][0]["joints"] = list(range(len(payload["nodes"])))

        declared = resolve_humanoid(payload)

        self.assertEqual(declared["roles"]["left_upper_leg"], "anon_left_upper_leg")
        self.assertEqual(declared["roles"]["left_foot"], "anon_left_foot")
        self.assertEqual(declared["roles"]["right_upper_leg"], "anon_right_upper_leg")
        self.assertEqual(declared["roles"]["right_foot"], "anon_right_foot")

    def test_leg_selection_uses_descendant_downward_evidence_when_roots_are_near_center(self) -> None:
        payload = minimal_semantic_payload(prefix="anon")
        replacements = {
            "anon_left_upper_leg": [-0.01, -0.45, 0.0],
            "anon_left_lower_leg": [-0.22, -0.7, 0.0],
            "anon_left_foot": [-0.08, -0.25, 0.22],
            "anon_right_upper_leg": [0.01, -0.45, 0.0],
            "anon_right_lower_leg": [0.22, -0.7, 0.0],
            "anon_right_foot": [0.08, -0.25, 0.22],
        }
        for node in payload["nodes"]:
            if node["name"] in replacements:
                node["translation"] = replacements[node["name"]]
        hips_index = next(index for index, node in enumerate(payload["nodes"]) if node["name"] == "anon_hips")
        for name, translation in (
            ("anon_helper_left", [-0.02, -0.08, 0.0]),
            ("anon_helper_left_tip", [-0.02, -0.04, 0.0]),
            ("anon_helper_right", [0.02, -0.08, 0.0]),
            ("anon_helper_right_tip", [0.02, -0.04, 0.0]),
            ("anon_center_tail", [0.0, -0.2, 0.0]),
            ("anon_center_tail_tip", [0.0, -0.2, 0.0]),
        ):
            payload["nodes"].append({"name": name, "translation": translation})
        index_by_name = {node["name"]: index for index, node in enumerate(payload["nodes"])}
        payload["nodes"][hips_index].setdefault("children", []).extend(
            [index_by_name["anon_helper_left"], index_by_name["anon_helper_right"], index_by_name["anon_center_tail"]]
        )
        payload["nodes"][index_by_name["anon_helper_left"]]["children"] = [index_by_name["anon_helper_left_tip"]]
        payload["nodes"][index_by_name["anon_helper_right"]]["children"] = [index_by_name["anon_helper_right_tip"]]
        payload["nodes"][index_by_name["anon_center_tail"]]["children"] = [index_by_name["anon_center_tail_tip"]]
        payload["skins"][0]["joints"] = list(range(len(payload["nodes"])))

        declared = resolve_humanoid(payload)

        self.assertEqual(declared["roles"]["left_upper_leg"], "anon_left_upper_leg")
        self.assertEqual(declared["roles"]["left_lower_leg"], "anon_left_lower_leg")
        self.assertEqual(declared["roles"]["left_foot"], "anon_left_foot")
        self.assertEqual(declared["roles"]["right_upper_leg"], "anon_right_upper_leg")
        self.assertEqual(declared["roles"]["right_lower_leg"], "anon_right_lower_leg")
        self.assertEqual(declared["roles"]["right_foot"], "anon_right_foot")

    def test_leg_selection_scans_lower_trunk_when_hips_root_only_contains_pelvis_child(self) -> None:
        payload = minimal_semantic_payload(prefix="anon")
        index_by_name = {node["name"]: index for index, node in enumerate(payload["nodes"])}
        hips = payload["nodes"][index_by_name["anon_hips"]]
        spine = payload["nodes"][index_by_name["anon_spine"]]
        hips["children"] = [index_by_name["anon_spine"]]
        spine.setdefault("children", []).extend([index_by_name["anon_left_upper_leg"], index_by_name["anon_right_upper_leg"]])

        declared = resolve_humanoid(payload)

        self.assertEqual(declared["roles"]["hips"], "anon_hips")
        self.assertEqual(declared["roles"]["left_upper_leg"], "anon_left_upper_leg")
        self.assertEqual(declared["roles"]["left_foot"], "anon_left_foot")
        self.assertEqual(declared["roles"]["right_upper_leg"], "anon_right_upper_leg")
        self.assertEqual(declared["roles"]["right_foot"], "anon_right_foot")

    def test_leg_selection_fails_closed_when_multiple_leg_pairs_have_no_clear_margin(self) -> None:
        payload = minimal_semantic_payload(prefix="anon")
        hips_index = next(index for index, node in enumerate(payload["nodes"]) if node["name"] == "anon_hips")
        for name, translation in (
            ("anon_left_spare_upper_leg", [-0.19, -0.8, 0.0]),
            ("anon_left_spare_lower_leg", [0.0, -0.8, 0.0]),
            ("anon_left_spare_foot", [0.0, -0.25, 0.35]),
            ("anon_right_spare_upper_leg", [0.19, -0.8, 0.0]),
            ("anon_right_spare_lower_leg", [0.0, -0.8, 0.0]),
            ("anon_right_spare_foot", [0.0, -0.25, 0.35]),
        ):
            payload["nodes"].append({"name": name, "translation": translation})
        index_by_name = {node["name"]: index for index, node in enumerate(payload["nodes"])}
        payload["nodes"][hips_index].setdefault("children", []).extend(
            [index_by_name["anon_left_spare_upper_leg"], index_by_name["anon_right_spare_upper_leg"]]
        )
        payload["nodes"][index_by_name["anon_left_spare_upper_leg"]]["children"] = [index_by_name["anon_left_spare_lower_leg"]]
        payload["nodes"][index_by_name["anon_left_spare_lower_leg"]]["children"] = [index_by_name["anon_left_spare_foot"]]
        payload["nodes"][index_by_name["anon_right_spare_upper_leg"]]["children"] = [index_by_name["anon_right_spare_lower_leg"]]
        payload["nodes"][index_by_name["anon_right_spare_lower_leg"]]["children"] = [index_by_name["anon_right_spare_foot"]]
        payload["skins"][0]["joints"] = list(range(len(payload["nodes"])))

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_leg_symmetry_ambiguous") as raised:
            resolve_humanoid(payload)

        self.assertIn("multiple plausible leg pairs", raised.exception.diagnostics[0]["message"])

    def test_offset_humanoid_symmetry_uses_relative_center_not_global_x_origin(self) -> None:
        payload = shoulderless_semantic_payload(prefix="anon")
        payload["nodes"][0]["translation"] = [10.0, 1.0, 0.0]

        declared = resolve_humanoid(payload)

        self.assertEqual(declared["roles"]["left_upper_arm"], "anon_left_upper_arm")
        self.assertEqual(declared["roles"]["right_upper_arm"], "anon_right_upper_arm")
        self.assertEqual(declared["roles"]["left_upper_leg"], "anon_left_upper_leg")
        self.assertEqual(declared["roles"]["right_upper_leg"], "anon_right_upper_leg")

    def test_lateral_symmetry_can_be_inferred_from_z_axis_when_global_x_is_not_split(self) -> None:
        payload = shoulderless_semantic_payload(prefix="anon")
        z_lateral = {
            "anon_left_upper_arm": [0.0, 0.0, -0.45],
            "anon_left_lower_arm": [0.0, -0.05, -0.45],
            "anon_left_hand": [0.0, -0.05, -0.25],
            "anon_right_upper_arm": [0.0, 0.0, 0.45],
            "anon_right_lower_arm": [0.0, -0.05, 0.45],
            "anon_right_hand": [0.0, -0.05, 0.25],
            "anon_left_upper_leg": [0.0, -0.8, -0.18],
            "anon_left_lower_leg": [0.0, -0.8, 0.0],
            "anon_left_foot": [0.35, -0.25, 0.0],
            "anon_right_upper_leg": [0.0, -0.8, 0.18],
            "anon_right_lower_leg": [0.0, -0.8, 0.0],
            "anon_right_foot": [0.35, -0.25, 0.0],
        }
        for node in payload["nodes"]:
            if node["name"] in z_lateral:
                node["translation"] = z_lateral[node["name"]]

        declared = resolve_humanoid(payload)

        self.assertEqual(declared["roles"]["left_hand"], "anon_left_hand")
        self.assertEqual(declared["roles"]["right_hand"], "anon_right_hand")
        self.assertEqual(declared["roles"]["left_foot"], "anon_left_foot")
        self.assertEqual(declared["roles"]["right_foot"], "anon_right_foot")

    def test_too_short_shoulderless_arm_chain_still_fails_closed(self) -> None:
        payload = shoulderless_semantic_payload(prefix="anon")
        left_hand = next(index for index, node in enumerate(payload["nodes"]) if node["name"] == "anon_left_hand")
        payload["skins"][0]["joints"].remove(left_hand)

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_arm_chain_missing"):
            resolve_humanoid(payload)

    def test_short_trunk_output_reports_graph_metrics_in_spine_diagnostic(self) -> None:
        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_spine_missing") as raised:
            resolve_humanoid(short_trunk_output_payload(prefix="out"))

        diagnostic = raised.exception.diagnostics[0]
        self.assertEqual(diagnostic["code"], "semantic_spine_missing")
        self.assertEqual(diagnostic["joint_count"], 8)
        self.assertEqual(diagnostic["root_count"], 1)
        self.assertEqual(diagnostic["highest_path_length"], 2)
        self.assertEqual(diagnostic["minimum_trunk_length"], 5)
        self.assertEqual(diagnostic["highest_path"], ["out_hips", "out_spine"])

    def test_real_unirig_52_variant_matches_profile_role_oracle_through_semantics(self) -> None:
        declared = resolve_humanoid(real_unirig_52_payload())

        self.assertEqual({role: declared["roles"][role] for role in REAL_UNIRIG_52_ROLE_MAP}, REAL_UNIRIG_52_ROLE_MAP)
        self.assertEqual(declared["provenance"]["method"], "semantic-graph-rest-symmetry")

    def test_real_unirig_40_variant_matches_profile_role_oracle_through_semantics(self) -> None:
        declared = resolve_humanoid(real_unirig_40_payload())

        self.assertEqual({role: declared["roles"][role] for role in REAL_UNIRIG_40_ROLE_MAP}, REAL_UNIRIG_40_ROLE_MAP)
        self.assertEqual(declared["provenance"]["source"], "semantic_humanoid_resolver")

    def test_shifted_numbering_equivalent_topology_does_not_depend_on_exact_bone_names(self) -> None:
        declared = resolve_humanoid(shifted_real_payload(real_unirig_40_payload(), REAL_UNIRIG_40_EDGES, offset=100))

        self.assertEqual(declared["roles"]["hips"], "joint_100")
        self.assertEqual(declared["roles"]["left_hand"], "joint_109")
        self.assertEqual(declared["roles"]["right_foot"], "joint_139")

    def test_ambiguous_lateral_symmetry_fails_closed_with_diagnostics(self) -> None:
        payload = minimal_semantic_payload()
        for node in payload["nodes"]:
            if node["name"] in {"joint_left_upper_leg", "joint_right_upper_leg", "joint_left_shoulder", "joint_right_shoulder"}:
                node["translation"][0] = 0.0

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_.*symmetry_ambiguous") as raised:
            resolve_humanoid(payload)

        self.assertIn(
            raised.exception.diagnostics[0]["code"],
            {"semantic_symmetry_ambiguous", "semantic_leg_symmetry_ambiguous"},
        )

    def test_disconnected_or_malformed_skin_fails_closed_before_contract_emission(self) -> None:
        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_skin_missing"):
            resolve_humanoid({"asset": {"version": "2.0"}, "nodes": [{"name": "only"}], "skins": []})

    def test_optional_semantic_body_report_drives_roles_without_rewalking_topology_heuristics(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="unirig-resolver-report-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "avatar.glb")
            container = read_glb_container(glb)
            semantic_report = build_semantic_body_report(container, _declared_roles())

            declared = resolve_humanoid(container.json, semantic_report=semantic_report)

        self.assertEqual(declared["roles"], semantic_report.core_roles)
        self.assertEqual(declared["provenance"]["method"], "semantic-body-report")
        self.assertEqual(declared["confidence"]["overall"], semantic_report.contract_core_confidence)

    def test_optional_nonpublishable_semantic_body_report_fails_before_contract_roles(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="unirig-resolver-report-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "avatar.glb", sleeve=True)
            container = read_glb_container(glb)
            semantic_report = build_semantic_body_report(container, _declared_roles())

            with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_body_graph_not_publishable") as raised:
                resolve_humanoid(container.json, semantic_report=semantic_report)

        self.assertIn("semantic_body_graph", raised.exception.diagnostics[0])


if __name__ == "__main__":
    unittest.main()
