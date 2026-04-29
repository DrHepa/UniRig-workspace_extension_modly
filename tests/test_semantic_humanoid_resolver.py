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
from unirig_ext.humanoid_contract import build_contract_from_declared_data
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

        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_symmetry_ambiguous") as raised:
            resolve_humanoid(payload)

        self.assertIn("semantic_symmetry_ambiguous", raised.exception.diagnostics[0]["code"])

    def test_disconnected_or_malformed_skin_fails_closed_before_contract_emission(self) -> None:
        with self.assertRaisesRegex(SemanticHumanoidResolutionError, "semantic_skin_missing"):
            resolve_humanoid({"asset": {"version": "2.0"}, "nodes": [{"name": "only"}], "skins": []})


if __name__ == "__main__":
    unittest.main()
