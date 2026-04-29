from __future__ import annotations

import json
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

from fixtures.unirig_real_topology import real_unirig_40_payload, real_unirig_52_payload
from unirig_ext.humanoid_contract import build_contract_from_declared_data
from unirig_ext.topology_profiles import TopologyProfileError, build_declared_data_from_known_profile


class TopologyProfileTests(unittest.TestCase):
    def test_exact_known_profile_generates_stable_declared_data(self) -> None:
        payload = json.loads((ROOT / "tests" / "fixtures" / "unirig_topology_profile_minimal.json").read_text(encoding="utf-8"))

        first = build_declared_data_from_known_profile(payload)
        second = build_declared_data_from_known_profile(payload)

        self.assertEqual(first, second)
        self.assertEqual(first["roles"]["hips"], "hips")
        self.assertEqual(first["roles"]["right_foot"], "right_foot")
        self.assertEqual(first["provenance"]["source"], "known-unirig-topology-profile")
        self.assertEqual(first["confidence"]["roles"]["hips"], 1.0)

    def test_unknown_topology_is_rejected_fail_closed(self) -> None:
        payload = {"asset": {"version": "2.0"}, "nodes": [{"name": "mystery"}], "skins": []}

        with self.assertRaisesRegex(TopologyProfileError, "Unknown or ambiguous UniRig topology"):
            build_declared_data_from_known_profile(payload)

    def test_real_unirig_52_bone_profile_generates_declared_humanoid_data_with_real_indices(self) -> None:
        payload = real_unirig_52_payload()

        declared = build_declared_data_from_known_profile(payload)

        self.assertEqual(declared["provenance"]["profile_id"], "unirig-anonymous-bone-52")
        self.assertEqual(declared["roles"]["hips"], "bone_0")
        self.assertEqual(declared["roles"]["chest"], "bone_3")
        self.assertEqual(declared["roles"]["right_foot"], "bone_51")
        nodes = {node["id"]: node for node in declared["nodes"]}
        self.assertEqual(nodes["bone_0"]["index"], 2)
        self.assertEqual(nodes["bone_51"]["index"], 53)
        self.assertEqual(nodes["bone_1"]["parent"], "bone_0")
        self.assertEqual(nodes["bone_0"]["rest_local"], nodes["bone_0"]["rest_world"])
        self.assertEqual(nodes["bone_1"]["rest_local"][1][3], 0.7)
        self.assertEqual(nodes["bone_1"]["rest_world"][1][3], 1.7)
        self.assertNotEqual(nodes["bone_1"]["rest_local"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def test_real_unirig_52_bone_profile_declares_proximal_shoulders_before_arm_chain(self) -> None:
        declared = build_declared_data_from_known_profile(real_unirig_52_payload())

        self.assertEqual(declared["roles"]["left_shoulder"], "bone_6")
        self.assertEqual(declared["roles"]["left_upper_arm"], "bone_7")
        self.assertEqual(declared["roles"]["left_lower_arm"], "bone_8")
        self.assertEqual(declared["roles"]["left_hand"], "bone_9")
        self.assertEqual(declared["roles"]["right_shoulder"], "bone_25")
        self.assertEqual(declared["roles"]["right_upper_arm"], "bone_26")
        self.assertEqual(declared["roles"]["right_lower_arm"], "bone_27")
        self.assertEqual(declared["roles"]["right_hand"], "bone_28")
        self.assertEqual(declared["confidence"]["roles"]["left_shoulder"], 1.0)
        self.assertEqual(declared["confidence"]["roles"]["right_shoulder"], 1.0)

    def test_real_unirig_52_bone_declared_data_validates_as_humanoid_contract(self) -> None:
        declared = build_declared_data_from_known_profile(real_unirig_52_payload())

        contract = build_contract_from_declared_data(declared, source_hash="0" * 64, output_hash="1" * 64)

        self.assertEqual(contract["required_roles"]["left_hand"], "bone_9")
        self.assertEqual(contract["optional_roles"]["left_shoulder"], "bone_6")
        self.assertEqual(contract["optional_roles"]["right_shoulder"], "bone_25")
        self.assertEqual(contract["chains"]["spine"], ["bone_0", "bone_1", "bone_3", "bone_4", "bone_5"])
        self.assertEqual(contract["chains"]["left_arm_proximal"], ["bone_6", "bone_7", "bone_8", "bone_9"])
        self.assertEqual(contract["chains"]["right_arm_proximal"], ["bone_25", "bone_26", "bone_27", "bone_28"])
        self.assertEqual(contract["nodes"]["bone_1"]["transforms"]["rest_world"][1][3], 1.7)

    def test_real_unirig_52_bone_profile_rejects_ambiguous_variants(self) -> None:
        payload = real_unirig_52_payload()
        payload["skins"][0]["joints"] = list(reversed(payload["skins"][0]["joints"]))

        with self.assertRaisesRegex(TopologyProfileError, "Unknown or ambiguous UniRig topology"):
            build_declared_data_from_known_profile(payload)

    def test_real_unirig_40_bone_profile_emits_shifted_shoulder_and_arm_roles(self) -> None:
        declared = build_declared_data_from_known_profile(real_unirig_40_payload())

        self.assertEqual(declared["provenance"]["profile_id"], "unirig-anonymous-bone-40")
        self.assertEqual(declared["roles"]["hips"], "bone_0")
        self.assertEqual(declared["roles"]["left_shoulder"], "bone_6")
        self.assertEqual(declared["roles"]["left_upper_arm"], "bone_7")
        self.assertEqual(declared["roles"]["left_lower_arm"], "bone_8")
        self.assertEqual(declared["roles"]["left_hand"], "bone_9")
        self.assertEqual(declared["roles"]["right_shoulder"], "bone_19")
        self.assertEqual(declared["roles"]["right_upper_arm"], "bone_20")
        self.assertEqual(declared["roles"]["right_lower_arm"], "bone_21")
        self.assertEqual(declared["roles"]["right_hand"], "bone_22")
        self.assertEqual(declared["roles"]["left_upper_leg"], "bone_32")
        self.assertEqual(declared["roles"]["left_lower_leg"], "bone_33")
        self.assertEqual(declared["roles"]["left_foot"], "bone_35")
        self.assertEqual(declared["roles"]["right_upper_leg"], "bone_36")
        self.assertEqual(declared["roles"]["right_lower_leg"], "bone_37")
        self.assertEqual(declared["roles"]["right_foot"], "bone_39")

    def test_real_unirig_40_bone_declared_data_validates_as_humanoid_contract(self) -> None:
        declared = build_declared_data_from_known_profile(real_unirig_40_payload())

        contract = build_contract_from_declared_data(declared, source_hash="0" * 64, output_hash="1" * 64)

        self.assertEqual(contract["required_roles"]["left_hand"], "bone_9")
        self.assertEqual(contract["optional_roles"]["left_shoulder"], "bone_6")
        self.assertEqual(contract["optional_roles"]["right_shoulder"], "bone_19")
        self.assertEqual(contract["chains"]["spine"], ["bone_0", "bone_1", "bone_3", "bone_4", "bone_5"])
        self.assertEqual(contract["chains"]["left_arm_proximal"], ["bone_6", "bone_7", "bone_8", "bone_9"])
        self.assertEqual(contract["chains"]["right_arm_proximal"], ["bone_19", "bone_20", "bone_21", "bone_22"])
        self.assertEqual(contract["chains"]["left_leg"], ["bone_32", "bone_33", "bone_35"])
        self.assertEqual(contract["chains"]["right_leg"], ["bone_36", "bone_37", "bone_39"])

    def test_real_unirig_40_bone_profile_rejects_parent_child_variants(self) -> None:
        payload = real_unirig_40_payload()
        nodes_by_name = {node["name"]: node for node in payload["nodes"]}
        bone_3_children = nodes_by_name["bone_3"]["children"]
        bone_6_index = payload["skins"][0]["joints"][6]
        bone_3_children.remove(bone_6_index)
        nodes_by_name["bone_0"]["children"].append(bone_6_index)

        with self.assertRaisesRegex(TopologyProfileError, "Unknown or ambiguous UniRig topology"):
            build_declared_data_from_known_profile(payload)


if __name__ == "__main__":
    unittest.main()
