from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_humanoid_quality_gate import _declared_roles, write_embedded_skin_glb
from unirig_ext.gltf_skin_analysis import read_glb_container
from unirig_ext.semantic_body_graph import build_semantic_body_report


class SemanticBodyGraphTests(unittest.TestCase):
    def _report(self, name: str = "avatar.glb", **options):
        temp_dir = tempfile.TemporaryDirectory(prefix="unirig-sbg-")
        self.addCleanup(temp_dir.cleanup)
        glb = write_embedded_skin_glb(Path(temp_dir.name) / name, **options)
        return build_semantic_body_report(read_glb_container(glb), _declared_roles())

    def test_clean_simple_humanoid_classifies_core_anatomy_and_publishable_roles(self) -> None:
        report = self._report()

        self.assertTrue(report.publishable, report.diagnostic)
        self.assertEqual(report.predicates["has_clear_spine"], True)
        self.assertEqual(report.predicates["has_left_right_arm_pair"], True)
        self.assertEqual(report.predicates["has_leg_pair"], True)
        self.assertGreaterEqual(report.contract_core_confidence, 0.9)
        self.assertEqual(report.nodes["hips"].classes, ("root", "body_core"))
        self.assertIn("spine", report.nodes["spine"].classes)
        self.assertIn("hand", report.nodes["left_hand"].classes)
        self.assertIn("foot", report.nodes["right_foot"].classes)
        self.assertEqual(report.core_roles["left_hand"], "left_hand")
        self.assertNotIn("unused", set(report.core_roles.values()))

    def test_separable_passive_subtrees_are_noncontract_evidence(self) -> None:
        report = self._report(sleeve=True, hand_leaf=True)

        self.assertFalse(report.publishable)
        self.assertEqual(report.predicates["has_passive_noncontract_subtrees"], True)
        self.assertIn("passive", report.nodes["left_sleeve"].classes)
        self.assertIn("clothing", report.nodes["left_sleeve"].classes)
        self.assertIn("accessory", report.nodes["left_watch_leaf"].classes)
        self.assertFalse(report.nodes["left_sleeve"].is_contract_candidate)
        self.assertNotIn("left_sleeve", set(report.core_roles.values()))
        self.assertIn("sleeve_branch_under_arm", {reason["code"] for reason in report.diagnostic["reasons"]})

    def test_high_region_contamination_is_diagnostic_not_predicate_only_decision(self) -> None:
        report = self._report(hair_contamination=True)

        self.assertFalse(report.publishable)
        self.assertEqual(report.predicates["has_high_region_contamination"], True)
        self.assertLess(report.contract_core_confidence, 0.9)
        self.assertIn("high_region_weighted_by_torso_or_arm", {reason["code"] for reason in report.diagnostic["reasons"]})

    def test_unknown_heavy_required_roles_fail_closed_with_actionable_diagnostics(self) -> None:
        declared = _declared_roles()
        declared["roles"] = dict(declared["roles"])
        declared["roles"].pop("left_hand")

        temp_dir = tempfile.TemporaryDirectory(prefix="unirig-sbg-")
        self.addCleanup(temp_dir.cleanup)
        glb = write_embedded_skin_glb(Path(temp_dir.name) / "unknown.glb")
        report = build_semantic_body_report(read_glb_container(glb), declared)

        self.assertFalse(report.publishable)
        self.assertEqual(report.predicates["unknown_near_required_roles"], True)
        self.assertIn("required_role_semantic_evidence_missing", {reason["code"] for reason in report.diagnostic["reasons"]})
        self.assertIn("left_hand", report.diagnostic["unknown_required_roles"])

    def test_serializable_diagnostic_contains_predicates_and_weight_evidence(self) -> None:
        report = self._report(mixed_classes=True)

        diagnostic = report.as_diagnostic()
        self.assertEqual(diagnostic["code"], "semantic_body_graph")
        self.assertEqual(diagnostic["predicates"]["has_passive_noncontract_subtrees"], True)
        self.assertIn("left_hair_strand", diagnostic["nodes"])
        self.assertEqual(diagnostic["nodes"]["left_hair_strand"]["classes"], ["hair", "passive"])
        self.assertIn("weighted_joints", diagnostic)


if __name__ == "__main__":
    unittest.main()
