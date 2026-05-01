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
    def _report(self, name: str = "avatar.glb", declared: dict | None = None, **options):
        temp_dir = tempfile.TemporaryDirectory(prefix="unirig-sbg-")
        self.addCleanup(temp_dir.cleanup)
        glb = write_embedded_skin_glb(Path(temp_dir.name) / name, **options)
        return build_semantic_body_report(read_glb_container(glb), declared or _declared_roles())

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

    def test_synthetic_basic_humanoid_with_shoulders_and_chest_high_region_is_publishable(self) -> None:
        report = self._report(shoulder_connectors=True, hair_contamination=True, declared=_declared_roles(include_shoulders=True))

        self.assertTrue(report.publishable, report.diagnostic)
        self.assertEqual(report.predicates["has_high_region_contamination"], False)
        self.assertEqual(report.predicates["has_high_region_warning"], True)
        self.assertGreaterEqual(report.contract_core_confidence, 0.9)
        self.assertEqual(report.nodes["left_shoulder"].is_contract_candidate, True)
        self.assertEqual(report.nodes["right_shoulder"].is_contract_candidate, True)
        self.assertNotIn("left_shoulder", report.core_roles)
        self.assertNotIn("right_shoulder", report.core_roles)
        self.assertNotIn("anatomical_role_not_separable", {reason["code"] for reason in report.diagnostic["reasons"]})

    def test_isolated_chest_high_region_is_serialized_as_warning_not_blocking_reason(self) -> None:
        report = self._report(hair_contamination=True)

        warning_codes = {warning["code"] for warning in report.diagnostic["warnings"]}
        reason_codes = {reason["code"] for reason in report.diagnostic["reasons"]}
        self.assertTrue(report.publishable, report.diagnostic)
        self.assertEqual(report.predicates["has_high_region_warning"], True)
        self.assertIn("high_region_weighted_by_torso_or_arm", warning_codes)
        self.assertNotIn("high_region_weighted_by_torso_or_arm", reason_codes)

    def test_fourth_real_failure_chest_and_moderate_upper_arm_high_region_is_warning(self) -> None:
        report = self._report(hair_contamination=True, high_region_role="right_upper_arm", high_region_y=1.584)

        reason_codes = {reason["code"] for reason in report.diagnostic["reasons"]}
        self.assertTrue(report.publishable, report.diagnostic)
        self.assertEqual(report.predicates["has_high_region_contamination"], False)
        self.assertEqual(report.predicates["has_high_region_warning"], True)
        self.assertGreaterEqual(report.contract_core_confidence, 0.9)
        self.assertNotIn("high_region_weighted_by_torso_or_arm", reason_codes)

    def test_upper_arm_high_region_warning_serializes_thresholds_and_cutoff_context(self) -> None:
        report = self._report(hair_contamination=True, high_region_role="right_upper_arm", high_region_y=1.584)

        warnings = [warning for warning in report.diagnostic["warnings"] if warning["code"] == "high_region_weighted_by_torso_or_arm"]
        self.assertEqual(len(warnings), 1)
        warning = warnings[0]
        joints_by_role = {joint["role"]: joint for joint in warning["joints"]}
        self.assertEqual(warning["severity"], "warning")
        self.assertEqual(warning["threshold_normalized"], 0.8)
        self.assertEqual(warning["upper_arm_warning_cutoff_normalized"], 0.85)
        self.assertEqual(set(joints_by_role), {"chest", "right_upper_arm"})
        self.assertEqual(joints_by_role["right_upper_arm"]["joint"], "right_upper_arm")
        self.assertAlmostEqual(joints_by_role["right_upper_arm"]["normalized_max_y"], 0.8085, places=4)

    def test_arm_high_region_contamination_remains_blocking(self) -> None:
        report = self._report(arm_high_region=True)

        self.assertFalse(report.publishable)
        self.assertEqual(report.predicates["has_high_region_contamination"], True)
        self.assertIn("high_region_weighted_by_torso_or_arm", {reason["code"] for reason in report.diagnostic["reasons"]})

    def test_upper_arm_at_or_above_warning_cutoff_remains_blocking_and_confidence_capped(self) -> None:
        report = self._report(high_region_role="left_upper_arm", high_region_y=1.67)

        high_region = [reason for reason in report.diagnostic["reasons"] if reason["code"] == "high_region_weighted_by_torso_or_arm"]
        self.assertFalse(report.publishable)
        self.assertEqual(report.predicates["has_high_region_contamination"], True)
        self.assertLess(report.contract_core_confidence, 0.85)
        self.assertEqual(high_region[0]["joints"][0]["role"], "left_upper_arm")
        self.assertGreaterEqual(high_region[0]["joints"][0]["normalized_max_y"], 0.85)

    def test_lower_arm_and_hand_high_region_remain_blocking(self) -> None:
        for role in ("left_lower_arm", "right_lower_arm", "left_hand", "right_hand"):
            with self.subTest(role=role):
                report = self._report(high_region_role=role, high_region_y=1.51)
                high_region = [reason for reason in report.diagnostic["reasons"] if reason["code"] == "high_region_weighted_by_torso_or_arm"]

                self.assertFalse(report.publishable)
                self.assertEqual(report.predicates["has_high_region_contamination"], True)
                self.assertEqual(high_region[0]["joints"][0]["role"], role)

    def test_passive_branch_prevents_moderate_upper_arm_high_region_tolerance(self) -> None:
        report = self._report(sleeve=True, hair_contamination=True, high_region_role="right_upper_arm", high_region_y=1.584)

        reason_codes = {reason["code"] for reason in report.diagnostic["reasons"]}
        self.assertFalse(report.publishable)
        self.assertEqual(report.predicates["has_passive_noncontract_subtrees"], True)
        self.assertIn("sleeve_branch_under_arm", reason_codes)
        self.assertIn("high_region_weighted_by_torso_or_arm", reason_codes)

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
