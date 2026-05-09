from __future__ import annotations

import json
import shutil
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

from test_metadata import MetadataTests, complete_humanoid_source
from test_humanoid_source import write_glb_json
from test_humanoid_quality_gate import _declared_roles, _write_glb, write_embedded_skin_glb
from fixtures.unirig_real_topology import real_unirig_52_payload
from unirig_ext.metadata import build_sidecar
from unirig_ext.humanoid_mapping_candidates import SEMANTIC_CANDIDATES_SCHEMA


class MetadataSidecarModeTests(unittest.TestCase):
    def setUp(self) -> None:
        base = MetadataTests(methodName="test_sidecar_payload_is_deterministic")
        base.setUp()
        self._base = base
        self.temp_dir = base.temp_dir
        self.input_mesh = base.input_mesh
        self.output_mesh = base.output_mesh
        self.context = base.context

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_legacy_mode_suppresses_humanoid_fields_even_with_companion(self) -> None:
        self.output_mesh.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(complete_humanoid_source()), encoding="utf-8")

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="legacy")

        self.assertNotIn("metadata_mode", payload)
        self.assertNotIn("humanoid_contract", payload)
        self.assertNotIn("humanoid_source_kind", payload)
        self.assertNotIn("humanoid_provenance", payload)
        self.assertNotIn("humanoid_warnings", payload)

    def test_auto_mode_without_source_completes_with_deterministic_fallback_warning(self) -> None:
        first = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")
        second = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")

        self.assertEqual(first, second)
        self.assertEqual(first["metadata_mode"], "auto")
        self.assertEqual(first["humanoid_source_kind"], "fallback")
        self.assertEqual(first["humanoid_warnings"][0]["code"], "humanoid_metadata_unavailable")
        self.assertNotIn("humanoid_contract", first)

    def test_humanoid_mode_with_companion_emits_contract_and_provenance(self) -> None:
        self.output_mesh.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(complete_humanoid_source()), encoding="utf-8")

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")

        self.assertEqual(payload["metadata_mode"], "humanoid")
        self.assertEqual(payload["humanoid_source_kind"], "companion")
        self.assertEqual(payload["humanoid_contract"]["schema"], "modly.humanoid.v1")
        self.assertEqual(payload["humanoid_provenance"]["source_kind"], "companion")

    def test_humanoid_mode_without_valid_source_fails_actionably(self) -> None:
        with self.assertRaisesRegex(Exception, "metadata_mode=humanoid.*valid humanoid"):
            build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")

    def test_humanoid_mode_emits_sidecar_from_real_unirig_52_bone_semantic_resolver(self) -> None:
        write_glb_json(self.output_mesh, real_unirig_52_payload())

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")

        self.assertEqual(payload["metadata_mode"], "humanoid")
        self.assertEqual(payload["humanoid_source_kind"], "semantic_resolver")
        self.assertEqual(payload["humanoid_contract"]["required_roles"]["hips"], "bone_0")
        self.assertEqual(payload["humanoid_contract"]["nodes"]["bone_1"]["transforms"]["rest_world"][1][3], 1.7)

    def test_humanoid_mode_fails_closed_when_quality_gate_reports_sleeve_branch(self) -> None:
        write_embedded_skin_glb(self.output_mesh, sleeve=True)
        self.output_mesh.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(_declared_roles()), encoding="utf-8")

        with self.assertRaisesRegex(Exception, "unsafe_for_humanoid_retarget.*sleeve_branch_under_arm"):
            build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")

    def test_basic_humanoid_mode_accepts_shoulders_without_auto_or_legacy_fallback(self) -> None:
        write_embedded_skin_glb(self.output_mesh, shoulder_connectors=True, hair_contamination=True)
        self.output_mesh.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(complete_humanoid_source(include_shoulders=True, include_fingers=False)), encoding="utf-8")

        humanoid = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")
        auto = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")
        legacy = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="legacy")

        self.assertEqual(humanoid["metadata_mode"], "humanoid")
        self.assertEqual(humanoid["humanoid_source_kind"], "companion")
        self.assertEqual(humanoid["humanoid_contract"]["schema"], "modly.humanoid.v1")
        self.assertEqual(humanoid["humanoid_contract"]["optional_roles"]["left_shoulder"], "left_shoulder")
        self.assertEqual(humanoid["humanoid_provenance"]["quality_gate"]["semantic_body_graph"]["warnings"][0]["code"], "high_region_weighted_by_torso_or_arm")
        self.assertEqual(auto["metadata_mode"], "auto")
        self.assertEqual(auto["humanoid_source_kind"], "companion")
        self.assertNotIn("metadata_mode", legacy)
        self.assertNotIn("humanoid_contract", legacy)

    def test_upper_arm_high_region_tolerance_keeps_metadata_modes_explicit(self) -> None:
        write_embedded_skin_glb(
            self.output_mesh,
            hair_contamination=True,
            high_region_role="right_upper_arm",
            high_region_y=1.584,
        )
        self.output_mesh.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(complete_humanoid_source(include_fingers=False)), encoding="utf-8")

        humanoid = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")
        auto = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")
        legacy = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="legacy")

        self.assertEqual(humanoid["metadata_mode"], "humanoid")
        self.assertEqual(humanoid["humanoid_source_kind"], "companion")
        self.assertEqual(humanoid["humanoid_provenance"]["quality_gate"]["semantic_body_graph"]["warnings"][0]["severity"], "warning")
        self.assertEqual(auto["metadata_mode"], "auto")
        self.assertEqual(auto["humanoid_source_kind"], "companion")
        self.assertNotIn("metadata_mode", legacy)
        self.assertNotIn("humanoid_contract", legacy)

    def test_explicit_humanoid_source_is_still_gated_when_output_contains_skin_evidence(self) -> None:
        write_embedded_skin_glb(self.output_mesh, sleeve=True)

        with self.assertRaisesRegex(Exception, "unsafe_for_humanoid_retarget.*sleeve_branch_under_arm"):
            build_sidecar(
                self.output_mesh,
                self.input_mesh,
                12345,
                self.context,
                humanoid_source=_declared_roles(),
                metadata_mode="humanoid",
            )

    def test_explicit_humanoid_source_without_gate_evidence_is_marked_trusted_unverified(self) -> None:
        payload = build_sidecar(
            self.output_mesh,
            self.input_mesh,
            12345,
            self.context,
            humanoid_source=complete_humanoid_source(include_fingers=False),
            metadata_mode="humanoid",
        )

        self.assertEqual(payload["humanoid_source_kind"], "provided")
        self.assertEqual(payload["humanoid_provenance"]["quality_gate"]["status"], "trusted_source_unverified")
        self.assertEqual(payload["humanoid_provenance"]["safe_retargeting_evidence"], "unverified")
        self.assertIn("trusted_humanoid_source_unverified", {warning["code"] for warning in payload["humanoid_warnings"]})

    def test_auto_mode_avoids_unsafe_contract_and_surfaces_quality_gate_diagnostics(self) -> None:
        write_embedded_skin_glb(self.output_mesh, sleeve=True)
        self.output_mesh.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(_declared_roles()), encoding="utf-8")

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")

        self.assertNotIn("humanoid_contract", payload)
        self.assertEqual(payload["humanoid_source_kind"], "fallback")
        self.assertEqual(payload["humanoid_provenance"]["diagnostic"]["code"], "unsafe_for_humanoid_retarget")
        self.assertIn("semantic_body_graph", payload["humanoid_provenance"]["diagnostic"])
        self.assertIn("sleeve_branch_under_arm", {reason["code"] for reason in payload["humanoid_provenance"]["diagnostic"]["reasons"]})

    def test_auto_mode_with_unsafe_accessory_evidence_emits_untrusted_semantic_candidates(self) -> None:
        write_embedded_skin_glb(self.output_mesh, sleeve=True, semantic_connected=True)
        self.output_mesh.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(_declared_roles()), encoding="utf-8")

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")

        self.assertNotIn("humanoid_contract", payload)
        self.assertEqual(payload["humanoid_contract_status"], "contract_missing_or_untrusted")
        candidates = payload["semantic_candidates"]
        self.assertEqual(candidates["schema"], SEMANTIC_CANDIDATES_SCHEMA)
        self.assertEqual(candidates["trust"], {
            "trusted": False,
            "stabilization_eligible": False,
            "reason": "contract_missing_or_untrusted",
        })
        self.assertEqual(candidates["status"], "candidate")
        for role in ("hips", "spine", "head"):
            with self.subTest(role=role):
                self.assertIn(role, candidates["roles"])
                self.assertGreaterEqual(candidates["roles"][role]["confidence"], 0.75)
                self.assertEqual(candidates["roles"][role]["unsafe_flags"], [])
        self.assertEqual(candidates["chains"]["torso"], ["hips", "spine", "chest", "neck", "head"])
        self.assertGreaterEqual(candidates["topology"]["joint_count"], 17)
        self.assertLessEqual(len(candidates["rejected_candidates"]), 5)
        self.assertEqual(candidates["transforms"]["matrix_order"], "row-major")
        self.assertIn(candidates["roles"]["hips"]["node_id"], candidates["transforms"]["nodes"])

    def test_auto_mode_with_too_few_no_core_joints_emits_blocked_candidate_diagnostics(self) -> None:
        self._write_too_few_no_core_glb(self.output_mesh)

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")

        self.assertNotIn("humanoid_contract", payload)
        self.assertEqual(payload["humanoid_contract_status"], "contract_missing_or_untrusted")
        candidates = payload["semantic_candidates"]
        self.assertEqual(candidates["schema"], SEMANTIC_CANDIDATES_SCHEMA)
        self.assertEqual(candidates["status"], "blocked")
        self.assertFalse(candidates["trust"]["trusted"])
        self.assertFalse(candidates["trust"]["stabilization_eligible"])
        self.assertEqual(candidates["topology"]["joint_count"], 5)
        self.assertIn("insufficient_core_candidates", {item["code"] for item in candidates["diagnostics"]})
        missing_core = next(item for item in candidates["diagnostics"] if item["code"] == "insufficient_core_candidates")
        self.assertEqual(missing_core["missing_roles"], ["head", "hips", "spine"])
        self.assertIn("too_few_joints", missing_core["reasons"])

    def _write_too_few_no_core_glb(self, target: Path) -> Path:
        nodes = [
            {"name": "accessory_root", "translation": [0.0, 0.0, 0.0], "children": [1, 2]},
            {"name": "left_leaf", "translation": [-0.1, 0.1, 0.0]},
            {"name": "right_leaf", "translation": [0.1, 0.1, 0.0], "children": [3]},
            {"name": "tip", "translation": [0.0, 0.1, 0.0], "children": [4]},
            {"name": "ornament", "translation": [0.0, 0.1, 0.0]},
        ]
        positions = [(0.0, 0.0, 0.0), (-0.1, 0.1, 0.0), (0.1, 0.1, 0.0)]
        joint_rows = [(0, 0, 0, 0), (1, 0, 0, 0), (4, 0, 0, 0)]
        weight_rows = [(1.0, 0.0, 0.0, 0.0) for _ in positions]
        return _write_glb(target, nodes=nodes, joints=list(range(len(nodes))), positions=positions, joint_rows=joint_rows, weight_rows=weight_rows)

    def test_legacy_mode_suppresses_humanoid_fields_even_with_unsafe_semantic_evidence(self) -> None:
        write_embedded_skin_glb(self.output_mesh, sleeve=True)

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="legacy")

        self.assertNotIn("metadata_mode", payload)
        self.assertNotIn("humanoid_contract", payload)
        self.assertNotIn("humanoid_provenance", payload)


if __name__ == "__main__":
    unittest.main()
