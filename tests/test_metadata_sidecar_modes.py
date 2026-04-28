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
from fixtures.unirig_real_topology import real_unirig_52_payload
from unirig_ext.metadata import build_sidecar


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

    def test_humanoid_mode_emits_sidecar_from_real_unirig_52_bone_profile(self) -> None:
        write_glb_json(self.output_mesh, real_unirig_52_payload())

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")

        self.assertEqual(payload["metadata_mode"], "humanoid")
        self.assertEqual(payload["humanoid_source_kind"], "topology_profile")
        self.assertEqual(payload["humanoid_contract"]["required_roles"]["hips"], "bone_0")
        self.assertEqual(payload["humanoid_contract"]["nodes"]["bone_1"]["transforms"]["rest_world"][1][3], 3.0)


if __name__ == "__main__":
    unittest.main()
