from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_metadata import MetadataTests, complete_humanoid_source  # noqa: E402
from test_humanoid_quality_gate import _declared_roles, write_embedded_skin_glb  # noqa: E402
from unirig_ext.humanoid_contract import build_contract_from_declared_data, validate_humanoid_contract  # noqa: E402
from unirig_ext.metadata import build_sidecar  # noqa: E402


class HumanoidContractPublicationTests(unittest.TestCase):
    def test_strict_contract_publishes_version_validation_producer_and_hash_association(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(include_fingers=False),
            source_hash="a" * 64,
            output_hash="b" * 64,
            producer={"extension_id": "unirig-process-extension", "node_id": "rig-mesh", "version": "test-ref"},
        )

        self.assertEqual(contract["schema"], "modly.humanoid.v1")
        self.assertEqual(contract["contract_version"], 1)
        self.assertEqual(
            contract["producer"],
            {"extension_id": "unirig-process-extension", "node_id": "rig-mesh", "version": "test-ref"},
        )
        self.assertEqual(contract["validation"]["status"], "validated")
        self.assertEqual(contract["validation"]["validator"], "unirig_ext.humanoid_contract.validate_humanoid_contract")
        self.assertEqual(contract["validation"]["unsafe_flags"], [])
        self.assertEqual(contract["hashes"]["source_sha256"], "a" * 64)
        self.assertEqual(contract["hashes"]["output_sha256"], "b" * 64)
        self.assertEqual(contract["hashes"]["sidecar_payload_sha256"], "pending_sidecar_payload")

        validate_humanoid_contract(contract)

    def test_sidecar_embeds_validated_contract_payload_hash_and_keeps_warnings_diagnostic(self) -> None:
        base = MetadataTests(methodName="test_sidecar_payload_is_deterministic")
        base.setUp()
        try:
            payload = build_sidecar(
                base.output_mesh,
                base.input_mesh,
                12345,
                base.context,
                humanoid_source=complete_humanoid_source(include_fingers=False),
                metadata_mode="humanoid",
            )
        finally:
            base.tearDown()

        contract = payload["humanoid_contract"]
        self.assertEqual(contract["validation"]["status"], "validated")
        self.assertEqual(contract["hashes"]["sidecar_payload_sha256"], payload["sidecar_payload_sha256"])
        self.assertEqual(contract["producer"]["extension_id"], "unirig-process-extension")
        self.assertEqual(contract["producer"]["node_id"], "rig-mesh")
        self.assertEqual(contract["producer"]["version"], "test-ref")
        self.assertEqual(contract["validation"]["warnings"], contract["warnings"])

    def test_auto_mode_unsafe_sidecar_is_trusted_stabilization_ineligible_not_runtime_failure(self) -> None:
        base = MetadataTests(methodName="test_sidecar_payload_is_deterministic")
        base.setUp()
        try:
            write_embedded_skin_glb(base.output_mesh, sleeve=True)
            base.output_mesh.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(_declared_roles()), encoding="utf-8")
            payload = build_sidecar(base.output_mesh, base.input_mesh, 12345, base.context, metadata_mode="auto")
        finally:
            base.tearDown()

        self.assertNotIn("humanoid_contract", payload)
        self.assertEqual(payload["humanoid_contract_status"], "contract_missing_or_untrusted")
        self.assertEqual(payload["trusted_stabilization_status"], "ineligible")
        self.assertEqual(payload["runtime_status"], "success_capable")
        self.assertEqual(payload["humanoid_provenance"]["diagnostic"]["code"], "unsafe_for_humanoid_retarget")


if __name__ == "__main__":
    unittest.main()
