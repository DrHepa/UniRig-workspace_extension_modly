# pyright: reportMissingImports=false

from __future__ import annotations

import json
import hashlib
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))
if str(TESTS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(TESTS))

from unirig_ext import bootstrap
from unirig_ext.bootstrap import RuntimeContext
from unirig_ext.generation_profile import normalize_generation_profile, resolve_generation_profile
from unirig_ext.io import derive_output_path
from unirig_ext.humanoid_contract import (
    HUMANOID_SCHEMA,
    HumanoidContractError,
    build_humanoid_contract,
    build_contract_from_declared_data,
    validate_humanoid_contract,
)
from unirig_ext.metadata import build_sidecar, sidecar_path_for, write_sidecar
from unirig_ext.humanoid_source import HumanoidResolutionFailure
from fixtures.unirig_real_topology import real_unirig_40_payload
from test_semantic_humanoid_resolver import short_trunk_output_payload


def write_glb_json(target: Path, payload: dict) -> Path:
    json_chunk = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    blob = bytearray(b"glTF")
    blob += struct.pack("<I", 2)
    blob += struct.pack("<I", 12 + 8 + len(json_chunk))
    blob += struct.pack("<I", len(json_chunk))
    blob += b"JSON"
    blob += json_chunk
    target.write_bytes(blob)
    return target


def complete_humanoid_source(
    *,
    include_fingers: bool = True,
    include_toes: bool = True,
    include_shoulders: bool = False,
    shoulder_aliases: bool = False,
    malformed_shoulder: bool = False,
    partial_finger_chain: bool = False,
    basis_status: str = "asserted",
) -> dict:
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    roles = {
        "hips": "hips",
        "spine": "spine",
        "chest": "chest",
        "neck": "neck",
        "head": "head",
        "left_upper_arm": "left_upper_arm",
        "left_lower_arm": "left_lower_arm",
        "left_hand": "left_hand",
        "right_upper_arm": "right_upper_arm",
        "right_lower_arm": "right_lower_arm",
        "right_hand": "right_hand",
        "left_upper_leg": "left_upper_leg",
        "left_lower_leg": "left_lower_leg",
        "left_foot": "left_foot",
        "right_upper_leg": "right_upper_leg",
        "right_lower_leg": "right_lower_leg",
        "right_foot": "right_foot",
    }
    parents = {
        "hips": None,
        "spine": "hips",
        "chest": "spine",
        "neck": "chest",
        "head": "neck",
        "left_upper_arm": "chest",
        "left_lower_arm": "left_upper_arm",
        "left_hand": "left_lower_arm",
        "right_upper_arm": "chest",
        "right_lower_arm": "right_upper_arm",
        "right_hand": "right_lower_arm",
        "left_upper_leg": "hips",
        "left_lower_leg": "left_upper_leg",
        "left_foot": "left_lower_leg",
        "right_upper_leg": "hips",
        "right_lower_leg": "right_upper_leg",
        "right_foot": "right_lower_leg",
    }
    if include_shoulders:
        left_role = "left_clavicle" if shoulder_aliases else "left_shoulder"
        right_role = "right_clavicle" if shoulder_aliases else "right_shoulder"
        roles[left_role] = "missing_left_shoulder" if malformed_shoulder else "left_shoulder"
        roles[right_role] = "right_shoulder"
        parents["left_shoulder"] = "chest"
        parents["left_upper_arm"] = "left_shoulder"
        parents["right_shoulder"] = "chest"
        parents["right_upper_arm"] = "right_shoulder"
    if include_toes:
        roles["left_toe"] = "left_toe"
        roles["right_toe"] = "right_toe"
        parents["left_toe"] = "left_foot"
        parents["right_toe"] = "right_foot"
    if include_fingers:
        roles["left_thumb_1"] = "left_thumb_1"
        roles["left_thumb_2"] = "left_thumb_2"
        if not partial_finger_chain:
            roles["left_thumb_3"] = "left_thumb_3"
        parents["left_thumb_1"] = "left_hand"
        parents["left_thumb_2"] = "left_thumb_1"
        if not partial_finger_chain:
            parents["left_thumb_3"] = "left_thumb_2"
    nodes = [
        {"id": node_id, "name": node_id, "parent": parent, "rest_local": identity, "rest_world": identity}
        for node_id, parent in parents.items()
    ]
    return {
        "roles": roles,
        "nodes": nodes,
        "basis": {"up": "Y", "forward": "Z", "handedness": "right", "status": basis_status},
        "provenance": {"source": "unit-fixture", "method": "declared"},
    }


class MetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unirig-meta-"))
        self.input_mesh = self.temp_dir / "avatar.glb"
        self.input_mesh.write_bytes(b"input")
        self.output_mesh = derive_output_path(self.input_mesh)
        self.output_mesh.write_bytes(b"output")
        self.context = RuntimeContext(
            extension_root=self.temp_dir,
            runtime_root=self.temp_dir / ".unirig-runtime",
            cache_dir=self.temp_dir / ".unirig-runtime" / "cache",
            assets_dir=self.temp_dir / ".unirig-runtime" / "assets",
            logs_dir=self.temp_dir / ".unirig-runtime" / "logs",
            state_path=self.temp_dir / ".unirig-runtime" / "bootstrap_state.json",
            venv_dir=self.temp_dir / "venv",
            venv_python=self.temp_dir / "venv" / "bin" / "python",
            runtime_vendor_dir=self.temp_dir / ".unirig-runtime" / "vendor",
            unirig_dir=self.temp_dir / ".unirig-runtime" / "vendor" / "unirig",
            hf_home=self.temp_dir / ".unirig-runtime" / "hf-home",
            extension_id="unirig-process-extension",
            runtime_mode="real",
            allow_local_stub_runtime=False,
            bootstrap_version=bootstrap.BOOTSTRAP_VERSION,
            vendor_source="fixture",
            source_ref="test-ref",
            host_python=sys.executable,
            platform_tag="linux-aarch64",
            python_version="3.12.3",
            platform_policy=bootstrap.resolve_platform_policy("linux", "aarch64"),
            source_build={"status": "ready", "mode": "source-build", "dependencies": {}},
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sidecar_payload_is_deterministic(self) -> None:
        first = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context)
        second = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context)
        self.assertEqual(first, second)
        self.assertEqual(first["output_mesh"], "avatar_unirig.glb")

    def test_write_sidecar_creates_adjacent_rigmeta_json(self) -> None:
        destination = write_sidecar(self.output_mesh, self.input_mesh, 7, self.context)
        self.assertEqual(destination, sidecar_path_for(self.output_mesh))
        payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata_version"], 1)
        self.assertEqual(payload["node_id"], "rig-mesh")
        self.assertEqual(
            payload["runtime"],
            {
                "mode": "real",
                "python_version": "3.12.3",
                "source_ref": "test-ref",
            },
        )
        self.assertNotIn("platform_policy", payload["runtime"])
        self.assertNotIn("source_build", payload["runtime"])
        self.assertNotIn("vendor_source", payload["runtime"])
        self.assertNotIn("runtime_root", payload["runtime"])
        self.assertNotIn("bootstrap_version", payload["runtime"])

    def test_default_sidecar_reports_stable_generation_profile_without_trust_effect(self) -> None:
        payload = build_sidecar(self.output_mesh, self.input_mesh, 7, self.context, metadata_mode="legacy")

        self.assertEqual(payload["generation_profile"], "articulationxl")
        self.assertEqual(payload["generation_profile_status"], "stable")
        self.assertEqual(
            payload["generation_diagnostics"],
            {
                "skeleton_prior": "articulationxl",
                "profile_config_source": "upstream_task",
                "trust_effect": "none",
            },
        )
        self.assertNotIn("humanoid_contract", payload)

    def test_vroid_sidecar_reports_experimental_generated_config_without_trust_upgrade(self) -> None:
        source_path = self.context.unirig_dir / "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            json.dumps(
                {
                    "task": {"name": "skeleton", "assign_cls": "articulationxl"},
                    "system": {"skeleton_prior": "articulationxl"},
                    "generate_kwargs": {"cls": "articulationxl"},
                    "tokenizer": {"skeleton_order": ["hips", "spine", "head"]},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        profile = resolve_generation_profile(
            normalize_generation_profile({"generation_profile": "vroid"}),
            context=self.context,
            run_dir=self.temp_dir / "run-vroid",
        )

        payload = build_sidecar(self.output_mesh, self.input_mesh, 7, self.context, metadata_mode="legacy", generation_profile=profile)

        self.assertEqual(payload["generation_profile"], "vroid")
        self.assertEqual(payload["generation_profile_status"], "experimental")
        self.assertEqual(payload["generation_diagnostics"]["skeleton_prior"], "vroid")
        self.assertEqual(payload["generation_diagnostics"]["profile_config_source"], "generated_run_config")
        self.assertEqual(payload["generation_diagnostics"]["trust_effect"], "none")
        self.assertEqual(payload["generation_diagnostics"]["generated_config_sha256"], profile.generated_config_sha256)
        self.assertNotIn("humanoid_contract", payload)

    def test_build_contract_from_declared_data_covers_core_roles_and_optional_fingers(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(include_fingers=True),
            source_hash="a" * 64,
            output_hash="b" * 64,
        )

        self.assertEqual(contract["schema"], HUMANOID_SCHEMA)
        self.assertEqual(contract["chains"]["spine"], ["hips", "spine", "chest", "neck", "head"])
        self.assertEqual(contract["chains"]["left_arm"], ["left_upper_arm", "left_lower_arm", "left_hand"])
        self.assertEqual(contract["chains"]["left_thumb"], ["left_thumb_1", "left_thumb_2", "left_thumb_3"])
        self.assertEqual(contract["chains"]["left_leg"], ["left_upper_leg", "left_lower_leg", "left_foot", "left_toe"])
        self.assertEqual(contract["nodes"]["left_hand"]["parent"], "left_lower_arm")
        self.assertEqual(contract["nodes"]["hips"]["transforms"]["matrix_order"], "row-major")
        self.assertEqual(contract["hashes"]["source_sha256"], "a" * 64)
        self.assertEqual(contract["hashes"]["output_sha256"], "b" * 64)
        self.assertEqual(contract["warnings"], [])

    def test_contract_missing_optional_fingers_succeeds_with_deterministic_warning(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(include_fingers=False),
            source_hash="c" * 64,
            output_hash="d" * 64,
        )

        self.assertEqual(contract["chains"]["left_thumb"], [])
        self.assertEqual(
            contract["warnings"],
            [
                {
                    "code": "optional_fingers_missing",
                    "message": "Optional finger chains were not declared; full-body contract remains valid.",
                    "severity": "warning",
                }
            ],
        )

    def test_contract_emits_optional_shoulder_roles_and_proximal_arm_chains(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(include_shoulders=True, include_fingers=False),
            source_hash="5" * 64,
            output_hash="6" * 64,
        )

        self.assertEqual(contract["required_roles"]["left_upper_arm"], "left_upper_arm")
        self.assertEqual(contract["optional_roles"]["left_shoulder"], "left_shoulder")
        self.assertEqual(contract["optional_roles"]["right_shoulder"], "right_shoulder")
        self.assertEqual(contract["chains"]["left_arm"], ["left_upper_arm", "left_lower_arm", "left_hand"])
        self.assertEqual(contract["chains"]["left_arm_proximal"], ["left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand"])
        self.assertEqual(contract["chains"]["right_arm_proximal"], ["right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand"])

    def test_contract_canonicalizes_clavicle_aliases_to_optional_shoulder_roles(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(include_shoulders=True, shoulder_aliases=True, include_fingers=False),
            source_hash="5" * 64,
            output_hash="6" * 64,
        )

        self.assertEqual(contract["optional_roles"], {"left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder"})
        self.assertNotIn("left_clavicle", contract["optional_roles"])
        self.assertNotIn("right_clavicle", contract["optional_roles"])

    def test_malformed_optional_shoulder_role_fails_closed(self) -> None:
        with self.assertRaisesRegex(HumanoidContractError, "Optional humanoid role 'left_shoulder' references unknown node 'missing_left_shoulder'"):
            build_contract_from_declared_data(
                complete_humanoid_source(include_shoulders=True, malformed_shoulder=True),
                source_hash="5" * 64,
                output_hash="6" * 64,
            )

    def test_partial_optional_finger_chain_succeeds_with_deterministic_warning(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(include_fingers=True, partial_finger_chain=True),
            source_hash="8" * 64,
            output_hash="9" * 64,
        )

        self.assertEqual(contract["chains"]["left_thumb"], ["left_thumb_1", "left_thumb_2"])
        self.assertIn(
            {
                "code": "optional_finger_chain_partial",
                "message": "Optional finger chain 'left_thumb' is partially declared; missing roles: left_thumb_3.",
                "severity": "warning",
            },
            contract["warnings"],
        )

    def test_missing_optional_toes_keeps_core_leg_chains_valid(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(include_toes=False),
            source_hash="6" * 64,
            output_hash="7" * 64,
        )

        self.assertNotIn("left_toe", contract["required_roles"])
        self.assertNotIn("right_toe", contract["required_roles"])
        self.assertEqual(contract["chains"]["left_leg"], ["left_upper_leg", "left_lower_leg", "left_foot"])
        self.assertEqual(contract["chains"]["right_leg"], ["right_upper_leg", "right_lower_leg", "right_foot"])

    def test_contract_records_inferred_basis_warning(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(include_fingers=False, basis_status="inferred"),
            source_hash="e" * 64,
            output_hash="f" * 64,
        )

        self.assertEqual(contract["basis"]["status"], "inferred")
        self.assertIn(
            {
                "code": "basis_inferred",
                "message": "Coordinate basis is inferred rather than asserted by the source metadata.",
                "severity": "warning",
            },
            contract["warnings"],
        )

    def test_required_role_failure_is_actionable(self) -> None:
        source = complete_humanoid_source()
        del source["roles"]["hips"]

        with self.assertRaisesRegex(HumanoidContractError, "Missing required humanoid role 'hips'"):
            build_contract_from_declared_data(source, source_hash="0" * 64, output_hash="1" * 64)

    def test_output_without_declared_humanoid_metadata_fails_loudly(self) -> None:
        with self.assertRaisesRegex(HumanoidContractError, "No declared humanoid metadata"):
            build_humanoid_contract(self.output_mesh, source_hash="4" * 64)

    def test_unknown_chain_node_failure_is_actionable(self) -> None:
        contract = build_contract_from_declared_data(
            complete_humanoid_source(),
            source_hash="2" * 64,
            output_hash="3" * 64,
        )
        contract["chains"]["left_arm"] = ["left_upper_arm", "missing_node", "left_hand"]

        with self.assertRaisesRegex(HumanoidContractError, "Chain 'left_arm' references unknown node 'missing_node'"):
            validate_humanoid_contract(contract)

    def test_sidecar_with_declared_humanoid_source_is_byte_deterministic(self) -> None:
        source = complete_humanoid_source(include_fingers=False)
        first = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, humanoid_source=source)
        second = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, humanoid_source=source)

        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["metadata_version"], 1)
        self.assertEqual(first["humanoid_contract"]["schema"], HUMANOID_SCHEMA)
        self.assertEqual(first["humanoid_contract"]["hashes"]["source_sha256"], first["source_sha256"])

    def test_legacy_mode_skips_semantic_resolver_and_emits_no_humanoid_metadata(self) -> None:
        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="legacy")

        self.assertNotIn("metadata_mode", payload)
        self.assertNotIn("humanoid_source_kind", payload)
        self.assertNotIn("humanoid_contract", payload)

    def test_auto_mode_semantic_failure_writes_legacy_warning(self) -> None:
        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")

        self.assertEqual(payload["humanoid_source_kind"], "fallback")
        self.assertEqual(payload["humanoid_warnings"][0]["code"], "humanoid_metadata_unavailable")
        self.assertIn("metadata_mode=auto", payload["humanoid_warnings"][0]["message"])

    def test_humanoid_mode_semantic_failure_is_actionable_before_done(self) -> None:
        with self.assertRaisesRegex(HumanoidResolutionFailure, "semantic resolver"):
            build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")

    def test_humanoid_mode_with_resolving_input_and_insufficient_output_fails_closed_without_sidecar(self) -> None:
        write_glb_json(self.input_mesh, real_unirig_40_payload())
        write_glb_json(self.output_mesh, short_trunk_output_payload(prefix="out"))

        with self.assertRaises(HumanoidResolutionFailure) as raised:
            write_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")

        message = str(raised.exception)
        self.assertIn("humanoid_output_contract_insufficient", message)
        self.assertIn("semantic_spine_missing", message)
        self.assertIn("source evidence resolved", message)
        self.assertFalse(sidecar_path_for(self.output_mesh).exists())

    def test_input_path_source_probe_is_diagnostic_only_and_never_publishes_source_roles(self) -> None:
        write_glb_json(self.input_mesh, real_unirig_40_payload())
        write_glb_json(self.output_mesh, short_trunk_output_payload(prefix="out"))

        with self.assertRaises(HumanoidResolutionFailure) as raised:
            build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="humanoid")

        diagnostic = json.loads(str(raised.exception))
        self.assertEqual(diagnostic["code"], "humanoid_output_contract_insufficient")
        self.assertEqual(diagnostic["source"]["status"], "resolved")
        self.assertEqual(diagnostic["transfer"], {"status": "absent", "required": True})
        self.assertEqual(diagnostic["output"]["joint_count"], 8)
        self.assertEqual(diagnostic["output"]["root_count"], 1)
        self.assertEqual(diagnostic["output"]["highest_path_length"], 2)
        self.assertEqual(diagnostic["output"]["minimum_trunk_length"], 5)
        self.assertNotIn("humanoid_contract", diagnostic)
        self.assertNotIn("roles", diagnostic["source"])

    def test_auto_mode_success_records_semantic_resolver_source_kind(self) -> None:
        write_glb_json(self.output_mesh, real_unirig_40_payload())

        payload = build_sidecar(self.output_mesh, self.input_mesh, 12345, self.context, metadata_mode="auto")

        self.assertEqual(payload["humanoid_source_kind"], "semantic_resolver")
        self.assertEqual(payload["humanoid_contract"]["required_roles"]["right_foot"], "bone_39")

    def test_write_sidecar_records_deterministic_payload_hash_semantics(self) -> None:
        destination = write_sidecar(
            self.output_mesh,
            self.input_mesh,
            31,
            self.context,
            humanoid_source=complete_humanoid_source(include_fingers=False),
        )
        payload = json.loads(destination.read_text(encoding="utf-8"))
        recorded_hash = payload.pop("sidecar_payload_sha256")
        payload["humanoid_contract"]["hashes"]["sidecar_payload_sha256"] = "pending_sidecar_payload"
        canonical = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")

        self.assertEqual(recorded_hash, hashlib.sha256(canonical).hexdigest())


if __name__ == "__main__":
    unittest.main()
