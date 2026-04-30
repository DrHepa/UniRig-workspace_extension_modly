from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext.gltf_skin_analysis import read_glb_container
from unirig_ext.humanoid_quality_gate import HumanoidQualityGateError, run_humanoid_quality_gate
from unirig_ext.semantic_body_graph import build_semantic_body_report


COMPONENT_FLOAT = 5126
COMPONENT_USHORT = 5123


def write_embedded_skin_glb(
    target: Path,
    *,
    sleeve: bool = False,
    hand_leaf: bool = False,
    non_local_weights: bool = False,
    hair_contamination: bool = False,
    mixed_classes: bool = False,
    missing_weights: bool = False,
    extras: dict | None = None,
    semantic_connected: bool = False,
) -> Path:
    nodes = [
        {"name": "hips", "translation": [0.0, 0.0, 0.0], "children": [1, 5, 8]},
        {"name": "spine", "translation": [0.0, 1.0, 0.0], "children": [2]},
        {"name": "chest", "translation": [0.0, 0.7, 0.0], "children": [3, 12, 16]},
        {"name": "neck", "translation": [0.0, 0.45, 0.0], "children": [4]},
        {"name": "head", "translation": [0.0, 0.3, 0.0]},
        {"name": "left_upper_leg", "translation": [-0.25, -0.1, 0.0], "children": [6]},
        {"name": "left_lower_leg", "translation": [0.0, -0.9, 0.0], "children": [7]},
        {"name": "left_foot", "translation": [0.0, -0.8, 0.2]},
        {"name": "right_upper_leg", "translation": [0.25, -0.1, 0.0], "children": [9]},
        {"name": "right_lower_leg", "translation": [0.0, -0.9, 0.0], "children": [10]},
        {"name": "right_foot", "translation": [0.0, -0.8, 0.2]},
        {"name": "unused", "translation": [0.0, 0.0, 0.0]},
        {"name": "left_upper_arm", "translation": [-0.65, 0.25, 0.0], "children": [13]},
        {"name": "left_lower_arm", "translation": [-0.6, -0.15, 0.0], "children": [14]},
        {"name": "left_hand", "translation": [-0.45, -0.15, 0.0]},
        {"name": "left_sleeve", "translation": [-0.05, -0.55, 0.0]},
        {"name": "right_upper_arm", "translation": [0.65, 0.25, 0.0], "children": [17]},
        {"name": "right_lower_arm", "translation": [0.6, -0.15, 0.0], "children": [18]},
        {"name": "right_hand", "translation": [0.45, -0.15, 0.0]},
    ]
    if semantic_connected:
        nodes[0].setdefault("children", []).append(11)
    if sleeve or mixed_classes:
        nodes[13]["children"].append(15)
    if hand_leaf or mixed_classes:
        nodes[13].setdefault("children", []).append(len(nodes))
        nodes.append({"name": "left_watch_leaf", "translation": [-0.05, -0.3, 0.0]})
    if hair_contamination or mixed_classes:
        nodes[13].setdefault("children", []).append(len(nodes))
        nodes.append({"name": "left_hair_strand", "translation": [0.2, 0.75, 0.0]})
    joints = list(range(len(nodes)))

    positions = [
        (-0.95, 1.45, -0.05), (-1.85, 1.25, 0.05), (0.95, 1.45, -0.05), (1.85, 1.25, 0.05),
        (0.0, 2.25, 0.0), (0.0, 0.3, 0.0), (-0.25, -1.65, 0.2), (0.25, -1.65, 0.2),
    ]
    joint_rows = [(14, 0, 0, 0), (14, 0, 0, 0), (18, 0, 0, 0), (18, 0, 0, 0), (4, 0, 0, 0), (0, 0, 0, 0), (7, 0, 0, 0), (10, 0, 0, 0)]
    if sleeve or mixed_classes:
        positions.append((-1.2, 0.65, 0.18))
        joint_rows.append((15, 0, 0, 0))
    if hand_leaf or mixed_classes:
        positions.append((-1.3, 1.8, 0.03))
        joint_rows.append((19, 0, 0, 0))
    if hair_contamination:
        positions.append((0.0, 2.35, 0.0))
        joint_rows.append((2, 0, 0, 0))
    if mixed_classes:
        positions.append((-0.4, 2.35, 0.0))
        joint_rows.append((20, 0, 0, 0))
    if non_local_weights:
        positions.append((0.0, -1.65, 0.0))
        joint_rows.append((14, 0, 0, 0))
    weight_rows = [(1.0, 0.0, 0.0, 0.0) for _ in positions]
    return _write_glb(target, nodes=nodes, joints=joints, positions=positions, joint_rows=joint_rows, weight_rows=None if missing_weights else weight_rows, extras=extras)


def _write_glb(target: Path, *, nodes: list[dict], joints: list[int], positions: list[tuple[float, float, float]], joint_rows: list[tuple[int, int, int, int]], weight_rows: list[tuple[float, float, float, float]] | None, extras: dict | None = None) -> Path:
    binary = bytearray()
    buffer_views = []
    accessors = []

    def add_blob(blob: bytes, *, component_type: int, type_name: str, count: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(blob)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        accessors.append({"bufferView": len(buffer_views) - 1, "componentType": component_type, "count": count, "type": type_name})
        return len(accessors) - 1

    position_accessor = add_blob(b"".join(struct.pack("<fff", *row) for row in positions), component_type=COMPONENT_FLOAT, type_name="VEC3", count=len(positions))
    joints_accessor = add_blob(b"".join(struct.pack("<HHHH", *row) for row in joint_rows), component_type=COMPONENT_USHORT, type_name="VEC4", count=len(joint_rows))
    attributes = {"POSITION": position_accessor, "JOINTS_0": joints_accessor}
    if weight_rows is not None:
        attributes["WEIGHTS_0"] = add_blob(b"".join(struct.pack("<ffff", *row) for row in weight_rows), component_type=COMPONENT_FLOAT, type_name="VEC4", count=len(weight_rows))
    gltf = {
        "asset": {"version": "2.0"},
        "nodes": nodes,
        "skins": [{"joints": joints}],
        "meshes": [{"primitives": [{"attributes": attributes}]}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "buffers": [{"byteLength": len(binary)}],
    }
    if extras is not None:
        gltf["extras"] = extras
    return _write_glb_chunks(target, gltf, bytes(binary))


def _write_glb_chunks(target: Path, gltf: dict, bin_chunk: bytes) -> Path:
    json_chunk = json.dumps(gltf, separators=(",", ":"), sort_keys=True).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    while len(bin_chunk) % 4:
        bin_chunk += b"\x00"
    blob = bytearray(b"glTF")
    blob += struct.pack("<I", 2)
    blob += struct.pack("<I", 12 + 8 + len(json_chunk) + 8 + len(bin_chunk))
    blob += struct.pack("<I", len(json_chunk)) + b"JSON" + json_chunk
    blob += struct.pack("<I", len(bin_chunk)) + b"BIN\x00" + bin_chunk
    target.write_bytes(blob)
    return target


class HumanoidQualityGateTests(unittest.TestCase):
    def test_sleeve_branch_under_forearm_fails_closed_with_joint_classification(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-qgate-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "sleeved.glb", sleeve=True)
            declared = _declared_roles()

            with self.assertRaisesRegex(HumanoidQualityGateError, "unsafe_for_humanoid_retarget") as raised:
                run_humanoid_quality_gate(read_glb_container(glb), declared)

            diagnostics = raised.exception.diagnostic
            self.assertEqual(diagnostics["code"], "unsafe_for_humanoid_retarget")
            self.assertIn("sleeve_branch_under_arm", {reason["code"] for reason in diagnostics["reasons"]})
            self.assertEqual(diagnostics["joint_classes"]["left_sleeve"], "clothing")

    def test_clean_simple_humanoid_passes_and_records_body_classes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-qgate-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "clean.glb")

            report = run_humanoid_quality_gate(read_glb_container(glb), _declared_roles())

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.diagnostic["semantic_body_graph"]["publishable"], True)
            self.assertEqual(report.diagnostic["semantic_body_graph"]["predicates"]["has_clear_spine"], True)
            self.assertEqual(report.diagnostic["joint_classes"]["left_hand"], "body")
            self.assertEqual(report.diagnostic["weight_summary"]["vertex_count"], 8)

    def test_mesh_with_missing_weight_attributes_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-qgate-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "missing.glb", missing_weights=True)

            with self.assertRaisesRegex(HumanoidQualityGateError, "skin_weight_data_unavailable") as raised:
                run_humanoid_quality_gate(read_glb_container(glb), _declared_roles())

            self.assertIn("skin_weight_data_unavailable", {reason["code"] for reason in raised.exception.diagnostic["reasons"]})

    def test_non_anatomical_hand_leaf_fails_closed_with_accessory_classification(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-qgate-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "hand-leaf.glb", hand_leaf=True)

            with self.assertRaisesRegex(HumanoidQualityGateError, "unsafe_for_humanoid_retarget") as raised:
                run_humanoid_quality_gate(read_glb_container(glb), _declared_roles())

            diagnostics = raised.exception.diagnostic
            self.assertIn("non_anatomical_leaf_under_arm", {reason["code"] for reason in diagnostics["reasons"]})
            self.assertIn("semantic_passive_noncontract_subtree", {reason["code"] for reason in diagnostics["reasons"]})
            self.assertEqual(diagnostics["joint_classes"]["left_watch_leaf"], "accessory")

    def test_non_local_hand_weights_fail_closed_with_spread_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-qgate-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "non-local.glb", non_local_weights=True)

            with self.assertRaisesRegex(HumanoidQualityGateError, "unsafe_for_humanoid_retarget") as raised:
                run_humanoid_quality_gate(read_glb_container(glb), _declared_roles())

            non_local = [reason for reason in raised.exception.diagnostic["reasons"] if reason["code"] == "non_local_weight_spread"]
            self.assertEqual([reason["role"] for reason in non_local], ["left_hand"])
            self.assertGreater(non_local[0]["spread_y"], 2.0)

    def test_high_region_torso_or_arm_weighting_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-qgate-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "high-region.glb", hair_contamination=True)

            with self.assertRaisesRegex(HumanoidQualityGateError, "unsafe_for_humanoid_retarget") as raised:
                run_humanoid_quality_gate(read_glb_container(glb), _declared_roles())

            high_region = [reason for reason in raised.exception.diagnostic["reasons"] if reason["code"] == "high_region_weighted_by_torso_or_arm"]
            self.assertEqual(high_region[0]["joints"], [{"role": "chest", "joint": "chest", "max_y": 2.35}])

    def test_mixed_diagnostics_keep_body_clothing_hair_accessory_and_unknown_classes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-qgate-") as temp_dir:
            glb = write_embedded_skin_glb(Path(temp_dir) / "mixed.glb", mixed_classes=True)

            with self.assertRaisesRegex(HumanoidQualityGateError, "unsafe_for_humanoid_retarget") as raised:
                run_humanoid_quality_gate(read_glb_container(glb), _declared_roles())

            joint_classes = raised.exception.diagnostic["joint_classes"]
            self.assertEqual(joint_classes["left_hand"], "body")
            self.assertEqual(joint_classes["left_sleeve"], "clothing")
            self.assertEqual(joint_classes["left_hair_strand"], "hair")
            self.assertEqual(joint_classes["left_watch_leaf"], "accessory")
            self.assertEqual(joint_classes["unused"], "unknown")

    def test_quality_gate_reuses_provided_semantic_report_instead_of_rebuilding_graph(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-qgate-") as temp_dir:
            container = read_glb_container(write_embedded_skin_glb(Path(temp_dir) / "clean.glb"))
            declared = _declared_roles()
            semantic_report = build_semantic_body_report(container, declared)

            with patch("unirig_ext.humanoid_quality_gate.build_semantic_body_report") as builder:
                report = run_humanoid_quality_gate(container, declared, semantic_report=semantic_report)

            builder.assert_not_called()
            self.assertEqual(report.status, "passed")
            self.assertEqual(report.diagnostic["semantic_body_graph"], semantic_report.as_diagnostic())


def _declared_roles() -> dict:
    return {
        "roles": {
            "hips": "hips", "spine": "spine", "chest": "chest", "neck": "neck", "head": "head",
            "left_upper_arm": "left_upper_arm", "left_lower_arm": "left_lower_arm", "left_hand": "left_hand",
            "right_upper_arm": "right_upper_arm", "right_lower_arm": "right_lower_arm", "right_hand": "right_hand",
            "left_upper_leg": "left_upper_leg", "left_lower_leg": "left_lower_leg", "left_foot": "left_foot",
            "right_upper_leg": "right_upper_leg", "right_lower_leg": "right_lower_leg", "right_foot": "right_foot",
        }
    }


if __name__ == "__main__":
    unittest.main()
