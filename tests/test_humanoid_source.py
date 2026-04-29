from __future__ import annotations

import json
import struct
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

from test_metadata import complete_humanoid_source
from fixtures.unirig_real_topology import real_unirig_40_payload, real_unirig_52_payload
from unirig_ext.humanoid_contract import build_contract_from_declared_data
from unirig_ext.humanoid_source import HumanoidResolutionFailure, resolve_humanoid_source


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


class HumanoidSourceTests(unittest.TestCase):
    def test_companion_source_wins_over_glb_extras(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-source-") as temp_dir:
            output_path = Path(temp_dir) / "avatar_unirig.glb"
            companion_payload = complete_humanoid_source(basis_status="asserted")
            extras_payload = complete_humanoid_source(basis_status="inferred")
            write_glb_json(output_path, {"asset": {"version": "2.0"}, "extras": {"unirig_humanoid": extras_payload}})
            output_path.with_name("avatar_unirig.humanoid.json").write_text(json.dumps(companion_payload), encoding="utf-8")

            resolved = resolve_humanoid_source(output_path)

            self.assertEqual(resolved.kind, "companion")
            self.assertEqual(resolved.payload["basis"]["status"], "asserted")
            self.assertEqual(resolved.provenance["path"], str(output_path.with_name("avatar_unirig.humanoid.json")))

    def test_glb_extras_source_is_used_when_companion_is_absent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-source-") as temp_dir:
            output_path = Path(temp_dir) / "avatar_unirig.glb"
            extras_payload = complete_humanoid_source(basis_status="inferred")
            write_glb_json(output_path, {"asset": {"version": "2.0"}, "extras": {"unirig_humanoid": extras_payload}})

            resolved = resolve_humanoid_source(output_path)

            self.assertEqual(resolved.kind, "glb_extras")
            self.assertEqual(resolved.payload["basis"]["status"], "inferred")

    def test_invalid_companion_fails_without_falling_through_to_lower_priority_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-source-") as temp_dir:
            output_path = Path(temp_dir) / "avatar_unirig.glb"
            write_glb_json(output_path, {"asset": {"version": "2.0"}, "extras": {"unirig_humanoid": complete_humanoid_source()}})
            output_path.with_name("avatar_unirig.humanoid.json").write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(HumanoidResolutionFailure, "companion.*JSON object"):
                resolve_humanoid_source(output_path)

    def test_real_unirig_52_bone_profile_resolves_as_semantic_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-source-") as temp_dir:
            output_path = Path(temp_dir) / "avatar_unirig.glb"
            write_glb_json(output_path, real_unirig_52_payload())

            resolved = resolve_humanoid_source(output_path)

            self.assertEqual(resolved.kind, "semantic_resolver")
            self.assertEqual(resolved.provenance["method"], "semantic-graph-rest-symmetry")
            self.assertEqual(resolved.payload["roles"]["hips"], "bone_0")
            self.assertEqual(resolved.payload["roles"]["right_foot"], "bone_51")
            self.assertEqual(resolved.warnings[0]["code"], "humanoid_source_from_semantic_resolver")

    def test_real_unirig_40_bone_profile_resolves_as_contract_ready_semantic_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-source-") as temp_dir:
            output_path = Path(temp_dir) / "avatar_unirig.glb"
            write_glb_json(output_path, real_unirig_40_payload())

            resolved = resolve_humanoid_source(output_path)
            contract = build_contract_from_declared_data(resolved.payload, source_hash="0" * 64, output_hash="1" * 64)

            self.assertEqual(resolved.kind, "semantic_resolver")
            self.assertEqual(resolved.provenance["method"], "semantic-graph-rest-symmetry")
            self.assertEqual(contract["required_roles"]["right_hand"], "bone_22")
            self.assertEqual(contract["optional_roles"]["right_shoulder"], "bone_19")
            self.assertEqual(resolved.warnings[0]["code"], "humanoid_source_from_semantic_resolver")


if __name__ == "__main__":
    unittest.main()
