from __future__ import annotations

import json
import os
import subprocess
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

from test_humanoid_corpus_profiler import write_resolver_ready_glb
from test_humanoid_quality_gate import _write_glb
from unirig_ext.humanoid_mapping_candidates import (
    CANDIDATE_SCHEMA,
    FULL_TOPOLOGY_SUFFICIENT_ASSETS,
    REAL_CORPUS_ENV_VAR,
    REPRESENTATIVE_CORPUS_ASSETS,
    CandidateInputError,
    CandidateOutputError,
    build_candidate_for_glb,
    build_candidate_reports,
    build_full_topology_sufficient_corpus_manifest,
    build_representative_corpus_manifest,
    dumps_candidate_json,
    render_run_suggestions,
    select_candidate_inputs,
    write_candidates_json,
    write_candidates_jsonl,
)
from unirig_ext.kimodo_probe import FAILURE_LAYERS, KimodoProbeBackend, ProbeResult, classify_probe_failure


class HumanoidMappingCandidateSchemaTests(unittest.TestCase):
    def test_candidate_schema_is_deterministic_and_contains_evidence_roles_and_projection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-schema-") as temp_dir:
            root = Path(temp_dir)
            glb = write_resolver_ready_glb(root / "ready.glb")

            candidate = build_candidate_for_glb(glb, source="synthetic-test")
            encoded = dumps_candidate_json(candidate)
            decoded = json.loads(encoded)

            self.assertEqual(decoded["schema"], CANDIDATE_SCHEMA)
            self.assertEqual(decoded["asset"]["path"], str(glb.resolve()))
            self.assertEqual(decoded["asset"]["source"], "synthetic-test")
            self.assertEqual(len(decoded["asset"]["sha256"]), 64)
            self.assertEqual(decoded["topology"]["joint_count"], 17)
            self.assertEqual(decoded["topology"]["root_count"], 1)
            self.assertGreaterEqual(decoded["topology"]["max_depth"], 3)
            self.assertIn("chest", decoded["topology"]["branch_points"])
            self.assertEqual(decoded["topology"]["hierarchy"]["hips"]["children"], ["spine", "left_upper_leg", "right_upper_leg"])
            self.assertEqual(decoded["topology"]["hierarchy"]["chest"]["children"], ["neck", "left_upper_arm", "right_upper_arm"])
            self.assertEqual(decoded["roles"]["left_lower_arm"]["bone"], "left_lower_arm")
            self.assertGreaterEqual(decoded["roles"]["left_lower_arm"]["confidence"], 0.75)
            self.assertIn("semantic_humanoid_resolver", decoded["roles"]["left_lower_arm"]["evidence"])
            self.assertEqual(decoded["roles"]["left_lower_arm"]["fail_reasons"], [])
            self.assertEqual(decoded["kimodo_projection"]["role_aliases"]["left_lower_arm"], "left_forearm")
            self.assertEqual(decoded["kimodo_projection"]["role_aliases"]["spine"], "torso")
            self.assertEqual(decoded["kimodo_projection"]["probe"]["status"], "not_run")
            self.assertEqual(decoded["status"], "contract_ready")
            self.assertEqual(encoded, dumps_candidate_json(candidate))

    def test_ambiguous_candidate_fails_closed_without_guessing_final_role_bones(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-ambiguous-") as temp_dir:
            root = Path(temp_dir)
            glb = self._write_short_trunk_glb(root / "short.glb")

            candidate = build_candidate_for_glb(glb, source="negative-control")

            self.assertIn(candidate["status"], {"ambiguous", "blocked"})
            self.assertEqual(candidate["roles"]["left_lower_arm"]["bone"], None)
            self.assertEqual(candidate["roles"]["left_lower_arm"]["confidence"], 0.0)
            self.assertIn("semantic_spine_missing", candidate["roles"]["left_lower_arm"]["fail_reasons"])
            self.assertIn("semantic_spine_missing", [item["code"] for item in candidate["diagnostics"]])

    def _write_short_trunk_glb(self, target: Path) -> Path:
        nodes = [
            {"name": "bone_0", "translation": [0.0, 0.0, 0.0], "children": [1, 2, 5]},
            {"name": "bone_1", "translation": [0.0, 0.5, 0.0]},
            {"name": "bone_2", "translation": [-0.2, -0.5, 0.0], "children": [3]},
            {"name": "bone_3", "translation": [0.0, -0.5, 0.0], "children": [4]},
            {"name": "bone_4", "translation": [0.0, -0.2, 0.2]},
            {"name": "bone_5", "translation": [0.2, -0.5, 0.0], "children": [6]},
            {"name": "bone_6", "translation": [0.0, -0.5, 0.0], "children": [7]},
            {"name": "bone_7", "translation": [0.0, -0.2, 0.2]},
        ]
        positions = [(0.0, 0.0, 0.0), (-0.2, -1.0, 0.0), (0.2, -1.0, 0.0)]
        joint_rows = [(0, 0, 0, 0), (4, 0, 0, 0), (7, 0, 0, 0)]
        weight_rows = [(1.0, 0.0, 0.0, 0.0) for _ in positions]
        return _write_glb(target, nodes=nodes, joints=list(range(len(nodes))), positions=positions, joint_rows=joint_rows, weight_rows=weight_rows)


class HumanoidMappingCandidateReadOnlyTests(unittest.TestCase):
    def test_glb_analysis_is_read_only_and_invalid_inputs_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-readonly-") as temp_dir:
            root = Path(temp_dir)
            glb = write_resolver_ready_glb(root / "source.glb")
            before_bytes = glb.read_bytes()
            before_mtime = glb.stat().st_mtime_ns

            candidate = build_candidate_for_glb(glb)

            self.assertEqual(candidate["transforms"]["status"], "available")
            self.assertEqual(candidate["transforms"]["matrix_order"], "row-major")
            self.assertEqual(candidate["symmetry"]["lateral_axis"], "x")
            self.assertGreater(candidate["symmetry"]["left_right_confidence"], 0.0)
            self.assertEqual(glb.read_bytes(), before_bytes)
            self.assertEqual(glb.stat().st_mtime_ns, before_mtime)
            self.assertFalse(glb.with_suffix(".rigmeta.json").exists())

            with self.assertRaisesRegex(CandidateInputError, "input path does not exist"):
                select_candidate_inputs([root / "missing.glb"])
            text = root / "not-glb.txt"
            text.write_text("not glb", encoding="utf-8")
            with self.assertRaisesRegex(CandidateInputError, "explicit input is not a .glb file"):
                select_candidate_inputs([text])
            with self.assertRaisesRegex(CandidateInputError, "not an embedded GLB"):
                build_candidate_for_glb(text)


class HumanoidMappingCandidateKimodoProbeTests(unittest.TestCase):
    def test_complete_source_bvh_fixture_covers_required_kimodo_roles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-complete-source-") as temp_dir:
            source_bvh = _write_complete_source_bvh(Path(temp_dir) / "complete_source.bvh")

            text = source_bvh.read_text(encoding="utf-8")

            for joint_name in (
                "Hips",
                "Spine",
                "Chest",
                "Neck",
                "Head",
                "LeftUpperArm",
                "LeftForeArm",
                "LeftHand",
                "RightUpperArm",
                "RightForeArm",
                "RightHand",
                "LeftUpperLeg",
                "LeftLowerLeg",
                "LeftFoot",
                "RightUpperLeg",
                "RightLowerLeg",
                "RightFoot",
            ):
                self.assertIn(f"JOINT {joint_name}", text.replace("ROOT Hips", "JOINT Hips"))
            self.assertNotIn("/home/drhepa", str(source_bvh))

    def test_probe_unavailable_is_non_fatal_and_keeps_base_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-probe-unavailable-") as temp_dir:
            glb = write_resolver_ready_glb(Path(temp_dir) / "ready.glb")

            candidate = build_candidate_for_glb(glb, kimodo_backend=UnavailableBackend())

            self.assertEqual(candidate["status"], "contract_ready")
            self.assertEqual(candidate["kimodo_projection"]["probe"]["status"], "unavailable")
            self.assertEqual(candidate["kimodo_projection"]["probe"]["primary_failure_layer"], None)
            self.assertNotIn("probe_unavailable", FAILURE_LAYERS)

    def test_probe_results_classify_exactly_one_primary_failure_layer(self) -> None:
        cases = [
            ("accepted", None, "accepted"),
            ("contract_error", "contract_invalid", "rejected"),
            ("MAPPING_INCOMPATIBLE", "mapping_incompatible", "rejected"),
            ("CALIBRATION_UNAVAILABLE", "mapping_incompatible", "rejected"),
            ("retarget_exception", "retarget_failed", "rejected"),
            ("export_invalid", "export_invalid", "rejected"),
            ("validator_failed", "structural_validation_failed", "rejected"),
            ("visual_quality_unknown", "visual_quality_unknown", "accepted_with_unknown_visual_quality"),
        ]
        for code, expected_layer, expected_status in cases:
            with self.subTest(code=code):
                result = classify_probe_failure(code, f"{code} happened")
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.primary_failure_layer, expected_layer)
                populated = [result.primary_failure_layer] if result.primary_failure_layer else []
                self.assertLessEqual(len(populated), 1)

    def test_missing_source_role_calibration_error_is_source_incomplete_not_mapping_incompatible(self) -> None:
        result = classify_probe_failure(
            "CALIBRATION_UNAVAILABLE",
            "CALIBRATION_UNAVAILABLE: fail_closed_reasons=['missing_source_role:left_hand']",
            diagnostics=[{"code": "retarget_calibration", "message": "missing_source_role:left_hand"}],
        )

        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.primary_failure_layer, "source_calibration_incomplete")
        self.assertEqual(result.code, "CALIBRATION_UNAVAILABLE")
        self.assertIn("missing_source_role:left_hand", result.message)

    def test_probe_acceptance_records_disposable_copy_without_source_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-probe-accepted-") as temp_dir:
            root = Path(temp_dir)
            glb = write_resolver_ready_glb(root / "source.glb")
            backend = AcceptedBackend()

            candidate = build_candidate_for_glb(glb, kimodo_backend=backend)

            probe = candidate["kimodo_projection"]["probe"]
            self.assertEqual(probe["status"], "accepted")
            self.assertEqual(probe["primary_failure_layer"], None)
            self.assertEqual(len(backend.seen_source_paths), 1)
            self.assertNotEqual(backend.seen_source_paths[0], glb.resolve())
            self.assertFalse(glb.with_suffix(".rigmeta.json").exists())

    def test_kimodo_backend_executes_inspect_and_build_mapping_from_configured_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-kimodo-root-") as temp_dir:
            root = Path(temp_dir)
            kimodo_root = root / "kimodo"
            _write_fake_kimodo_root(kimodo_root, mode="accepted")
            glb = write_resolver_ready_glb(root / "source.glb")
            backend = KimodoProbeBackend(kimodo_root)

            candidate = build_candidate_for_glb(glb, kimodo_backend=backend)

            probe = candidate["kimodo_projection"]["probe"]
            self.assertEqual(probe["status"], "accepted")
            self.assertEqual(probe["primary_failure_layer"], None)
            self.assertEqual(probe["diagnostics"][0]["mapping_confidence"], "compatible")
            self.assertEqual(probe["diagnostics"][0]["stages"], ["parse_bvh", "inspect_rig", "build_joint_mapping"])
            self.assertFalse(glb.with_suffix(".rigmeta.json").exists())

    def test_kimodo_backend_runs_calibrated_source_sequence_on_disposable_probe_copy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-kimodo-calibrated-") as temp_dir:
            root = Path(temp_dir)
            kimodo_root = root / "kimodo"
            calls_log = root / "probe_calls.json"
            _write_fake_kimodo_root(kimodo_root, mode="accepted", calls_log=calls_log)
            source_bvh = _write_complete_source_bvh(root / "complete_source.bvh")
            probe_output_root = root / "probe-output"
            probe_output_root.mkdir()
            glb = write_resolver_ready_glb(root / "source.glb")
            backend = KimodoProbeBackend(kimodo_root, source_bvh=source_bvh, probe_output_root=probe_output_root)

            candidate = build_candidate_for_glb(glb, kimodo_backend=backend, probe_retarget=True)

            probe = candidate["kimodo_projection"]["probe"]
            self.assertEqual(probe["probe_status"], "accepted_with_unknown_visual_quality")
            self.assertEqual(probe["calibration_status"], "complete")
            self.assertEqual(probe["mapping_confidence"], "calibrated")
            self.assertEqual(probe["chain_coverage"], {"torso": "5/5", "left_arm": "3/3", "right_arm": "3/3", "left_leg": "3/3", "right_leg": "3/3"})
            self.assertEqual(probe["diagnostics"][0]["stages"], ["parse_bvh", "inspect_rig", "build_joint_mapping", "build_retarget_clip"])
            self.assertEqual(probe["source_coverage"]["missing_roles"], [])
            self.assertIn("left_hand", probe["source_coverage"]["present_roles"])
            self.assertFalse(glb.with_suffix(".rigmeta.json").exists())
            staged_sidecars = sorted(probe_output_root.glob("**/*.rigmeta.json"))
            self.assertEqual(len(staged_sidecars), 1)
            calls = json.loads(calls_log.read_text(encoding="utf-8"))
            self.assertEqual([item["stage"] for item in calls], ["parse_bvh", "inspect_rig", "build_joint_mapping", "build_retarget_clip"])
            self.assertEqual(Path(calls[0]["path"]).resolve(), source_bvh.resolve())
            self.assertNotEqual(Path(calls[1]["path"]).resolve(), glb.resolve())

    def test_kimodo_backend_contract_failure_from_inspect_rig_stays_contract_invalid(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-kimodo-contract-invalid-") as temp_dir:
            root = Path(temp_dir)
            kimodo_root = root / "kimodo"
            _write_fake_kimodo_root(kimodo_root, mode="contract_invalid")
            glb = write_resolver_ready_glb(root / "source.glb")

            candidate = build_candidate_for_glb(glb, kimodo_backend=KimodoProbeBackend(kimodo_root))

            probe = candidate["kimodo_projection"]["probe"]
            self.assertEqual(probe["status"], "rejected")
            self.assertEqual(probe["primary_failure_layer"], "contract_invalid")
            self.assertEqual(probe["failure"]["layer"], "contract_invalid")
            self.assertEqual(probe["code"], "INVALID_CANDIDATE")
            self.assertIn("fake target contract failure", probe["message"])
            self.assertEqual(probe["diagnostics"][0]["code"], "INVALID_CANDIDATE")
            self.assertIn("fake target contract failure", probe["diagnostics"][0]["message"])
            self.assertFalse(glb.with_suffix(".rigmeta.json").exists())

    def test_kimodo_backend_rejection_flows_through_disposable_probe_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-kimodo-reject-") as temp_dir:
            root = Path(temp_dir)
            kimodo_root = root / "kimodo"
            _write_fake_kimodo_root(kimodo_root, mode="mapping_incompatible")
            glb = write_resolver_ready_glb(root / "source.glb")

            candidate = build_candidate_for_glb(glb, kimodo_backend=KimodoProbeBackend(kimodo_root))

            probe = candidate["kimodo_projection"]["probe"]
            self.assertEqual(probe["status"], "rejected")
            self.assertEqual(probe["primary_failure_layer"], "mapping_incompatible")
            self.assertEqual(probe["code"], "MAPPING_INCOMPATIBLE")
            self.assertIn("fake incompatible mapping", probe["message"])


class HumanoidMappingCandidateCliTests(unittest.TestCase):
    def test_cli_writes_json_and_jsonl_for_explicit_inputs_and_manifest_only_to_requested_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-cli-") as temp_dir:
            root = Path(temp_dir)
            first = write_resolver_ready_glb(root / "b.glb")
            second = write_resolver_ready_glb(root / "a.glb")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"assets": [str(first), str(second)]}), encoding="utf-8")
            json_out = root / "report.json"
            jsonl_out = root / "report.jsonl"

            result = self._run_cli(["--manifest", str(manifest), "--json-out", str(json_out), "--jsonl-out", str(jsonl_out)])

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["total_assets"], 2)
            self.assertEqual([Path(item["asset"]["path"]).name for item in report["candidates"]], ["a.glb", "b.glb"])
            jsonl_rows = [json.loads(line) for line in jsonl_out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([Path(item["asset"]["path"]).name for item in jsonl_rows], ["a.glb", "b.glb"])
            self.assertFalse(first.with_suffix(".rigmeta.json").exists())
            self.assertFalse(second.with_suffix(".rigmeta.json").exists())

            missing_parent = root / "missing" / "report.json"
            failed = self._run_cli([str(first), "--json-out", str(missing_parent)])
            self.assertEqual(failed.returncode, 2)
            self.assertIn("output parent does not exist", failed.stderr)

    def test_cli_accepts_probe_source_and_output_root_and_fails_loudly_for_bad_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-cli-probe-") as temp_dir:
            root = Path(temp_dir)
            glb = write_resolver_ready_glb(root / "candidate.glb")
            kimodo_root = root / "kimodo"
            _write_fake_kimodo_root(kimodo_root, mode="accepted")
            source_bvh = _write_complete_source_bvh(root / "complete_source.bvh")
            probe_output_root = root / "probe-output"
            probe_output_root.mkdir()
            json_out = root / "report.json"

            ok = self._run_cli([
                str(glb),
                "--json-out",
                str(json_out),
                "--kimodo-root",
                str(kimodo_root),
                "--source-bvh",
                str(source_bvh),
                "--probe-output-root",
                str(probe_output_root),
                "--probe-retarget",
            ])
            self.assertEqual(ok.returncode, 0, ok.stderr)
            probe = json.loads(json_out.read_text(encoding="utf-8"))["candidates"][0]["kimodo_projection"]["probe"]
            self.assertEqual(probe["probe_status"], "accepted_with_unknown_visual_quality")
            self.assertEqual(probe["calibration_status"], "complete")
            self.assertFalse(glb.with_suffix(".rigmeta.json").exists())

            bad_source = self._run_cli([str(glb), "--json-out", str(root / "bad-source.json"), "--kimodo-root", str(kimodo_root), "--source-bvh", str(root / "missing.bvh")])
            self.assertEqual(bad_source.returncode, 2)
            self.assertIn("source BVH path does not exist", bad_source.stderr)
            self.assertIn("missing.bvh", bad_source.stderr)

            bad_root = self._run_cli([str(glb), "--json-out", str(root / "bad-root.json"), "--kimodo-root", str(kimodo_root), "--source-bvh", str(source_bvh), "--probe-output-root", str(root / "missing-output-root")])
            self.assertEqual(bad_root.returncode, 2)
            self.assertIn("probe output root does not exist", bad_root.stderr)

    def test_core_writers_emit_sorted_json_and_jsonl_to_existing_parents(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-writers-") as temp_dir:
            root = Path(temp_dir)
            paths = [write_resolver_ready_glb(root / "z.glb"), write_resolver_ready_glb(root / "a.glb")]
            candidates = build_candidate_reports(paths)["candidates"]
            json_out = root / "candidates.json"
            jsonl_out = root / "candidates.jsonl"

            write_candidates_json(candidates, json_out)
            write_candidates_jsonl(candidates, jsonl_out)

            decoded = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual([Path(item["asset"]["path"]).name for item in decoded["candidates"]], ["a.glb", "z.glb"])
            self.assertEqual(len(jsonl_out.read_text(encoding="utf-8").splitlines()), 2)
            with self.assertRaisesRegex(CandidateOutputError, "output parent does not exist"):
                write_candidates_json(candidates, root / "missing" / "bad.json")

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "unirig_ext.humanoid_mapping_candidates_cli", *args],
            cwd=ROOT,
            env={"PYTHONPATH": str(SRC)},
            text=True,
            capture_output=True,
            check=False,
        )


class HumanoidMappingCandidatePhase4Tests(unittest.TestCase):
    def test_representative_corpus_manifest_is_env_gated_and_covers_expected_asset_groups(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-corpus-gate-") as temp_dir:
            root = Path(temp_dir)
            for group in REPRESENTATIVE_CORPUS_ASSETS.values():
                for name in group:
                    (root / name).write_bytes(b"glTF placeholder")

            manifest = build_representative_corpus_manifest(root)

            self.assertEqual(manifest["env_var"], REAL_CORPUS_ENV_VAR)
            self.assertEqual(manifest["root"], str(root.resolve()))
            self.assertEqual(manifest["total_assets"], 11)
            self.assertEqual(set(manifest["groups"]), {"direct_ok", "resolver_ok_direct_fail", "ambiguous", "negative_controls"})
            self.assertEqual(len(manifest["groups"]["direct_ok"]), 2)
            self.assertEqual(len(manifest["groups"]["resolver_ok_direct_fail"]), 4)
            self.assertEqual(len(manifest["groups"]["ambiguous"]), 3)
            self.assertEqual(len(manifest["groups"]["negative_controls"]), 2)
            self.assertEqual([Path(path).name for path in manifest["assets"]], sorted(Path(path).name for path in manifest["assets"]))

            missing = root / REPRESENTATIVE_CORPUS_ASSETS["negative_controls"][0]
            missing.unlink()
            with self.assertRaisesRegex(CandidateInputError, "representative corpus asset is missing"):
                build_representative_corpus_manifest(root)

    def test_full_topology_sufficient_manifest_covers_all_34_expected_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-map-full-corpus-") as temp_dir:
            root = Path(temp_dir)
            for name in FULL_TOPOLOGY_SUFFICIENT_ASSETS:
                (root / name).write_bytes(b"glTF placeholder")

            manifest = build_full_topology_sufficient_corpus_manifest(root)

            self.assertEqual(manifest["env_var"], REAL_CORPUS_ENV_VAR)
            self.assertEqual(manifest["total_assets"], 34)
            self.assertEqual([Path(path).name for path in manifest["assets"]], sorted(FULL_TOPOLOGY_SUFFICIENT_ASSETS))
            self.assertEqual(manifest["topology_sufficient"], True)

    def test_run_suggestions_are_deterministic_and_keep_outputs_under_tmp_opencode(self) -> None:
        suggestions = render_run_suggestions(exports_root="/exports", output_root="/tmp/opencode/unirig-map-candidates")

        self.assertIn("python3 -m unittest discover -s tests -p test_humanoid_mapping_candidates.py -v", suggestions)
        self.assertIn(REAL_CORPUS_ENV_VAR, suggestions)
        self.assertIn("/tmp/opencode/unirig-map-candidates", suggestions)
        self.assertIn("python3 -m unirig_ext.humanoid_mapping_candidates_cli", suggestions)
        self.assertNotIn("pytest", suggestions)
        self.assertNotIn("/exports/report", suggestions)

    @unittest.skipUnless(os.environ.get(REAL_CORPUS_ENV_VAR), f"set {REAL_CORPUS_ENV_VAR} to run read-only real export corpus validation")
    def test_env_gated_real_corpus_classifies_representative_exports_read_only(self) -> None:
        exports_root = Path(os.environ[REAL_CORPUS_ENV_VAR]).expanduser().resolve()
        manifest = build_representative_corpus_manifest(exports_root)
        before = {Path(path): (Path(path).read_bytes(), Path(path).stat().st_mtime_ns) for path in manifest["assets"]}

        report = build_candidate_reports(manifest["assets"], source="env-gated-real-corpus")

        self.assertEqual(report["summary"]["total_assets"], 11)
        by_name = {Path(item["asset"]["path"]).name: item for item in report["candidates"]}
        for name in REPRESENTATIVE_CORPUS_ASSETS["direct_ok"] + REPRESENTATIVE_CORPUS_ASSETS["resolver_ok_direct_fail"]:
            self.assertIn(by_name[name]["status"], {"candidate", "contract_ready"})
            self.assertEqual(by_name[name]["kimodo_projection"]["probe"]["status"], "not_run")
        for name in REPRESENTATIVE_CORPUS_ASSETS["ambiguous"]:
            self.assertIn(by_name[name]["status"], {"ambiguous", "blocked"})
        for name in REPRESENTATIVE_CORPUS_ASSETS["negative_controls"]:
            self.assertEqual(by_name[name]["status"], "blocked")
        for path, (content, mtime_ns) in before.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, mtime_ns)
            self.assertFalse(path.with_suffix(".rigmeta.json").exists())

    @unittest.skipUnless(os.environ.get(REAL_CORPUS_ENV_VAR), f"set {REAL_CORPUS_ENV_VAR} to run read-only 34-export corpus validation")
    def test_env_gated_full_topology_sufficient_corpus_reports_34_assets_read_only(self) -> None:
        exports_root = Path(os.environ[REAL_CORPUS_ENV_VAR]).expanduser().resolve()
        manifest = build_full_topology_sufficient_corpus_manifest(exports_root)
        before = {Path(path): (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in manifest["assets"]}

        report = build_candidate_reports(manifest["assets"], source="env-gated-full-topology-sufficient-corpus")

        self.assertEqual(report["summary"]["total_assets"], 34)
        self.assertEqual(len(report["candidates"]), 34)
        self.assertEqual([Path(item["asset"]["path"]).name for item in report["candidates"]], sorted(FULL_TOPOLOGY_SUFFICIENT_ASSETS))
        for candidate in report["candidates"]:
            self.assertIn(candidate["status"], {"candidate", "contract_ready", "ambiguous", "blocked"})
            self.assertGreater(candidate["topology"]["joint_count"], 0)
            self.assertIn("hierarchy", candidate["topology"])
            self.assertEqual(candidate["kimodo_projection"]["probe"]["status"], "not_run")
        for path, (size, mtime_ns) in before.items():
            self.assertEqual(path.stat().st_size, size)
            self.assertEqual(path.stat().st_mtime_ns, mtime_ns)
            self.assertFalse(path.with_suffix(".rigmeta.json").exists())


class UnavailableBackend:
    def available(self) -> bool:
        return False

    def probe(self, *_args: object, **_kwargs: object) -> ProbeResult:
        raise AssertionError("unavailable backend must not be probed")


class AcceptedBackend:
    def __init__(self) -> None:
        self.seen_source_paths: list[Path] = []

    def available(self) -> bool:
        return True

    def probe(self, copied_glb: Path, _sidecar_payload: dict, *, probe_retarget: bool = False) -> ProbeResult:
        self.seen_source_paths.append(copied_glb.resolve())
        self.assert_sidecar_exists(copied_glb)
        return ProbeResult(status="accepted", primary_failure_layer=None, code=None, message="accepted", diagnostics=[{"code": "kimodo_probe_accepted"}])

    def assert_sidecar_exists(self, copied_glb: Path) -> None:
        if not copied_glb.with_suffix(".rigmeta.json").exists():
            raise AssertionError("probe sidecar was not written beside disposable copy")


def _write_complete_source_bvh(path: Path) -> Path:
    path.write_text(
        """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.0 10.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Chest
    {
      OFFSET 0.0 10.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT Neck
      {
        OFFSET 0.0 5.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Head
        {
          OFFSET 0.0 5.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
            OFFSET 0.0 3.0 0.0
          }
        }
      }
      JOINT LeftUpperArm
      {
        OFFSET -5.0 4.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftForeArm
        {
          OFFSET -8.0 0.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT LeftHand
          {
            OFFSET -6.0 0.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
              OFFSET -2.0 0.0 0.0
            }
          }
        }
      }
      JOINT RightUpperArm
      {
        OFFSET 5.0 4.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightForeArm
        {
          OFFSET 8.0 0.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT RightHand
          {
            OFFSET 6.0 0.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
              OFFSET 2.0 0.0 0.0
            }
          }
        }
      }
    }
  }
  JOINT LeftUpperLeg
  {
    OFFSET -3.0 -8.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLowerLeg
    {
      OFFSET 0.0 -10.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftFoot
      {
        OFFSET 0.0 -8.0 3.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.0 -2.0 3.0
        }
      }
    }
  }
  JOINT RightUpperLeg
  {
    OFFSET 3.0 -8.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLowerLeg
    {
      OFFSET 0.0 -10.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightFoot
      {
        OFFSET 0.0 -8.0 3.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.0 -2.0 3.0
        }
      }
    }
  }
}
MOTION
Frames: 1
Frame Time: 0.0333333
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
""",
        encoding="utf-8",
    )
    return path


def _write_fake_kimodo_root(root: Path, *, mode: str, calls_log: Path | None = None) -> None:
    (root / "tests" / "fixtures").mkdir(parents=True)
    (root / "tests" / "fixtures" / "synthetic_walk.bvh").write_text("fake bvh", encoding="utf-8")
    calls_path = str(calls_log) if calls_log is not None else ""
    mapping_confidence = "calibrated" if calls_log is not None else "compatible"
    (root / "probe_calls.py").write_text(
        f"import json\n"
        f"from pathlib import Path\n"
        f"CALLS_PATH = {calls_path!r}\n"
        f"def record(stage, **payload):\n"
        f"    if not CALLS_PATH:\n"
        f"        return\n"
        f"    path = Path(CALLS_PATH)\n"
        f"    calls = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []\n"
        f"    calls.append(dict(stage=stage, **payload))\n"
        f"    path.write_text(json.dumps(calls, sort_keys=True), encoding='utf-8')\n",
        encoding="utf-8",
    )
    (root / "retarget_errors.py").write_text(
        "class RetargetError(RuntimeError):\n"
        "    def __init__(self, code, message, diagnostics=None):\n"
        "        super().__init__(f'{code}: {message}')\n"
        "        self.code = code\n"
        "        self.message = message\n"
        "        self.diagnostics = diagnostics or {}\n"
        "MAPPING_INCOMPATIBLE = 'MAPPING_INCOMPATIBLE'\n",
        encoding="utf-8",
    )
    (root / "kimodo_bvh.py").write_text(
        "from probe_calls import record\n"
        "class Motion:\n"
        "    joints = {'Hips': object(), 'Spine': object(), 'Chest': object(), 'Neck': object(), 'Head': object(), 'LeftUpperArm': object(), 'LeftForeArm': object(), 'LeftHand': object(), 'RightUpperArm': object(), 'RightForeArm': object(), 'RightHand': object(), 'LeftUpperLeg': object(), 'LeftLowerLeg': object(), 'LeftFoot': object(), 'RightUpperLeg': object(), 'RightLowerLeg': object(), 'RightFoot': object()}\n"
        "def parse_bvh(path):\n"
        "    record('parse_bvh', path=str(path))\n"
        "    return Motion()\n",
        encoding="utf-8",
    )
    (root / "rig_inspector.py").write_text(
        "import json\n"
        "from probe_calls import record\n"
        "class Rig:\n"
        "    joint_names = ('hips', 'spine', 'chest', 'neck', 'head', 'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_leg', 'left_lower_leg', 'left_foot', 'right_upper_leg', 'right_lower_leg', 'right_foot')\n"
        "    humanoid_diagnostics = ('humanoid_contract_validated',)\n"
        "def inspect_rig(path, source_kind='unknown_mesh_source'):\n"
        "    record('inspect_rig', path=str(path), source_kind=source_kind)\n"
        "    sidecar = path.with_suffix('.rigmeta.json')\n"
        "    payload = json.loads(sidecar.read_text(encoding='utf-8'))\n"
        "    if payload.get('humanoid_contract', {}).get('schema') != 'modly.humanoid.v1':\n"
        "        raise RuntimeError('contract was not staged beside disposable copy')\n"
        + ("    raise RuntimeError('INVALID_CANDIDATE: fake target contract failure from inspect_rig')\n" if mode == "contract_invalid" else "")
        + "    return Rig()\n",
        encoding="utf-8",
    )
    build_mapping = (
        "from retarget_errors import RetargetError, MAPPING_INCOMPATIBLE\n"
        "class Mapping:\n"
        f"    confidence = {mapping_confidence!r}\n"
        "    warnings = ('fake_mapping',)\n"
        "    pairs = {'Hips': 'hips'}\n"
        "    coverage = {'torso': '5/5', 'left_arm': '3/3', 'right_arm': '3/3', 'left_leg': '3/3', 'right_leg': '3/3'}\n"
        "class Clip:\n"
        "    calibration = {'chain_coverage': {'torso': '5/5', 'left_arm': '3/3', 'right_arm': '3/3', 'left_leg': '3/3', 'right_leg': '3/3'}, 'version': 'kimodo.retarget_calibration.v1'}\n"
        "    channel_count = 18\n"
        "def build_joint_mapping(source, target):\n"
    )
    if mode == "mapping_incompatible":
        build_mapping += "    raise RetargetError(MAPPING_INCOMPATIBLE, 'fake incompatible mapping')\n"
    else:
        build_mapping += "    from probe_calls import record\n    record('build_joint_mapping')\n    return Mapping()\n"
    build_mapping += "def build_retarget_clip(source, target):\n    from probe_calls import record\n    record('build_retarget_clip')\n    return Clip()\n"
    (root / "retargeting.py").write_text(build_mapping, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
