# pyright: reportMissingImports=false

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext import bootstrap


def load_setup_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("unirig_setup_module", SETUP)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load setup.py module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_config_relpath() -> Path:
    module = load_setup_module()
    return module.WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH


def module_skin_config_relpath() -> Path:
    module = load_setup_module()
    return module.WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH


def module_model_parse_relpath() -> Path:
    module = load_setup_module()
    return module.RUNTIME_MODEL_PARSE_RELATIVE_PATH


def module_unirig_skin_relpath() -> Path:
    module = load_setup_module()
    return module.RUNTIME_UNIRIG_SKIN_RELATIVE_PATH


class SetupBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unirig-setup-"))
        self.ext_dir = self.temp_dir / "extension-root"
        self.ext_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_minimal_upstream_tree(self, root: Path) -> None:
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "configs" / "model").mkdir(parents=True, exist_ok=True)
        (root / "configs" / "task").mkdir(parents=True, exist_ok=True)
        (root / "run.py").write_text("print('ok')\n", encoding="utf-8")
        (root / "requirements.txt").write_text("trimesh\n", encoding="utf-8")
        (root / module_config_relpath()).write_text(
            "model:\n  _attn_implementation: flash_attention_2\n  torch_dtype: float32\n",
            encoding="utf-8",
        )
        (root / module_skin_config_relpath()).write_text(
            "mesh_encoder:\n  enable_flash: true\n  width: 256\n",
            encoding="utf-8",
        )

    def test_resolve_source_defaults_to_pinned_upstream_ref(self) -> None:
        module = load_setup_module()

        source, source_ref = module._resolve_source({})

        self.assertEqual(source_ref, bootstrap.UPSTREAM_REF_DEFAULT)
        self.assertIn(bootstrap.UPSTREAM_REF_DEFAULT, source)

    def test_archive_url_requires_full_commit_sha(self) -> None:
        module = load_setup_module()

        with self.assertRaises(RuntimeError) as ctx:
            module._archive_url_for_ref("a6a4e2d")

        self.assertIn("immutable upstream commit ref", str(ctx.exception))
        full_ref = "a6a4e2d6c23b88eb79b4396c0bae558aaad4744b"
        self.assertIn(full_ref, module._archive_url_for_ref(full_ref))

    def test_resolve_linux_arm64_blender_candidate_prefers_override_over_path_visible_blender(self) -> None:
        module = load_setup_module()
        override = self.temp_dir / "override-blender"
        override.write_text("#!/bin/sh\n", encoding="utf-8")

        with mock.patch.dict(module.os.environ, {}, clear=True), mock.patch.object(
            module.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/blender" if name == "blender" else None,
        ):
            candidate = module._resolve_linux_arm64_blender_candidate({"blender_exe": str(override)})

        self.assertEqual(candidate["source"], "override")
        self.assertEqual(candidate["path"], str(override.resolve()))
        self.assertIn("override", str(candidate["selected_because"]).lower())

    def test_resolve_linux_arm64_blender_candidate_falls_back_to_default_path_name_without_override(self) -> None:
        module = load_setup_module()

        with mock.patch.dict(module.os.environ, {}, clear=True), mock.patch.object(
            module.shutil,
            "which",
            side_effect=lambda name: "/opt/blender/blender" if name == "blender" else None,
        ):
            candidate = module._resolve_linux_arm64_blender_candidate({})

        self.assertEqual(candidate["source"], "path")
        self.assertEqual(candidate["path"], "/opt/blender/blender")
        self.assertIn("blender", str(candidate["selected_because"]).lower())

    def test_probe_linux_arm64_blender_bpy_returns_stable_success_payload(self) -> None:
        module = load_setup_module()
        blender_exe = Path("/opt/blender/blender")
        probe_marker = module.LINUX_ARM64_BLENDER_PROBE_MARKER
        stdout = "Blender 4.0.2\nboot line\n" + probe_marker + '{"blender_version": "4.0.2", "python_version": "3.12.1", "smoke_result": "passed"}\n'

        with mock.patch.object(
            module.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=[str(blender_exe)],
                returncode=0,
                stdout=stdout,
                stderr="warn line\nsecond warn\n",
            ),
        ) as run_mock:
            probe = module._probe_linux_arm64_blender_bpy(blender_exe)

        self.assertEqual(probe["status"], "ok")
        self.assertEqual(probe["blender_version"], "4.0.2")
        self.assertEqual(probe["python_version"], "3.12.1")
        self.assertEqual(probe["smoke_result"], "passed")
        self.assertEqual(probe["returncode"], 0)
        self.assertEqual(probe["stdout_tail"], ["Blender 4.0.2", "boot line", probe_marker + '{"blender_version": "4.0.2", "python_version": "3.12.1", "smoke_result": "passed"}'])
        self.assertEqual(probe["stderr_tail"], ["warn line", "second warn"])
        self.assertEqual(probe["command"][:3], [str(blender_exe), "--background", "--factory-startup"])
        self.assertIn("--python-expr", probe["command"])
        run_mock.assert_called_once()

    def test_probe_linux_arm64_blender_bpy_returns_stable_error_payload_on_nonzero_exit(self) -> None:
        module = load_setup_module()
        blender_exe = Path("/opt/blender/blender")
        probe_marker = module.LINUX_ARM64_BLENDER_PROBE_MARKER
        stdout = probe_marker + '{"blender_version": "4.0.2", "python_version": "3.12.1", "smoke_result": "failed"}\n'

        with mock.patch.object(
            module.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=[str(blender_exe)],
                returncode=7,
                stdout=stdout,
                stderr="Traceback line 1\nTraceback line 2\n",
            ),
        ):
            probe = module._probe_linux_arm64_blender_bpy(blender_exe)

        self.assertEqual(probe["status"], "error")
        self.assertEqual(probe["blender_version"], "4.0.2")
        self.assertEqual(probe["python_version"], "3.12.1")
        self.assertEqual(probe["smoke_result"], "failed")
        self.assertEqual(probe["returncode"], 7)
        self.assertEqual(probe["stdout_tail"], [probe_marker + '{"blender_version": "4.0.2", "python_version": "3.12.1", "smoke_result": "failed"}'])
        self.assertEqual(probe["stderr_tail"], ["Traceback line 1", "Traceback line 2"])

    def test_probe_linux_arm64_blender_bpy_returns_stable_error_payload_on_launch_failure(self) -> None:
        module = load_setup_module()
        blender_exe = Path("/opt/blender/blender")

        with mock.patch.object(module.subprocess, "run", side_effect=OSError("permission denied")):
            probe = module._probe_linux_arm64_blender_bpy(blender_exe)

        self.assertEqual(probe["status"], "error")
        self.assertEqual(probe["blender_version"], "")
        self.assertEqual(probe["python_version"], "")
        self.assertEqual(probe["smoke_result"], "launch-failed")
        self.assertIsNone(probe["returncode"])
        self.assertEqual(probe["stdout_tail"], [])
        self.assertEqual(probe["stderr_tail"], ["permission denied"])

    def test_classify_linux_arm64_bpy_viability_marks_missing_blender_as_missing(self) -> None:
        module = load_setup_module()

        classification = module._classify_linux_arm64_bpy_viability(
            {
                "source": "missing",
                "path": "",
                "selected_because": "No PATH-visible Blender candidate.",
            },
            None,
        )

        blockers = {item["code"]: item for item in classification["blockers"]}

        self.assertEqual(classification["status"], "missing")
        self.assertFalse(classification["ready"])
        self.assertEqual(classification["evidence_kind"], "external-blender")
        self.assertEqual(classification["boundary"], "environment")
        self.assertEqual(classification["owner"], "environment")
        self.assertEqual(classification["blocker_codes"], ["blender-executable-missing"])
        self.assertEqual(blockers["blender-executable-missing"]["boundary"], "environment")
        self.assertEqual(blockers["blender-executable-missing"]["owner"], "environment")

    def test_classify_linux_arm64_bpy_viability_marks_python_mismatch_as_discovered_incompatible(self) -> None:
        module = load_setup_module()

        classification = module._classify_linux_arm64_bpy_viability(
            {
                "source": "path",
                "path": "/opt/blender/blender",
                "selected_because": "Selected PATH-visible Blender candidate.",
            },
            {
                "status": "ok",
                "command": ["/opt/blender/blender", "--background"],
                "blender_version": "4.0.2",
                "python_version": "3.12.1",
                "smoke_result": "passed",
                "returncode": 0,
                "stdout_tail": ["probe ok"],
                "stderr_tail": [],
            },
            wrapper_python_version="3.11.9",
        )

        blockers = {item["code"]: item for item in classification["blockers"]}

        self.assertEqual(classification["status"], "discovered-incompatible")
        self.assertFalse(classification["ready"])
        self.assertEqual(classification["boundary"], "upstream")
        self.assertEqual(classification["owner"], "upstream-package")
        self.assertEqual(classification["blocker_codes"], ["blender-python-abi-mismatch"])
        self.assertEqual(blockers["blender-python-abi-mismatch"]["boundary"], "upstream")
        self.assertEqual(blockers["blender-python-abi-mismatch"]["owner"], "upstream-package")
        self.assertIn("3.12.1", blockers["blender-python-abi-mismatch"]["message"])
        self.assertIn("3.11", blockers["blender-python-abi-mismatch"]["message"])

    def test_classify_linux_arm64_bpy_viability_marks_matching_external_smoke_as_external_smoke_ready(self) -> None:
        module = load_setup_module()

        classification = module._classify_linux_arm64_bpy_viability(
            {
                "source": "override",
                "path": "/custom/blender",
                "selected_because": "Selected explicit override.",
            },
            {
                "status": "ok",
                "command": ["/custom/blender", "--background"],
                "blender_version": "3.6.9",
                "python_version": "3.11.8",
                "smoke_result": "passed",
                "returncode": 0,
                "stdout_tail": ["probe ok"],
                "stderr_tail": [],
            },
            wrapper_python_version="3.11.9",
        )

        self.assertEqual(classification["status"], "external-bpy-smoke-ready")
        self.assertFalse(classification["ready"])
        self.assertEqual(classification["evidence_kind"], "external-blender")
        self.assertEqual(classification["boundary"], "wrapper")
        self.assertEqual(classification["owner"], "wrapper")
        self.assertEqual(classification["blocker_codes"], [])
        self.assertEqual(classification["blockers"], [])

    def test_classify_linux_arm64_bpy_viability_marks_probe_errors_as_error(self) -> None:
        module = load_setup_module()

        classification = module._classify_linux_arm64_bpy_viability(
            {
                "source": "path",
                "path": "/opt/blender/blender",
                "selected_because": "Selected PATH-visible Blender candidate.",
            },
            {
                "status": "error",
                "command": ["/opt/blender/blender", "--background"],
                "blender_version": "",
                "python_version": "",
                "smoke_result": "launch-failed",
                "returncode": None,
                "stdout_tail": [],
                "stderr_tail": ["permission denied"],
            },
        )

        blockers = {item["code"]: item for item in classification["blockers"]}

        self.assertEqual(classification["status"], "error")
        self.assertFalse(classification["ready"])
        self.assertEqual(classification["boundary"], "wrapper")
        self.assertEqual(classification["owner"], "wrapper")
        self.assertEqual(classification["blocker_codes"], ["blender-bpy-smoke-error"])
        self.assertEqual(blockers["blender-bpy-smoke-error"]["boundary"], "wrapper")
        self.assertEqual(blockers["blender-bpy-smoke-error"]["owner"], "wrapper")
        self.assertIn("permission denied", "\n".join(blockers["blender-bpy-smoke-error"].get("details", [])))

    def test_linux_arm64_bpy_evidence_matrix_maps_each_case_to_exactly_one_allowed_class(self) -> None:
        module = load_setup_module()

        cases = [
            (
                "override precedence",
                {"blender_exe": "/custom/blender"},
                {
                    "source": "override",
                    "path": "/custom/blender",
                    "selected_because": "Selected explicit override before PATH fallback.",
                },
                {
                    "status": "ok",
                    "command": ["/custom/blender", "--background"],
                    "blender_version": "3.6.9",
                    "python_version": "3.11.8",
                    "smoke_result": "passed",
                    "returncode": 0,
                    "stdout_tail": ["probe ok"],
                    "stderr_tail": [],
                },
                "external-bpy-smoke-ready",
            ),
            (
                "missing Blender",
                {},
                {
                    "source": "missing",
                    "path": "",
                    "selected_because": "No PATH-visible Blender candidate.",
                },
                {},
                "missing",
            ),
            (
                "python mismatch",
                {},
                {
                    "source": "path",
                    "path": "/opt/blender/blender",
                    "selected_because": "Selected PATH-visible Blender candidate.",
                },
                {
                    "status": "ok",
                    "command": ["/opt/blender/blender", "--background"],
                    "blender_version": "4.0.2",
                    "python_version": "3.12.1",
                    "smoke_result": "passed",
                    "returncode": 0,
                    "stdout_tail": ["probe ok"],
                    "stderr_tail": [],
                },
                "discovered-incompatible",
            ),
            (
                "smoke-ready-but-blocked",
                {"blender_exe": "/custom/blender"},
                {
                    "source": "override",
                    "path": "/custom/blender",
                    "selected_because": "Selected explicit override before PATH fallback.",
                },
                {
                    "status": "ok",
                    "command": ["/custom/blender", "--background"],
                    "blender_version": "3.6.9",
                    "python_version": "3.11.8",
                    "smoke_result": "passed",
                    "returncode": 0,
                    "stdout_tail": ["probe ok"],
                    "stderr_tail": [],
                },
                "external-bpy-smoke-ready",
            ),
            (
                "subprocess error",
                {},
                {
                    "source": "path",
                    "path": "/opt/blender/blender",
                    "selected_because": "Selected PATH-visible Blender candidate.",
                },
                {
                    "status": "error",
                    "command": ["/opt/blender/blender", "--background"],
                    "blender_version": "",
                    "python_version": "",
                    "smoke_result": "launch-failed",
                    "returncode": None,
                    "stdout_tail": [],
                    "stderr_tail": ["permission denied"],
                },
                "error",
            ),
        ]

        for label, payload, candidate, probe, expected_status in cases:
            with self.subTest(label=label):
                with mock.patch.object(module, "_resolve_linux_arm64_blender_candidate", return_value=candidate, create=True), mock.patch.object(
                    module, "_probe_linux_arm64_blender_bpy", return_value=probe, create=True
                ) as probe_mock:
                    evidence = module._linux_arm64_bpy_evidence(payload, wrapper_python_version="3.11.9")

                classification = evidence["classification"]
                self.assertEqual(classification["status"], expected_status)
                self.assertIn(classification["status"], module.LINUX_ARM64_BPY_EVIDENCE_CLASSES)
                self.assertEqual(sum(1 for status in module.LINUX_ARM64_BPY_EVIDENCE_CLASSES if status == classification["status"]), 1)
                self.assertEqual(evidence["candidate"]["source"], candidate["source"])
                if label == "override precedence":
                    self.assertEqual(evidence["candidate"]["source"], "override")
                    self.assertEqual(evidence["candidate"]["path"], "/custom/blender")
                    self.assertIn("override", evidence["candidate"]["selected_because"].lower())
                if label == "missing Blender":
                    probe_mock.assert_not_called()
                    self.assertEqual(evidence["probe"], {})
                else:
                    probe_mock.assert_called_once_with(Path(candidate["path"]))

    def test_linux_arm64_bpy_evidence_boundary_ownership_matches_case_and_smoke_ready_stays_blocked(self) -> None:
        module = load_setup_module()

        cases = [
            (
                "missing Blender",
                {
                    "source": "missing",
                    "path": "",
                    "selected_because": "No PATH-visible Blender candidate.",
                },
                None,
                "missing",
                "environment",
                "environment",
                "blender-executable-missing",
            ),
            (
                "python mismatch",
                {
                    "source": "path",
                    "path": "/opt/blender/blender",
                    "selected_because": "Selected PATH-visible Blender candidate.",
                },
                {
                    "status": "ok",
                    "command": ["/opt/blender/blender", "--background"],
                    "blender_version": "4.0.2",
                    "python_version": "3.12.1",
                    "smoke_result": "passed",
                    "returncode": 0,
                    "stdout_tail": ["probe ok"],
                    "stderr_tail": [],
                },
                "discovered-incompatible",
                "upstream",
                "upstream-package",
                "blender-python-abi-mismatch",
            ),
            (
                "subprocess error",
                {
                    "source": "path",
                    "path": "/opt/blender/blender",
                    "selected_because": "Selected PATH-visible Blender candidate.",
                },
                {
                    "status": "error",
                    "command": ["/opt/blender/blender", "--background"],
                    "blender_version": "",
                    "python_version": "",
                    "smoke_result": "launch-failed",
                    "returncode": None,
                    "stdout_tail": [],
                    "stderr_tail": ["permission denied"],
                },
                "error",
                "wrapper",
                "wrapper",
                "blender-bpy-smoke-error",
            ),
        ]

        for label, candidate, probe, expected_status, expected_boundary, expected_owner, expected_blocker_code in cases:
            with self.subTest(label=label):
                classification = module._classify_linux_arm64_bpy_viability(
                    candidate,
                    probe,
                    wrapper_python_version="3.11.9",
                )

                self.assertEqual(classification["status"], expected_status)
                self.assertEqual(classification["boundary"], expected_boundary)
                self.assertEqual(classification["owner"], expected_owner)
                self.assertEqual(classification["blocker_codes"], [expected_blocker_code])

        normalized = bootstrap.normalize_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "python_version": "3.11.9",
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "spconv": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "bpy": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "candidate": {
                                "source": "override",
                                "path": "/custom/blender",
                                "selected_because": "Selected explicit override before PATH fallback.",
                            },
                            "blender": {"version": "3.6.9", "python_version": "3.11.8"},
                            "verification": "blender-background-bpy-smoke",
                            "checks": [{"label": "bpy", "status": "external-bpy-smoke-ready", "returncode": 0}],
                            "blockers": [],
                            "blocker_codes": [],
                            "boundary": "wrapper",
                            "owner": "wrapper",
                        },
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "deferred_work": [],
                    "non_blender_runtime_ready": True,
                    "bpy_evidence_class": "external-bpy-smoke-ready",
                    "external_blender": {
                        "candidate": {
                            "source": "override",
                            "path": "/custom/blender",
                            "selected_because": "Selected explicit override before PATH fallback.",
                        },
                        "probe": {
                            "status": "ok",
                            "command": ["/custom/blender", "--background"],
                            "blender_version": "3.6.9",
                            "python_version": "3.11.8",
                            "smoke_result": "passed",
                            "returncode": 0,
                            "stdout_tail": ["probe ok"],
                            "stderr_tail": [],
                        },
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "blockers": [],
                            "blocker_codes": [],
                            "boundary": "wrapper",
                            "owner": "wrapper",
                        },
                    },
                },
            },
            self.ext_dir,
        )

        self.assertEqual(normalized["install_state"], "blocked")
        self.assertFalse(normalized["last_verification"]["runtime_ready"])
        smoke_blocker = next(item for item in normalized["source_build"]["blockers"] if item["code"] == "external-bpy-evidence-only")
        self.assertEqual(normalized["source_build"]["bpy_evidence_class"], "external-bpy-smoke-ready")
        self.assertEqual(smoke_blocker["boundary"], "wrapper")
        self.assertEqual(smoke_blocker["owner"], "wrapper")

    def test_prepare_runtime_source_reuses_matching_staged_runtime(self) -> None:
        module = load_setup_module()
        source_dir = self.temp_dir / "upstream-source"
        self._write_minimal_upstream_tree(source_dir)
        staged_unirig_dir = module._runtime_unirig_dir(self.ext_dir)
        self._write_minimal_upstream_tree(staged_unirig_dir)
        sentinel = staged_unirig_dir / "reuse-sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        module._runtime_stage_manifest_path(self.ext_dir).write_text(
            json.dumps(
                {
                    "vendor_source": "local-directory",
                    "source": str(source_dir.resolve()),
                    "source_ref": "directory",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        with mock.patch.object(module, "_resolve_source", return_value=(str(source_dir), "directory")), mock.patch.object(
            module.shutil, "copytree", wraps=module.shutil.copytree
        ) as copytree_mock:
            unirig_dir, vendor_source, source_ref = module._prepare_runtime_source(self.ext_dir, {})

        self.assertEqual(unirig_dir, staged_unirig_dir)
        self.assertEqual(vendor_source, "local-directory")
        self.assertEqual(source_ref, "directory")
        self.assertTrue(sentinel.exists())
        copytree_mock.assert_not_called()

    def test_prepare_runtime_source_restages_when_matching_stage_is_incomplete(self) -> None:
        module = load_setup_module()
        source_dir = self.temp_dir / "upstream-source"
        self._write_minimal_upstream_tree(source_dir)
        staged_unirig_dir = module._runtime_unirig_dir(self.ext_dir)
        staged_unirig_dir.mkdir(parents=True, exist_ok=True)
        (staged_unirig_dir / "run.py").write_text("stale\n", encoding="utf-8")
        module._runtime_stage_manifest_path(self.ext_dir).write_text(
            json.dumps(
                {
                    "vendor_source": "local-directory",
                    "source": str(source_dir.resolve()),
                    "source_ref": "directory",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        with mock.patch.object(module, "_resolve_source", return_value=(str(source_dir), "directory")), mock.patch.object(
            module.shutil, "copytree", wraps=module.shutil.copytree
        ) as copytree_mock:
            unirig_dir, vendor_source, source_ref = module._prepare_runtime_source(self.ext_dir, {})

        self.assertEqual(unirig_dir, staged_unirig_dir)
        self.assertEqual(vendor_source, "local-directory")
        self.assertEqual(source_ref, "directory")
        self.assertTrue((staged_unirig_dir / "requirements.txt").exists())
        self.assertGreater(copytree_mock.call_count, 0)

    def test_prepare_runtime_source_patches_windows_flash_attention_config_and_is_idempotent(self) -> None:
        module = load_setup_module()
        source_dir = self.temp_dir / "upstream-source"
        self._write_minimal_upstream_tree(source_dir)
        config_relpath = module.WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH
        skin_config_relpath = module.WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH

        with mock.patch.object(module, "_is_windows_host", return_value=True), mock.patch.object(
            module, "_classify_host", return_value="windows-x86_64"
        ), mock.patch.object(
            module, "_resolve_source", return_value=(str(source_dir), "directory")
        ):
            staged_dir, _, _ = module._prepare_runtime_source(self.ext_dir, {"unirig_source_dir": str(source_dir)})
            staged_config = staged_dir / config_relpath
            staged_skin_config = staged_dir / skin_config_relpath
            first_pass = staged_config.read_text(encoding="utf-8")
            first_skin_pass = staged_skin_config.read_text(encoding="utf-8")
            first_report = json.loads(module._runtime_stage_patch_report_path(self.ext_dir).read_text(encoding="utf-8"))

            staged_dir_second, _, _ = module._prepare_runtime_source(self.ext_dir, {"unirig_source_dir": str(source_dir)})
            second_pass = staged_config.read_text(encoding="utf-8")
            second_skin_pass = staged_skin_config.read_text(encoding="utf-8")
            second_report = json.loads(module._runtime_stage_patch_report_path(self.ext_dir).read_text(encoding="utf-8"))

        self.assertEqual(staged_dir_second, staged_dir)
        self.assertNotIn("flash_attention_2", first_pass)
        self.assertIn("_attn_implementation: eager", first_pass)
        self.assertIn("enable_flash: False", first_skin_pass)
        self.assertNotIn("enable_flash: true", first_skin_pass)
        self.assertEqual(first_pass, second_pass)
        self.assertEqual(first_skin_pass, second_skin_pass)
        self.assertEqual(first_report[0]["status"], "applied")
        self.assertEqual(first_report[1]["status"], "applied")
        self.assertEqual(first_report[1]["setting"], "mesh_encoder.enable_flash")
        self.assertEqual(first_report[1]["path"], str(skin_config_relpath))
        self.assertEqual(second_report[0]["status"], "already-patched")
        self.assertEqual(second_report[1]["status"], "already-patched")
        self.assertEqual(second_report[0]["path"], str(config_relpath))
        self.assertEqual(second_report[1]["path"], str(skin_config_relpath))

    def test_prepare_runtime_source_keeps_flash_attention_config_unchanged_on_linux(self) -> None:
        module = load_setup_module()
        source_dir = self.temp_dir / "upstream-source"
        self._write_minimal_upstream_tree(source_dir)
        config_relpath = module.WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH
        skin_config_relpath = module.WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH

        with mock.patch.object(module, "_classify_host", return_value="linux-x86_64"):
            staged_dir, _, _ = module._prepare_runtime_source(self.ext_dir, {"unirig_source_dir": str(source_dir)})

        staged_config = (staged_dir / config_relpath).read_text(encoding="utf-8")
        staged_skin_config = (staged_dir / skin_config_relpath).read_text(encoding="utf-8")
        self.assertIn("flash_attention_2", staged_config)
        self.assertIn("enable_flash: true", staged_skin_config)
        self.assertFalse(module._runtime_stage_patch_report_path(self.ext_dir).exists())

    def test_prepare_runtime_source_patches_float32_attention_config_on_linux_arm64(self) -> None:
        module = load_setup_module()
        source_dir = self.temp_dir / "upstream-source"
        self._write_minimal_upstream_tree(source_dir)

        with mock.patch.object(module, "_classify_host", return_value="linux-arm64"):
            staged_dir, _, _ = module._prepare_runtime_source(self.ext_dir, {"unirig_source_dir": str(source_dir)})

        staged_config = (staged_dir / module.WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
        self.assertNotIn("flash_attention_2", staged_config)
        self.assertIn("_attn_implementation: eager", staged_config)
        self.assertFalse(module._runtime_stage_patch_report_path(self.ext_dir).exists())

    def test_prepare_runtime_source_patches_skin_enable_flash_on_linux_arm64(self) -> None:
        module = load_setup_module()
        source_dir = self.temp_dir / "upstream-source"
        self._write_minimal_upstream_tree(source_dir)

        with mock.patch.object(module, "_classify_host", return_value="linux-arm64"):
            staged_dir, _, _ = module._prepare_runtime_source(self.ext_dir, {"unirig_source_dir": str(source_dir)})

        staged_skin_config = (staged_dir / module.WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8")
        self.assertIn("enable_flash: False", staged_skin_config)
        self.assertNotIn("enable_flash: true", staged_skin_config)
        self.assertFalse(module._runtime_stage_patch_report_path(self.ext_dir).exists())

    def test_prepare_runtime_source_patches_skin_runtime_to_fallback_when_flash_attn_is_missing_on_linux_arm64(self) -> None:
        module = load_setup_module()
        source_dir = self.temp_dir / "upstream-source"
        self._write_minimal_upstream_tree(source_dir)
        skin_model_path = source_dir / module_unirig_skin_relpath()
        skin_model_path.parent.mkdir(parents=True, exist_ok=True)
        skin_model_path.write_text(
            "import torch\n"
            "from torch import nn, Tensor\n"
            "from flash_attn.modules.mha import MHA\n\n"
            "class ResidualCrossAttn(nn.Module):\n"
            "    def __init__(self, feat_dim: int, num_heads: int):\n"
            "        super().__init__()\n"
            "        self.attention = MHA(embed_dim=feat_dim, num_heads=num_heads, cross_attn=True)\n",
            encoding="utf-8",
        )

        with mock.patch.object(module, "_classify_host", return_value="linux-arm64"):
            staged_dir, _, _ = module._prepare_runtime_source(self.ext_dir, {"unirig_source_dir": str(source_dir)})

        patched = (staged_dir / module.RUNTIME_UNIRIG_SKIN_RELATIVE_PATH).read_text(encoding="utf-8")
        self.assertIn(module.RUNTIME_UNIRIG_SKIN_FLASH_ATTN_PATCH_MARKER, patched)
        self.assertIn("try:\n    from flash_attn.modules.mha import MHA", patched)
        self.assertIn("MHA = None", patched)
        self.assertIn("class _FallbackCrossAttention(nn.Module):", patched)
        self.assertIn("self.Wq = nn.Linear(embed_dim, embed_dim)", patched)
        self.assertIn("self.Wkv = nn.Linear(embed_dim, embed_dim * 2)", patched)
        self.assertIn("F.scaled_dot_product_attention", patched)
        self.assertIn("self.attention = _build_cross_attention(embed_dim=feat_dim, num_heads=num_heads)", patched)

    def test_prepare_runtime_source_patches_model_parse_to_lazy_import_skin_runtime(self) -> None:
        module = load_setup_module()
        source_dir = self.temp_dir / "upstream-source"
        self._write_minimal_upstream_tree(source_dir)
        parse_path = source_dir / module_model_parse_relpath()
        parse_path.parent.mkdir(parents=True, exist_ok=True)
        parse_path.write_text(
            "from .unirig_ar import UniRigAR\n"
            "from .unirig_skin import UniRigSkin\n\n"
            "from .spec import ModelSpec\n\n"
            "def get_model(**kwargs) -> ModelSpec:\n"
            "    MAP = {\n"
            "        'unirig_ar': UniRigAR,\n"
            "        'unirig_skin': UniRigSkin,\n"
            "    }\n"
            "    __target__ = kwargs['__target__']\n"
            "    del kwargs['__target__']\n"
            "    assert __target__ in MAP, f\"expect: [{','.join(MAP.keys())}], found: {__target__}\"\n"
            "    return MAP[__target__](**kwargs)\n",
            encoding="utf-8",
        )

        staged_dir, _, _ = module._prepare_runtime_source(self.ext_dir, {"unirig_source_dir": str(source_dir)})

        patched = (staged_dir / module.RUNTIME_MODEL_PARSE_RELATIVE_PATH).read_text(encoding="utf-8")
        self.assertIn(module.RUNTIME_MODEL_PARSE_LAZY_IMPORT_PATCH_MARKER, patched)
        self.assertIn("if __target__ == 'unirig_ar':", patched)
        self.assertIn("if __target__ == 'unirig_skin':", patched)
        self.assertNotIn("from .unirig_skin import UniRigSkin\n\nfrom .spec import ModelSpec", patched)

    def test_preflight_report_is_host_facts_only(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="x86_64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.12.3"):
            report = module._preflight_check_summary(Path(sys.executable), {})

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["host"], {"arch": "x86_64", "os": "linux", "platform_tag": "linux-x86_64"})
        self.assertEqual(report["observed"]["python_version"], "3.12.3")
        self.assertNotIn("platform_policy", report)
        self.assertNotIn("manifest", report)
        self.assertNotIn("host_constraints", report)
        self.assertNotIn("bootstrap_python", report)

    def test_preflight_blocks_missing_bootstrap_python(self) -> None:
        module = load_setup_module()
        missing = self.temp_dir / "missing-python"

        report = module._preflight_check_summary(missing, {})

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("missing bootstrap python" in item.lower() for item in report["blocked"]))

    def test_collect_linux_arm64_baseline_prerequisites_reports_actionable_environment_blockers(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(
            module,
            "_probe_linux_arm64_build_environment",
            return_value={
                "gpu": {"present": False, "vendor": ""},
                "nvcc": {"path": "/usr/local/bin/nvcc-wrapper", "version": "wrapper", "is_real": False},
                "cuda": {"home": "", "version": "", "facts_ready": False},
                "compiler": {"path": "", "kind": ""},
                "python_headers": {"ready": False, "header": ""},
            },
            create=True,
        ):
            baseline = module._collect_linux_arm64_baseline_prerequisites(python_exe, python_version="3.12.3")

        blockers = {item["code"]: item for item in baseline["blockers"]}

        self.assertFalse(baseline["ready"])
        self.assertEqual(baseline["python"]["required"], "3.12")
        self.assertEqual(baseline["python"]["version"], "3.12.3")
        self.assertEqual(blockers["missing-nvidia-gpu"]["boundary"], "environment")
        self.assertEqual(blockers["missing-real-nvcc"]["boundary"], "environment")
        self.assertEqual(blockers["missing-system-cuda-facts"]["boundary"], "environment")
        self.assertEqual(blockers["missing-cxx-compiler"]["boundary"], "environment")
        self.assertEqual(blockers["missing-python-headers"]["boundary"], "environment")
        self.assertTrue(any("nvidia" in item.lower() for item in baseline["blocked"]))
        self.assertTrue(any("nvcc" in item.lower() for item in baseline["blocked"]))

    def test_preflight_linux_arm64_includes_baseline_prerequisites_and_detailed_blockers(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.10.9"), mock.patch.object(
            module,
            "_probe_linux_arm64_build_environment",
            return_value={
                "gpu": {"present": False, "vendor": ""},
                "nvcc": {"path": "", "version": "", "is_real": False},
                "cuda": {"home": "", "version": "", "facts_ready": False},
                "compiler": {"path": "", "kind": ""},
                "python_headers": {"ready": False, "header": ""},
            },
            create=True,
        ):
            report = module._preflight_check_summary(python_exe, {})

        blockers = {item["code"]: item for item in report["blockers"]}

        self.assertEqual(report["status"], "blocked")
        self.assertIn("baseline", report)
        self.assertFalse(report["baseline"]["ready"])
        self.assertEqual(report["baseline"]["python"]["version"], "3.10.9")
        self.assertEqual(blockers["missing-nvidia-gpu"]["boundary"], "environment")
        self.assertEqual(blockers["missing-real-nvcc"]["boundary"], "environment")
        self.assertEqual(blockers["missing-system-cuda-facts"]["boundary"], "environment")
        self.assertEqual(blockers["missing-cxx-compiler"]["boundary"], "environment")
        self.assertEqual(blockers["missing-python-headers"]["boundary"], "environment")

    def _linux_arm64_ready_probe(self) -> dict[str, object]:
        return {
            "gpu": {"present": True, "vendor": "NVIDIA", "model": "RTX", "nvidia_smi_path": "/usr/bin/nvidia-smi"},
            "nvcc": {"path": "/usr/local/cuda/bin/nvcc", "version": "Cuda compilation tools, release 12.4", "is_real": True},
            "cuda": {"home": "/usr/local/cuda", "version": "12.4", "facts_ready": True},
            "compiler": {"path": "/usr/bin/g++", "kind": "g++"},
            "python_headers": {"ready": True, "header": "/usr/include/python3.11/Python.h"},
        }

    def _assert_linux_arm64_baseline_case(
        self,
        *,
        python_version: str,
        probe: dict[str, object],
        expected_code: str,
        expected_check_id: str,
    ) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value=python_version), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=probe, create=True
        ):
            report = module._preflight_check_summary(python_exe, {})

        blockers = {item["code"]: item for item in report["blockers"]}
        checks = {item["id"]: item for item in report["checks"]}
        source_build = report["source_build"]
        baseline_stage = source_build["stages"]["baseline"]
        baseline_stage_blockers = {item["code"]: item for item in baseline_stage["blockers"]}

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["baseline"]["ready"])
        self.assertEqual(checks[expected_check_id]["status"], "fail")
        self.assertEqual(blockers[expected_code]["boundary"], "environment")
        self.assertEqual(blockers[expected_code]["owner"], "environment")
        self.assertEqual(source_build["status"], "blocked")
        self.assertEqual(source_build["current_stage"], "baseline")
        self.assertFalse(source_build["non_blender_runtime_ready"])
        self.assertEqual(baseline_stage["status"], "blocked")
        self.assertFalse(baseline_stage["ready"])
        self.assertIn(expected_code, baseline_stage["blocker_codes"])
        self.assertEqual(baseline_stage_blockers[expected_code]["boundary"], "environment")
        self.assertEqual(baseline_stage_blockers[expected_code]["owner"], "environment")

    def test_preflight_linux_arm64_reports_blocked_baseline_stage_for_each_missing_prerequisite(self) -> None:
        ready_probe = self._linux_arm64_ready_probe()
        cases = [
            (
                "missing nvcc",
                "3.11.9",
                {**ready_probe, "nvcc": {"path": "", "version": "", "is_real": False}},
                "missing-real-nvcc",
                "linux-arm64-real-nvcc",
            ),
            (
                "missing compiler",
                "3.11.9",
                {**ready_probe, "compiler": {"path": "", "kind": ""}},
                "missing-cxx-compiler",
                "linux-arm64-cxx-compiler",
            ),
            (
                "missing python headers",
                "3.11.9",
                {**ready_probe, "python_headers": {"ready": False, "header": ""}},
                "missing-python-headers",
                "linux-arm64-python-headers-baseline",
            ),
            (
                "missing cuda facts",
                "3.11.9",
                {**ready_probe, "cuda": {"home": "", "version": "", "facts_ready": False}},
                "missing-system-cuda-facts",
                "linux-arm64-system-cuda-facts",
            ),
            (
                "missing gpu facts",
                "3.11.9",
                {**ready_probe, "gpu": {"present": False, "vendor": "", "model": "", "nvidia_smi_path": ""}},
                "missing-nvidia-gpu",
                "linux-arm64-nvidia-gpu",
            ),
        ]

        for label, python_version, probe, expected_code, expected_check_id in cases:
            with self.subTest(label=label):
                self._assert_linux_arm64_baseline_case(
                    python_version=python_version,
                    probe=probe,
                    expected_code=expected_code,
                    expected_check_id=expected_check_id,
                )

    def test_preflight_linux_arm64_treats_selected_bootstrap_python_as_diagnostic_not_baseline_blocker(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.12.3"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "source-build-only",
                "ready": False,
                "verification": "deferred",
                "blockers": [],
                "blocker_codes": [],
                "checks": [],
            },
            create=True,
        ):
            report = module._preflight_check_summary(python_exe, {})

        checks = {item["id"]: item for item in report["checks"]}

        self.assertEqual(report["baseline"]["python"]["required"], "3.12")
        self.assertTrue(report["baseline"]["ready"])
        self.assertEqual(checks["linux-arm64-bootstrap-python"]["status"], "pass")
        self.assertEqual(checks["linux-arm64-bootstrap-python"]["code"], "bootstrap-python-selected")
        self.assertEqual(report["source_build"]["current_stage"], "pyg")

    def test_preflight_state_payload_keeps_linux_arm64_baseline_stage_blocked_and_not_runtime_ready(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        probe = self._linux_arm64_ready_probe()
        probe["nvcc"] = {"path": "", "version": "", "is_real": False}

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=probe, create=True
        ):
            preflight = module._preflight_check_summary(python_exe, {})
            planner = module.build_install_plan(host_os="linux", host_arch="aarch64")

        state = bootstrap.normalize_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "blocked",
                "planner": planner,
                "preflight": module._preflight_state_payload(preflight, planner),
                "install_plan": {
                    "summary": module._install_plan_summary(planner, install_state="blocked")
                },
                "deferred_work": planner["deferred"],
            },
            self.ext_dir,
            include_runtime_fields=True,
        )

        self.assertEqual(state["source_build"]["status"], "blocked")
        self.assertEqual(state["source_build"]["mode"], "staged-source-build")
        self.assertEqual(state["source_build"]["host_class"], "linux-arm64")
        blockers = {item["code"]: item for item in state["source_build"]["blockers"]}
        self.assertEqual(blockers["missing-real-nvcc"]["boundary"], "environment")
        self.assertFalse(state["last_verification"]["runtime_ready"])

    def test_write_error_state_persists_linux_arm64_staged_source_build_payload_when_baseline_is_blocked(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        probe = self._linux_arm64_ready_probe()
        probe["nvcc"] = {"path": "", "version": "", "is_real": False}

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=probe, create=True
        ):
            preflight = module._preflight_check_summary(python_exe, {})
            planner = module.build_install_plan(host_os="linux", host_arch="aarch64")

        module._write_error_state(
            self.ext_dir,
            "blocked for staged Linux ARM64 validation",
            preflight,
            requested_host_python=python_exe,
            bootstrap_resolution={"selected_source": "requested-host-python"},
            planner=planner,
        )

        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        source_build = state["source_build"]

        self.assertEqual(source_build["host_class"], "linux-arm64")
        self.assertEqual(source_build["mode"], "staged-source-build")
        self.assertFalse(source_build["baseline"]["ready"])
        self.assertEqual(source_build["baseline"]["python"]["version"], "3.11.9")
        self.assertEqual(source_build["stages"]["baseline"]["status"], "blocked")
        self.assertEqual(source_build["stages"]["pyg"]["status"], "blocked")
        self.assertEqual(source_build["stages"]["spconv"]["status"], "blocked")
        self.assertIn("missing-real-nvcc", source_build["stages"]["baseline"]["blocker_codes"])
        self.assertEqual(source_build["blocked_reasons"], preflight["blocked"])
        self.assertEqual(source_build["deferred_work"], ["bpy-portability"])
        self.assertFalse(source_build["non_blender_runtime_ready"])

    def test_write_error_state_persists_linux_arm64_stage_results_when_non_blender_stages_advance(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
            },
            create=True,
        ):
            preflight = module._preflight_check_summary(python_exe, {})
            planner = module.build_install_plan(host_os="linux", host_arch="aarch64")

        module._write_error_state(
            self.ext_dir,
            "blocked by deferred Linux ARM64 full-runtime work",
            preflight,
            requested_host_python=python_exe,
            bootstrap_resolution={"selected_source": "requested-host-python"},
            planner=planner,
        )

        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        source_build = state["source_build"]

        self.assertTrue(source_build["baseline"]["ready"])
        self.assertEqual(source_build["stages"]["baseline"]["status"], "ready")
        self.assertEqual(source_build["stages"]["pyg"]["status"], "ready")
        self.assertEqual(source_build["stages"]["pyg"]["verification"], "import-smoke")
        self.assertEqual(source_build["stages"]["spconv"]["status"], "build-ready")
        self.assertEqual(source_build["stages"]["spconv"]["verification"], "import-smoke")
        self.assertIn("bpy remains a likely Linux ARM64 portability risk", source_build["blocked_reasons"][-1])
        self.assertEqual(source_build["deferred_work"], ["bpy-portability"])
        self.assertFalse(source_build["non_blender_runtime_ready"])

    def test_preflight_linux_arm64_blocks_pyg_stage_until_baseline_prerequisites_are_ready(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        probe = self._linux_arm64_ready_probe()
        probe["nvcc"] = {"path": "", "version": "", "is_real": False}

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=probe, create=True
        ):
            report = module._preflight_check_summary(python_exe, {})

        pyg_stage = report["source_build"]["stages"]["pyg"]
        spconv_stage = report["source_build"]["stages"]["spconv"]
        blockers = {item["code"]: item for item in pyg_stage["blockers"]}

        self.assertEqual(pyg_stage["status"], "blocked")
        self.assertEqual(pyg_stage["mode"], "source-build-only")
        self.assertEqual(pyg_stage["blocked_by_stage"], "baseline")
        self.assertEqual(pyg_stage["packages"], ["torch_scatter", "torch_cluster"])
        self.assertEqual(pyg_stage["blocker_codes"], ["missing-real-nvcc"])
        self.assertEqual(blockers["missing-real-nvcc"]["boundary"], "environment")
        self.assertEqual(spconv_stage["status"], "blocked")
        self.assertEqual(spconv_stage["mode"], "source-build")
        self.assertEqual(spconv_stage["verification"], "import-smoke")
        self.assertEqual(spconv_stage["blocked_by_stage"], "baseline")
        self.assertEqual(spconv_stage["blocker_codes"], ["missing-real-nvcc"])

    def test_preflight_linux_arm64_reports_pyg_stage_as_source_build_only_when_baseline_is_ready(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "source-build-only",
                "ready": False,
                "verification": "deferred",
                "blockers": [],
                "blocker_codes": [],
                "checks": [],
            },
            create=True,
        ):
            report = module._preflight_check_summary(python_exe, {})

        checks = {item["id"]: item for item in report["checks"]}
        pyg_stage = report["source_build"]["stages"]["pyg"]
        spconv_stage = report["source_build"]["stages"]["spconv"]

        self.assertEqual(checks["linux-arm64-pyg-source-build"]["status"], "pass")
        self.assertEqual(checks["linux-arm64-spconv-guarded-source-build"]["status"], "pass")
        self.assertEqual(report["source_build"]["status"], "blocked")
        self.assertEqual(report["source_build"]["current_stage"], "pyg")
        self.assertFalse(report["source_build"]["non_blender_runtime_ready"])
        self.assertEqual(pyg_stage["status"], "source-build-only")
        self.assertEqual(pyg_stage["mode"], "source-build-only")
        self.assertEqual(pyg_stage["verification"], "deferred")
        self.assertFalse(pyg_stage["ready"])
        self.assertEqual(pyg_stage["packages"], ["torch_scatter", "torch_cluster"])
        self.assertEqual(pyg_stage["blocker_codes"], [])
        self.assertEqual(spconv_stage["status"], "deferred")
        self.assertEqual(spconv_stage["mode"], "source-build")
        self.assertEqual(spconv_stage["verification"], "import-smoke")
        self.assertFalse(spconv_stage["ready"])
        self.assertEqual(spconv_stage["blocked_by_stage"], "pyg")
        self.assertEqual(spconv_stage["blocker_codes"], [])

    def test_preflight_linux_arm64_marks_pyg_stage_ready_after_import_smoke_verification(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
            },
            create=True,
        ):
            report = module._preflight_check_summary(python_exe, {})

        pyg_stage = report["source_build"]["stages"]["pyg"]
        spconv_stage = report["source_build"]["stages"]["spconv"]

        self.assertEqual(report["source_build"]["status"], "blocked")
        self.assertFalse(report["source_build"]["non_blender_runtime_ready"])
        self.assertEqual(report["source_build"]["current_stage"], "spconv")
        self.assertEqual(pyg_stage["status"], "ready")
        self.assertTrue(pyg_stage["ready"])
        self.assertEqual(pyg_stage["verification"], "import-smoke")
        self.assertEqual([item["status"] for item in pyg_stage["checks"]], ["ready", "ready"])
        self.assertEqual(pyg_stage["blocker_codes"], [])
        self.assertEqual(spconv_stage["status"], "build-ready")
        self.assertFalse(spconv_stage["ready"])
        self.assertEqual(spconv_stage["mode"], "source-build")
        self.assertEqual(spconv_stage["verification"], "import-smoke")
        self.assertEqual(spconv_stage["blocker_codes"], [])

    def test_preflight_linux_arm64_marks_spconv_build_ready_after_pyg_ready_and_persists_stage_fields(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
            },
            create=True,
        ), mock.patch.object(module, "_run_linux_arm64_spconv_guarded_bringup", create=True) as spconv_mock:
            report = module._preflight_check_summary(python_exe, {})

        spconv_stage = report["source_build"]["stages"]["spconv"]

        spconv_mock.assert_not_called()
        self.assertEqual(spconv_stage["status"], "build-ready")
        self.assertFalse(spconv_stage["ready"])
        self.assertEqual(spconv_stage["packages"], ["cumm", "spconv"])
        self.assertEqual([item["label"] for item in spconv_stage["checks"]], ["spconv-guarded-source-build"])
        self.assertEqual(spconv_stage["blockers"], [])
        self.assertEqual(spconv_stage["blocker_codes"], [])
        self.assertEqual(spconv_stage["boundary"], "wrapper")
        self.assertNotIn("blocked_by_stage", spconv_stage)

    def test_preflight_linux_arm64_keeps_spconv_deferred_without_running_stage_before_pyg_ready(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "source-build-only",
                "ready": False,
                "verification": "deferred",
                "blockers": [],
                "blocker_codes": [],
                "checks": [],
            },
            create=True,
        ), mock.patch.object(module, "_run_linux_arm64_spconv_guarded_bringup", create=True) as spconv_mock:
            report = module._preflight_check_summary(python_exe, {})

        spconv_stage = report["source_build"]["stages"]["spconv"]

        spconv_mock.assert_not_called()
        self.assertEqual(spconv_stage["status"], "deferred")
        self.assertFalse(spconv_stage["ready"])
        self.assertEqual(spconv_stage["packages"], ["cumm", "spconv"])
        self.assertEqual(spconv_stage["blocked_by_stage"], "pyg")

    def test_probe_linux_arm64_spconv_preparation_uses_staged_gate_statuses(self) -> None:
        module = load_setup_module()

        cases = [
            (
                "baseline blocked",
                [{"code": "missing-real-nvcc", "boundary": "environment", "owner": "environment"}],
                {"status": "blocked", "boundary": "environment"},
                "blocked",
                "baseline",
            ),
            (
                "waiting for pyg",
                [],
                {"status": "source-build-only", "boundary": "wrapper"},
                "deferred",
                "pyg",
            ),
            (
                "prep ready only",
                [],
                {"status": "ready", "boundary": "wrapper"},
                "build-ready",
                "",
            ),
        ]

        for label, baseline_blockers, pyg_stage, expected_status, expected_blocked_by_stage in cases:
            with self.subTest(label=label):
                result = module._probe_linux_arm64_spconv_preparation(
                    baseline_blockers=baseline_blockers,
                    pyg_stage=pyg_stage,
                    spconv_dependency={
                        "verification": "import-smoke",
                        "allowed_statuses": ["blocked", "deferred", "build-ready", "ready"],
                    },
                )

                self.assertEqual(result["status"], expected_status)
                self.assertFalse(result["ready"])
                self.assertIn(result["status"], ["blocked", "deferred", "build-ready", "ready"])
                if expected_blocked_by_stage:
                    self.assertEqual(result["blocked_by_stage"], expected_blocked_by_stage)
                else:
                    self.assertNotIn("blocked_by_stage", result)

    def test_linux_arm64_dependency_blocker_describes_guarded_spconv_cumm_risk_as_upstream_package_owned(self) -> None:
        module = load_setup_module()

        blocker = module._linux_arm64_dependency_blocker(
            {
                "name": "spconv",
                "strategy": module.LINUX_ARM64_SPCONV_STRATEGY,
                "reason_code": module.LINUX_ARM64_SPCONV_REASON_CODE,
            }
        )

        self.assertIsNotNone(blocker)
        self.assertEqual(blocker["code"], module.LINUX_ARM64_SPCONV_REASON_CODE)
        self.assertEqual(blocker["boundary"], "upstream-package")
        self.assertEqual(blocker["owner"], "upstream-package")
        self.assertIn("cumm", blocker["message"].lower())
        self.assertIn("experimental and unvalidated", blocker["message"].lower())
        self.assertNotIn("supported", blocker["message"].lower())

    def test_preflight_linux_arm64_persists_wrapper_owned_spconv_check_failure_with_conservative_wording(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        malformed_plan = module.build_install_plan(host_os="linux", host_arch="aarch64")
        for dependency in malformed_plan["dependencies"]:
            if dependency["name"] == "spconv":
                dependency["strategy"] = "generic-prebuilt-package"
                dependency["reason_code"] = "incorrect-test-strategy"

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "build_install_plan", return_value=malformed_plan), mock.patch.object(
            module, "_probe_python_version", return_value="3.11.9"
        ), mock.patch.object(module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True):
            checks, blockers, blocked, _baseline = module._linux_arm64_preflight_checks(python_exe, python_version="3.11.9")

        spconv_check = next(item for item in checks if item["id"] == "linux-arm64-spconv-guarded-source-build")
        spconv_blocker = next(item for item in blockers if item["code"] == module.LINUX_ARM64_SPCONV_REASON_CODE)

        self.assertEqual(spconv_check["status"], "fail")
        self.assertEqual(spconv_check["boundary"], "wrapper")
        self.assertEqual(spconv_check["owner"], "wrapper")
        self.assertIn("cumm", spconv_check["message"].lower())
        self.assertIn("experimental and unvalidated", spconv_check["message"].lower())
        self.assertEqual(spconv_blocker["boundary"], "wrapper")
        self.assertEqual(spconv_blocker["owner"], "wrapper")
        self.assertIn(spconv_check["message"], blocked)
        self.assertNotIn("supported", spconv_check["message"].lower())

    def test_build_install_plan_keeps_linux_arm64_bpy_explicitly_deferred(self) -> None:
        module = load_setup_module()

        plan = module.build_install_plan(host_os="linux", host_arch="aarch64")
        dependency_map = {item["name"]: item for item in plan["dependencies"]}

        self.assertEqual(plan["host_class"], "linux-arm64")
        self.assertEqual(plan["stages"], ["baseline", "pyg", "spconv", "bpy-deferred"])
        self.assertNotIn("spconv-port", plan["deferred"])
        self.assertIn("bpy-portability", plan["deferred"])
        self.assertEqual(dependency_map["bpy"]["strategy"], "deferred-portability-review")
        self.assertEqual(dependency_map["bpy"]["reason_code"], "bpy-portability")

    def test_build_install_plan_describes_linux_arm64_spconv_as_guarded_source_build_import_smoke(self) -> None:
        module = load_setup_module()

        plan = module.build_install_plan(host_os="linux", host_arch="aarch64")
        spconv_dependency = next(item for item in plan["dependencies"] if item["name"] == "spconv")

        self.assertEqual(spconv_dependency["strategy"], "linux-arm64-guarded-source-build")
        self.assertEqual(spconv_dependency["reason_code"], "spconv-guarded-source-build")
        self.assertEqual(spconv_dependency["stage"], "spconv")
        self.assertEqual(spconv_dependency["verification"], "import-smoke")
        self.assertEqual(spconv_dependency["allowed_statuses"], ["blocked", "deferred", "build-ready", "ready"])

    def test_preflight_linux_arm64_keeps_bpy_deferred_even_when_non_blender_stages_advance(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
            },
            create=True,
        ):
            report = module._preflight_check_summary(python_exe, {})

        checks = {item["id"]: item for item in report["checks"]}
        blockers = {item["code"]: item for item in report["blockers"]}

        self.assertEqual(report["source_build"]["stages"]["spconv"]["status"], "build-ready")
        self.assertEqual(checks["linux-arm64-bpy-portability"]["status"], "fail")
        self.assertEqual(blockers["bpy-portability-risk"]["boundary"], "upstream")
        self.assertEqual(blockers["bpy-portability-risk"]["owner"], "upstream")
        self.assertTrue(any("bpy" in item.lower() for item in report["blocked"]))

    def test_preflight_linux_arm64_persists_external_blender_evidence_without_changing_non_blender_stage_gating(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        candidate = {
            "source": "override",
            "path": "/custom/blender",
            "selected_because": "Selected explicit override.",
        }
        probe = {
            "status": "ok",
            "command": ["/custom/blender", "--background"],
            "blender_version": "3.6.9",
            "python_version": "3.11.8",
            "smoke_result": "passed",
            "returncode": 0,
            "stdout_tail": ["probe ok"],
            "stderr_tail": [],
        }
        classification = module._classify_linux_arm64_bpy_viability(candidate, probe, wrapper_python_version="3.11.9")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
            },
            create=True,
        ), mock.patch.object(module, "_resolve_linux_arm64_blender_candidate", return_value=candidate, create=True), mock.patch.object(
            module, "_probe_linux_arm64_blender_bpy", return_value=probe, create=True
        ), mock.patch.object(module, "_classify_linux_arm64_bpy_viability", return_value=classification, create=True):
            report = module._preflight_check_summary(python_exe, {"blender_exe": "/custom/blender"})

        source_build = report["source_build"]
        checks = {item["id"]: item for item in report["checks"]}

        self.assertEqual(source_build["stages"]["baseline"]["status"], "ready")
        self.assertEqual(source_build["stages"]["pyg"]["status"], "ready")
        self.assertEqual(source_build["stages"]["spconv"]["status"], "build-ready")
        self.assertEqual(source_build["current_stage"], "spconv")
        self.assertFalse(source_build["non_blender_runtime_ready"])
        self.assertEqual(checks["linux-arm64-bpy-portability"]["status"], "pass")
        self.assertEqual(source_build["bpy_evidence_class"], "external-bpy-smoke-ready")
        self.assertEqual(source_build["stages"]["bpy"]["status"], "external-bpy-smoke-ready")
        self.assertFalse(source_build["stages"]["bpy"]["ready"])
        self.assertEqual(source_build["external_blender"]["candidate"], candidate)
        self.assertEqual(source_build["external_blender"]["probe"], probe)
        self.assertEqual(source_build["external_blender"]["classification"]["status"], "external-bpy-smoke-ready")
        extract_merge = source_build["executable_boundary"]["extract_merge"]
        self.assertFalse(extract_merge["enabled"])
        self.assertFalse(extract_merge["ready"])
        self.assertEqual(extract_merge["status"], "missing")

    def test_preflight_linux_arm64_seeds_extract_merge_boundary_metadata_from_existing_blender_candidate(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        candidate = {
            "source": "override",
            "path": "/custom/blender",
            "selected_because": "Selected explicit override.",
        }
        probe = {
            "status": "ok",
            "command": ["/custom/blender", "--background"],
            "blender_version": "3.6.9",
            "python_version": "3.11.8",
            "smoke_result": "passed",
            "returncode": 0,
            "stdout_tail": ["probe ok"],
            "stderr_tail": [],
        }
        classification = module._classify_linux_arm64_bpy_viability(candidate, probe, wrapper_python_version="3.11.9")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
            },
            create=True,
        ), mock.patch.object(module, "_resolve_linux_arm64_blender_candidate", return_value=candidate, create=True), mock.patch.object(
            module, "_probe_linux_arm64_blender_bpy", return_value=probe, create=True
        ), mock.patch.object(module, "_classify_linux_arm64_bpy_viability", return_value=classification, create=True):
            report = module._preflight_check_summary(python_exe, {"blender_exe": "/custom/blender"})

        extract_merge = report["source_build"]["executable_boundary"]["extract_merge"]

        self.assertFalse(extract_merge["enabled"])
        self.assertFalse(extract_merge["ready"])
        self.assertEqual(extract_merge["status"], "missing")
        self.assertEqual(extract_merge["candidate"], candidate)
        self.assertEqual(extract_merge["evidence_kind"], "external-blender")
        self.assertEqual(extract_merge["external_blender_status"], "external-bpy-smoke-ready")
        self.assertEqual(extract_merge["default_owner"], "context.venv_python")
        self.assertEqual(extract_merge["optional_owner"], "blender-subprocess")
        self.assertEqual(extract_merge["supported_stages"], ["extract-prepare", "extract-skin", "merge"])
        self.assertTrue(extract_merge["requires_explicit_gate"])
        self.assertTrue(extract_merge["requires_executable_boundary_proof"])

    def test_linux_arm64_qualification_fixture_declarations_cover_scoped_stages_categories_and_modes(self) -> None:
        module = load_setup_module()

        fixtures = module._linux_arm64_qualification_fixture_declarations()

        self.assertTrue(fixtures)
        self.assertEqual(
            len(fixtures),
            len(module.LINUX_ARM64_EXTRACT_MERGE_BOUNDARY_STAGES) * len(module.LINUX_ARM64_QUALIFICATION_FIXTURE_CLASSES),
        )
        self.assertEqual(
            {fixture["stage"] for fixture in fixtures},
            {"extract-prepare", "extract-skin", "merge"},
        )
        self.assertEqual(
            {fixture["fixture_class"] for fixture in fixtures},
            {"known-good", "normalization-sensitive", "realistic", "intentionally-bad"},
        )
        self.assertEqual(
            {(fixture["stage"], fixture["fixture_class"]) for fixture in fixtures},
            {
                (stage, fixture_class)
                for stage in module.LINUX_ARM64_EXTRACT_MERGE_BOUNDARY_STAGES
                for fixture_class in module.LINUX_ARM64_QUALIFICATION_FIXTURE_CLASSES
            },
        )
        self.assertNotIn("predict-skin", {fixture["stage"] for fixture in fixtures})
        for fixture in fixtures:
            self.assertEqual(fixture["execution_modes"], ["wrapper", "seam", "forced-fallback"])

    def test_linux_arm64_qualification_evidence_record_preserves_stable_failure_codes(self) -> None:
        module = load_setup_module()

        record = module._linux_arm64_qualification_evidence_record(
            fixture={
                "fixture_id": "merge-known-good",
                "fixture_class": "known-good",
                "stage": "merge",
            },
            run_label="merge-known-good-seam",
            selected_mode="seam",
            status="failed",
            failure_code="marker-missing",
            host_facts={"os": "linux", "arch": "aarch64"},
            blender_facts={"path": "/opt/blender/blender", "version": "4.0.2"},
            outputs={"produced": []},
            logs={"stdout_tail": ["trace"], "stderr_tail": ["error"]},
        )

        self.assertEqual(record["fixture_id"], "merge-known-good")
        self.assertEqual(record["selected_mode"], "seam")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure_code"], "marker-missing")
        self.assertEqual(record["host"], {"os": "linux", "arch": "aarch64"})
        self.assertEqual(record["blender"], {"path": "/opt/blender/blender", "version": "4.0.2"})

    def test_linux_arm64_qualification_evidence_record_accepts_comparison_failure_codes(self) -> None:
        module = load_setup_module()

        record = module._linux_arm64_qualification_evidence_record(
            fixture={
                "fixture_id": "extract-prepare-realistic",
                "fixture_class": "realistic",
                "stage": "extract-prepare",
            },
            run_label="extract-prepare-realistic-seam",
            selected_mode="seam",
            status="failed",
            failure_code="output-mismatch",
        )

        self.assertEqual(record["stage"], "extract-prepare")
        self.assertEqual(record["failure_code"], "output-mismatch")

    def test_linux_arm64_qualification_comparison_summary_counts_outcomes_and_failure_codes(self) -> None:
        module = load_setup_module()

        summary = module._linux_arm64_qualification_comparison_summary(
            [
                {"stage": "extract-prepare", "status": "passed", "selected_mode": "wrapper", "failure_code": ""},
                {"stage": "extract-prepare", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                {"stage": "merge", "status": "failed", "selected_mode": "seam", "failure_code": "output-mismatch"},
                {"stage": "extract-skin", "status": "failed", "selected_mode": "forced-fallback", "failure_code": " marker-missing "},
            ]
        )

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["by_failure_code"], {"marker-missing": 1, "output-mismatch": 1})
        self.assertEqual(
            summary["by_mode"],
            {
                "wrapper": {"passed": 1, "failed": 0},
                "seam": {"passed": 1, "failed": 1},
                "forced-fallback": {"passed": 0, "failed": 1},
            },
        )
        self.assertEqual(summary["required_stage_coverage"], {"extract-prepare": True, "extract-skin": False, "merge": False})
        self.assertEqual(summary["seam_failures"], 1)
        self.assertEqual(summary["risk_codes"], ["output-mismatch"])

    def test_reduce_linux_arm64_qualification_verdict_uses_real_evidence_for_fail_mixed_and_clean_outcomes(self) -> None:
        module = load_setup_module()

        cases = [
            (
                "failed seam evidence keeps tranche not ready",
                [
                    {"stage": "extract-prepare", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                    {"stage": "extract-skin", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                    {"stage": "merge", "status": "failed", "selected_mode": "seam", "failure_code": "stage-failed"},
                ],
                "not-ready",
            ),
            (
                "mixed evidence with full seam coverage but non-seam qualification risk stays candidate only",
                [
                    {"stage": "extract-prepare", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                    {"stage": "extract-skin", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                    {"stage": "merge", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                    {
                        "stage": "merge",
                        "status": "failed",
                        "selected_mode": "forced-fallback",
                        "failure_code": "upstream-package-mismatch",
                    },
                ],
                "candidate-with-known-risks",
            ),
            (
                "fully covered clean seam tranche is ready for separate defaulting change",
                [
                    {"stage": "extract-prepare", "status": "passed", "selected_mode": "wrapper", "failure_code": ""},
                    {"stage": "extract-prepare", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                    {"stage": "extract-skin", "status": "passed", "selected_mode": "wrapper", "failure_code": ""},
                    {"stage": "extract-skin", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                    {"stage": "merge", "status": "passed", "selected_mode": "wrapper", "failure_code": ""},
                    {"stage": "merge", "status": "passed", "selected_mode": "seam", "failure_code": ""},
                ],
                "ready-for-separate-defaulting-change",
            ),
        ]

        for label, records, expected in cases:
            with self.subTest(label=label):
                summary = module._linux_arm64_qualification_comparison_summary(records)
                self.assertEqual(module._reduce_linux_arm64_qualification_verdict(summary), expected)

    def test_reduce_linux_arm64_qualification_verdict_stays_conservative_across_fail_mixed_and_clean_runs(self) -> None:
        module = load_setup_module()

        cases = [
            (
                "failed seam run keeps tranche not ready",
                {
                    "passed": 2,
                    "failed": 1,
                    "required_stage_coverage": {"extract-prepare": True, "extract-skin": True, "merge": True},
                    "seam_failures": 1,
                    "risk_codes": [],
                },
                "not-ready",
            ),
            (
                "all seam stages pass but risks remain candidate only",
                {
                    "passed": 6,
                    "failed": 0,
                    "required_stage_coverage": {"extract-prepare": True, "extract-skin": True, "merge": True},
                    "seam_failures": 0,
                    "risk_codes": ["upstream-package-mismatch"],
                },
                "candidate-with-known-risks",
            ),
            (
                "fully covered clean seam tranche is ready for separate defaulting change",
                {
                    "passed": 9,
                    "failed": 0,
                    "required_stage_coverage": {"extract-prepare": True, "extract-skin": True, "merge": True},
                    "seam_failures": 0,
                    "risk_codes": [],
                },
                "ready-for-separate-defaulting-change",
            ),
        ]

        for label, summary, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(module._reduce_linux_arm64_qualification_verdict(summary), expected)

    def test_linux_arm64_qualification_runner_coordinator_emits_fixture_backed_records_and_comparisons(self) -> None:
        module = load_setup_module()

        fixtures = [
            {
                "fixture_id": "extract-prepare-known-good",
                "fixture_class": "known-good",
                "stage": "extract-prepare",
                "execution_modes": ["wrapper", "seam", "forced-fallback"],
            },
            {
                "fixture_id": "merge-realistic",
                "fixture_class": "realistic",
                "stage": "merge",
                "execution_modes": ["wrapper", "seam", "forced-fallback"],
            },
        ]
        host_facts = {"os": "linux", "arch": "aarch64", "platform_tag": "linux-aarch64"}
        blender_facts = {
            "candidate": {"path": "/opt/blender/blender", "source": "path"},
            "classification": {"status": "external-bpy-smoke-ready", "ready": False},
        }

        def fake_runner(*, fixture: dict[str, object], selected_mode: str, host_facts: dict[str, object], blender_facts: dict[str, object]) -> dict[str, object]:
            del host_facts, blender_facts
            fixture_id = str(fixture["fixture_id"])
            outputs = {"artifact": f"{fixture_id}-{selected_mode}.json"}
            if fixture_id == "merge-realistic" and selected_mode == "seam":
                return {
                    "status": "failed",
                    "failure_code": "output-mismatch",
                    "outputs": outputs,
                    "logs": {"stdout_tail": [f"{fixture_id}:{selected_mode}:stdout"], "stderr_tail": ["compare failed"]},
                }
            return {
                "status": "passed",
                "outputs": outputs,
                "logs": {"stdout_tail": [f"{fixture_id}:{selected_mode}:stdout"], "stderr_tail": []},
            }

        def fake_compare(*, fixture: dict[str, object], baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
            wrapper_artifact = baseline["outputs"]["artifact"]
            candidate_artifact = candidate["outputs"]["artifact"]
            failed = fixture["fixture_id"] == "merge-realistic" and candidate["selected_mode"] == "seam"
            return {
                "status": "failed" if failed else "passed",
                "failure_code": "output-mismatch" if failed else "",
                "compared_modes": [baseline["selected_mode"], candidate["selected_mode"]],
                "artifacts": [wrapper_artifact, candidate_artifact],
            }

        qualification = module._coordinate_linux_arm64_qualification_runs(
            fixture_declarations=fixtures,
            host_facts=host_facts,
            blender_facts=blender_facts,
            run_fixture=fake_runner,
            compare_fixture_runs=fake_compare,
        )

        self.assertEqual(qualification["host"]["os"], "linux")
        self.assertEqual(qualification["host"]["arch"], "aarch64")
        self.assertEqual(qualification["host"]["platform_tag"], "linux-aarch64")
        self.assertEqual(qualification["host"]["host_class"], "linux-arm64")
        self.assertEqual(qualification["blender"], blender_facts)
        self.assertEqual(qualification["default_owner"], "context.venv_python")
        self.assertEqual(qualification["optional_owner"], "blender-subprocess")
        self.assertEqual(len(qualification["records"]), 6)
        self.assertEqual(qualification["summary"]["total"], 6)
        self.assertEqual(qualification["summary"]["failed"], 1)
        self.assertEqual(qualification["verdict"], "not-ready")
        merge_seam = next(
            record
            for record in qualification["records"]
            if record["fixture_id"] == "merge-realistic" and record["selected_mode"] == "seam"
        )
        self.assertEqual(merge_seam["failure_code"], "output-mismatch")
        self.assertEqual(merge_seam["outputs"]["artifact"], "merge-realistic-seam.json")
        self.assertEqual(merge_seam["logs"]["stderr_tail"], ["compare failed"])
        merge_fixture = next(item for item in qualification["fixtures"] if item["fixture_id"] == "merge-realistic")
        self.assertEqual(merge_fixture["comparison"]["wrapper_vs_seam"]["status"], "failed")
        self.assertEqual(merge_fixture["comparison"]["wrapper_vs_seam"]["failure_code"], "output-mismatch")
        self.assertEqual(merge_fixture["comparison"]["wrapper_vs_forced_fallback"]["status"], "passed")

    def test_linux_arm64_qualification_runner_coordinator_stays_additive_off_linux_arm64_hosts(self) -> None:
        module = load_setup_module()
        run_fixture = mock.Mock()
        compare_fixture_runs = mock.Mock()

        for label, host_facts in [
            ("windows-x86_64", {"os": "windows", "arch": "x86_64", "platform_tag": "windows-x86_64"}),
            ("linux-x86_64", {"os": "linux", "arch": "x86_64", "platform_tag": "linux-x86_64"}),
        ]:
            with self.subTest(label=label):
                qualification = module._coordinate_linux_arm64_qualification_runs(
                    host_facts=host_facts,
                    fixture_declarations=[
                        {
                            "fixture_id": "extract-prepare-known-good",
                            "fixture_class": "known-good",
                            "stage": "extract-prepare",
                            "execution_modes": ["wrapper", "seam", "forced-fallback"],
                        }
                    ],
                    run_fixture=run_fixture,
                    compare_fixture_runs=compare_fixture_runs,
                )

                self.assertEqual(qualification, {})

        run_fixture.assert_not_called()
        compare_fixture_runs.assert_not_called()

    def test_linux_arm64_state_source_build_payload_preserves_verified_boundary_while_backfilling_candidate_metadata(self) -> None:
        module = load_setup_module()

        payload = module._linux_arm64_state_source_build_payload(
            {
                "status": "blocked",
                "baseline": {"ready": True},
                "blockers": [],
                "blocked": [],
                "source_build": {
                    "status": "blocked",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "external_blender": {
                        "candidate": {
                            "source": "override",
                            "path": "/custom/blender",
                            "selected_because": "Selected explicit override.",
                        },
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        },
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "status": "verified",
                            "proof_kind": "blender-subprocess",
                        }
                    },
                    "stages": {
                        "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []}
                    },
                },
            },
            {
                "host_class": "linux-arm64",
                "support_posture": "experimental-unvalidated",
                "install_mode": "staged-source-build",
                "deferred": ["bpy-portability"],
                "dependencies": [{"name": "bpy", "reason_code": "bpy-portability-risk"}],
            },
            deferred_work=["bpy-portability"],
        )

        extract_merge = payload["executable_boundary"]["extract_merge"]

        self.assertTrue(extract_merge["enabled"])
        self.assertTrue(extract_merge["ready"])
        self.assertEqual(extract_merge["status"], "verified")
        self.assertEqual(extract_merge["proof_kind"], "blender-subprocess")
        self.assertEqual(extract_merge["candidate"]["path"], "/custom/blender")
        self.assertEqual(extract_merge["external_blender_status"], "external-bpy-smoke-ready")

    def test_linux_arm64_state_source_build_payload_keeps_windows_x86_64_unchanged(self) -> None:
        module = load_setup_module()

        payload = module._linux_arm64_state_source_build_payload(
            {"status": "ready", "baseline": {}, "blockers": [], "blocked": []},
            {
                "host_class": "windows-x86_64",
                "support_posture": "validated",
                "install_mode": "pinned-prebuilt",
                "dependencies": [],
                "deferred": [],
            },
            deferred_work=[],
        )

        self.assertEqual(payload, {})

    def test_normalize_state_preserves_external_blender_separately_from_executable_boundary_extract_merge(self) -> None:
        normalized = bootstrap.normalize_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "python_version": "3.11.9",
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "bpy_evidence_class": "external-bpy-smoke-ready",
                    "external_blender": {
                        "candidate": {"source": "override", "path": "/custom/blender", "selected_because": "Selected explicit override."},
                        "probe": {
                            "status": "ok",
                            "command": ["/custom/blender", "--background"],
                            "blender_version": "3.6.9",
                            "python_version": "3.11.8",
                            "smoke_result": "passed",
                            "returncode": 0,
                            "stdout_tail": ["probe ok"],
                            "stderr_tail": [],
                        },
                        "classification": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "status": "verified",
                            "proof_kind": "blender-subprocess",
                        }
                    },
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "spconv": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "deferred_work": [],
                    "non_blender_runtime_ready": True,
                },
            },
            self.ext_dir,
        )

        self.assertEqual(normalized["source_build"]["external_blender"]["classification"]["status"], "external-bpy-smoke-ready")
        self.assertEqual(normalized["source_build"]["executable_boundary"]["extract_merge"]["status"], "verified")
        self.assertTrue(normalized["source_build"]["executable_boundary"]["extract_merge"]["ready"])
        self.assertEqual(normalized["install_state"], "partial")

    def test_load_state_exposes_linux_arm64_partial_runtime_when_extract_merge_boundary_proof_exists(self) -> None:
        state = {
            "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
            "install_state": "ready",
            "runtime_root": str(self.ext_dir / ".unirig-runtime"),
            "logs_dir": str(self.ext_dir / ".unirig-runtime" / "logs"),
            "runtime_vendor_dir": str(self.ext_dir / ".unirig-runtime" / "vendor"),
            "unirig_dir": str(self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"),
            "hf_home": str(self.ext_dir / ".unirig-runtime" / "hf-home"),
            "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
            "source_build": {
                "status": "ready",
                "mode": "staged-source-build",
                "host_class": "linux-arm64",
                "bpy_evidence_class": "external-bpy-smoke-ready",
                "external_blender": {
                    "classification": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []}
                },
                "executable_boundary": {
                    "extract_merge": {"enabled": True, "ready": True, "status": "verified", "proof_kind": "blender-subprocess"}
                },
                "stages": {
                    "baseline": {"status": "ready", "ready": True},
                    "pyg": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                    "spconv": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                    "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                },
                "blockers": [],
                "blocked_reasons": [],
                "deferred_work": [],
                "non_blender_runtime_ready": True,
            },
        }
        bootstrap.save_state(state, extension_root=self.ext_dir)

        loaded = bootstrap.load_state(self.ext_dir)

        self.assertEqual(loaded["install_state"], "partial")
        self.assertFalse(loaded["last_verification"]["runtime_ready"])
        self.assertEqual(loaded["source_build"]["executable_boundary"]["extract_merge"]["status"], "verified")

    def test_normalize_state_backfills_extract_merge_boundary_proof_metadata_without_collapsing_external_blender_evidence(self) -> None:
        normalized = bootstrap.normalize_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "python_version": "3.11.9",
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "bpy_evidence_class": "external-bpy-smoke-ready",
                    "external_blender": {
                        "candidate": {
                            "source": "override",
                            "path": "/custom/blender",
                            "selected_because": "Selected explicit override.",
                        },
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        },
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "proof_kind": "blender-subprocess",
                        }
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "non_blender_runtime_ready": True,
                },
            },
            self.ext_dir,
        )

        extract_merge = normalized["source_build"]["executable_boundary"]["extract_merge"]

        self.assertEqual(normalized["source_build"]["external_blender"]["classification"]["status"], "external-bpy-smoke-ready")
        self.assertEqual(extract_merge["status"], "verified")
        self.assertEqual(extract_merge["proof_kind"], "blender-subprocess")
        self.assertEqual(extract_merge["candidate"]["path"], "/custom/blender")
        self.assertEqual(extract_merge["external_blender_status"], "external-bpy-smoke-ready")
        self.assertEqual(extract_merge["evidence_kind"], "external-blender")
        self.assertEqual(extract_merge["default_owner"], "context.venv_python")
        self.assertEqual(extract_merge["optional_owner"], "blender-subprocess")
        self.assertEqual(extract_merge["supported_stages"], ["extract-prepare", "extract-skin", "merge"])
        self.assertTrue(extract_merge["requires_explicit_gate"])
        self.assertTrue(extract_merge["requires_executable_boundary_proof"])
        self.assertEqual(normalized["install_state"], "partial")

    def test_save_state_persists_linux_arm64_seam_proof_separately_from_external_blender_evidence(self) -> None:
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(self.ext_dir / ".unirig-runtime"),
                "logs_dir": str(self.ext_dir / ".unirig-runtime" / "logs"),
                "runtime_vendor_dir": str(self.ext_dir / ".unirig-runtime" / "vendor"),
                "unirig_dir": str(self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"),
                "hf_home": str(self.ext_dir / ".unirig-runtime" / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "external_blender": {
                        "candidate": {
                            "source": "override",
                            "path": "/custom/blender",
                            "selected_because": "Selected explicit override.",
                        },
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        },
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "proof_kind": "blender-subprocess",
                        }
                    },
                },
            },
            extension_root=self.ext_dir,
        )

        loaded = bootstrap.load_state(self.ext_dir)
        extract_merge = loaded["source_build"]["executable_boundary"]["extract_merge"]

        self.assertEqual(loaded["source_build"]["external_blender"]["classification"]["status"], "external-bpy-smoke-ready")
        self.assertEqual(extract_merge["status"], "verified")
        self.assertEqual(extract_merge["proof_kind"], "blender-subprocess")
        self.assertEqual(extract_merge["candidate"]["path"], "/custom/blender")
        self.assertEqual(extract_merge["external_blender_status"], "external-bpy-smoke-ready")
        self.assertEqual(loaded["install_state"], "blocked")
        self.assertFalse(loaded["last_verification"]["runtime_ready"])

    def test_save_state_persists_linux_arm64_qualification_comparison_separately_from_discovery_boundary_and_runtime_block(self) -> None:
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(self.ext_dir / ".unirig-runtime"),
                "logs_dir": str(self.ext_dir / ".unirig-runtime" / "logs"),
                "runtime_vendor_dir": str(self.ext_dir / ".unirig-runtime" / "vendor"),
                "unirig_dir": str(self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"),
                "hf_home": str(self.ext_dir / ".unirig-runtime" / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "external_blender": {
                        "candidate": {
                            "source": "override",
                            "path": "/custom/blender",
                            "selected_because": "Selected explicit override.",
                        },
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        },
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "status": "verified",
                            "proof_kind": "blender-subprocess",
                        }
                    },
                    "qualification": {
                        "extract_merge": {
                            "host": {
                                "os": "linux",
                                "arch": "aarch64",
                                "platform_tag": "linux-aarch64",
                                "host_class": "linux-arm64",
                            },
                            "fixtures": [
                                {
                                    "fixture_id": "merge-realistic",
                                    "fixture_class": "realistic",
                                    "stage": "merge",
                                    "execution_modes": ["wrapper", "seam", "forced-fallback"],
                                    "comparison": {
                                        "wrapper_vs_seam": {
                                            "status": "failed",
                                            "failure_code": "output-mismatch",
                                            "compared_modes": ["wrapper", "seam"],
                                        },
                                        "wrapper_vs_forced_fallback": {
                                            "status": "passed",
                                            "failure_code": "",
                                            "compared_modes": ["wrapper", "forced-fallback"],
                                        },
                                    },
                                }
                            ],
                            "records": [
                                {
                                    "fixture_id": "merge-realistic",
                                    "fixture_class": "realistic",
                                    "stage": "merge",
                                    "run_label": "merge-realistic-seam",
                                    "selected_mode": "seam",
                                    "status": "failed",
                                    "failure_code": "output-mismatch",
                                },
                                {
                                    "fixture_id": "merge-realistic",
                                    "fixture_class": "realistic",
                                    "stage": "merge",
                                    "run_label": "merge-realistic-forced-fallback",
                                    "selected_mode": "forced-fallback",
                                    "status": "passed",
                                    "failure_code": "",
                                },
                            ],
                            "summary": {
                                "total": 2,
                                "passed": 1,
                                "failed": 1,
                                "by_failure_code": {"output-mismatch": 1},
                            },
                            "verdict": "candidate-with-known-risks",
                            "windows_non_regression": {
                                "host": "windows-x86_64",
                                "seam_selected": False,
                                "status": "passed",
                            },
                        }
                    },
                },
            },
            extension_root=self.ext_dir,
        )

        loaded = bootstrap.load_state(self.ext_dir)
        qualification = loaded["source_build"]["qualification"]["extract_merge"]
        comparison = qualification["fixtures"][0]["comparison"]
        extract_merge = loaded["source_build"]["executable_boundary"]["extract_merge"]
        external_blender = loaded["source_build"]["external_blender"]

        self.assertEqual(external_blender["classification"]["status"], "external-bpy-smoke-ready")
        self.assertNotIn("comparison", external_blender)
        self.assertNotIn("comparison", external_blender["classification"])
        self.assertEqual(extract_merge["status"], "verified")
        self.assertNotIn("comparison", extract_merge)
        self.assertEqual(qualification["verdict"], "candidate-with-known-risks")
        self.assertEqual(comparison["wrapper_vs_seam"]["status"], "failed")
        self.assertEqual(comparison["wrapper_vs_seam"]["failure_code"], "output-mismatch")
        self.assertEqual(comparison["wrapper_vs_forced_fallback"]["status"], "passed")
        self.assertEqual(comparison["wrapper_vs_forced_fallback"]["failure_code"], "")
        self.assertEqual(loaded["install_state"], "blocked")
        self.assertFalse(loaded["last_verification"]["runtime_ready"])

    def test_normalize_state_normalizes_linux_arm64_qualification_extract_merge_without_unlocking_runtime(self) -> None:
        normalized = bootstrap.normalize_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "python_version": "3.11.9",
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "bpy_evidence_class": "external-bpy-smoke-ready",
                    "external_blender": {
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        }
                    },
                    "qualification": {
                        "extract_merge": {
                            "host": {"os": "linux", "arch": "aarch64", "platform_tag": "linux-aarch64", "host_class": "linux-arm64"},
                            "blender": {"candidate": {"path": "/opt/blender/blender", "source": "path"}},
                            "fixtures": [
                                {
                                    "fixture_id": "merge-realistic",
                                    "fixture_class": "realistic",
                                    "stage": "merge",
                                    "execution_modes": ["wrapper", "seam", "forced-fallback"],
                                    "runs": [
                                        {
                                            "fixture_id": "merge-realistic",
                                            "fixture_class": "realistic",
                                            "stage": "merge",
                                            "run_label": "merge-realistic-wrapper",
                                            "selected_mode": "wrapper",
                                            "status": "passed",
                                            "failure_code": "",
                                            "host": {"os": "linux", "arch": "aarch64"},
                                            "blender": {"candidate": {"path": "/opt/blender/blender"}},
                                            "outputs": {"artifact": "wrapper.json"},
                                            "logs": {"stdout_tail": ["wrapper ok"], "stderr_tail": []},
                                        },
                                        {
                                            "fixture_id": "merge-realistic",
                                            "fixture_class": "realistic",
                                            "stage": "merge",
                                            "run_label": "merge-realistic-seam",
                                            "selected_mode": "seam",
                                            "status": "failed",
                                            "failure_code": "output-mismatch",
                                            "host": {"os": "linux", "arch": "aarch64"},
                                            "blender": {"candidate": {"path": "/opt/blender/blender"}},
                                            "outputs": {"artifact": "seam.json"},
                                            "logs": {"stdout_tail": ["seam ok"], "stderr_tail": ["compare failed"]},
                                        },
                                    ],
                                    "comparison": {
                                        "fixture_id": "merge-realistic",
                                        "stage": "merge",
                                        "wrapper_vs_seam": {
                                            "status": "failed",
                                            "failure_code": "output-mismatch",
                                            "compared_modes": ["wrapper", "seam"],
                                        },
                                        "wrapper_vs_forced_fallback": {
                                            "status": "skipped",
                                            "failure_code": "",
                                            "compared_modes": ["wrapper", "forced-fallback"],
                                        },
                                    },
                                }
                            ],
                            "records": [
                                {
                                    "fixture_id": "merge-realistic",
                                    "fixture_class": "realistic",
                                    "stage": "merge",
                                    "run_label": "merge-realistic-seam",
                                    "selected_mode": "seam",
                                    "status": "failed",
                                    "failure_code": "output-mismatch",
                                    "host": {"os": "linux", "arch": "aarch64"},
                                    "blender": {"candidate": {"path": "/opt/blender/blender"}},
                                    "outputs": {"artifact": "seam.json"},
                                    "logs": {"stdout_tail": ["seam ok"], "stderr_tail": ["compare failed"]},
                                }
                            ],
                            "summary": {
                                "total": 2,
                                "passed": 1,
                                "failed": 1,
                                "by_failure_code": {"output-mismatch": 1},
                            },
                            "verdict": "candidate-with-known-risks",
                            "windows_non_regression": {
                                "host": "windows-x86_64",
                                "seam_selected": False,
                                "status": "passed",
                            },
                        }
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "status": "verified",
                            "proof_kind": "blender-subprocess",
                        }
                    },
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "spconv": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "deferred_work": [],
                    "non_blender_runtime_ready": True,
                },
            },
            self.ext_dir,
        )

        qualification = normalized["source_build"]["qualification"]["extract_merge"]

        self.assertEqual(qualification["schema_version"], 1)
        self.assertEqual(qualification["default_owner"], "context.venv_python")
        self.assertEqual(qualification["optional_owner"], "blender-subprocess")
        self.assertEqual(qualification["verdict"], "candidate-with-known-risks")
        self.assertEqual(qualification["summary"]["failed"], 1)
        self.assertEqual(qualification["summary"]["by_failure_code"], {"output-mismatch": 1})
        self.assertEqual(qualification["records"][0]["selected_mode"], "seam")
        self.assertEqual(qualification["records"][0]["failure_code"], "output-mismatch")
        self.assertEqual(qualification["fixtures"][0]["comparison"]["wrapper_vs_seam"]["failure_code"], "output-mismatch")
        self.assertFalse(qualification["windows_non_regression"]["seam_selected"])
        self.assertEqual(qualification["windows_non_regression"]["status"], "passed")
        self.assertEqual(normalized["source_build"]["external_blender"]["classification"]["status"], "external-bpy-smoke-ready")
        self.assertEqual(normalized["source_build"]["executable_boundary"]["extract_merge"]["status"], "verified")
        self.assertEqual(normalized["install_state"], "partial")
        self.assertFalse(normalized["last_verification"]["runtime_ready"])

    def test_normalize_state_keeps_ready_for_separate_defaulting_verdict_separate_from_full_runtime_readiness(self) -> None:
        normalized = bootstrap.normalize_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "python_version": "3.11.9",
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "external_blender": {
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        }
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "status": "verified",
                            "proof_kind": "blender-subprocess",
                        }
                    },
                    "qualification": {
                        "extract_merge": {
                            "records": [
                                {
                                    "fixture_id": "merge-realistic",
                                    "fixture_class": "realistic",
                                    "stage": "merge",
                                    "run_label": "merge-realistic-seam",
                                    "selected_mode": "seam",
                                    "status": "passed",
                                    "failure_code": "",
                                },
                                {
                                    "fixture_id": "merge-realistic",
                                    "fixture_class": "realistic",
                                    "stage": "merge",
                                    "run_label": "merge-realistic-wrapper",
                                    "selected_mode": "wrapper",
                                    "status": "passed",
                                    "failure_code": "",
                                },
                            ],
                            "fixtures": [
                                {
                                    "fixture_id": "merge-realistic",
                                    "fixture_class": "realistic",
                                    "stage": "merge",
                                    "execution_modes": ["wrapper", "seam", "forced-fallback"],
                                    "comparison": {
                                        "wrapper_vs_seam": {
                                            "status": "passed",
                                            "failure_code": "",
                                            "compared_modes": ["wrapper", "seam"],
                                        },
                                        "wrapper_vs_forced_fallback": {
                                            "status": "skipped",
                                            "failure_code": "",
                                            "compared_modes": ["wrapper", "forced-fallback"],
                                        },
                                    },
                                }
                            ],
                            "summary": {
                                "total": 2,
                                "passed": 2,
                                "failed": 0,
                                "by_failure_code": {},
                            },
                            "verdict": "ready-for-separate-defaulting-change",
                        }
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                },
            },
            self.ext_dir,
        )

        qualification = normalized["source_build"]["qualification"]["extract_merge"]

        self.assertEqual(qualification["verdict"], "ready-for-separate-defaulting-change")
        self.assertEqual(qualification["fixtures"][0]["comparison"]["wrapper_vs_seam"]["status"], "passed")
        self.assertEqual(normalized["source_build"]["external_blender"]["classification"]["status"], "external-bpy-smoke-ready")
        self.assertEqual(normalized["source_build"]["executable_boundary"]["extract_merge"]["status"], "verified")
        self.assertEqual(normalized["install_state"], "blocked")
        self.assertFalse(normalized["last_verification"]["runtime_ready"])

    def test_normalize_state_keeps_non_linux_arm64_hosts_unchanged_when_qualification_is_absent(self) -> None:
        for label, host_class in (("windows-x86_64", "windows-x86_64"), ("linux-x86_64", "linux-x86_64")):
            with self.subTest(label=label):
                normalized = bootstrap.normalize_state(
                    {
                        "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                        "install_state": "ready",
                        "python_version": "3.11.9",
                        "source_build": {
                            "status": "ready",
                            "mode": "prebuilt" if host_class == "windows-x86_64" else "source-build",
                            "host_class": host_class,
                            "blockers": [],
                            "blocked_reasons": [],
                        },
                    },
                    self.ext_dir,
                )

                self.assertEqual(normalized["install_state"], "ready")
                self.assertNotIn("qualification", normalized["source_build"])

    def test_write_error_state_persists_linux_arm64_blender_evidence_but_runtime_stays_not_ready(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        candidate = {
            "source": "path",
            "path": "/opt/blender/blender",
            "selected_because": "Selected PATH-visible Blender candidate.",
        }
        probe = {
            "status": "ok",
            "command": ["/opt/blender/blender", "--background"],
            "blender_version": "4.0.2",
            "python_version": "3.12.1",
            "smoke_result": "passed",
            "returncode": 0,
            "stdout_tail": ["probe ok"],
            "stderr_tail": [],
        }
        classification = module._classify_linux_arm64_bpy_viability(candidate, probe, wrapper_python_version="3.11.9")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
            },
            create=True,
        ), mock.patch.object(module, "_resolve_linux_arm64_blender_candidate", return_value=candidate, create=True), mock.patch.object(
            module, "_probe_linux_arm64_blender_bpy", return_value=probe, create=True
        ), mock.patch.object(module, "_classify_linux_arm64_bpy_viability", return_value=classification, create=True):
            preflight = module._preflight_check_summary(python_exe, {})
            planner = module.build_install_plan(host_os="linux", host_arch="aarch64")

        module._write_error_state(
            self.ext_dir,
            "blocked by external Blender mismatch evidence",
            preflight,
            requested_host_python=python_exe,
            bootstrap_resolution={"selected_source": "requested-host-python"},
            planner=planner,
        )

        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        source_build = state["source_build"]

        self.assertEqual(source_build["bpy_evidence_class"], "discovered-incompatible")
        self.assertEqual(source_build["stages"]["bpy"]["status"], "discovered-incompatible")
        self.assertEqual(source_build["external_blender"]["candidate"], candidate)
        self.assertEqual(source_build["external_blender"]["probe"], probe)
        self.assertEqual(source_build["external_blender"]["classification"]["blocker_codes"], ["blender-python-abi-mismatch"])
        self.assertFalse(source_build["stages"]["bpy"]["ready"])
        self.assertFalse(state["last_verification"]["runtime_ready"])

    def test_preflight_linux_arm64_blocks_pyg_stage_when_import_smoke_verification_fails(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=self._linux_arm64_ready_probe(), create=True
        ), mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "blocked",
                "ready": False,
                "verification": "import-smoke",
                "blockers": [
                    {
                        "category": "distribution",
                        "code": "pyg-import-smoke-failed",
                        "message": "torch_cluster import smoke failed on Linux ARM64 after source-build prerequisites were satisfied.",
                        "boundary": "upstream",
                        "owner": "upstream",
                    }
                ],
                "blocker_codes": ["pyg-import-smoke-failed"],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "error", "returncode": 1},
                ],
            },
            create=True,
        ):
            report = module._preflight_check_summary(python_exe, {})

        pyg_stage = report["source_build"]["stages"]["pyg"]
        blockers = {item["code"]: item for item in pyg_stage["blockers"]}

        self.assertEqual(report["source_build"]["status"], "blocked")
        self.assertFalse(report["source_build"]["non_blender_runtime_ready"])
        self.assertEqual(pyg_stage["status"], "blocked")
        self.assertFalse(pyg_stage["ready"])
        self.assertEqual(pyg_stage["verification"], "import-smoke")
        self.assertEqual(pyg_stage["blocker_codes"], ["pyg-import-smoke-failed"])
        self.assertEqual(blockers["pyg-import-smoke-failed"]["boundary"], "upstream")

    def test_verify_linux_arm64_pyg_import_smoke_marks_ready_when_all_checks_pass(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(
            module.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
            ],
        ):
            verification = module._verify_linux_arm64_pyg_import_smoke(python_exe)

        self.assertEqual(verification["status"], "ready")
        self.assertTrue(verification["ready"])
        self.assertEqual(verification["verification"], "import-smoke")
        self.assertEqual([item["status"] for item in verification["checks"]], ["ready", "ready"])
        self.assertEqual(verification["blockers"], [])

    def test_verify_linux_arm64_pyg_import_smoke_reports_wrapper_boundary_when_runner_fails(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module.subprocess, "run", side_effect=OSError("launcher failed")):
            verification = module._verify_linux_arm64_pyg_import_smoke(python_exe)

        blocker = verification["blockers"][0]

        self.assertEqual(verification["status"], "blocked")
        self.assertFalse(verification["ready"])
        self.assertEqual(blocker["code"], "pyg-import-smoke-failed")
        self.assertEqual(blocker["boundary"], "wrapper")
        self.assertEqual(blocker["owner"], "wrapper")
        self.assertEqual(blocker["failed_check"], "torch_scatter")
        self.assertEqual(blocker["details"], ["launcher failed"])

    def test_verify_linux_arm64_pyg_import_smoke_reports_environment_boundary_for_loader_failure(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(
            module.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(
                    ["python"],
                    1,
                    stdout="",
                    stderr="ImportError: libcusparse.so.12: cannot open shared object file: No such file or directory",
                ),
            ],
        ):
            verification = module._verify_linux_arm64_pyg_import_smoke(python_exe)

        blocker = verification["blockers"][0]

        self.assertEqual(verification["status"], "blocked")
        self.assertFalse(verification["ready"])
        self.assertEqual(blocker["boundary"], "environment")
        self.assertEqual(blocker["owner"], "environment")
        self.assertEqual(blocker["failed_check"], "torch_cluster")
        self.assertEqual(
            blocker["details"],
            ["ImportError: libcusparse.so.12: cannot open shared object file: No such file or directory"],
        )

    def test_verify_linux_arm64_pyg_import_smoke_reports_upstream_boundary_for_missing_module(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(
            module.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                ["python"],
                1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'torch_scatter'",
            ),
        ):
            verification = module._verify_linux_arm64_pyg_import_smoke(python_exe)

        blocker = verification["blockers"][0]

        self.assertEqual(verification["status"], "blocked")
        self.assertFalse(verification["ready"])
        self.assertEqual(blocker["boundary"], "upstream")
        self.assertEqual(blocker["owner"], "upstream")
        self.assertEqual(blocker["failed_check"], "torch_scatter")
        self.assertEqual(blocker["details"], ["ModuleNotFoundError: No module named 'torch_scatter'"])

    def test_linux_arm64_source_build_environment_prefers_cuda_12_8_when_present(self) -> None:
        module = load_setup_module()
        cuda_home = self.temp_dir / "cuda-12.8"
        python_exe = self.temp_dir / "venv" / "bin" / "python"
        (cuda_home / "bin").mkdir(parents=True, exist_ok=True)
        (cuda_home / "include").mkdir(parents=True, exist_ok=True)
        (cuda_home / "lib64").mkdir(parents=True, exist_ok=True)
        (cuda_home / "targets" / "sbsa-linux" / "include").mkdir(parents=True, exist_ok=True)
        (cuda_home / "targets" / "sbsa-linux" / "lib").mkdir(parents=True, exist_ok=True)
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        (cuda_home / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
        python_exe.write_text("#!/bin/sh\n", encoding="utf-8")

        with mock.patch.object(module, "LINUX_ARM64_CUDA_12_8_HOME", cuda_home), mock.patch.dict(
            module.os.environ,
            {
                "PATH": "/usr/bin",
                "LD_LIBRARY_PATH": "/usr/lib",
                "LIBRARY_PATH": "/usr/local/lib",
                "CPATH": "/usr/include",
                "C_INCLUDE_PATH": "/usr/include/c",
                "CPLUS_INCLUDE_PATH": "/usr/include/c++",
            },
            clear=True,
        ):
            env = module._linux_arm64_source_build_environment(python_exe=python_exe)

        self.assertIsNotNone(env)
        assert env is not None
        self.assertEqual(env["CUDA_HOME"], str(cuda_home))
        self.assertEqual(env["CUDA_PATH"], str(cuda_home))
        self.assertEqual(env["CUDACXX"], str(cuda_home / "bin" / "nvcc"))
        self.assertEqual(env["CUDA_BIN_PATH"], str(cuda_home))
        self.assertEqual(env["CUMM_CUDA_VERSION"], "12.8")
        self.assertEqual(env["CUMM_DISABLE_JIT"], "1")
        self.assertEqual(env["SPCONV_DISABLE_JIT"], "1")
        self.assertEqual(env["CUDAFLAGS"], "-allow-unsupported-compiler")
        self.assertEqual(env["CMAKE_CUDA_FLAGS"], "-allow-unsupported-compiler")
        self.assertEqual(env["PATH"], os.pathsep.join([str(python_exe.parent), str(cuda_home / "bin"), "/usr/bin"]))
        self.assertEqual(
            env["CPATH"],
            os.pathsep.join(
                [
                    str(cuda_home / "include"),
                    str(cuda_home / "targets" / "sbsa-linux" / "include"),
                    "/usr/include",
                ]
            ),
        )
        self.assertEqual(
            env["C_INCLUDE_PATH"],
            os.pathsep.join(
                [
                    str(cuda_home / "include"),
                    str(cuda_home / "targets" / "sbsa-linux" / "include"),
                    "/usr/include/c",
                ]
            ),
        )
        self.assertEqual(
            env["CPLUS_INCLUDE_PATH"],
            os.pathsep.join(
                [
                    str(cuda_home / "include"),
                    str(cuda_home / "targets" / "sbsa-linux" / "include"),
                    "/usr/include/c++",
                ]
            ),
        )
        self.assertEqual(
            env["LIBRARY_PATH"],
            os.pathsep.join(
                [
                    str(cuda_home / "lib64"),
                    str(cuda_home / "targets" / "sbsa-linux" / "lib"),
                    "/usr/local/lib",
                ]
            ),
        )
        self.assertEqual(
            env["LD_LIBRARY_PATH"],
            os.pathsep.join(
                [
                    str(cuda_home / "lib64"),
                    str(cuda_home / "targets" / "sbsa-linux" / "lib"),
                    "/usr/lib",
                ]
            ),
        )

    def test_linux_arm64_source_build_environment_keeps_requested_venv_bin_on_path_even_if_python_is_symlink(self) -> None:
        module = load_setup_module()
        cuda_home = self.temp_dir / "cuda-12.8"
        requested_python = self.temp_dir / "venv" / "bin" / "python"
        target_python = self.temp_dir / "system" / "python3.12"
        (cuda_home / "bin").mkdir(parents=True, exist_ok=True)
        (cuda_home / "include").mkdir(parents=True, exist_ok=True)
        requested_python.parent.mkdir(parents=True, exist_ok=True)
        target_python.parent.mkdir(parents=True, exist_ok=True)
        (cuda_home / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
        target_python.write_text("#!/bin/sh\n", encoding="utf-8")
        requested_python.symlink_to(target_python)

        with mock.patch.object(module, "LINUX_ARM64_CUDA_12_8_HOME", cuda_home), mock.patch.dict(
            module.os.environ,
            {"PATH": "/usr/bin"},
            clear=True,
        ):
            env = module._linux_arm64_source_build_environment(python_exe=requested_python)

        self.assertIsNotNone(env)
        assert env is not None
        self.assertTrue(env["PATH"].startswith(str(requested_python.parent) + os.pathsep))

    def test_patched_linux_arm64_cumm_common_source_prefers_cuda_env_and_sbsa_dirs(self) -> None:
        module = load_setup_module()
        original = '''        else:
            try:
                nvcc_path = subprocess.check_output(["which", "nvcc"
                                                    ]).decode("utf-8").strip()
                lib = Path(nvcc_path).parent.parent / "lib"
                include = Path(nvcc_path).parent.parent / "targets/x86_64-linux/include"
                if lib.exists() and include.exists():
                    if (lib / "libcudart.so").exists() and (include / "cuda.h").exists():
                        # should be nvidia conda package
                        _CACHED_CUDA_INCLUDE_LIB = ([include], lib)
                        return _CACHED_CUDA_INCLUDE_LIB
            except:
                pass 

            linux_cuda_root = Path("/usr/local/cuda")
            include = linux_cuda_root / f"include"
            lib64 = linux_cuda_root / f"lib64"
            assert linux_cuda_root.exists(), f"can't find cuda in {linux_cuda_root} install via cuda installer or conda first."
        _CACHED_CUDA_INCLUDE_LIB = ([include], lib64)
'''

        patched = module._patched_linux_arm64_cumm_common_source(original)

        self.assertIn(module.LINUX_ARM64_CUMM_PATCH_MARKER, patched)
        self.assertIn('os.getenv("CUDA_HOME", "").strip() or os.getenv("CUDA_PATH", "").strip()', patched)
        self.assertIn('resolved_root / "targets" / "sbsa-linux" / "include"', patched)
        self.assertIn('resolved_root / "targets" / "sbsa-linux" / "lib"', patched)
        self.assertIn('_CACHED_CUDA_INCLUDE_LIB = (valid_includes, valid_libs[0])', patched)

    def test_linux_arm64_source_build_environment_sets_conservative_cumm_arch_list_for_newer_gpu_capability(self) -> None:
        module = load_setup_module()
        cuda_home = self.temp_dir / "cuda-12.8"
        python_exe = self.temp_dir / "venv" / "bin" / "python"
        (cuda_home / "bin").mkdir(parents=True, exist_ok=True)
        (cuda_home / "include").mkdir(parents=True, exist_ok=True)
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        (cuda_home / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
        python_exe.write_text("#!/bin/sh\n", encoding="utf-8")

        with mock.patch.object(module, "LINUX_ARM64_CUDA_12_8_HOME", cuda_home), mock.patch.object(
            module, "_probe_linux_arm64_gpu_compute_capability", return_value=(12, 1)
        ), mock.patch.dict(module.os.environ, {"PATH": "/usr/bin"}, clear=True):
            env = module._linux_arm64_source_build_environment(python_exe=python_exe)

        self.assertIsNotNone(env)
        assert env is not None
        self.assertEqual(env["CUMM_CUDA_ARCH_LIST"], module.LINUX_ARM64_CUMM_FALLBACK_CUDA_ARCH_LIST)

    def test_linux_arm64_source_build_environment_preserves_explicit_cumm_arch_list_override(self) -> None:
        module = load_setup_module()
        cuda_home = self.temp_dir / "cuda-12.8"
        python_exe = self.temp_dir / "venv" / "bin" / "python"
        (cuda_home / "bin").mkdir(parents=True, exist_ok=True)
        (cuda_home / "include").mkdir(parents=True, exist_ok=True)
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        (cuda_home / "bin" / "nvcc").write_text("#!/bin/sh\n", encoding="utf-8")
        python_exe.write_text("#!/bin/sh\n", encoding="utf-8")

        with mock.patch.object(module, "LINUX_ARM64_CUDA_12_8_HOME", cuda_home), mock.patch.object(
            module, "_probe_linux_arm64_gpu_compute_capability", return_value=(12, 1)
        ), mock.patch.dict(module.os.environ, {"PATH": "/usr/bin", "CUMM_CUDA_ARCH_LIST": "8.7+PTX"}, clear=True):
            env = module._linux_arm64_source_build_environment(python_exe=python_exe)

        self.assertIsNotNone(env)
        assert env is not None
        self.assertEqual(env["CUMM_CUDA_ARCH_LIST"], "8.7+PTX")

    def test_install_linux_arm64_pyg_source_build_passes_preferred_cuda_env(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        pip = [str(python_exe), "-m", "pip"]
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        source_build_env = {
            "CUDA_HOME": "/usr/local/cuda-12.8",
            "CUDA_PATH": "/usr/local/cuda-12.8",
            "CUDACXX": "/usr/local/cuda-12.8/bin/nvcc",
        }

        with mock.patch.object(module, "_linux_arm64_source_build_environment", return_value=source_build_env), mock.patch.object(
            module, "_run"
        ) as run_mock:
            packages = module._install_linux_arm64_pyg_source_build(
                pip,
                unirig_dir,
                {"packages": ["torch_scatter", "torch_cluster"]},
            )

        self.assertEqual(packages, ["torch_scatter", "torch_cluster"])
        self.assertEqual(run_mock.call_args.kwargs["env"], source_build_env)

    def test_run_linux_arm64_spconv_guarded_bringup_passes_preferred_cuda_env(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        pip = [str(python_exe), "-m", "pip"]
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        source_build_env = {
            "CUDA_HOME": "/usr/local/cuda-12.8",
            "CUDA_PATH": "/usr/local/cuda-12.8",
            "CUDACXX": "/usr/local/cuda-12.8/bin/nvcc",
        }

        with mock.patch.object(module, "_linux_arm64_source_build_environment", return_value=source_build_env), mock.patch.object(
            module.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
            ],
        ) as run_mock, mock.patch.object(
            module,
            "_verify_linux_arm64_spconv_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "checks": [{"label": "spconv.pytorch", "status": "ready", "returncode": 0}],
                "blockers": [],
                "blocker_codes": [],
                "boundary": "wrapper",
            },
        ), mock.patch.object(module, "_patch_linux_arm64_installed_cumm_common", return_value=self.ext_dir / "cumm" / "common.py"):
            result = module._run_linux_arm64_spconv_guarded_bringup(pip, python_exe, unirig_dir)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(run_mock.call_args_list[0].kwargs["env"], source_build_env)
        self.assertEqual(run_mock.call_args_list[1].kwargs["env"], source_build_env)
        self.assertEqual(run_mock.call_args_list[2].kwargs["env"], source_build_env)
        self.assertEqual(run_mock.call_args_list[3].kwargs["env"], source_build_env)

    def test_run_linux_arm64_spconv_guarded_bringup_marks_ready_after_cumm_spconv_and_import_smoke(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        pip = [str(python_exe), "-m", "pip"]
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        source_build_env = {"CUDA_HOME": "/usr/local/cuda-12.8"}

        with mock.patch.object(module, "_linux_arm64_source_build_environment", return_value=source_build_env), mock.patch.object(
            module.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
            ],
        ) as run_mock, mock.patch.object(
            module, "_patch_linux_arm64_installed_cumm_common", return_value=self.ext_dir / "cumm" / "common.py"
        ):
            result = module._run_linux_arm64_spconv_guarded_bringup(pip, python_exe, unirig_dir)

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["ready"])
        self.assertEqual(result["verification"], "import-smoke")
        self.assertEqual(result["packages"], ["cumm", "spconv"])
        self.assertEqual([item["label"] for item in result["checks"]], ["cumm", "spconv", "spconv.pytorch"])
        self.assertEqual([item["status"] for item in result["checks"]], ["ready", "ready", "ready"])
        self.assertEqual(result["blockers"], [])
        self.assertEqual(
            run_mock.call_args_list[0].args[0],
            [str(python_exe), "-m", "pip", "uninstall", "-y", "cumm", "cumm-cu128", "spconv", "spconv-cu128"],
        )
        self.assertEqual(
            run_mock.call_args_list[1].args[0],
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                module.LINUX_ARM64_PCCM_PACKAGE,
                *module.LINUX_ARM64_SPCONV_BUILD_PREREQUISITES,
            ],
        )
        self.assertEqual(
            run_mock.call_args_list[2].args[0],
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-build-isolation",
                "--no-binary=cumm,spconv",
                module.LINUX_ARM64_CUMM_SOURCE_URL,
            ],
        )
        self.assertEqual(
            run_mock.call_args_list[3].args[0],
            [
                str(python_exe),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-build-isolation",
                "--no-binary=cumm,spconv",
                module.LINUX_ARM64_SPCONV_SOURCE_URL,
            ],
        )

    def test_run_linux_arm64_spconv_guarded_bringup_classifies_wrapper_failures(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        pip = [str(python_exe), "-m", "pip"]
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(module.subprocess, "run", side_effect=OSError("pip launcher missing")):
            result = module._run_linux_arm64_spconv_guarded_bringup(pip, python_exe, unirig_dir)

        blocker = result["blockers"][0]
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["ready"])
        self.assertEqual(blocker["boundary"], "wrapper")
        self.assertEqual(blocker["owner"], "wrapper")
        self.assertEqual(blocker["failed_check"], "cumm")
        self.assertEqual(blocker["details"], ["pip launcher missing"])

    def test_run_linux_arm64_spconv_guarded_bringup_classifies_environment_failures(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        pip = [str(python_exe), "-m", "pip"]
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(
            module.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(
                    ["python"],
                    1,
                    stdout="",
                    stderr="ImportError: libcusparse.so.12: cannot open shared object file: No such file or directory",
                ),
            ],
        ), mock.patch.object(module, "_patch_linux_arm64_installed_cumm_common", return_value=self.ext_dir / "cumm" / "common.py"):
            result = module._run_linux_arm64_spconv_guarded_bringup(pip, python_exe, unirig_dir)

        blocker = result["blockers"][0]
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["ready"])
        self.assertEqual(result["verification"], "import-smoke")
        self.assertEqual(blocker["boundary"], "environment")
        self.assertEqual(blocker["owner"], "environment")
        self.assertEqual(blocker["failed_check"], "spconv.pytorch")

    def test_run_linux_arm64_spconv_guarded_bringup_classifies_upstream_package_failures(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        pip = [str(python_exe), "-m", "pip"]
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(
            module.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                ["python"],
                1,
                stdout="",
                stderr="ERROR: No matching distribution found for cumm",
            ),
        ):
            result = module._run_linux_arm64_spconv_guarded_bringup(pip, python_exe, unirig_dir)

        blocker = result["blockers"][0]
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["ready"])
        self.assertEqual(blocker["boundary"], "upstream-package")
        self.assertEqual(blocker["owner"], "upstream-package")
        self.assertEqual(blocker["failed_check"], "cumm")
        self.assertEqual(blocker["details"], ["ERROR: No matching distribution found for cumm"])

    def test_preflight_reports_linux_arm64_blocker_families_from_mocked_probe_facts(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        probe = {
            "nvcc_path": "",
            "cxx_compiler": "",
            "python_headers_ready": False,
            "python_header": "",
            "cuda_home": "",
        }

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.12.3"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=probe, create=True
        ), mock.patch.object(
            module,
            "_linux_arm64_bpy_evidence",
            return_value={"candidate": {"source": "missing", "path": "", "selected_because": ""}, "probe": {}, "classification": {"status": "missing"}},
            create=True,
        ):
            report = module._preflight_check_summary(python_exe, {})

        checks = {item["id"]: item for item in report["checks"]}
        blockers = {item["code"]: item for item in report["blockers"]}

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(checks["linux-arm64-missing-distributions"]["status"], "fail")
        self.assertEqual(checks["linux-arm64-pyg-source-build"]["status"], "pass")
        self.assertEqual(checks["linux-arm64-spconv-guarded-source-build"]["status"], "pass")
        self.assertEqual(checks["linux-arm64-nvcc-toolchain"]["status"], "fail")
        self.assertEqual(checks["linux-arm64-python-headers"]["status"], "fail")
        self.assertEqual(checks["linux-arm64-bpy-portability"]["status"], "fail")
        self.assertEqual(blockers["missing-distribution"]["category"], "distribution")
        self.assertEqual(blockers["missing-nvcc-toolchain"]["category"], "toolchain")
        self.assertEqual(blockers["missing-python-headers"]["category"], "headers")
        self.assertEqual(blockers["bpy-portability-risk"]["category"], "portability")
        self.assertTrue(any("distribution" in item.lower() for item in report["blocked"]))
        self.assertFalse(any("pyg" in item.lower() for item in report["blocked"]))
        self.assertTrue(any("nvcc" in item.lower() for item in report["blocked"]))
        self.assertTrue(any("python.h" in item.lower() or "headers" in item.lower() for item in report["blocked"]))
        self.assertTrue(any("bpy" in item.lower() for item in report["blocked"]))

    def test_preflight_linux_arm64_keeps_toolchain_ready_checks_green_while_blocking_dependency_gaps(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python"
        python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        probe = {
            "nvcc_path": "/usr/local/cuda/bin/nvcc",
            "cxx_compiler": "/usr/bin/g++",
            "python_headers_ready": True,
            "python_header": "/usr/include/python3.12/Python.h",
            "cuda_home": "/usr/local/cuda",
        }

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="arm64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.12.3"), mock.patch.object(
            module, "_probe_linux_arm64_build_environment", return_value=probe, create=True
        ), mock.patch.object(
            module,
            "_linux_arm64_bpy_evidence",
            return_value={"candidate": {"source": "missing", "path": "", "selected_because": ""}, "probe": {}, "classification": {"status": "missing"}},
            create=True,
        ):
            report = module._preflight_check_summary(python_exe, {})

        checks = {item["id"]: item for item in report["checks"]}

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(checks["linux-arm64-nvcc-toolchain"]["status"], "pass")
        self.assertEqual(checks["linux-arm64-python-headers"]["status"], "pass")
        self.assertEqual(checks["linux-arm64-missing-distributions"]["status"], "fail")
        self.assertEqual(checks["linux-arm64-pyg-source-build"]["status"], "pass")
        self.assertEqual(checks["linux-arm64-spconv-guarded-source-build"]["status"], "pass")
        self.assertEqual(checks["linux-arm64-bpy-portability"]["status"], "fail")
        self.assertFalse(any("python.h" in item.lower() for item in report["blocked"]))
        self.assertFalse(any("nvcc" in item.lower() for item in report["blocked"]))
        self.assertFalse(any("pyg" in item.lower() for item in report["blocked"]))

    def test_windows_bootstrap_uses_requested_interpreter_without_wrapper_resolution(self) -> None:
        module = load_setup_module()
        requested = Path(r"C:\Python312\python.exe")

        requested_host_python, bootstrap_python, resolution = module._resolve_bootstrap_python({"python_exe": str(requested)})

        self.assertEqual(requested_host_python, requested)
        self.assertEqual(bootstrap_python, requested)
        self.assertEqual(resolution["selected_python"], str(requested))
        self.assertEqual(resolution["selected_source"], "requested-host-python")

    def test_windows_x86_64_install_policy_locks_validated_pinned_prebuilt_stack(self) -> None:
        module = load_setup_module()

        policy = module._build_host_install_policy("windows", "x86_64")

        self.assertEqual(policy["host_class"], "windows-x86_64")
        self.assertEqual(policy["support_posture"], "validated")
        self.assertEqual(policy["install_mode"], "pinned-prebuilt")
        self.assertEqual(policy["profile"], "pinned-upstream-wrapper")
        self.assertEqual(policy["spconv_package"], module.WINDOWS_SPCONV_PACKAGE_DEFAULT)
        self.assertEqual(policy["cumm_package"], module.WINDOWS_CUMM_PACKAGE_DEFAULT)
        self.assertEqual(policy["triton_package"], module.TRITON_WINDOWS_PACKAGE_DEFAULT)
        self.assertEqual(policy["flash_attn_wheel"], module.FLASH_ATTN_WHEEL_DEFAULT)
        self.assertEqual(policy["smoke_check_policy"], "windows-runtime-imports")
        self.assertEqual(
            [check["label"] for check in policy["smoke_checks"]],
            [check["label"] for check in module.WINDOWS_RUNTIME_SMOKE_CHECKS],
        )

    def test_windows_validated_contract_exposes_pinned_dependency_entries_and_smoke_policy(self) -> None:
        module = load_setup_module()

        contract = module._windows_validated_runtime_contract()

        self.assertEqual(contract["install_mode"], "pinned-prebuilt")
        self.assertEqual(contract["profile"], "pinned-upstream-wrapper")
        self.assertEqual(contract["smoke_check_policy"], "windows-runtime-imports")
        self.assertEqual(
            [item["name"] for item in contract["dependency_entries"]],
            ["cumm", "spconv", "triton", "flash_attn", "sitecustomize"],
        )
        self.assertEqual(contract["dependency_entries"][0]["package"], module.WINDOWS_CUMM_PACKAGE_DEFAULT)
        self.assertEqual(contract["dependency_entries"][1]["package"], module.WINDOWS_SPCONV_PACKAGE_DEFAULT)
        self.assertEqual(contract["dependency_entries"][2]["package"], module.TRITON_WINDOWS_PACKAGE_DEFAULT)
        self.assertEqual(contract["dependency_entries"][3]["wheel"], module.FLASH_ATTN_WHEEL_DEFAULT)
        self.assertEqual(
            [check["label"] for check in contract["smoke_checks"]],
            [check["label"] for check in module.WINDOWS_RUNTIME_SMOKE_CHECKS],
        )
        for dependency in contract["dependency_entries"]:
            self.assertFalse(any(key in dependency for key in ["stage", "verification", "allowed_statuses", "reason_code"]))

    def test_windows_amd64_build_plan_reuses_validated_contract_dependency_entries(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Windows"), mock.patch.object(
            module.platform, "machine", return_value="AMD64"
        ):
            plan = module.build_install_plan()

        contract = module._windows_validated_runtime_contract()

        self.assertEqual(plan["host_class"], "windows-x86_64")
        self.assertEqual(plan["install_mode"], contract["install_mode"])
        self.assertEqual(plan["support_posture"], "validated")
        self.assertEqual(plan["stages"], [])
        self.assertEqual(plan["deferred"], [])
        self.assertEqual(plan["dependencies"][-5:], contract["dependency_entries"])
        for dependency in plan["dependencies"][-5:]:
            self.assertFalse(any(key in dependency for key in ["stage", "verification", "allowed_statuses", "reason_code"]))

    def test_windows_amd64_alias_uses_the_same_validated_pinned_prebuilt_policy(self) -> None:
        module = load_setup_module()

        policy = module._build_host_install_policy("Windows", "AMD64")

        self.assertEqual(policy["host_class"], "windows-x86_64")
        self.assertEqual(policy["install_mode"], "pinned-prebuilt")
        self.assertEqual(policy["profile"], "pinned-upstream-wrapper")
        self.assertEqual(policy["spconv_package"], module.WINDOWS_SPCONV_PACKAGE_DEFAULT)
        self.assertEqual(policy["cumm_package"], module.WINDOWS_CUMM_PACKAGE_DEFAULT)

    def test_windows_x86_64_policy_is_not_classified_as_linux_or_linux_arm64(self) -> None:
        module = load_setup_module()

        windows_policy = module._build_host_install_policy("windows", "x86_64")
        linux_policy = module._build_host_install_policy("linux", "x86_64")
        linux_arm64_policy = module._build_host_install_policy("linux", "aarch64")

        self.assertEqual(windows_policy["host_class"], "windows-x86_64")
        self.assertEqual(linux_policy["host_class"], "linux-x86_64")
        self.assertEqual(linux_arm64_policy["host_class"], "linux-arm64")
        self.assertNotEqual(windows_policy["host_class"], linux_policy["host_class"])
        self.assertNotEqual(windows_policy["host_class"], linux_arm64_policy["host_class"])
        self.assertNotEqual(windows_policy["install_mode"], linux_arm64_policy["install_mode"])
        self.assertEqual(linux_arm64_policy["install_mode"], "staged-source-build")
        self.assertEqual(linux_arm64_policy["profile"], "linux-arm64-runtime-bringup")
        self.assertEqual(linux_arm64_policy["stages"], ["baseline", "pyg", "spconv", "bpy-deferred"])
        self.assertEqual(windows_policy["smoke_check_policy"], "windows-runtime-imports")
        self.assertEqual(linux_policy["smoke_check_policy"], "standard")
        self.assertEqual(linux_arm64_policy["smoke_check_policy"], "blocked")

    def test_classify_host_detects_windows_x86_64_from_platform_facts(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Windows"), mock.patch.object(
            module.platform, "machine", return_value="AMD64"
        ):
            host_class = module._classify_host()

        self.assertEqual(host_class, "windows-x86_64")

    def test_classify_host_detects_linux_x86_64_from_platform_facts(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="x86_64"
        ):
            host_class = module._classify_host()

        self.assertEqual(host_class, "linux-x86_64")

    def test_classify_host_detects_linux_arm64_from_platform_facts(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="arm64"
        ):
            host_class = module._classify_host()

        self.assertEqual(host_class, "linux-arm64")

    def test_classify_host_marks_other_platforms_as_unsupported(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Darwin"), mock.patch.object(
            module.platform, "machine", return_value="x86_64"
        ):
            host_class = module._classify_host()

        self.assertEqual(host_class, "unsupported")

    def test_build_install_plan_returns_validated_windows_x86_64_plan(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Windows"), mock.patch.object(
            module.platform, "machine", return_value="AMD64"
        ):
            plan = module.build_install_plan()

        dependency_strategies = {item["name"]: item["strategy"] for item in plan["dependencies"]}

        self.assertEqual(plan["host_class"], "windows-x86_64")
        self.assertEqual(plan["support_posture"], "validated")
        self.assertEqual(plan["install_mode"], "pinned-prebuilt")
        self.assertEqual(plan["stages"], [])
        self.assertEqual(plan["deferred"], [])
        self.assertEqual(dependency_strategies["spconv"], "windows-pinned-prebuilt")
        self.assertEqual(dependency_strategies["cumm"], "windows-pinned-prebuilt")
        self.assertEqual(dependency_strategies["triton"], "windows-pinned-package")
        self.assertEqual(dependency_strategies["flash_attn"], "windows-pinned-wheel")
        spconv_dependency = next(item for item in plan["dependencies"] if item["name"] == "spconv")
        self.assertEqual(spconv_dependency["package"], module.WINDOWS_SPCONV_PACKAGE_DEFAULT)
        self.assertFalse(any(key in spconv_dependency for key in ["stage", "verification", "allowed_statuses", "reason_code"]))

    def test_build_install_plan_keeps_windows_x86_64_isolated_from_linux_arm64_blender_helpers(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module, "_resolve_linux_arm64_blender_candidate", autospec=True) as resolve_mock, mock.patch.object(
            module, "_probe_linux_arm64_blender_bpy", autospec=True
        ) as probe_mock:
            plan = module.build_install_plan(host_os="Windows", host_arch="AMD64")

        self.assertEqual(plan["host_class"], "windows-x86_64")
        self.assertEqual(plan["install_mode"], "pinned-prebuilt")
        resolve_mock.assert_not_called()
        probe_mock.assert_not_called()

    def test_preflight_windows_x86_64_never_invokes_linux_arm64_blender_helpers(self) -> None:
        module = load_setup_module()
        python_exe = self.ext_dir / "python.exe"
        python_exe.write_text("#!python\n", encoding="utf-8")

        with mock.patch.object(module.platform, "system", return_value="Windows"), mock.patch.object(
            module.platform, "machine", return_value="AMD64"
        ), mock.patch.object(module, "_probe_python_version", return_value="3.11.9"), mock.patch.object(
            module, "_resolve_linux_arm64_blender_candidate", autospec=True
        ) as resolve_mock, mock.patch.object(module, "_probe_linux_arm64_blender_bpy", autospec=True) as probe_mock:
            report = module._preflight_check_summary(python_exe, {})

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["host"]["platform_tag"], "windows-amd64")
        resolve_mock.assert_not_called()
        probe_mock.assert_not_called()

    def test_build_install_plan_returns_supported_linux_x86_64_plan(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="x86_64"
        ):
            plan = module.build_install_plan()

        dependency_strategies = {item["name"]: item["strategy"] for item in plan["dependencies"]}

        self.assertEqual(plan["host_class"], "linux-x86_64")
        self.assertEqual(plan["support_posture"], "supported")
        self.assertEqual(plan["install_mode"], "prebuilt")
        self.assertEqual(plan["deferred"], [])
        self.assertEqual(dependency_strategies["torch"], "torch-index-cu128")
        self.assertEqual(dependency_strategies["pyg"], "pyg-wheel-index")
        self.assertEqual(dependency_strategies["spconv"], "generic-prebuilt-package")
        self.assertNotIn("flash_attn", dependency_strategies)

    def test_build_install_plan_returns_staged_linux_arm64_bringup_plan(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.platform, "system", return_value="Linux"), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ):
            plan = module.build_install_plan()

        dependency_strategies = {item["name"]: item["strategy"] for item in plan["dependencies"]}

        self.assertEqual(plan["host_class"], "linux-arm64")
        self.assertEqual(plan["support_posture"], "experimental-unvalidated")
        self.assertEqual(plan["install_mode"], "staged-source-build")
        self.assertEqual(plan["profile"], "linux-arm64-runtime-bringup")
        self.assertEqual(plan["stages"], ["baseline", "pyg", "spconv", "bpy-deferred"])
        self.assertEqual(dependency_strategies["torch"], "torch-index-cu128")
        self.assertEqual(dependency_strategies["pyg"], "linux-arm64-source-build-only")
        self.assertEqual(dependency_strategies["spconv"], "linux-arm64-guarded-source-build")
        self.assertEqual(dependency_strategies["bpy"], "deferred-portability-review")
        pyg_dependency = next(item for item in plan["dependencies"] if item["name"] == "pyg")
        spconv_dependency = next(item for item in plan["dependencies"] if item["name"] == "spconv")
        self.assertEqual(pyg_dependency["packages"], ["torch_scatter", "torch_cluster"])
        self.assertEqual(pyg_dependency["stage"], "pyg")
        self.assertEqual(pyg_dependency["verification"], "deferred")
        self.assertEqual(spconv_dependency["stage"], "spconv")
        self.assertEqual(spconv_dependency["verification"], "import-smoke")
        self.assertEqual(spconv_dependency["allowed_statuses"], ["blocked", "deferred", "build-ready", "ready"])
        self.assertEqual(plan["deferred"], ["bpy-portability"])

    def test_install_runtime_packages_keeps_spconv_deferred_until_pyg_import_smoke_is_ready(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")
        plan = {
            "host_class": "linux-arm64",
            "support_posture": "experimental-unvalidated",
            "install_mode": "staged-source-build",
            "dependencies": [
                {"name": "torch", "strategy": "torch-index-cu128", "index_url": module.TORCH_INDEX_URL, "packages": list(module.TORCH_PACKAGES)},
                {"name": "upstream-requirements", "strategy": "upstream-requirements-file"},
                {
                    "name": "pyg",
                    "strategy": "linux-arm64-source-build-only",
                    "reason_code": "pyg-source-build",
                    "stage": "pyg",
                    "packages": ["torch_scatter", "torch_cluster"],
                    "verification": "deferred",
                },
                {"name": "spconv", "strategy": "blocked-missing-distribution", "reason_code": "missing-distribution"},
                {"name": "bpy", "strategy": "deferred-portability-review", "reason_code": "bpy-portability"},
            ],
            "deferred": ["spconv-port", "bpy-portability"],
        }

        with mock.patch.object(module, "_venv_python", return_value=self.ext_dir / "venv" / "bin" / "python"), mock.patch.object(
            module, "build_install_plan", return_value=plan
        ), mock.patch.object(module, "_run") as run_mock, mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "source-build-only",
                "ready": False,
                "verification": "deferred",
                "blockers": [],
                "blocker_codes": [],
                "checks": [],
                "boundary": "wrapper",
            },
            create=True,
        ), mock.patch.object(module, "_run_linux_arm64_spconv_guarded_bringup", create=True) as spconv_helper_mock:
            result = module._install_runtime_packages(self.ext_dir, {})

        blocked = {item["code"]: item for item in result["blocked"]}
        commands = [call.args[0] for call in run_mock.mock_calls]
        spconv_stage = result["source_build"]["stages"]["spconv"]

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["profile"], "linux-arm64-runtime-bringup")
        self.assertEqual(result["host_class"], "linux-arm64")
        self.assertEqual(result["deferred_work"], ["spconv-port", "bpy-portability"])
        self.assertEqual(result["steps"], ["bootstrap-python", "torch", "upstream-requirements", "linux-arm64-pyg-source-build"])
        self.assertEqual(spconv_stage["status"], "deferred")
        self.assertEqual(spconv_stage["blocked_by_stage"], "pyg")
        self.assertEqual(spconv_stage["packages"], ["cumm", "spconv"])
        self.assertEqual(blocked["bpy-portability-risk"]["action"], "stop")
        spconv_helper_mock.assert_not_called()
        self.assertEqual(commands[0], [str(self.ext_dir / "venv" / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        self.assertIn(
            [
                str(self.ext_dir / "venv" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "-r",
                str(self.ext_dir / ".unirig-runtime" / "requirements.upstream.linux-arm64.txt"),
            ],
            commands,
        )
        self.assertIn(
            [
                str(self.ext_dir / "venv" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-build-isolation",
                "--no-binary=torch_scatter,torch_cluster",
                "torch_scatter",
                "torch_cluster",
            ],
            commands,
        )

    def test_main_stops_before_runtime_provisioning_when_linux_arm64_baseline_is_blocked(self) -> None:
        module = load_setup_module()
        blocked_preflight = {
            "status": "blocked",
            "checked_at": "2026-04-12T00:00:00+00:00",
            "host": {"os": "linux", "arch": "aarch64", "platform_tag": "linux-aarch64"},
            "observed": {
                "python_exe": sys.executable,
                "python_version": "3.12.3",
                "requested_host_python": sys.executable,
            },
            "checks": [
                {
                    "id": "linux-arm64-real-nvcc",
                    "label": "Real nvcc compiler",
                    "required": "linux-arm64 diagnostic preflight",
                    "status": "fail",
                    "message": "Linux ARM64 bringup requires the real NVIDIA nvcc compiler; wrapper shims or missing executables are not sufficient.",
                }
            ],
            "blockers": [
                {
                    "category": "toolchain",
                    "code": "missing-real-nvcc",
                    "message": "Linux ARM64 bringup requires the real NVIDIA nvcc compiler; wrapper shims or missing executables are not sufficient.",
                    "action": "stop",
                    "boundary": "environment",
                    "owner": "environment",
                }
            ],
            "blocked": ["Linux ARM64 bringup requires the real NVIDIA nvcc compiler; wrapper shims or missing executables are not sufficient."],
            "baseline": {
                "ready": False,
                "blockers": [
                    {
                        "category": "toolchain",
                        "code": "missing-real-nvcc",
                        "message": "Linux ARM64 bringup requires the real NVIDIA nvcc compiler; wrapper shims or missing executables are not sufficient.",
                        "action": "stop",
                        "boundary": "environment",
                        "owner": "environment",
                    }
                ],
            },
            "source_build": {
                "status": "blocked",
                "current_stage": "baseline",
                "non_blender_runtime_ready": False,
                "stages": {
                    "baseline": {"status": "blocked", "ready": False, "blocker_codes": ["missing-real-nvcc"]},
                    "pyg": {"status": "blocked", "ready": False, "blocked_by_stage": "baseline"},
                    "spconv": {"status": "blocked", "ready": False, "blocked_by_stage": "baseline"},
                },
            },
            "repeatability": {
                "checklist_file": "logs/bootstrap-preflight-checklist.txt",
                "report_file": "logs/bootstrap-preflight.json",
            },
        }
        planner = {
            "host_class": "linux-arm64",
            "support_posture": "experimental-unvalidated",
            "install_mode": "staged-source-build",
            "dependencies": [
                {"name": "torch", "strategy": "torch-index-cu128"},
                {"name": "upstream-requirements", "strategy": "upstream-requirements-file"},
                {"name": "pyg", "strategy": "linux-arm64-source-build-only", "reason_code": "pyg-source-build"},
                {"name": "spconv", "strategy": "linux-arm64-guarded-source-build", "reason_code": "spconv-guarded-source-build"},
                {"name": "bpy", "strategy": "deferred-portability-review", "reason_code": "bpy-portability"},
            ],
            "deferred": ["bpy-portability"],
        }

        with mock.patch.object(module, "_resolve_bootstrap_python", return_value=(Path(sys.executable), Path(sys.executable), {"selected_source": "requested-host-python"})), mock.patch.object(
            module, "_preflight_check_summary", return_value=blocked_preflight
        ), mock.patch.object(module, "build_install_plan", return_value=planner), mock.patch.object(
            module, "_probe_python_version", return_value="3.12.3"
        ), mock.patch.object(module, "_prepare_runtime_source") as prepare_mock, mock.patch.object(
            module, "_install_runtime_packages"
        ) as install_mock, mock.patch.object(module, "_run_post_setup_smoke_checks") as smoke_mock:
            with self.assertRaises(SystemExit) as ctx:
                module.main([str(SETUP), json.dumps({"ext_dir": str(self.ext_dir), "python_exe": sys.executable})])

        self.assertIn("Bootstrap preflight blocked", str(ctx.exception))
        prepare_mock.assert_not_called()
        install_mock.assert_not_called()
        smoke_mock.assert_not_called()
        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["install_state"], "blocked")
        self.assertEqual(state["last_verification"]["status"], "blocked")
        self.assertTrue(any("nvcc" in item.lower() for item in state["last_verification"]["errors"]))
        self.assertEqual(state["planner"]["host_class"], "linux-arm64")
        self.assertEqual(state["planner"]["support_posture"], "experimental-unvalidated")
        self.assertEqual(state["preflight"]["host_class"], "linux-arm64")
        self.assertEqual(state["preflight"]["support_posture"], "experimental-unvalidated")
        self.assertEqual(state["preflight"]["blockers"][0]["code"], "missing-real-nvcc")
        self.assertEqual(state["install_plan"]["summary"]["install_mode"], "staged-source-build")
        self.assertEqual(state["install_plan"]["summary"]["status"], "blocked")
        self.assertEqual(state["deferred_work"], ["bpy-portability"])

    def test_main_runs_linux_arm64_staged_provisioning_before_reporting_remaining_blockers(self) -> None:
        module = load_setup_module()
        blocked_preflight = {
            "status": "blocked",
            "checked_at": "2026-04-13T00:00:00+00:00",
            "host": {"os": "linux", "arch": "aarch64", "platform_tag": "linux-aarch64"},
            "observed": {
                "python_exe": sys.executable,
                "python_version": "3.11.9",
                "requested_host_python": sys.executable,
            },
            "checks": [
                {
                    "id": "linux-arm64-spconv-guarded-source-build",
                    "label": "spconv guarded source-build contract",
                    "required": "linux-arm64 diagnostic preflight",
                    "status": "fail",
                    "message": "spconv on Linux ARM64 is planned as a guarded source-build/import-smoke stage; setup may record blocked, deferred, build-ready, or ready states without claiming full runtime support.",
                },
                {
                    "id": "linux-arm64-bpy-portability",
                    "label": "bpy portability risk",
                    "required": "linux-arm64 diagnostic preflight",
                    "status": "fail",
                    "message": "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists; this change only records the blocker.",
                },
            ],
            "blockers": [
                {
                    "category": "source-build",
                    "code": "spconv-guarded-source-build",
                    "message": "spconv on Linux ARM64 is planned as a guarded source-build/import-smoke stage; setup may record blocked, deferred, build-ready, or ready states without claiming full runtime support.",
                    "action": "stop",
                    "boundary": "wrapper",
                    "owner": "wrapper",
                },
                {
                    "category": "portability",
                    "code": "bpy-portability-risk",
                    "message": "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists; this change only records the blocker.",
                    "action": "stop",
                    "boundary": "upstream",
                    "owner": "upstream",
                },
            ],
            "blocked": [
                "spconv on Linux ARM64 is planned as a guarded source-build/import-smoke stage; setup may record blocked, deferred, build-ready, or ready states without claiming full runtime support.",
                "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists; this change only records the blocker.",
            ],
            "baseline": {"ready": True, "blockers": []},
            "source_build": {
                "status": "blocked",
                "current_stage": "spconv",
                "non_blender_runtime_ready": False,
                "stages": {
                    "baseline": {"status": "ready", "ready": True, "blocker_codes": []},
                    "pyg": {
                        "status": "ready",
                        "ready": True,
                        "mode": "source-build-only",
                        "verification": "import-smoke",
                        "packages": ["torch_scatter", "torch_cluster"],
                        "blocker_codes": [],
                        "blockers": [],
                        "checks": [{"label": "torch_scatter import", "status": "ready"}],
                        "boundary": "wrapper",
                    },
                    "spconv": {
                        "status": "build-ready",
                        "ready": False,
                        "mode": "source-build",
                        "verification": "import-smoke",
                        "allowed_statuses": ["blocked", "deferred", "build-ready", "ready"],
                        "blocker_codes": [],
                        "blockers": [],
                        "checks": [{"label": "spconv-guarded-source-build", "status": "build-ready"}],
                        "boundary": "wrapper",
                    },
                },
            },
            "repeatability": {
                "checklist_file": "logs/bootstrap-preflight-checklist.txt",
                "report_file": "logs/bootstrap-preflight.json",
            },
        }
        planner = {
            "host_class": "linux-arm64",
            "support_posture": "experimental-unvalidated",
            "install_mode": "staged-source-build",
            "profile": "linux-arm64-runtime-bringup",
            "dependencies": [
                {"name": "torch", "strategy": "torch-index-cu128"},
                {"name": "upstream-requirements", "strategy": "upstream-requirements-file"},
                {"name": "pyg", "strategy": "linux-arm64-source-build-only", "reason_code": "pyg-source-build"},
                {"name": "spconv", "strategy": "linux-arm64-guarded-source-build", "reason_code": "spconv-guarded-source-build"},
                {"name": "bpy", "strategy": "deferred-portability-review", "reason_code": "bpy-portability"},
            ],
            "deferred": ["bpy-portability"],
        }
        install_result = {
            "status": "blocked",
            "profile": "linux-arm64-runtime-bringup",
            "host_class": "linux-arm64",
            "steps": ["bootstrap-python", "torch", "upstream-requirements", "linux-arm64-pyg-source-build"],
            "installed": {"pyg": ["torch_scatter", "torch_cluster"]},
            "deferred_work": ["bpy-portability"],
            "blocked": [
                {
                    "category": "distribution",
                    "code": "spconv-missing-distribution",
                    "message": "spconv remains blocked after staged PyG provisioning because no verified Linux ARM64 runtime-ready distribution exists.",
                    "action": "stop",
                    "dependency": "spconv",
                },
                {
                    "category": "portability",
                    "code": "bpy-portability-risk",
                    "message": "bpy remains deferred on Linux ARM64 after staged provisioning.",
                    "action": "stop",
                    "dependency": "bpy",
                },
            ],
        }

        with mock.patch.object(module, "_resolve_bootstrap_python", return_value=(Path(sys.executable), Path(sys.executable), {"selected_source": "requested-host-python"})), mock.patch.object(
            module, "_preflight_check_summary", return_value=blocked_preflight
        ), mock.patch.object(module, "build_install_plan", return_value=planner), mock.patch.object(
            module, "_probe_python_version", return_value="3.11.9"
        ), mock.patch.object(module, "_prepare_runtime_source", return_value=(self.ext_dir / ".unirig-runtime" / "vendor" / "unirig", "fixture", "test-ref")) as prepare_mock, mock.patch.object(
            module, "_install_runtime_packages", return_value=install_result
        ) as install_mock, mock.patch.object(module, "_run_post_setup_smoke_checks") as smoke_mock:
            with self.assertRaises(SystemExit) as ctx:
                module.main([str(SETUP), json.dumps({"ext_dir": str(self.ext_dir), "python_exe": sys.executable})])

        self.assertIn("remaining blockers", str(ctx.exception).lower())
        prepare_mock.assert_called_once()
        install_mock.assert_called_once()
        smoke_mock.assert_not_called()
        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["install_state"], "blocked")
        self.assertEqual(state["last_verification"]["status"], "blocked")
        self.assertFalse(state["last_verification"]["runtime_ready"])
        self.assertEqual(state["install_plan"]["summary"]["install_mode"], "staged-source-build")
        self.assertEqual(state["install_plan"]["summary"]["status"], "blocked")
        self.assertEqual(state["source_build"]["stages"]["baseline"]["status"], "ready")
        self.assertEqual(state["source_build"]["stages"]["pyg"]["status"], "ready")
        self.assertEqual(state["source_build"]["stages"]["spconv"]["status"], "build-ready")
        self.assertEqual(state["deferred_work"], ["bpy-portability"])

    def test_install_runtime_packages_runs_single_pinned_profile(self) -> None:
        module = load_setup_module()
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        requirements = unirig_dir / "requirements.txt"
        requirements.write_text("trimesh\n", encoding="utf-8")

        with mock.patch.object(module, "_venv_python", return_value=self.ext_dir / "venv" / "bin" / "python"), mock.patch.object(
            module.platform, "machine", return_value="x86_64"
        ), mock.patch.object(module, "_run") as run_mock:
            result = module._install_runtime_packages(self.ext_dir, {})

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["profile"], "pinned-upstream-wrapper")
        commands = [call.args[0] for call in run_mock.mock_calls]
        self.assertEqual(commands[0], [str(self.ext_dir / "venv" / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        self.assertIn(
            [str(self.ext_dir / "venv" / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements)],
            commands,
        )
        self.assertFalse(any("source-build" in " ".join(command) for command in commands))

    def test_filtered_requirements_path_removes_flash_attn_only_on_windows(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn==9.9.9\nflash-attn\nspconv-cu120\n", encoding="utf-8")

        filtered = module._filtered_requirements_path(unirig_dir, runtime_root, host_class="windows-x86_64")

        self.assertEqual(filtered, runtime_root / "requirements.upstream.windows.txt")
        content = filtered.read_text(encoding="utf-8")
        self.assertIn("trimesh", content)
        self.assertIn("spconv-cu120", content)
        self.assertNotIn("flash_attn==9.9.9", content)
        self.assertNotIn("flash-attn", content)

    def test_filtered_requirements_path_removes_premature_runtime_requirements_on_linux_arm64(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nbpy==4.2\nflash_attn\nopen3d\npyrender\n", encoding="utf-8")

        filtered = module._filtered_requirements_path(unirig_dir, runtime_root, host_class="linux-arm64")

        self.assertEqual(filtered, runtime_root / "requirements.upstream.linux-arm64.txt")
        content = filtered.read_text(encoding="utf-8")
        self.assertIn("trimesh", content)
        self.assertIn("pyrender", content)
        self.assertNotIn("bpy==4.2", content)
        self.assertNotIn("flash_attn", content)
        self.assertNotIn("open3d", content)

    def test_install_windows_spconv_stack_installs_validated_prebuilt_pair(self) -> None:
        module = load_setup_module()
        pip = [str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-m", "pip"]
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(module, "_run") as run_mock:
            cumm_package, spconv_package = module._install_windows_spconv_stack(pip, unirig_dir)

        self.assertEqual(cumm_package, module.WINDOWS_CUMM_PACKAGE_DEFAULT)
        self.assertEqual(spconv_package, module.WINDOWS_SPCONV_PACKAGE_DEFAULT)
        run_mock.assert_called_once_with(
            pip + ["install", "--no-cache-dir", module.WINDOWS_CUMM_PACKAGE_DEFAULT, module.WINDOWS_SPCONV_PACKAGE_DEFAULT],
            cwd=unirig_dir,
        )

    def test_filtered_requirements_path_keeps_upstream_requirements_on_linux(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        requirements = unirig_dir / "requirements.txt"
        requirements.write_text("trimesh\nflash_attn==9.9.9\n", encoding="utf-8")

        self.assertEqual(module._filtered_requirements_path(unirig_dir, runtime_root, host_class="linux-x86_64"), requirements)

    def test_install_runtime_packages_filters_upstream_requirements_and_installs_windows_shims(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")

        with mock.patch.object(module.os, "name", "nt"), mock.patch.object(module.platform, "machine", return_value="AMD64"), mock.patch.object(
            module, "_venv_python", return_value=self.ext_dir / "venv" / "Scripts" / "python.exe"
        ), mock.patch.object(module, "_run") as run_mock, mock.patch.object(module.urllib.request, "urlretrieve") as urlretrieve_mock:
            result = module._install_runtime_packages(self.ext_dir, {})

        filtered_requirements = runtime_root / "requirements.upstream.windows.txt"
        flash_wheel = runtime_root / "cache" / module.FLASH_ATTN_WHEEL_DEFAULT.rsplit("/", 1)[-1]
        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            result["steps"][-4:],
            ["windows-cumm-spconv", "windows-triton", "windows-flash-attn", "windows-sitecustomize"],
        )
        self.assertTrue(filtered_requirements.exists())
        self.assertNotIn("flash_attn", filtered_requirements.read_text(encoding="utf-8"))
        commands = [call.args[0] for call in run_mock.mock_calls]
        self.assertIn([str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-m", "pip", "install", "-r", str(filtered_requirements)], commands)
        self.assertIn(
            [
                str(self.ext_dir / "venv" / "Scripts" / "python.exe"),
                "-m",
                "pip",
                "install",
                "-f",
                module.PYG_INDEX_URL,
                "--no-cache-dir",
                *module.PYG_PACKAGES,
                module.NUMPY_PIN,
                "pygltflib>=1.15.0",
            ],
            commands,
        )
        self.assertIn(
            [
                str(self.ext_dir / "venv" / "Scripts" / "python.exe"),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                module.WINDOWS_CUMM_PACKAGE_DEFAULT,
                module.WINDOWS_SPCONV_PACKAGE_DEFAULT,
            ],
            commands,
        )
        self.assertIn([str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-m", "pip", "install", module.TRITON_WINDOWS_PACKAGE_DEFAULT], commands)
        self.assertIn([str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-m", "pip", "install", str(flash_wheel)], commands)
        urlretrieve_mock.assert_called_once_with(module.FLASH_ATTN_WHEEL_DEFAULT, str(flash_wheel))

    def test_install_runtime_packages_uses_windows_dependency_entries_from_install_plan(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")
        plan = {
            "host_class": "windows-x86_64",
            "support_posture": "validated",
            "install_mode": "pinned-prebuilt",
            "dependencies": [
                {"name": "torch", "strategy": "torch-index-cu128", "index_url": module.TORCH_INDEX_URL, "packages": list(module.TORCH_PACKAGES)},
                {"name": "upstream-requirements", "strategy": "windows-filtered-requirements"},
                {
                    "name": "pyg",
                    "strategy": "pyg-wheel-index",
                    "index_url": module.PYG_INDEX_URL,
                    "packages": list(module.PYG_PACKAGES) + [module.NUMPY_PIN, "pygltflib>=1.15.0"],
                },
                {"name": "cumm", "strategy": "windows-pinned-prebuilt", "package": "cumm-cu126==9.9.9"},
                {"name": "spconv", "strategy": "windows-pinned-prebuilt", "package": "spconv-cu126==8.8.8"},
                {"name": "triton", "strategy": "windows-pinned-package", "package": "triton-windows==7.7.7"},
                {
                    "name": "flash_attn",
                    "strategy": "windows-pinned-wheel",
                    "wheel": "https://example.invalid/flash_attn-custom.whl",
                },
                {"name": "sitecustomize", "strategy": "windows-dll-shim"},
            ],
            "deferred": [],
        }

        with mock.patch.object(module.os, "name", "nt"), mock.patch.object(
            module, "_venv_python", return_value=self.ext_dir / "venv" / "Scripts" / "python.exe"
        ), mock.patch.object(module, "build_install_plan", return_value=plan), mock.patch.object(
            module, "_run"
        ) as run_mock, mock.patch.object(module.urllib.request, "urlretrieve") as urlretrieve_mock:
            module._install_runtime_packages(self.ext_dir, {})

        commands = [call.args[0] for call in run_mock.mock_calls]
        self.assertIn(
            [
                str(self.ext_dir / "venv" / "Scripts" / "python.exe"),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "cumm-cu126==9.9.9",
                "spconv-cu126==8.8.8",
            ],
            commands,
        )
        self.assertIn(
            [str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-m", "pip", "install", "triton-windows==7.7.7"],
            commands,
        )
        self.assertIn(
            [
                str(self.ext_dir / "venv" / "Scripts" / "python.exe"),
                "-m",
                "pip",
                "install",
                str(runtime_root / "cache" / "flash_attn-custom.whl"),
            ],
            commands,
        )
        urlretrieve_mock.assert_called_once_with(
            "https://example.invalid/flash_attn-custom.whl",
            str(runtime_root / "cache" / "flash_attn-custom.whl"),
        )

    def test_install_runtime_packages_never_calls_linux_arm64_spconv_helper_on_windows(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")

        with mock.patch.object(module.os, "name", "nt"), mock.patch.object(
            module, "_venv_python", return_value=self.ext_dir / "venv" / "Scripts" / "python.exe"
        ), mock.patch.object(
            module.platform, "machine", return_value="AMD64"
        ), mock.patch.object(module, "_run") as run_mock, mock.patch.object(
            module.urllib.request, "urlretrieve"
        ), mock.patch.object(module, "_install_linux_arm64_spconv_stage") as spconv_helper_mock:
            result = module._install_runtime_packages(self.ext_dir, {})

        self.assertEqual(result["status"], "ready")
        self.assertIn("windows-cumm-spconv", result["steps"])
        spconv_helper_mock.assert_not_called()
        self.assertTrue(any(module.WINDOWS_SPCONV_PACKAGE_DEFAULT in " ".join(command) for command in [call.args[0] for call in run_mock.mock_calls]))

    def test_install_runtime_packages_routes_linux_arm64_ready_pyg_through_guarded_spconv_helper(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")
        plan = {
            "host_class": "linux-arm64",
            "support_posture": "experimental-unvalidated",
            "install_mode": "staged-source-build",
            "dependencies": [
                {"name": "torch", "strategy": "torch-index-cu128", "index_url": module.TORCH_INDEX_URL, "packages": list(module.TORCH_PACKAGES)},
                {"name": "upstream-requirements", "strategy": "upstream-requirements-file"},
                {
                    "name": "pyg",
                    "strategy": "linux-arm64-source-build-only",
                    "reason_code": "pyg-source-build",
                    "stage": "pyg",
                    "packages": ["torch_scatter", "torch_cluster"],
                    "verification": "deferred",
                },
                {
                    "name": "spconv",
                    "strategy": "linux-arm64-guarded-source-build",
                    "reason_code": "spconv-guarded-source-build",
                    "stage": "spconv",
                    "verification": "import-smoke",
                    "allowed_statuses": ["blocked", "deferred", "build-ready", "ready"],
                },
                {"name": "bpy", "strategy": "deferred-portability-review", "reason_code": "bpy-portability"},
            ],
            "deferred": ["bpy-portability"],
        }

        with mock.patch.object(module, "_venv_python", return_value=self.ext_dir / "venv" / "bin" / "python"), mock.patch.object(
            module, "build_install_plan", return_value=plan
        ), mock.patch.object(module, "_run") as run_mock, mock.patch.object(
            module,
            "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
                "boundary": "wrapper",
            },
            create=True,
        ), mock.patch.object(
            module,
            "_run_linux_arm64_spconv_guarded_bringup",
            return_value={
                "status": "blocked",
                "ready": False,
                "mode": "source-build",
                "verification": "import-smoke",
                "packages": ["cumm", "spconv"],
                "checks": [{"label": "cumm", "status": "ready", "returncode": 0}],
                "blockers": [],
                "blocker_codes": [],
                "boundary": "wrapper",
            },
        ) as spconv_helper_mock:
            result = module._install_runtime_packages(self.ext_dir, {})

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["host_class"], "linux-arm64")
        self.assertEqual(result["deferred_work"], ["bpy-portability"])
        spconv_helper_mock.assert_called_once()
        self.assertTrue(any("torch_scatter" in " ".join(command) for command in [call.args[0] for call in run_mock.mock_calls]))

    def test_install_runtime_packages_runs_guarded_spconv_only_after_ready_pyg_and_returns_stage_payload(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")
        plan = {
            "host_class": "linux-arm64",
            "support_posture": "experimental-unvalidated",
            "install_mode": "staged-source-build",
            "dependencies": [
                {"name": "torch", "strategy": "torch-index-cu128", "index_url": module.TORCH_INDEX_URL, "packages": list(module.TORCH_PACKAGES)},
                {"name": "upstream-requirements", "strategy": "upstream-requirements-file"},
                {
                    "name": "pyg",
                    "strategy": "linux-arm64-source-build-only",
                    "reason_code": "pyg-source-build",
                    "stage": "pyg",
                    "packages": ["torch_scatter", "torch_cluster"],
                    "verification": "deferred",
                },
                {
                    "name": "spconv",
                    "strategy": "linux-arm64-guarded-source-build",
                    "reason_code": "spconv-guarded-source-build",
                    "stage": "spconv",
                    "verification": "import-smoke",
                    "allowed_statuses": ["blocked", "deferred", "build-ready", "ready"],
                },
                {"name": "bpy", "strategy": "deferred-portability-review", "reason_code": "bpy-portability"},
            ],
            "deferred": ["bpy-portability"],
        }
        spconv_result = {
            "status": "blocked",
            "ready": False,
            "mode": "source-build",
            "verification": "import-smoke",
            "packages": ["cumm", "spconv"],
            "checks": [
                {"label": "cumm", "status": "ready", "returncode": 0},
                {"label": "spconv", "status": "error", "returncode": 1},
            ],
            "blockers": [
                {
                    "category": "source-build",
                    "code": "spconv-guarded-bringup-failed",
                    "message": "Linux ARM64 guarded cumm/spconv orchestration failed before import verification could prove runtime readiness.",
                    "action": "stop",
                    "boundary": "upstream-package",
                    "owner": "upstream-package",
                }
            ],
            "blocker_codes": ["spconv-guarded-bringup-failed"],
            "boundary": "upstream-package",
        }

        with mock.patch.object(module, "_venv_python", return_value=self.ext_dir / "venv" / "bin" / "python"), mock.patch.object(
            module, "build_install_plan", return_value=plan
        ), mock.patch.object(module, "_run") as run_mock, mock.patch.object(
            module, "_run_linux_arm64_spconv_guarded_bringup", return_value=spconv_result
        ) as spconv_mock, mock.patch.object(
            module, "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
                "boundary": "wrapper",
            },
            create=True,
        ):
            result = module._install_runtime_packages(self.ext_dir, {})

        spconv_stage = result["source_build"]["stages"]["spconv"]

        spconv_mock.assert_called_once()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(spconv_stage["status"], "blocked")
        self.assertEqual(spconv_stage["packages"], ["cumm", "spconv"])
        self.assertEqual(spconv_stage["blocker_codes"], ["spconv-guarded-bringup-failed"])
        self.assertEqual(spconv_stage["boundary"], "upstream-package")
        self.assertNotIn("blocked_by_stage", spconv_stage)
        self.assertTrue(any("torch_scatter" in " ".join(command) for command in [call.args[0] for call in run_mock.mock_calls]))

    def test_install_runtime_packages_returns_partial_when_only_remaining_linux_arm64_block_is_wrapper_owned_bpy_runtime(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")
        plan = {
            "host_class": "linux-arm64",
            "support_posture": "experimental-unvalidated",
            "install_mode": "staged-source-build",
            "dependencies": [
                {"name": "torch", "strategy": "torch-index-cu128", "index_url": module.TORCH_INDEX_URL, "packages": list(module.TORCH_PACKAGES)},
                {"name": "upstream-requirements", "strategy": "upstream-requirements-file"},
                {
                    "name": "pyg",
                    "strategy": "linux-arm64-source-build-only",
                    "reason_code": "pyg-source-build",
                    "stage": "pyg",
                    "packages": ["torch_scatter", "torch_cluster"],
                    "verification": "deferred",
                },
                {
                    "name": "spconv",
                    "strategy": "linux-arm64-guarded-source-build",
                    "reason_code": "spconv-guarded-source-build",
                    "stage": "spconv",
                    "verification": "import-smoke",
                    "allowed_statuses": ["blocked", "deferred", "build-ready", "ready"],
                },
                {"name": "bpy", "strategy": "deferred-portability-review", "reason_code": "bpy-portability"},
            ],
            "deferred": ["bpy-portability"],
        }
        bpy_evidence = {
            "candidate": {"source": "path", "path": "/usr/bin/blender", "selected_because": "Selected PATH-visible Blender candidate."},
            "probe": {
                "status": "ok",
                "command": ["/usr/bin/blender", "--background"],
                "blender_version": "4.0.2",
                "python_version": "3.12.3",
                "smoke_result": "passed",
                "returncode": 0,
                "stdout_tail": ["probe ok"],
                "stderr_tail": [],
            },
            "classification": {
                "status": "external-bpy-smoke-ready",
                "ready": False,
                "evidence_kind": "external-blender",
                "blockers": [],
                "blocker_codes": [],
            },
        }
        spconv_result = {
            "status": "ready",
            "ready": True,
            "mode": "source-build",
            "verification": "import-smoke",
            "packages": ["cumm", "spconv"],
            "checks": [
                {"label": "cumm", "status": "ready", "returncode": 0},
                {"label": "spconv", "status": "ready", "returncode": 0},
                {"label": "spconv.pytorch", "status": "ready", "returncode": 0},
            ],
            "blockers": [],
            "blocker_codes": [],
            "boundary": "wrapper",
        }

        with mock.patch.object(module, "_venv_python", return_value=self.ext_dir / "venv" / "bin" / "python"), mock.patch.object(
            module, "build_install_plan", return_value=plan
        ), mock.patch.object(module, "_run") as run_mock, mock.patch.object(
            module, "_verify_linux_arm64_pyg_import_smoke",
            return_value={
                "status": "ready",
                "ready": True,
                "verification": "import-smoke",
                "blockers": [],
                "blocker_codes": [],
                "checks": [
                    {"label": "torch_scatter", "status": "ready", "returncode": 0},
                    {"label": "torch_cluster", "status": "ready", "returncode": 0},
                ],
                "boundary": "wrapper",
            },
            create=True,
        ), mock.patch.object(module, "_run_linux_arm64_spconv_guarded_bringup", return_value=spconv_result), mock.patch.object(
            module, "_linux_arm64_bpy_evidence", return_value=bpy_evidence
        ), mock.patch.object(module, "_probe_python_version", return_value="3.12.3"):
            result = module._install_runtime_packages(self.ext_dir, {})

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["deferred_work"], ["bpy-portability"])
        self.assertEqual(result["source_build"]["status"], "partial")
        self.assertEqual(result["source_build"]["stages"]["spconv"]["status"], "ready")
        self.assertEqual(result["source_build"]["stages"]["bpy"]["status"], "external-bpy-smoke-ready")
        self.assertEqual(result["source_build"]["external_blender"], bpy_evidence)
        self.assertEqual(result["blocked"], [])
        self.assertTrue(any("torch_scatter" in " ".join(command) for command in [call.args[0] for call in run_mock.mock_calls]))

    def test_write_error_state_persists_linux_arm64_install_spconv_stage_result(self) -> None:
        module = load_setup_module()
        preflight = {
            "status": "blocked",
            "blocked": ["bpy remains deferred"],
            "blockers": [],
            "baseline": {"ready": True, "python": {"version": "3.11.9"}, "blockers": []},
            "source_build": {
                "status": "blocked",
                "current_stage": "spconv",
                "non_blender_runtime_ready": False,
                "stages": {
                    "baseline": {"status": "ready", "ready": True, "blocker_codes": [], "blockers": []},
                    "pyg": {
                        "status": "ready",
                        "ready": True,
                        "mode": "source-build-only",
                        "verification": "import-smoke",
                        "packages": ["torch_scatter", "torch_cluster"],
                        "checks": [
                            {"label": "torch_scatter", "status": "ready", "returncode": 0},
                            {"label": "torch_cluster", "status": "ready", "returncode": 0},
                        ],
                        "blockers": [],
                        "blocker_codes": [],
                        "boundary": "wrapper",
                    },
                    "spconv": {
                        "status": "build-ready",
                        "ready": False,
                        "mode": "source-build",
                        "verification": "import-smoke",
                        "packages": ["cumm", "spconv"],
                        "checks": [{"label": "spconv-guarded-source-build", "status": "build-ready"}],
                        "blockers": [],
                        "blocker_codes": [],
                        "boundary": "wrapper",
                    },
                },
            },
        }
        planner = module.build_install_plan(host_os="linux", host_arch="aarch64")
        install_result = {
            "status": "blocked",
            "profile": "linux-arm64-runtime-bringup",
            "host_class": "linux-arm64",
            "steps": ["bootstrap-python", "torch", "upstream-requirements", "linux-arm64-pyg-source-build", "linux-arm64-spconv-guarded-source-build"],
            "deferred_work": ["bpy-portability"],
            "blocked": [
                {
                    "category": "source-build",
                    "code": "spconv-guarded-bringup-failed",
                    "message": "Linux ARM64 guarded cumm/spconv orchestration failed before import verification could prove runtime readiness.",
                    "action": "stop",
                    "boundary": "upstream-package",
                    "owner": "upstream-package",
                }
            ],
            "source_build": {
                "status": "blocked",
                "current_stage": "spconv",
                "non_blender_runtime_ready": False,
                "stages": {
                    "spconv": {
                        "status": "blocked",
                        "ready": False,
                        "mode": "source-build",
                        "verification": "import-smoke",
                        "packages": ["cumm", "spconv"],
                        "checks": [
                            {"label": "cumm", "status": "ready", "returncode": 0},
                            {"label": "spconv", "status": "error", "returncode": 1},
                        ],
                        "blockers": [
                            {
                                "category": "source-build",
                                "code": "spconv-guarded-bringup-failed",
                                "message": "Linux ARM64 guarded cumm/spconv orchestration failed before import verification could prove runtime readiness.",
                                "action": "stop",
                                "boundary": "upstream-package",
                                "owner": "upstream-package",
                            }
                        ],
                        "blocker_codes": ["spconv-guarded-bringup-failed"],
                        "boundary": "upstream-package",
                    }
                },
            },
        }

        module._write_error_state(
            self.ext_dir,
            "blocked by guarded Linux ARM64 spconv bringup",
            preflight,
            requested_host_python=Path(sys.executable),
            bootstrap_resolution={"selected_source": "requested-host-python"},
            planner=planner,
            install_result=install_result,
        )

        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        spconv_stage = state["source_build"]["stages"]["spconv"]

        self.assertEqual(spconv_stage["status"], "blocked")
        self.assertEqual(spconv_stage["packages"], ["cumm", "spconv"])
        self.assertEqual(spconv_stage["blocker_codes"], ["spconv-guarded-bringup-failed"])
        self.assertEqual(spconv_stage["boundary"], "upstream-package")
        self.assertEqual([item["label"] for item in spconv_stage["checks"]], ["cumm", "spconv"])

    def test_write_error_state_keeps_linux_arm64_full_runtime_blocked_when_spconv_becomes_import_ready(self) -> None:
        module = load_setup_module()
        preflight = {
            "status": "blocked",
            "blocked": ["bpy remains deferred on Linux ARM64 until upstream portability evidence exists."],
            "blockers": [
                {
                    "category": "portability",
                    "code": "bpy-portability-risk",
                    "message": "bpy remains deferred on Linux ARM64 until upstream portability evidence exists.",
                    "action": "stop",
                    "boundary": "upstream",
                    "owner": "upstream",
                    "dependency": "bpy",
                }
            ],
            "baseline": {"ready": True, "python": {"version": "3.11.9"}, "blockers": []},
            "source_build": {
                "status": "blocked",
                "current_stage": "spconv",
                "non_blender_runtime_ready": False,
                "stages": {
                    "baseline": {"status": "ready", "ready": True, "blocker_codes": [], "blockers": []},
                    "pyg": {
                        "status": "ready",
                        "ready": True,
                        "mode": "source-build-only",
                        "verification": "import-smoke",
                        "packages": ["torch_scatter", "torch_cluster"],
                        "checks": [
                            {"label": "torch_scatter", "status": "ready", "returncode": 0},
                            {"label": "torch_cluster", "status": "ready", "returncode": 0},
                        ],
                        "blockers": [],
                        "blocker_codes": [],
                        "boundary": "wrapper",
                    },
                    "spconv": {
                        "status": "build-ready",
                        "ready": False,
                        "mode": "source-build",
                        "verification": "import-smoke",
                        "packages": ["cumm", "spconv"],
                        "checks": [{"label": "spconv-guarded-source-build", "status": "build-ready"}],
                        "blockers": [],
                        "blocker_codes": [],
                        "boundary": "wrapper",
                    },
                },
            },
        }
        planner = module.build_install_plan(host_os="linux", host_arch="aarch64")
        install_result = {
            "status": "blocked",
            "profile": "linux-arm64-runtime-bringup",
            "host_class": "linux-arm64",
            "steps": [
                "bootstrap-python",
                "torch",
                "upstream-requirements",
                "linux-arm64-pyg-source-build",
                "linux-arm64-spconv-guarded-source-build",
            ],
            "deferred_work": ["bpy-portability"],
            "blocked": [
                {
                    "category": "portability",
                    "code": "bpy-portability-risk",
                    "message": "bpy remains deferred on Linux ARM64 until upstream portability evidence exists.",
                    "action": "stop",
                    "boundary": "upstream",
                    "owner": "upstream",
                    "dependency": "bpy",
                }
            ],
            "source_build": {
                "status": "blocked",
                "current_stage": "spconv",
                "non_blender_runtime_ready": False,
                "stages": {
                    "spconv": {
                        "status": "ready",
                        "ready": True,
                        "mode": "source-build",
                        "verification": "import-smoke",
                        "packages": ["cumm", "spconv"],
                        "checks": [
                            {"label": "cumm", "status": "ready", "returncode": 0},
                            {"label": "spconv", "status": "ready", "returncode": 0},
                            {"label": "spconv.pytorch", "status": "ready", "returncode": 0},
                        ],
                        "blockers": [],
                        "blocker_codes": [],
                        "boundary": "wrapper",
                    }
                },
            },
        }

        module._write_error_state(
            self.ext_dir,
            "blocked by deferred bpy after Linux ARM64 spconv import readiness",
            preflight,
            requested_host_python=Path(sys.executable),
            bootstrap_resolution={"selected_source": "requested-host-python"},
            planner=planner,
            install_result=install_result,
        )

        state = bootstrap.load_state(self.ext_dir)

        self.assertEqual(state["install_state"], "blocked")
        self.assertFalse(state["last_verification"]["runtime_ready"])
        self.assertFalse(state["source_build"]["non_blender_runtime_ready"])
        self.assertEqual(state["source_build"]["stages"]["spconv"]["status"], "ready")
        self.assertTrue(state["source_build"]["stages"]["spconv"]["ready"])
        self.assertEqual(state["source_build"]["stages"]["bpy"]["status"], "deferred")
        self.assertEqual(state["source_build"]["deferred_work"], ["bpy-portability"])
        self.assertIn("bpy remains deferred", state["source_build"]["blocked_reasons"][0])

    def test_ensure_ready_still_blocks_linux_arm64_when_spconv_is_import_ready_but_bpy_is_deferred(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        self._write_minimal_upstream_tree(unirig_dir)
        venv_python = self.ext_dir / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(unirig_dir),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(venv_python),
                "python_version": "3.11.9",
                "source_build": {
                    "status": "blocked",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True, "blocker_codes": [], "blockers": []},
                        "pyg": {"status": "ready", "ready": True, "verification": "import-smoke"},
                        "spconv": {"status": "ready", "ready": True, "verification": "import-smoke"},
                    },
                    "blockers": [
                        {
                            "category": "portability",
                            "code": "bpy-portability-risk",
                            "dependency": "bpy",
                            "message": "bpy remains deferred on Linux ARM64 until upstream portability evidence exists.",
                            "action": "stop",
                            "boundary": "upstream",
                            "owner": "upstream",
                        }
                    ],
                    "blocked_reasons": [
                        "bpy remains deferred on Linux ARM64 until upstream portability evidence exists."
                    ],
                    "deferred_work": ["bpy-portability"],
                    "non_blender_runtime_ready": False,
                },
            },
            extension_root=self.ext_dir,
        )

        with self.assertRaises(bootstrap.BootstrapError) as ctx:
            bootstrap.ensure_ready(self.ext_dir)

        message = str(ctx.exception)
        self.assertIn("UniRig runtime is blocked", message)
        self.assertIn("linux-arm64", message)
        self.assertIn("staged-source-build", message)
        self.assertIn("bpy-portability-risk", message)
        self.assertIn("bpy remains deferred", message)
        self.assertIn("deferred work", message.lower())
        self.assertIn("bpy-portability", message)

    def test_install_runtime_packages_rejects_windows_plan_missing_flash_attn_dependency(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")
        plan = {
            "host_class": "windows-x86_64",
            "support_posture": "validated",
            "install_mode": "pinned-prebuilt",
            "dependencies": [
                {"name": "torch", "strategy": "torch-index-cu128", "index_url": module.TORCH_INDEX_URL, "packages": list(module.TORCH_PACKAGES)},
                {"name": "upstream-requirements", "strategy": "windows-filtered-requirements"},
                {
                    "name": "pyg",
                    "strategy": "pyg-wheel-index",
                    "index_url": module.PYG_INDEX_URL,
                    "packages": list(module.PYG_PACKAGES) + [module.NUMPY_PIN, "pygltflib>=1.15.0"],
                },
                {"name": "cumm", "strategy": "windows-pinned-prebuilt", "package": module.WINDOWS_CUMM_PACKAGE_DEFAULT},
                {"name": "spconv", "strategy": "windows-pinned-prebuilt", "package": module.WINDOWS_SPCONV_PACKAGE_DEFAULT},
                {"name": "triton", "strategy": "windows-pinned-package", "package": module.TRITON_WINDOWS_PACKAGE_DEFAULT},
                {"name": "sitecustomize", "strategy": "windows-dll-shim"},
            ],
            "deferred": [],
        }

        with mock.patch.object(module.os, "name", "nt"), mock.patch.object(
            module.platform, "machine", return_value="AMD64"
        ), mock.patch.object(
            module, "_venv_python", return_value=self.ext_dir / "venv" / "Scripts" / "python.exe"
        ), mock.patch.object(module, "build_install_plan", return_value=plan), self.assertRaises(RuntimeError) as ctx:
            module._install_runtime_packages(self.ext_dir, {})

        self.assertIn("flash_attn", str(ctx.exception))

    def test_install_windows_sitecustomize_writes_silent_dll_shim(self) -> None:
        module = load_setup_module()
        venv_dir = self.ext_dir / "venv"
        torch_lib = venv_dir / "Lib" / "site-packages" / "torch" / "lib"
        cublas_bin = venv_dir / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
        for path in (torch_lib, cublas_bin):
            path.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(module.os, "name", "nt"):
            written = module._install_windows_sitecustomize(venv_dir)

        self.assertEqual(written, [venv_dir / "Lib" / "site-packages" / "sitecustomize.py"])
        content = written[0].read_text(encoding="utf-8")
        self.assertIn("torch/lib", content)
        self.assertIn("nvidia/*/bin", content)
        self.assertIn("os.add_dll_directory", content)
        self.assertIn("_MODLY_UNIRIG_DLL_HANDLES", content)
        self.assertIn("sitecustomize-unirig.log", content)
        self.assertIn("def _record_bootstrap_note(message):", content)
        self.assertIn("def _register_torch_safe_globals():", content)
        self.assertIn("from box.box import Box", content)
        self.assertIn('getattr(torch.serialization, "add_safe_globals", None)', content)
        self.assertIn("add_safe_globals([Box])", content)
        self.assertIn("_register_torch_safe_globals()", content)
        self.assertNotIn("print(", content)
        compile(content, str(written[0]), "exec")

    def test_windows_runtime_dll_search_paths_collects_torch_and_nvidia_bins(self) -> None:
        venv_dir = self.ext_dir / "venv"
        torch_lib = venv_dir / "Lib" / "site-packages" / "torch" / "lib"
        cublas_bin = venv_dir / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
        cudnn_bin = venv_dir / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
        for path in (torch_lib, cublas_bin, cudnn_bin):
            path.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(bootstrap.os, "name", "nt"):
            paths = bootstrap.windows_runtime_dll_search_paths(venv_dir)

        self.assertEqual(paths, [torch_lib, cublas_bin, cudnn_bin])

    def test_runtime_environment_keeps_linux_path_unchanged(self) -> None:
        original = os.environ.get("PATH", "")

        with mock.patch.object(bootstrap.os, "name", "posix"):
            env = bootstrap.runtime_environment(venv_dir=self.ext_dir / "venv")

        self.assertEqual(env["PATH"], original)

    def test_runtime_environment_drops_unrelated_host_variables(self) -> None:
        with mock.patch.dict(os.environ, {"UNRELATED_HOST_STATE": "secret-value"}):
            env = bootstrap.runtime_environment(venv_dir=self.ext_dir / "venv")

        self.assertNotIn("UNRELATED_HOST_STATE", env)

    def test_resolve_extension_root_ignores_env_override_by_default(self) -> None:
        override_root = self.ext_dir / "override-root"

        with mock.patch.dict(os.environ, {"UNIRIG_EXTENSION_ROOT": str(override_root)}):
            resolved = bootstrap.resolve_extension_root()

        self.assertEqual(resolved, ROOT.resolve())

    def test_resolve_extension_root_prefers_explicit_argument_over_opt_in_env_override(self) -> None:
        explicit_root = self.ext_dir / "explicit-root"
        override_root = self.ext_dir / "override-root"

        with mock.patch.dict(os.environ, {"UNIRIG_EXTENSION_ROOT": str(override_root)}):
            resolved = bootstrap.resolve_extension_root(explicit_root, allow_env_override=True)

        self.assertEqual(resolved, explicit_root.resolve())

    def test_resolve_extension_root_can_opt_in_to_env_override(self) -> None:
        override_root = self.ext_dir / "override-root"

        with mock.patch.dict(os.environ, {"UNIRIG_EXTENSION_ROOT": str(override_root)}):
            resolved = bootstrap.resolve_extension_root(allow_env_override=True)

        self.assertEqual(resolved, override_root.resolve())

    def test_ensure_ready_resolves_relative_runtime_paths_against_extension_root(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        venv_python = self.ext_dir / "venv" / "bin" / "python"
        for rel in bootstrap.REQUIRED_RUNTIME_PATHS:
            path = unirig_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok\n", encoding="utf-8")
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        state = {
            "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
            "install_state": "ready",
            "runtime_paths": {
                "runtime_root": ".unirig-runtime",
                "logs_dir": ".unirig-runtime/logs",
                "runtime_vendor_dir": ".unirig-runtime/vendor",
                "unirig_dir": ".unirig-runtime/vendor/unirig",
                "hf_home": ".unirig-runtime/hf-home",
                "venv_python": "venv/bin/python",
            },
            "last_verification": {"status": "ready", "runtime_ready": True, "host": {"os": "linux", "arch": "x86_64"}},
        }
        bootstrap.save_state(state, extension_root=self.ext_dir)

        context = bootstrap.ensure_ready(self.ext_dir)

        self.assertEqual(context.runtime_root, runtime_root)
        self.assertEqual(context.unirig_dir, unirig_dir)
        self.assertEqual(context.venv_python, venv_python)

    def test_run_post_setup_smoke_checks_runs_windows_import_smoke_suite(self) -> None:
        module = load_setup_module()
        torch_lib = self.ext_dir / "venv" / "Lib" / "site-packages" / "torch" / "lib"
        cublas_bin = self.ext_dir / "venv" / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
        for path in (torch_lib, cublas_bin):
            path.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(module.os, "name", "nt"), mock.patch.object(bootstrap.os, "name", "nt"), mock.patch.object(
            module, "_venv_python", return_value=self.ext_dir / "venv" / "Scripts" / "python.exe"
        ), mock.patch.object(module.subprocess, "run", return_value=subprocess.CompletedProcess(["python"], 0, stdout="", stderr="")) as run_mock:
            report = module._run_post_setup_smoke_checks(self.ext_dir)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(
            [call.args[0] for call in run_mock.mock_calls],
            [
                [str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-c", 'import importlib; importlib.import_module("flash_attn")'],
                [str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-c", 'import importlib; importlib.import_module("triton")'],
                [str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-c", "from flash_attn.layers.rotary import apply_rotary_emb"],
                [str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-c", 'import importlib; importlib.import_module("cumm.core_cc")'],
                [str(self.ext_dir / "venv" / "Scripts" / "python.exe"), "-c", 'import importlib; importlib.import_module("spconv.pytorch")'],
            ],
        )
        for call in run_mock.mock_calls:
            env = call.kwargs["env"]
            self.assertTrue(env["PATH"].startswith(str(torch_lib)))
            self.assertIn(str(cublas_bin), env["PATH"])

    def test_run_post_setup_smoke_checks_fails_early_on_triton_import(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.os, "name", "nt"), mock.patch.object(bootstrap.os, "name", "nt"), mock.patch.object(
            module, "_venv_python", return_value=self.ext_dir / "venv" / "Scripts" / "python.exe"
        ), mock.patch.object(
            module.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 1, stdout="", stderr="ModuleNotFoundError: No module named 'triton'"),
            ],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                module._run_post_setup_smoke_checks(self.ext_dir)

        message = str(ctx.exception)
        self.assertIn("Check: triton", message)
        self.assertIn("MODLY_UNIRIG_TRITON_PACKAGE", message)
        self.assertIn("No module named 'triton'", message)

    def test_run_post_setup_smoke_checks_reports_actionable_cumm_failure(self) -> None:
        module = load_setup_module()

        with mock.patch.object(module.os, "name", "nt"), mock.patch.object(bootstrap.os, "name", "nt"), mock.patch.object(
            module, "_venv_python", return_value=self.ext_dir / "venv" / "Scripts" / "python.exe"
        ), mock.patch.object(
            module.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
                subprocess.CompletedProcess(["python"], 1, stdout="", stderr="ImportError: DLL load failed while importing core_cc"),
            ],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                module._run_post_setup_smoke_checks(self.ext_dir)

        message = str(ctx.exception)
        self.assertIn("Check: cumm.core_cc", message)
        self.assertIn(module.WINDOWS_CUMM_PACKAGE_DEFAULT, message)
        self.assertIn(module.WINDOWS_SPCONV_PACKAGE_DEFAULT, message)
        self.assertIn("DLL load failed", message)

    def test_setup_prepares_fresh_runtime(self) -> None:
        module = load_setup_module()

        def fake_prepare_runtime_source(ext_dir: Path, payload: dict) -> tuple[Path, str, str]:
            unirig_dir = module._runtime_unirig_dir(ext_dir)
            self._write_minimal_upstream_tree(unirig_dir)
            return unirig_dir, "fixture", "test-ref"

        def fake_create_virtualenv(venv_dir: Path, bootstrap_python: Path) -> None:
            del bootstrap_python
            python_exe = module._venv_python(venv_dir)
            python_exe.parent.mkdir(parents=True, exist_ok=True)
            python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        with mock.patch.object(module, "_create_virtualenv", side_effect=fake_create_virtualenv), mock.patch.object(
            module, "_prepare_runtime_source", side_effect=fake_prepare_runtime_source
        ), mock.patch.object(module, "_install_runtime_packages", return_value={"status": "ready", "profile": "pinned-upstream-wrapper"}), mock.patch.object(
            module, "_run_post_setup_smoke_checks", return_value={"platform": "linux", "status": "skipped", "checks": []}
        ), mock.patch.object(
            module, "_probe_python_version", return_value="3.12.3"
        ), mock.patch.object(
            module.platform, "machine", return_value="x86_64"
        ):
            payload = json.dumps({"ext_dir": str(self.ext_dir), "python_exe": sys.executable})
            result = module.main([str(SETUP), payload])

        self.assertEqual(result, 0)
        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["install_state"], "ready")
        self.assertEqual(state["bootstrap_version"], bootstrap.BOOTSTRAP_VERSION)
        self.assertEqual(state["source_ref"], "test-ref")
        self.assertEqual(state["vendor_source"], "fixture")
        self.assertEqual(state["requested_host_python"], sys.executable)
        self.assertEqual(state["bootstrap_resolution"]["selected_source"], "requested-host-python")
        self.assertEqual(state["runtime_paths"]["unirig_dir"], str(self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"))
        self.assertEqual(state["last_verification"]["status"], "ready")
        self.assertEqual(state["last_verification"]["python_version"], "3.12.3")
        self.assertEqual(state["last_verification"]["host"]["os"], sys.platform.lower().replace("win32", "windows"))
        self.assertEqual(state["planner"]["host_class"], "linux-x86_64")
        self.assertEqual(state["preflight"]["host_class"], "linux-x86_64")
        self.assertEqual(state["preflight"]["support_posture"], "supported")
        self.assertEqual(state["preflight"]["blockers"], [])
        self.assertEqual(state["install_plan"]["summary"]["install_mode"], "prebuilt")
        self.assertEqual(state["install_plan"]["summary"]["status"], "ready")
        self.assertEqual(state["deferred_work"], [])
        self.assertNotIn("platform_policy", state)
        self.assertNotIn("source_build", state)
        self.assertTrue((self.ext_dir / ".unirig-runtime" / "logs" / "bootstrap-preflight.json").exists())

    def test_main_accepts_linux_arm64_partial_install_result_and_persists_partial_state(self) -> None:
        module = load_setup_module()

        def fake_prepare_runtime_source(ext_dir: Path, payload: dict) -> tuple[Path, str, str]:
            del payload
            unirig_dir = module._runtime_unirig_dir(ext_dir)
            self._write_minimal_upstream_tree(unirig_dir)
            return unirig_dir, "fixture", "test-ref"

        def fake_create_virtualenv(venv_dir: Path, bootstrap_python: Path) -> None:
            del bootstrap_python
            python_exe = module._venv_python(venv_dir)
            python_exe.parent.mkdir(parents=True, exist_ok=True)
            python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        install_result = {
            "status": "partial",
            "profile": "linux-arm64-runtime-bringup",
            "host_class": "linux-arm64",
            "deferred_work": ["bpy-portability"],
            "blocked": [],
            "source_build": {
                "status": "partial",
                "host_class": "linux-arm64",
                "stages": {
                    "baseline": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                    "pyg": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                    "spconv": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                    "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                },
                "external_blender": {
                    "classification": {
                        "status": "external-bpy-smoke-ready",
                        "ready": False,
                        "evidence_kind": "external-blender",
                        "blockers": [],
                        "blocker_codes": [],
                    }
                },
            },
        }

        with mock.patch.object(module, "_create_virtualenv", side_effect=fake_create_virtualenv), mock.patch.object(
            module, "_prepare_runtime_source", side_effect=fake_prepare_runtime_source
        ), mock.patch.object(module, "_install_runtime_packages", return_value=install_result), mock.patch.object(
            module, "_run_post_setup_smoke_checks"
        ) as smoke_mock, mock.patch.object(
            module, "_probe_python_version", return_value="3.12.3"
        ), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(
            module.platform, "system", return_value="Linux"
        ):
            payload = json.dumps({"ext_dir": str(self.ext_dir), "python_exe": sys.executable})
            result = module.main([str(SETUP), payload])

        self.assertEqual(result, 0)
        smoke_mock.assert_not_called()
        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["install_state"], "blocked")
        self.assertEqual(state["install_plan"]["summary"]["status"], "partial")
        self.assertEqual(state["source_build"]["stages"]["bpy"]["status"], "external-bpy-smoke-ready")

    def test_main_windows_x86_64_ready_path_keeps_pinned_prebuilt_defaults_and_skips_linux_arm64_proof_logic(self) -> None:
        module = load_setup_module()

        def fake_prepare_runtime_source(ext_dir: Path, payload: dict) -> tuple[Path, str, str]:
            del payload
            unirig_dir = module._runtime_unirig_dir(ext_dir)
            self._write_minimal_upstream_tree(unirig_dir)
            return unirig_dir, "fixture", "test-ref"

        def fake_create_virtualenv(venv_dir: Path, bootstrap_python: Path) -> None:
            del bootstrap_python
            python_exe = module._venv_python(venv_dir)
            python_exe.parent.mkdir(parents=True, exist_ok=True)
            python_exe.write_text("#!python\n", encoding="utf-8")

        with mock.patch.object(module, "_create_virtualenv", side_effect=fake_create_virtualenv), mock.patch.object(
            module, "_prepare_runtime_source", side_effect=fake_prepare_runtime_source
        ), mock.patch.object(module, "_install_runtime_packages", return_value={"status": "ready"}), mock.patch.object(
            module, "_run_post_setup_smoke_checks"
        ) as smoke_mock, mock.patch.object(
            module, "_linux_arm64_seed_partial_runtime_proofs", autospec=True
        ) as seed_mock, mock.patch.object(
            module, "_probe_python_version", return_value="3.11.9"
        ), mock.patch.object(
            module.platform, "machine", return_value="AMD64"
        ), mock.patch.object(
            module.platform, "system", return_value="Windows"
        ):
            payload = json.dumps({"ext_dir": str(self.ext_dir), "python_exe": sys.executable})
            result = module.main([str(SETUP), payload])

        self.assertEqual(result, 0)
        smoke_mock.assert_called_once_with(self.ext_dir)
        seed_mock.assert_not_called()
        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["install_state"], "ready")
        self.assertEqual(state["planner"]["host_class"], "windows-x86_64")
        self.assertEqual(state["planner"]["install_mode"], "pinned-prebuilt")
        self.assertEqual(state["install_plan"]["summary"]["install_mode"], "pinned-prebuilt")
        self.assertEqual(state["preflight"]["support_posture"], "validated")
        self.assertEqual(state["deferred_work"], [])
        self.assertNotIn("source_build", state)

    def test_linux_arm64_seed_partial_runtime_proofs_restores_supported_stage_artifacts_from_backup(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        backup_root = self.ext_dir / ".unirig-runtime.backup-github-sim-test"

        def write_stage(run_name: str, stage: str, produced_relative_path: str) -> None:
            run_dir = backup_root / "runs" / run_name
            produced_path = run_dir / produced_relative_path
            produced_path.parent.mkdir(parents=True, exist_ok=True)
            produced_path.write_text(stage, encoding="utf-8")
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "protocol_version": 1,
                        "stage": stage,
                        "status": "ok",
                        "produced": [str(runtime_root / "runs" / run_name / produced_relative_path)],
                    }
                ),
                encoding="utf-8",
            )

        write_stage("run-prepare", "extract-prepare", "skeleton_npz/.modly_stage_input_run-processor/raw_data.npz")
        write_stage("run-skin", "extract-skin", "skin_npz/skeleton_stage/raw_data.npz")
        write_stage("run-proof", "skin", "skeleton_stage.fbx")
        write_stage("run-ignore", "merge", "merged.glb")
        (backup_root / "logs" / "run-proof").mkdir(parents=True, exist_ok=True)
        (backup_root / "logs" / "run-proof" / "skeleton.log").write_text("ok\n", encoding="utf-8")

        proof_seed = module._linux_arm64_seed_partial_runtime_proofs(self.ext_dir)

        self.assertTrue(proof_seed["seeded"])
        self.assertEqual(proof_seed["source"], str(backup_root))
        self.assertTrue((runtime_root / "runs" / "run-prepare" / "result.json").exists())
        self.assertTrue(
            (runtime_root / "runs" / "run-prepare" / "skeleton_npz" / ".modly_stage_input_run-processor" / "raw_data.npz").exists()
        )
        self.assertTrue((runtime_root / "runs" / "run-skin" / "skin_npz" / "skeleton_stage" / "raw_data.npz").exists())
        self.assertTrue((runtime_root / "runs" / "run-proof" / "skeleton_stage.fbx").exists())
        self.assertTrue((runtime_root / "logs" / "run-proof" / "skeleton.log").exists())
        self.assertFalse((runtime_root / "runs" / "run-ignore").exists())
        self.assertTrue(module._linux_arm64_runtime_has_persisted_partial_proofs(runtime_root))

    def test_linux_arm64_seed_partial_runtime_proofs_keeps_clean_install_unseeded_without_backup_fixture(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"

        proof_seed = module._linux_arm64_seed_partial_runtime_proofs(self.ext_dir)

        self.assertEqual(proof_seed, {"seeded": False, "source": "", "reason": "no-compatible-backup-proofs"})
        self.assertFalse(module._linux_arm64_runtime_has_persisted_partial_proofs(runtime_root))

    def test_main_partial_install_on_clean_linux_arm64_does_not_attempt_to_seed_runtime_proofs_before_writing_state(self) -> None:
        module = load_setup_module()

        def fake_prepare_runtime_source(ext_dir: Path, payload: dict) -> tuple[Path, str, str]:
            del payload
            unirig_dir = module._runtime_unirig_dir(ext_dir)
            self._write_minimal_upstream_tree(unirig_dir)
            return unirig_dir, "fixture", "test-ref"

        def fake_create_virtualenv(venv_dir: Path, bootstrap_python: Path) -> None:
            del bootstrap_python
            python_exe = module._venv_python(venv_dir)
            python_exe.parent.mkdir(parents=True, exist_ok=True)
            python_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        install_result = {
            "status": "partial",
            "profile": "linux-arm64-runtime-bringup",
            "host_class": "linux-arm64",
            "deferred_work": ["bpy-portability"],
            "blocked": [],
            "source_build": {
                "status": "partial",
                "host_class": "linux-arm64",
                "stages": {
                    "baseline": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                    "pyg": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                    "spconv": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                    "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                },
                "external_blender": {
                    "classification": {
                        "status": "external-bpy-smoke-ready",
                        "ready": False,
                        "evidence_kind": "external-blender",
                        "blockers": [],
                        "blocker_codes": [],
                    }
                },
            },
        }

        with mock.patch.object(module, "_create_virtualenv", side_effect=fake_create_virtualenv), mock.patch.object(
            module, "_prepare_runtime_source", side_effect=fake_prepare_runtime_source
        ), mock.patch.object(module, "_install_runtime_packages", return_value=install_result), mock.patch.object(
            module, "_run_post_setup_smoke_checks"
        ) as smoke_mock, mock.patch.object(
            module, "_probe_python_version", return_value="3.12.3"
        ), mock.patch.object(
            module.platform, "machine", return_value="aarch64"
        ), mock.patch.object(
            module.platform, "system", return_value="Linux"
        ), mock.patch.object(
            module, "_linux_arm64_seed_partial_runtime_proofs", return_value={"seeded": True, "source": "backup-runtime"}
        ) as seed_mock:
            payload = json.dumps({"ext_dir": str(self.ext_dir), "python_exe": sys.executable})
            result = module.main([str(SETUP), payload])

        self.assertEqual(result, 0)
        smoke_mock.assert_not_called()
        seed_mock.assert_not_called()
        state = json.loads((self.ext_dir / ".unirig-runtime" / "bootstrap_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["install_state"], "blocked")
        self.assertEqual(state["install_plan"]["summary"]["status"], "partial")
        self.assertEqual(state["source_build"]["stages"]["bpy"]["status"], "external-bpy-smoke-ready")

    def test_bootstrap_requires_prepared_state(self) -> None:
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.ensure_ready(self.ext_dir)

    def test_bootstrap_rejects_outdated_or_incomplete_real_runtime(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
            },
            extension_root=self.ext_dir,
        )

        with self.assertRaises(bootstrap.BootstrapError) as ctx:
            bootstrap.ensure_ready(self.ext_dir)

        self.assertIn("incomplete", str(ctx.exception))

    def test_save_state_rewrites_legacy_runtime_state_to_minimal_contract(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "source_ref": "legacy-ref",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "python_version": "3.12.4",
                "platform_policy": bootstrap.resolve_platform_policy("linux", "x86_64"),
                "preflight": {
                    "status": "ready",
                    "checked_at": "2026-04-11T00:00:00+00:00",
                    "host": {"os": "linux", "arch": "x86_64"},
                    "blocked": [],
                },
                "source_build": {"status": "ready", "mode": "prebuilt", "dependencies": {}},
            },
            extension_root=self.ext_dir,
        )

        state = json.loads((runtime_root / "bootstrap_state.json").read_text(encoding="utf-8"))

        self.assertEqual(state["source_ref"], "legacy-ref")
        self.assertEqual(state["runtime_paths"]["runtime_root"], str(runtime_root))
        self.assertEqual(state["last_verification"]["status"], "ready")
        self.assertEqual(state["last_verification"]["python_version"], "3.12.4")
        self.assertEqual(state["last_verification"]["host"], {"arch": "x86_64", "os": "linux"})
        self.assertNotIn("platform_policy", state)
        self.assertNotIn("source_build", state)

    def test_load_state_normalizes_linux_arm64_blocked_planner_metadata_into_runtime_fields(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "blocked",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "planner": {
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "install_mode": "staged-source-build",
                    "dependencies": [
                        {"name": "spconv", "strategy": "blocked-missing-distribution", "reason_code": "missing-distribution"}
                    ],
                    "deferred": ["pyg-source-build", "spconv-port", "bpy-portability"],
                },
                "preflight": {
                    "status": "blocked",
                    "checked_at": "2026-04-12T00:00:00+00:00",
                    "host": {"os": "linux", "arch": "aarch64", "platform_tag": "linux-aarch64"},
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "blockers": [
                        {
                            "category": "distribution",
                            "code": "spconv-missing-distribution",
                            "dependency": "spconv",
                            "message": "spconv has no validated Linux ARM64 distribution in the current wrapper contract, so setup stops before attempting a source port.",
                            "action": "stop",
                        }
                    ],
                    "blocked": [
                        "spconv has no validated Linux ARM64 distribution in the current wrapper contract, so setup stops before attempting a source port."
                    ],
                },
                "install_plan": {"summary": {"host_class": "linux-arm64", "install_mode": "staged-source-build", "status": "blocked"}},
                "deferred_work": ["pyg-source-build", "spconv-port", "bpy-portability"],
            },
            extension_root=self.ext_dir,
        )

        state = bootstrap.load_state(self.ext_dir)

        self.assertEqual(state["planner"]["host_class"], "linux-arm64")
        self.assertEqual(state["platform_policy"]["selected"]["key"], "linux-arm64")
        self.assertEqual(state["platform_policy"]["selected"]["status"], "experimental-unvalidated")
        self.assertEqual(state["platform_policy"]["selected"]["install_mode"], "staged-source-build")
        self.assertEqual(state["source_build"]["status"], "blocked")
        self.assertEqual(state["source_build"]["mode"], "staged-source-build")
        self.assertEqual(state["source_build"]["host_class"], "linux-arm64")
        self.assertEqual(state["source_build"]["blockers"][0]["code"], "spconv-missing-distribution")
        self.assertEqual(state["source_build"]["blockers"][0]["category"], "distribution")
        self.assertEqual(state["source_build"]["blockers"][0]["dependency"], "spconv")
        self.assertEqual(state["source_build"]["blockers"][0]["action"], "stop")
        self.assertIn("source port", state["source_build"]["blockers"][0]["message"])
        self.assertEqual(state["source_build"]["deferred_work"], ["pyg-source-build", "spconv-port", "bpy-portability"])

    def test_ensure_ready_reports_linux_arm64_blockers_with_actionable_details(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "blocked",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "planner": {
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "install_mode": "staged-source-build",
                    "dependencies": [
                        {"name": "spconv", "strategy": "blocked-missing-distribution", "reason_code": "missing-distribution"}
                    ],
                    "deferred": ["pyg-source-build", "spconv-port", "bpy-portability"],
                },
                "preflight": {
                    "status": "blocked",
                    "checked_at": "2026-04-12T00:00:00+00:00",
                    "host": {"os": "linux", "arch": "aarch64", "platform_tag": "linux-aarch64"},
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "blockers": [
                        {
                            "category": "distribution",
                            "code": "spconv-missing-distribution",
                            "dependency": "spconv",
                            "message": "spconv has no validated Linux ARM64 distribution in the current wrapper contract, so setup stops before attempting a source port.",
                            "action": "stop",
                        },
                        {
                            "category": "toolchain",
                            "code": "missing-nvcc-toolchain",
                            "dependency": "spconv",
                            "message": "nvcc and a Linux ARM64 CUDA build toolchain were not detected, so the blocked source-build path stays out of scope.",
                            "action": "stop",
                        },
                    ],
                    "blocked": [
                        "spconv has no validated Linux ARM64 distribution in the current wrapper contract, so setup stops before attempting a source port.",
                        "nvcc and a Linux ARM64 CUDA build toolchain were not detected, so the blocked source-build path stays out of scope.",
                    ],
                },
                "install_plan": {"summary": {"host_class": "linux-arm64", "install_mode": "staged-source-build", "status": "blocked"}},
                "deferred_work": ["pyg-source-build", "spconv-port", "bpy-portability"],
            },
            extension_root=self.ext_dir,
        )

        with self.assertRaises(bootstrap.BootstrapError) as ctx:
            bootstrap.ensure_ready(self.ext_dir)

        message = str(ctx.exception)
        self.assertIn("UniRig runtime is blocked", message)
        self.assertIn("linux-arm64", message)
        self.assertIn("staged-source-build", message)
        self.assertIn("spconv-missing-distribution", message)
        self.assertIn("distribution", message)
        self.assertIn("missing-nvcc-toolchain", message)
        self.assertIn("toolchain", message)
        self.assertIn("deferred work", message.lower())
        self.assertIn("spconv-port", message)
        self.assertIn("Run setup.py again to repair the extension.", message)

    def test_load_state_normalizes_legacy_linux_arm64_staged_runtime_to_blocked_full_runtime(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "python_version": "3.11.9",
                "source_build": {
                    "status": "non-blender-ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True, "python": {"version": "3.11.9", "required": "3.11"}},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True, "blocker_codes": [], "blockers": []},
                        "pyg": {
                            "status": "ready",
                            "ready": True,
                            "verification": "import-smoke",
                            "packages": ["torch_scatter", "torch_cluster"],
                            "blocker_codes": [],
                            "blockers": [],
                        },
                        "spconv": {
                            "status": "prep-ready",
                            "ready": False,
                            "verification": "probe-only",
                            "blocker_codes": [],
                            "blockers": [],
                        },
                        "bpy": {
                            "status": "deferred",
                            "ready": False,
                            "reason_code": "bpy-portability-risk",
                            "message": "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists.",
                        },
                    },
                    "blocked_reasons": [
                        "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists."
                    ],
                    "deferred_work": ["bpy-portability"],
                    "non_blender_runtime_ready": True,
                },
            },
            extension_root=self.ext_dir,
        )

        state = bootstrap.load_state(self.ext_dir)

        self.assertEqual(state["install_state"], "blocked")
        self.assertFalse(state["last_verification"]["runtime_ready"])
        self.assertEqual(state["source_build"]["host_class"], "linux-arm64")
        self.assertTrue(state["source_build"]["non_blender_runtime_ready"])
        self.assertEqual(state["source_build"]["stages"]["pyg"]["status"], "ready")
        self.assertEqual(state["source_build"]["stages"]["spconv"]["status"], "prep-ready")
        self.assertEqual(state["source_build"]["stages"]["bpy"]["status"], "deferred")
        self.assertEqual(state["source_build"]["deferred_work"], ["bpy-portability"])
        self.assertIn("bpy remains a likely Linux ARM64 portability risk", state["source_build"]["blocked_reasons"][0])

    def test_ensure_ready_blocks_legacy_linux_arm64_non_blender_ready_state_when_bpy_is_deferred(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        self._write_minimal_upstream_tree(unirig_dir)
        venv_python = self.ext_dir / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(unirig_dir),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(venv_python),
                "python_version": "3.11.9",
                "source_build": {
                    "status": "non-blender-ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "verification": "import-smoke"},
                        "spconv": {"status": "prep-ready", "ready": False, "verification": "probe-only"},
                        "bpy": {
                            "status": "deferred",
                            "ready": False,
                            "reason_code": "bpy-portability-risk",
                            "message": "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists.",
                        },
                    },
                    "blocked_reasons": [
                        "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists."
                    ],
                    "deferred_work": ["bpy-portability"],
                    "non_blender_runtime_ready": True,
                },
            },
            extension_root=self.ext_dir,
        )

        with self.assertRaises(bootstrap.BootstrapError) as ctx:
            bootstrap.ensure_ready(self.ext_dir)

        message = str(ctx.exception)
        self.assertIn("UniRig runtime is blocked", message)
        self.assertIn("linux-arm64", message)
        self.assertIn("staged-source-build", message)
        self.assertIn("bpy", message.lower())
        self.assertIn("deferred work", message.lower())

    def test_load_state_maps_partial_linux_arm64_non_blender_progress_to_blocked_overall_state(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "partial",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "python_version": "3.11.9",
                "source_build": {
                    "status": "partial",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "verification": "import-smoke"},
                        "spconv": {"status": "prep-ready", "ready": False, "verification": "probe-only"},
                        "bpy": {
                            "status": "deferred",
                            "ready": False,
                            "reason_code": "bpy-portability-risk",
                            "message": "bpy remains deferred on Linux ARM64 until upstream portability evidence exists.",
                        },
                    },
                    "blocked_reasons": [
                        "bpy remains deferred on Linux ARM64 until upstream portability evidence exists."
                    ],
                    "deferred_work": ["bpy-portability"],
                    "non_blender_runtime_ready": True,
                },
            },
            extension_root=self.ext_dir,
        )

        state = bootstrap.load_state(self.ext_dir)

        self.assertEqual(state["install_state"], "blocked")
        self.assertEqual(state["source_build"]["status"], "partial")
        self.assertTrue(state["source_build"]["non_blender_runtime_ready"])
        self.assertEqual(state["source_build"]["stages"]["pyg"]["status"], "ready")
        self.assertEqual(state["source_build"]["stages"]["spconv"]["status"], "prep-ready")
        self.assertEqual(state["source_build"]["stages"]["bpy"]["status"], "deferred")

    def test_load_state_preserves_linux_arm64_spconv_stage_evidence_and_blocks_runtime_on_stage_blocker(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "python_version": "3.11.9",
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {
                            "status": "ready",
                            "ready": True,
                            "verification": "import-smoke",
                        },
                        "spconv": {
                            "status": "blocked",
                            "ready": False,
                            "mode": "source-build",
                            "verification": "import-smoke",
                            "packages": ["cumm", "spconv"],
                            "checks": [
                                {"label": "cumm", "status": "ready", "returncode": 0},
                                {"label": "spconv", "status": "ready", "returncode": 0},
                                {"label": "spconv.pytorch", "status": "blocked", "returncode": 1},
                            ],
                            "blockers": [
                                {
                                    "category": "import",
                                    "code": "spconv-import-smoke-failed",
                                    "dependency": "spconv",
                                    "message": "spconv.pytorch import smoke failed on Linux ARM64.",
                                    "action": "stop",
                                    "boundary": "upstream-package",
                                    "owner": "upstream-package",
                                }
                            ],
                            "blocker_codes": ["spconv-import-smoke-failed"],
                            "boundary": "upstream-package",
                        },
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "deferred_work": [],
                    "non_blender_runtime_ready": False,
                },
            },
            extension_root=self.ext_dir,
        )

        state = bootstrap.load_state(self.ext_dir)

        spconv_stage = state["source_build"]["stages"]["spconv"]
        self.assertEqual(state["install_state"], "blocked")
        self.assertFalse(state["last_verification"]["runtime_ready"])
        self.assertEqual(spconv_stage["status"], "blocked")
        self.assertEqual(spconv_stage["mode"], "source-build")
        self.assertEqual(spconv_stage["verification"], "import-smoke")
        self.assertEqual(spconv_stage["packages"], ["cumm", "spconv"])
        self.assertEqual([item["label"] for item in spconv_stage["checks"]], ["cumm", "spconv", "spconv.pytorch"])
        self.assertEqual(spconv_stage["blocker_codes"], ["spconv-import-smoke-failed"])
        self.assertEqual(spconv_stage["boundary"], "upstream-package")
        self.assertEqual(state["source_build"]["blockers"][0]["code"], "spconv-import-smoke-failed")
        self.assertIn("spconv.pytorch import smoke failed", state["source_build"]["blocked_reasons"][0])

    def test_load_state_preserves_linux_arm64_spconv_import_ready_evidence_through_save_load(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "python_version": "3.11.9",
                "source_build": {
                    "status": "blocked",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {
                            "status": "ready",
                            "ready": True,
                            "verification": "import-smoke",
                            "packages": ["torch_scatter", "torch_cluster"],
                        },
                        "spconv": {
                            "status": "ready",
                            "ready": True,
                            "mode": "source-build",
                            "verification": "import-smoke",
                            "packages": ["cumm", "spconv"],
                            "checks": [
                                {"label": "cumm", "status": "ready", "returncode": 0},
                                {"label": "spconv", "status": "ready", "returncode": 0},
                                {"label": "spconv.pytorch", "status": "ready", "returncode": 0},
                            ],
                            "blockers": [],
                            "blocker_codes": [],
                            "boundary": "wrapper",
                        },
                        "bpy": {
                            "status": "deferred",
                            "ready": False,
                            "reason_code": "bpy-portability-risk",
                            "message": "bpy remains deferred on Linux ARM64 until upstream portability evidence exists.",
                        },
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "deferred_work": ["bpy-portability"],
                    "non_blender_runtime_ready": False,
                },
            },
            extension_root=self.ext_dir,
        )

        state = bootstrap.load_state(self.ext_dir)

        spconv_stage = state["source_build"]["stages"]["spconv"]
        self.assertEqual(state["install_state"], "blocked")
        self.assertFalse(state["last_verification"]["runtime_ready"])
        self.assertEqual(spconv_stage["status"], "ready")
        self.assertTrue(spconv_stage["ready"])
        self.assertEqual(spconv_stage["mode"], "source-build")
        self.assertEqual(spconv_stage["verification"], "import-smoke")
        self.assertEqual(spconv_stage["packages"], ["cumm", "spconv"])
        self.assertEqual([item["label"] for item in spconv_stage["checks"]], ["cumm", "spconv", "spconv.pytorch"])
        self.assertEqual(state["source_build"]["blockers"][0]["code"], "bpy-portability-risk")
        self.assertIn("bpy remains deferred", state["source_build"]["blocked_reasons"][0])

    def test_load_state_keeps_linux_arm64_external_blender_evidence_blocked_for_full_runtime(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        base_state = {
            "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
            "install_state": "ready",
            "runtime_root": str(runtime_root),
            "logs_dir": str(runtime_root / "logs"),
            "runtime_vendor_dir": str(runtime_root / "vendor"),
            "unirig_dir": str(runtime_root / "vendor" / "unirig"),
            "hf_home": str(runtime_root / "hf-home"),
            "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
            "python_version": "3.11.9",
        }

        cases = [
            (
                "external-bpy-smoke-ready",
                [],
                [],
            ),
            (
                "discovered-incompatible",
                ["blender-python-abi-mismatch"],
                [
                    {
                        "category": "compatibility",
                        "code": "blender-python-abi-mismatch",
                        "dependency": "bpy",
                        "message": "Linux ARM64 discovered Blender external bpy evidence, but Blender Python 3.12.1 does not match the wrapper Python 3.11 expectation.",
                        "action": "stop",
                        "boundary": "upstream",
                        "owner": "upstream-package",
                    }
                ],
            ),
            (
                "missing",
                ["blender-executable-missing"],
                [
                    {
                        "category": "environment",
                        "code": "blender-executable-missing",
                        "dependency": "bpy",
                        "message": "Linux ARM64 requires an external Blender executable for bpy evidence, but none was discovered.",
                        "action": "stop",
                        "boundary": "environment",
                        "owner": "environment",
                    }
                ],
            ),
            (
                "error",
                ["blender-bpy-smoke-error"],
                [
                    {
                        "category": "probe",
                        "code": "blender-bpy-smoke-error",
                        "dependency": "bpy",
                        "message": "Linux ARM64 Blender background bpy smoke probe failed.",
                        "action": "stop",
                        "boundary": "wrapper",
                        "owner": "wrapper",
                    }
                ],
            ),
        ]

        for status, blocker_codes, blockers in cases:
            with self.subTest(status=status):
                bootstrap.save_state(
                    {
                        **base_state,
                        "source_build": {
                            "status": "ready",
                            "mode": "staged-source-build",
                            "host_class": "linux-arm64",
                            "support_posture": "experimental-unvalidated",
                            "baseline": {"ready": True},
                            "stages": {
                                "baseline": {"status": "ready", "ready": True},
                                "pyg": {
                                    "status": "ready",
                                    "ready": True,
                                    "verification": "import-smoke",
                                    "packages": ["torch_scatter", "torch_cluster"],
                                    "blockers": [],
                                    "blocker_codes": [],
                                },
                                "spconv": {
                                    "status": "ready",
                                    "ready": True,
                                    "mode": "source-build",
                                    "verification": "import-smoke",
                                    "packages": ["cumm", "spconv"],
                                    "checks": [
                                        {"label": "cumm", "status": "ready", "returncode": 0},
                                        {"label": "spconv", "status": "ready", "returncode": 0},
                                        {"label": "spconv.pytorch", "status": "ready", "returncode": 0},
                                    ],
                                    "blockers": [],
                                    "blocker_codes": [],
                                    "boundary": "wrapper",
                                },
                                "bpy": {
                                    "status": status,
                                    "ready": False,
                                    "evidence_kind": "external-blender",
                                    "candidate": {
                                        "source": "path",
                                        "path": "/opt/blender/blender",
                                        "selected_because": "Selected PATH-visible Blender candidate.",
                                    },
                                    "blender": {"version": "4.0.2", "python_version": "3.12.1"},
                                    "verification": "blender-background-bpy-smoke",
                                    "checks": [{"label": "bpy", "status": status, "returncode": 0}],
                                    "blockers": blockers,
                                    "blocker_codes": blocker_codes,
                                    "boundary": "wrapper" if status in {"external-bpy-smoke-ready", "error"} else "upstream",
                                },
                            },
                            "blockers": [],
                            "blocked_reasons": [],
                            "deferred_work": [],
                            "non_blender_runtime_ready": True,
                            "bpy_evidence_class": status,
                            "external_blender": {
                                "candidate": {
                                    "source": "path",
                                    "path": "/opt/blender/blender",
                                    "selected_because": "Selected PATH-visible Blender candidate.",
                                },
                                "probe": {
                                    "status": "ok",
                                    "command": ["/opt/blender/blender", "--background"],
                                    "blender_version": "4.0.2",
                                    "python_version": "3.12.1",
                                    "smoke_result": "passed",
                                    "returncode": 0,
                                    "stdout_tail": ["probe ok"],
                                    "stderr_tail": [],
                                },
                                "classification": {"status": status, "ready": False, "blockers": blockers, "blocker_codes": blocker_codes},
                            },
                        },
                    },
                    extension_root=self.ext_dir,
                )

                state = bootstrap.load_state(self.ext_dir)

                self.assertEqual(state["install_state"], "blocked")
                self.assertFalse(state["last_verification"]["runtime_ready"])
                self.assertEqual(state["source_build"]["bpy_evidence_class"], status)
                self.assertEqual(state["source_build"]["stages"]["bpy"]["status"], status)
                self.assertEqual(state["source_build"]["external_blender"]["classification"]["status"], status)

    def test_normalize_state_backfills_linux_arm64_bpy_stage_from_legacy_external_blender_summary(self) -> None:
        normalized = bootstrap.normalize_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "python_version": "3.11.9",
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True},
                        "spconv": {
                            "status": "ready",
                            "ready": True,
                            "mode": "source-build",
                            "verification": "import-smoke",
                            "packages": ["cumm", "spconv"],
                            "checks": [
                                {"label": "cumm", "status": "ready", "returncode": 0},
                                {"label": "spconv", "status": "ready", "returncode": 0},
                                {"label": "spconv.pytorch", "status": "ready", "returncode": 0},
                            ],
                            "blockers": [],
                            "blocker_codes": [],
                        },
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "deferred_work": [],
                    "non_blender_runtime_ready": True,
                    "bpy_evidence_class": "external-bpy-smoke-ready",
                    "external_blender": {
                        "candidate": {
                            "source": "override",
                            "path": "/custom/blender",
                            "selected_because": "Selected explicit override.",
                        },
                        "probe": {
                            "status": "ok",
                            "command": ["/custom/blender", "--background"],
                            "blender_version": "3.6.9",
                            "python_version": "3.11.8",
                            "smoke_result": "passed",
                            "returncode": 0,
                            "stdout_tail": ["probe ok"],
                            "stderr_tail": [],
                        },
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "candidate": {
                                "source": "override",
                                "path": "/custom/blender",
                                "selected_because": "Selected explicit override.",
                            },
                            "blender": {"version": "3.6.9", "python_version": "3.11.8"},
                            "verification": "blender-background-bpy-smoke",
                            "checks": [{"label": "bpy", "status": "external-bpy-smoke-ready", "returncode": 0}],
                            "blockers": [],
                            "blocker_codes": [],
                            "boundary": "wrapper",
                        },
                    },
                },
            },
            self.ext_dir,
        )

        self.assertEqual(normalized["install_state"], "blocked")
        self.assertFalse(normalized["last_verification"]["runtime_ready"])
        self.assertEqual(normalized["source_build"]["bpy_evidence_class"], "external-bpy-smoke-ready")
        self.assertEqual(normalized["source_build"]["stages"]["bpy"]["status"], "external-bpy-smoke-ready")
        self.assertEqual(
            normalized["source_build"]["external_blender"]["classification"]["candidate"]["path"],
            "/custom/blender",
        )

    def test_load_state_backfills_missing_linux_arm64_spconv_stage_conservatively_for_legacy_state(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(runtime_root / "vendor" / "unirig"),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(self.ext_dir / "venv" / "bin" / "python"),
                "python_version": "3.11.9",
                "source_build": {
                    "status": "ready",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {
                            "status": "ready",
                            "ready": True,
                            "verification": "import-smoke",
                            "packages": ["torch_scatter", "torch_cluster"],
                            "blockers": [],
                            "blocker_codes": [],
                        },
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "deferred_work": [],
                    "non_blender_runtime_ready": True,
                },
            },
            extension_root=self.ext_dir,
        )

        state = bootstrap.load_state(self.ext_dir)

        spconv_stage = state["source_build"]["stages"]["spconv"]
        self.assertEqual(state["install_state"], "blocked")
        self.assertFalse(state["last_verification"]["runtime_ready"])
        self.assertEqual(spconv_stage["status"], "deferred")
        self.assertFalse(spconv_stage["ready"])
        self.assertEqual(spconv_stage["mode"], "source-build")
        self.assertEqual(spconv_stage["verification"], "import-smoke")
        self.assertEqual(spconv_stage["packages"], ["cumm", "spconv"])
        self.assertEqual(spconv_stage["blocker_codes"], ["spconv-state-missing"])
        self.assertEqual(state["source_build"]["blockers"][0]["code"], "spconv-state-missing")
        self.assertIn("legacy Linux ARM64 state is missing spconv stage evidence", state["source_build"]["blocked_reasons"][0])

    def test_ensure_ready_reports_partial_linux_arm64_progress_honestly_while_remaining_blocked(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        self._write_minimal_upstream_tree(unirig_dir)
        venv_python = self.ext_dir / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "partial",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(unirig_dir),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(venv_python),
                "python_version": "3.11.9",
                "source_build": {
                    "status": "partial",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "baseline": {"ready": True},
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "verification": "import-smoke"},
                        "spconv": {"status": "prep-ready", "ready": False, "verification": "probe-only"},
                        "bpy": {
                            "status": "deferred",
                            "ready": False,
                            "reason_code": "bpy-portability-risk",
                            "message": "bpy remains deferred on Linux ARM64 until upstream portability evidence exists.",
                        },
                    },
                    "blocked_reasons": [
                        "bpy remains deferred on Linux ARM64 until upstream portability evidence exists."
                    ],
                    "deferred_work": ["bpy-portability"],
                    "non_blender_runtime_ready": True,
                },
            },
            extension_root=self.ext_dir,
        )

        with self.assertRaises(bootstrap.BootstrapError) as ctx:
            bootstrap.ensure_ready(self.ext_dir)

        message = str(ctx.exception)
        self.assertIn("UniRig runtime is blocked", message)
        self.assertIn("linux-arm64", message)
        self.assertIn("staged-source-build", message)
        self.assertIn("partial", message)
        self.assertIn("bpy", message.lower())
        self.assertIn("deferred work", message.lower())

    def test_ensure_ready_allows_linux_arm64_partial_runtime_when_seam_boundary_is_verified(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        self._write_minimal_upstream_tree(unirig_dir)
        venv_python = self.ext_dir / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "blocked",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(unirig_dir),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(venv_python),
                "python_version": "3.12.3",
                "preflight": {
                    "status": "blocked",
                    "host": {"os": "linux", "arch": "aarch64", "platform_tag": "linux-aarch64"},
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "blockers": [
                        {
                            "category": "distribution",
                            "code": "missing-distribution",
                            "dependency": "wrapper-runtime",
                            "message": "Linux ARM64 has no validated prebuilt distribution path for this wrapper; do not assume the Windows-pinned package route exists on this host.",
                            "action": "stop",
                            "boundary": "wrapper",
                            "owner": "wrapper",
                        }
                    ],
                    "blocked": [
                        "Linux ARM64 has no validated prebuilt distribution path for this wrapper; do not assume the Windows-pinned package route exists on this host."
                    ],
                },
                "source_build": {
                    "status": "partial",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "support_posture": "experimental-unvalidated",
                    "bpy_evidence_class": "external-bpy-smoke-ready",
                    "external_blender": {
                        "candidate": {"path": "/custom/blender", "source": "path"},
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        },
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "status": "verified",
                            "proof_kind": "blender-subprocess",
                            "optional_owner": "blender-subprocess",
                            "supported_stages": ["extract-prepare", "extract-skin", "merge"],
                        }
                    },
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "verification": "import-smoke", "blockers": [], "blocker_codes": []},
                        "spconv": {"status": "ready", "ready": True, "verification": "import-smoke", "blockers": [], "blocker_codes": []},
                        "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "deferred_work": ["bpy-portability"],
                    "non_blender_runtime_ready": True,
                },
            },
            extension_root=self.ext_dir,
        )

        context = bootstrap.ensure_ready(self.ext_dir)

        self.assertEqual(context.install_state, "partial")
        self.assertEqual(context.source_build["executable_boundary"]["extract_merge"]["status"], "verified")
        self.assertTrue(context.source_build["non_blender_runtime_ready"])
        self.assertEqual(context.source_build["stages"]["bpy"]["status"], "external-bpy-smoke-ready")

    def test_normalize_state_allows_linux_arm64_partial_runtime_for_persisted_stage_subset_without_merge(self) -> None:
        normalized = bootstrap.normalize_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "blocked",
                "python_version": "3.12.3",
                "source_build": {
                    "status": "blocked",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "bpy_evidence_class": "external-bpy-smoke-ready",
                    "external_blender": {
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        }
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": True,
                            "ready": True,
                            "status": "verified",
                            "proof_kind": "blender-subprocess",
                            "supported_stages": ["extract-prepare", "skeleton", "extract-skin", "skin"],
                        }
                    },
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "spconv": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                    },
                    "blockers": [],
                    "blocked_reasons": [],
                    "non_blender_runtime_ready": True,
                },
            },
            self.ext_dir,
        )

        self.assertEqual(normalized["install_state"], "partial")
        self.assertEqual(
            normalized["source_build"]["executable_boundary"]["extract_merge"]["supported_stages"],
            ["extract-prepare", "skeleton", "extract-skin", "skin"],
        )

    def test_ensure_ready_recovers_linux_arm64_partial_runtime_from_persisted_stage_proofs_and_live_imports(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        self._write_minimal_upstream_tree(unirig_dir)
        venv_python = self.ext_dir / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

        run_extract_prepare = runtime_root / "runs" / "run-aa"
        run_extract_prepare.mkdir(parents=True, exist_ok=True)
        (run_extract_prepare / "extract_npz").mkdir(parents=True, exist_ok=True)
        (run_extract_prepare / "extract_npz" / "raw_data.npz").write_bytes(b"npz")
        (run_extract_prepare / "result.json").write_text(
            json.dumps(
                {
                    "protocol_version": 1,
                    "stage": "extract-prepare",
                    "status": "ok",
                    "produced": [str(run_extract_prepare / "extract_npz" / "raw_data.npz")],
                }
            ),
            encoding="utf-8",
        )

        run_skeleton = runtime_root / "runs" / "run-bb"
        run_skeleton.mkdir(parents=True, exist_ok=True)
        (run_skeleton / "skeleton_stage.fbx").write_bytes(b"skeleton")
        skeleton_log = runtime_root / "logs" / "run-bb"
        skeleton_log.mkdir(parents=True, exist_ok=True)
        (skeleton_log / "skeleton.log").write_text("stage: skeleton\n", encoding="utf-8")

        run_extract_skin = runtime_root / "runs" / "run-cc"
        run_extract_skin.mkdir(parents=True, exist_ok=True)
        (run_extract_skin / "skin_npz").mkdir(parents=True, exist_ok=True)
        (run_extract_skin / "skin_npz" / "raw_data.npz").write_bytes(b"npz")
        (run_extract_skin / "result.json").write_text(
            json.dumps(
                {
                    "protocol_version": 1,
                    "stage": "extract-skin",
                    "status": "ok",
                    "produced": [str(run_extract_skin / "skin_npz" / "raw_data.npz")],
                }
            ),
            encoding="utf-8",
        )

        run_skin = runtime_root / "runs" / "run-dd"
        run_skin.mkdir(parents=True, exist_ok=True)
        (run_skin / "skin_stage.fbx").write_bytes(b"skin")
        (run_skin / "result.json").write_text(
            json.dumps(
                {
                    "protocol_version": 1,
                    "stage": "skin",
                    "status": "ok",
                    "produced": [str(run_skin / "skin_stage.fbx")],
                }
            ),
            encoding="utf-8",
        )

        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "blocked",
                "runtime_root": str(runtime_root),
                "logs_dir": str(runtime_root / "logs"),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(unirig_dir),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(venv_python),
                "python_version": "3.12.3",
                "source_build": {
                    "status": "blocked",
                    "mode": "staged-source-build",
                    "host_class": "linux-arm64",
                    "bpy_evidence_class": "external-bpy-smoke-ready",
                    "external_blender": {
                        "candidate": {"path": "/custom/blender", "source": "path"},
                        "classification": {
                            "status": "external-bpy-smoke-ready",
                            "ready": False,
                            "evidence_kind": "external-blender",
                            "blockers": [],
                            "blocker_codes": [],
                        },
                    },
                    "executable_boundary": {
                        "extract_merge": {
                            "enabled": False,
                            "ready": False,
                            "status": "missing",
                            "proof_kind": "blender-subprocess",
                            "supported_stages": ["extract-prepare", "extract-skin", "merge"],
                        }
                    },
                    "stages": {
                        "baseline": {"status": "ready", "ready": True},
                        "pyg": {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []},
                        "spconv": {"status": "build-ready", "ready": False, "blockers": [], "blocker_codes": []},
                        "bpy": {"status": "external-bpy-smoke-ready", "ready": False, "blockers": [], "blocker_codes": []},
                    },
                    "blockers": [],
                    "blocked_reasons": ["Linux ARM64 has no validated prebuilt distribution path for this wrapper."],
                    "deferred_work": ["bpy-portability"],
                    "non_blender_runtime_ready": False,
                },
            },
            extension_root=self.ext_dir,
        )

        with mock.patch.object(
            bootstrap.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr=""),
            ],
        ):
            context = bootstrap.ensure_ready(self.ext_dir)

        self.assertEqual(context.install_state, "partial")
        self.assertTrue(context.source_build["non_blender_runtime_ready"])
        self.assertEqual(
            context.source_build["executable_boundary"]["extract_merge"]["supported_stages"],
            ["extract-prepare", "skeleton", "extract-skin", "skin", "merge"],
        )
        self.assertTrue(context.source_build["executable_boundary"]["extract_merge"]["recovered_from_persisted_stage_proofs"])

    def test_bootstrap_reuses_legacy_state_keys_for_transition_readiness(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        self._write_minimal_upstream_tree(unirig_dir)
        venv_python = self.ext_dir / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(unirig_dir),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(venv_python),
                "python_version": "3.12.3",
                "preflight": {"status": "ready", "host": {"os": "linux", "arch": "x86_64"}, "blocked": []},
            },
            extension_root=self.ext_dir,
        )

        context = bootstrap.ensure_ready(self.ext_dir)

        self.assertEqual(context.install_state, "ready")
        self.assertEqual(context.source_ref, bootstrap.UPSTREAM_REF_DEFAULT)
        self.assertEqual(context.last_verification["host"], {"arch": "x86_64", "os": "linux"})
        self.assertEqual(context.last_verification["python_version"], "3.12.3")

    def test_bootstrap_ready_context_surfaces_normalized_runtime_fields_from_planner_metadata(self) -> None:
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        self._write_minimal_upstream_tree(unirig_dir)
        venv_python = self.ext_dir / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "ready",
                "runtime_root": str(runtime_root),
                "runtime_vendor_dir": str(runtime_root / "vendor"),
                "unirig_dir": str(unirig_dir),
                "hf_home": str(runtime_root / "hf-home"),
                "venv_python": str(venv_python),
                "python_version": "3.12.3",
                "planner": {
                    "host_class": "linux-x86_64",
                    "support_posture": "supported",
                    "install_mode": "prebuilt",
                    "dependencies": [{"name": "spconv", "strategy": "generic-prebuilt-package"}],
                    "deferred": [],
                },
                "preflight": {
                    "status": "ready",
                    "host": {"os": "linux", "arch": "x86_64", "platform_tag": "linux-x86_64"},
                    "host_class": "linux-x86_64",
                    "support_posture": "supported",
                    "blockers": [],
                    "blocked": [],
                },
                "install_plan": {"summary": {"host_class": "linux-x86_64", "install_mode": "prebuilt", "status": "ready"}},
                "deferred_work": [],
            },
            extension_root=self.ext_dir,
        )

        context = bootstrap.ensure_ready(self.ext_dir)

        self.assertEqual(context.platform_policy["selected"]["key"], "linux-x86_64")
        self.assertEqual(context.platform_policy["selected"]["status"], "supported")
        self.assertEqual(context.platform_policy["selected"]["install_mode"], "prebuilt")
        self.assertEqual(context.source_build["status"], "ready")
        self.assertEqual(context.source_build["mode"], "prebuilt")
        self.assertEqual(context.source_build["host_class"], "linux-x86_64")
        self.assertEqual(context.source_build["blockers"], [])

    def test_bootstrap_reports_actionable_readiness_errors_without_policy_noise(self) -> None:
        bootstrap.save_state(
            {
                "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
                "install_state": "blocked",
                "preflight": {
                    "status": "blocked",
                    "host": {"os": "linux", "arch": "aarch64"},
                    "blocked": ["Python development headers are missing"],
                },
            },
            extension_root=self.ext_dir,
        )

        with self.assertRaises(bootstrap.BootstrapError) as ctx:
            bootstrap.ensure_ready(self.ext_dir)

        message = str(ctx.exception)
        self.assertIn("Python development headers are missing", message)
        self.assertNotIn("platform policy", message.lower())
        self.assertNotIn("source-build", message)


if __name__ == "__main__":
    unittest.main()
