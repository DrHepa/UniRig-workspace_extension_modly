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

        staged_dir, _, _ = module._prepare_runtime_source(self.ext_dir, {"unirig_source_dir": str(source_dir)})

        staged_config = (staged_dir / config_relpath).read_text(encoding="utf-8")
        staged_skin_config = (staged_dir / skin_config_relpath).read_text(encoding="utf-8")
        self.assertIn("flash_attention_2", staged_config)
        self.assertIn("enable_flash: true", staged_skin_config)
        self.assertFalse(module._runtime_stage_patch_report_path(self.ext_dir).exists())

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

    def test_windows_bootstrap_uses_requested_interpreter_without_wrapper_resolution(self) -> None:
        module = load_setup_module()
        requested = Path(r"C:\Python312\python.exe")

        requested_host_python, bootstrap_python, resolution = module._resolve_bootstrap_python({"python_exe": str(requested)})

        self.assertEqual(requested_host_python, requested)
        self.assertEqual(bootstrap_python, requested)
        self.assertEqual(resolution["selected_python"], str(requested))
        self.assertEqual(resolution["selected_source"], "requested-host-python")

    def test_install_runtime_packages_runs_single_pinned_profile(self) -> None:
        module = load_setup_module()
        unirig_dir = self.ext_dir / ".unirig-runtime" / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        requirements = unirig_dir / "requirements.txt"
        requirements.write_text("trimesh\n", encoding="utf-8")

        with mock.patch.object(module, "_venv_python", return_value=self.ext_dir / "venv" / "bin" / "python"), mock.patch.object(
            module, "_run"
        ) as run_mock:
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

        with mock.patch.object(module.os, "name", "nt"):
            filtered = module._filtered_requirements_path(unirig_dir, runtime_root)

        self.assertEqual(filtered, runtime_root / "requirements.upstream.windows.txt")
        content = filtered.read_text(encoding="utf-8")
        self.assertIn("trimesh", content)
        self.assertIn("spconv-cu120", content)
        self.assertNotIn("flash_attn==9.9.9", content)
        self.assertNotIn("flash-attn", content)

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

        self.assertEqual(module._filtered_requirements_path(unirig_dir, runtime_root), requirements)

    def test_install_runtime_packages_filters_upstream_requirements_and_installs_windows_shims(self) -> None:
        module = load_setup_module()
        runtime_root = self.ext_dir / ".unirig-runtime"
        unirig_dir = runtime_root / "vendor" / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        (unirig_dir / "requirements.txt").write_text("trimesh\nflash_attn\n", encoding="utf-8")

        with mock.patch.object(module.os, "name", "nt"), mock.patch.object(
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
        self.assertNotIn("platform_policy", state)
        self.assertNotIn("source_build", state)
        self.assertTrue((self.ext_dir / ".unirig-runtime" / "logs" / "bootstrap-preflight.json").exists())

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
