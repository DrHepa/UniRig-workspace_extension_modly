from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext import bootstrap, io
from unirig_ext.bootstrap import RuntimeContext


class PrepareInputMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unirig-io-"))
        self.run_dir = self.temp_dir / "run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.context = self._context(host_os="linux")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _context(self, *, host_os: str) -> RuntimeContext:
        return RuntimeContext(
            extension_root=self.temp_dir,
            runtime_root=self.temp_dir / ".unirig-runtime",
            cache_dir=self.temp_dir / ".unirig-runtime" / "cache",
            assets_dir=self.temp_dir / ".unirig-runtime" / "assets",
            logs_dir=self.temp_dir / ".unirig-runtime" / "logs",
            state_path=self.temp_dir / ".unirig-runtime" / "bootstrap_state.json",
            venv_dir=self.temp_dir / "venv",
            venv_python=Path(sys.executable),
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
            platform_tag=f"{host_os}-x86_64",
            python_version="3.12.3",
            platform_policy=bootstrap.resolve_platform_policy(host_os, "x86_64"),
            source_build={"status": "ready", "mode": "prebuilt", "dependencies": {}},
        )

    def test_prepare_input_mesh_normalizes_windows_glb_before_extract(self) -> None:
        mesh_path = self.run_dir / "input.glb"
        mesh_path.write_bytes(b"source-glb")
        prepared_path = self.run_dir / "prepared.glb"

        def _fake_run(*args, **kwargs):
            prepared_path.write_bytes(b"normalized-glb")
            return subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr="")

        with mock.patch("unirig_ext.io.subprocess.run", side_effect=_fake_run) as run_mock, mock.patch(
            "unirig_ext.io.shutil.copy2"
        ) as copy_mock:
            result = io.prepare_input_mesh(mesh_path, self.run_dir, self._context(host_os="windows"))

        self.assertEqual(result, prepared_path)
        self.assertEqual(result.read_bytes(), b"normalized-glb")
        copy_mock.assert_not_called()
        run_mock.assert_called_once()

    def test_prepare_input_mesh_normalizes_windows_gltf_before_extract(self) -> None:
        mesh_path = self.run_dir / "input.gltf"
        mesh_path.write_text('{"asset":{"version":"2.0"}}', encoding="utf-8")
        prepared_path = self.run_dir / "prepared.glb"

        def _fake_run(*args, **kwargs):
            prepared_path.write_bytes(b"normalized-gltf")
            return subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr="")

        with mock.patch("unirig_ext.io.subprocess.run", side_effect=_fake_run) as run_mock:
            result = io.prepare_input_mesh(mesh_path, self.run_dir, self._context(host_os="windows"))

        self.assertEqual(result, prepared_path)
        self.assertEqual(result.read_bytes(), b"normalized-gltf")
        run_mock.assert_called_once()

    def test_prepare_input_mesh_keeps_linux_glb_direct_copy_path(self) -> None:
        mesh_path = self.run_dir / "input.glb"
        mesh_path.write_bytes(b"linux-direct")

        with mock.patch("unirig_ext.io.subprocess.run") as run_mock:
            result = io.prepare_input_mesh(mesh_path, self.run_dir, self.context)

        self.assertEqual(result, self.run_dir / "prepared.glb")
        self.assertEqual(result.read_bytes(), b"linux-direct")
        run_mock.assert_not_called()


class PublishOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unirig-publish-"))
        self.source_path = self.temp_dir / "merged.glb"
        self.source_path.write_bytes(b"rigged-output")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _context(self, *, host_os: str, host_arch: str) -> RuntimeContext:
        return RuntimeContext(
            extension_root=self.temp_dir,
            runtime_root=self.temp_dir / ".unirig-runtime",
            cache_dir=self.temp_dir / ".unirig-runtime" / "cache",
            assets_dir=self.temp_dir / ".unirig-runtime" / "assets",
            logs_dir=self.temp_dir / ".unirig-runtime" / "logs",
            state_path=self.temp_dir / ".unirig-runtime" / "bootstrap_state.json",
            venv_dir=self.temp_dir / "venv",
            venv_python=Path(sys.executable),
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
            platform_tag=f"{host_os}-{host_arch}",
            python_version="3.12.3",
            platform_policy=bootstrap.resolve_platform_policy(host_os, host_arch),
            source_build={"status": "ready", "mode": "prebuilt", "dependencies": {}},
        )

    def test_publish_output_keeps_deterministic_unirig_path_by_default(self) -> None:
        mesh_path = self.temp_dir / "avatar.glb"
        mesh_path.write_bytes(b"original-input")

        published = io.publish_output(self.source_path, mesh_path, context=self._context(host_os="linux", host_arch="x86_64"))

        self.assertEqual(published, self.temp_dir / "avatar_unirig.glb")
        self.assertEqual(published.read_bytes(), b"rigged-output")
        self.assertEqual(mesh_path.read_bytes(), b"original-input")

    def test_publish_output_mirrors_linux_arm64_glb_back_to_input_path(self) -> None:
        mesh_path = self.temp_dir / "avatar.glb"
        mesh_path.write_bytes(b"original-input")

        published = io.publish_output(self.source_path, mesh_path, context=self._context(host_os="linux", host_arch="aarch64"))

        self.assertEqual(published, self.temp_dir / "avatar_unirig.glb")
        self.assertEqual(published.read_bytes(), b"rigged-output")
        self.assertEqual(mesh_path.read_bytes(), b"rigged-output")


if __name__ == "__main__":
    unittest.main()
