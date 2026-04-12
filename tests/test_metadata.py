# pyright: reportMissingImports=false

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from unirig_ext import bootstrap
from unirig_ext.bootstrap import RuntimeContext
from unirig_ext.io import derive_output_path
from unirig_ext.metadata import build_sidecar, sidecar_path_for, write_sidecar


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


if __name__ == "__main__":
    unittest.main()
