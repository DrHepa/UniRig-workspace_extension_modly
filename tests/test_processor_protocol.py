from __future__ import annotations

import json
import importlib.util
import io as text_io
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from contextlib import nullcontext
from pathlib import Path
from typing import Callable
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PROCESSOR = ROOT / "processor.py"
MANIFEST = ROOT / "manifest.json"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext import blender_bridge, bootstrap, pipeline
from unirig_ext.bootstrap import RuntimeContext
from unirig_ext.humanoid_contract import HumanoidContractError
from fixtures.unirig_real_topology import real_unirig_40_payload
from test_semantic_humanoid_resolver import short_trunk_output_payload


def write_minimal_valid_glb(target: Path) -> Path:
    json_chunk = json.dumps(
        {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0}]}],
            "buffers": [{"byteLength": 44}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 6, "target": 34963},
                {"buffer": 0, "byteOffset": 8, "byteLength": 36, "target": 34962},
            ],
            "accessors": [
                {"bufferView": 0, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR", "max": [2], "min": [0]},
                {
                    "bufferView": 1,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3",
                    "max": [1.0, 1.0, 0.0],
                    "min": [0.0, 0.0, 0.0],
                },
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "

    binary_chunk = struct.pack("<3H", 0, 1, 2) + b"\x00\x00" + struct.pack(
        "<9f",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )

    blob = bytearray(b"glTF")
    blob += struct.pack("<I", 2)
    blob += struct.pack("<I", 12 + 8 + len(json_chunk) + 8 + len(binary_chunk))
    blob += struct.pack("<I", len(json_chunk))
    blob += b"JSON"
    blob += json_chunk
    blob += struct.pack("<I", len(binary_chunk))
    blob += b"BIN\x00"
    blob += binary_chunk

    target.write_bytes(blob)
    return target


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


def create_fake_unirig_runtime(extension_root: Path) -> Path:
    runtime_root = extension_root / ".unirig-runtime"
    unirig_dir = runtime_root / "vendor" / "unirig"
    (unirig_dir / "src" / "data").mkdir(parents=True, exist_ok=True)
    (unirig_dir / "src" / "inference").mkdir(parents=True, exist_ok=True)
    (unirig_dir / "configs" / "task").mkdir(parents=True, exist_ok=True)
    (unirig_dir / "configs" / "data").mkdir(parents=True, exist_ok=True)
    for package_dir in (unirig_dir / "src", unirig_dir / "src" / "data", unirig_dir / "src" / "inference"):
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

    (unirig_dir / "requirements.txt").write_text("trimesh\n", encoding="utf-8")
    (unirig_dir / "configs" / "task" / "quick_inference_skeleton_articulationxl_ar_256.yaml").write_text("skeleton\n", encoding="utf-8")
    (unirig_dir / "configs" / "task" / "quick_inference_unirig_skin.yaml").write_text("skin\n", encoding="utf-8")
    (unirig_dir / "configs" / "data" / "quick_inference.yaml").write_text("extract\n", encoding="utf-8")
    (unirig_dir / "run.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "args = {}\n"
        "for item in sys.argv[1:]:\n"
        "    if item.startswith('--') and '=' in item:\n"
        "        key, value = item[2:].split('=', 1)\n"
        "        args[key] = value\n"
        "output = Path(args['output'])\n"
        "stage = 'skeleton' if 'skeleton' in args.get('task', '') else 'skin'\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_bytes((Path(args['input']).read_bytes() if Path(args['input']).exists() else args['input'].encode('utf-8')) + f'::{stage}'.encode('utf-8'))\n",
        encoding="utf-8",
    )
    (unirig_dir / "src" / "data" / "extract.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "args = {}\n"
        "for item in sys.argv[1:]:\n"
        "    if item.startswith('--') and '=' in item:\n"
        "        key, value = item[2:].split('=', 1)\n"
        "        args[key] = value\n"
        "input_name = Path(args['input']).stem\n"
        "destination = Path(args['output_dir']) / input_name\n"
        "destination.mkdir(parents=True, exist_ok=True)\n"
        "(destination / 'raw_data.npz').write_bytes(b'npz')\n",
        encoding="utf-8",
    )
    (unirig_dir / "src" / "inference" / "merge.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "args = {}\n"
        "for item in sys.argv[1:]:\n"
        "    if item.startswith('--') and '=' in item:\n"
        "        key, value = item[2:].split('=', 1)\n"
        "        args[key] = value\n"
        "source = Path(args['source']).read_bytes()\n"
        "target = Path(args['target']).read_bytes()\n"
        "output = Path(args['output'])\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_bytes(target + b'::merged::' + source)\n",
        encoding="utf-8",
    )

    state = {
        "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
        "install_state": "ready",
        "runtime_mode": "real",
        "runtime_root": str(runtime_root),
        "cache_dir": str(runtime_root / "cache"),
        "assets_dir": str(runtime_root / "assets"),
        "logs_dir": str(runtime_root / "logs"),
        "runtime_vendor_dir": str(runtime_root / "vendor"),
        "unirig_dir": str(unirig_dir),
        "hf_home": str(runtime_root / "hf-home"),
        "venv_dir": str(extension_root / "venv"),
        "venv_python": sys.executable,
        "vendor_source": "test-fixture",
        "source_ref": "test-fixture",
        "python_version": "3.12.3",
        "bootstrap_manifest": bootstrap.arm64_prerequisite_manifest(),
        "source_build": {
            "status": "ready",
            "mode": "source-build",
            "dependencies": {
                "cumm": {"name": "cumm", "build_mode": "source-build", "result": {"status": "ready"}},
                "spconv": {"name": "spconv", "build_mode": "source-build", "result": {"status": "ready"}},
            },
        },
        "preflight": {
            "status": "ready",
            "checked_at": "2026-01-01T00:00:00+00:00",
            "manifest": bootstrap.arm64_prerequisite_manifest(),
            "checks": [],
            "blocked": [],
        },
    }
    state_path = runtime_root / "bootstrap_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return runtime_root


class ProcessorProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unirig-protocol-"))
        self.ext_dir = self.temp_dir / "extension-root"
        self.ext_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROCESSOR, self.ext_dir / PROCESSOR.name)
        shutil.copytree(SRC, self.ext_dir / "src")
        create_fake_unirig_runtime(self.ext_dir)
        self.ready_context: RuntimeContext | None = None
        self.input_mesh = self.temp_dir / "mesh.glb"
        write_minimal_valid_glb(self.input_mesh)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _runtime_state_path(self) -> Path:
        return self.ext_dir / ".unirig-runtime" / "bootstrap_state.json"

    def _read_runtime_state(self) -> dict:
        return json.loads(self._runtime_state_path().read_text(encoding="utf-8"))

    def _write_runtime_state(self, state: dict) -> None:
        self._runtime_state_path().write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _make_fake_blender_executable(self, *, trace_path: Path, scenario: str) -> Path:
        blender_path = self.temp_dir / f"fake-blender-{scenario}.py"
        blender_path.write_text(
            "#!/usr/bin/env python3\n"
            "from __future__ import annotations\n"
            "import json\n"
            "import os\n"
            "import sys\n"
            "import time\n"
            "from pathlib import Path\n"
            f"SCENARIO = {scenario!r}\n"
            f"TRACE_PATH = Path({str(trace_path)!r})\n"
            "MARKER = 'UNIRIG_BLENDER_STAGE_RESULT='\n"
            "if '--' not in sys.argv:\n"
            "    raise SystemExit('missing payload delimiter')\n"
            "payload_path = Path(sys.argv[sys.argv.index('--') + 1])\n"
            "payload = json.loads(payload_path.read_text(encoding='utf-8'))\n"
            "TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
            "with TRACE_PATH.open('a', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps({'stage': payload['stage'], 'scenario': SCENARIO}) + '\\n')\n"
            "if SCENARIO == 'timeout' and payload['stage'] == 'merge':\n"
            "    time.sleep(float(os.environ.get('UNIRIG_FAKE_BLENDER_SLEEP_SECONDS', '0.05')))\n"
            "    raise SystemExit(0)\n"
            "result_path = Path(payload['run_dir']) / 'result.json'\n"
            "if SCENARIO == 'parse-failure' and payload['stage'] == 'merge':\n"
            "    result_path.write_text('{not-json', encoding='utf-8')\n"
            "    sys.stdout.write(MARKER + str(result_path.resolve()) + '\\n')\n"
            "    raise SystemExit(0)\n"
            "if SCENARIO == 'skeleton-shape-mismatch' and payload['stage'] == 'skeleton':\n"
            "    result = {\n"
            "        'protocol_version': 1,\n"
            "        'stage': payload['stage'],\n"
            "        'status': 'stage-failed',\n"
            "        'produced': [],\n"
            "        'error_code': 'expected-output-missing',\n"
            "        'message': 'Expected Blender stage outputs were not created on disk.',\n"
            "        'stdout_tail': ['all input arrays must have the same shape'],\n"
            "        'stderr_tail': [],\n"
            "    }\n"
            "    result_path.write_text(json.dumps(result), encoding='utf-8')\n"
            "    sys.stdout.write('all input arrays must have the same shape\\n')\n"
            "    sys.stdout.write(MARKER + str(result_path.resolve()) + '\\n')\n"
            "    raise SystemExit(0)\n"
            "output_path = Path(payload['input']['output'])\n"
            "output_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "if payload['stage'].startswith('extract-'):\n"
            "    output_path.write_bytes(f\"npz::{payload['stage']}\".encode('utf-8'))\n"
            "elif payload['stage'] in {'skeleton', 'skin'}:\n"
            "    source = Path(payload['input']['source']).read_bytes()\n"
            "    output_path.write_bytes(source + f\"::blender-subprocess::{payload['stage']}\".encode('utf-8'))\n"
            "else:\n"
            "    source = Path(payload['input']['source']).read_bytes()\n"
            "    target = Path(payload['input']['target']).read_bytes()\n"
            "    output_path.write_bytes(target + b'::blender-subprocess::' + source)\n"
            "result = {\n"
            "    'protocol_version': 1,\n"
            "    'stage': payload['stage'],\n"
            "    'status': 'ok',\n"
            "    'produced': [str(output_path)],\n"
            "    'error_code': '',\n"
            "    'message': '',\n"
            "    'stdout_tail': [f\"{payload['stage']} ok\"],\n"
            "    'stderr_tail': [],\n"
            "}\n"
            "result_path.write_text(json.dumps(result), encoding='utf-8')\n"
            "sys.stdout.write('boot\\n')\n"
            "sys.stdout.write(MARKER + str(result_path.resolve()) + '\\n')\n",
            encoding="utf-8",
        )
        blender_path.chmod(0o755)
        return blender_path

    def _build_runtime_context(self, *, host_os: str, host_arch: str, source_build: dict) -> RuntimeContext:
        runtime_root = self.ext_dir / ".unirig-runtime"
        return RuntimeContext(
            extension_root=self.ext_dir,
            runtime_root=runtime_root,
            cache_dir=runtime_root / "cache",
            assets_dir=runtime_root / "assets",
            logs_dir=runtime_root / "logs",
            state_path=runtime_root / "bootstrap_state.json",
            venv_dir=self.ext_dir / "venv",
            venv_python=Path(sys.executable),
            runtime_vendor_dir=runtime_root / "vendor",
            unirig_dir=runtime_root / "vendor" / "unirig",
            hf_home=runtime_root / "hf-home",
            extension_id="unirig-process-extension",
            runtime_mode="real",
            allow_local_stub_runtime=False,
            bootstrap_version=bootstrap.BOOTSTRAP_VERSION,
            vendor_source="test-fixture",
            source_ref="test-fixture",
            host_python=sys.executable,
            platform_tag=f"{host_os}-{host_arch.lower()}",
            python_version="3.12.3",
            platform_policy=bootstrap.resolve_platform_policy(host_os, host_arch),
            source_build=source_build,
            install_state="ready",
        )

    def _enable_linux_arm64_blender_seam(self, *, blender_path: Path) -> None:
        source_build = {
            "status": "ready",
            "mode": "staged-source-build",
            "host_class": "linux-arm64",
            "dependencies": {
                "cumm": {"name": "cumm", "build_mode": "source-build", "result": {"status": "ready"}},
                "spconv": {"name": "spconv", "build_mode": "source-build", "result": {"status": "ready"}},
            },
            "external_blender": {
                "candidate": {"source": "path", "path": str(blender_path)},
                "classification": {"status": "external-bpy-smoke-ready", "ready": False},
            },
            "executable_boundary": {
                "extract_merge": {
                    "enabled": True,
                    "ready": True,
                    "status": "verified",
                    "proof_kind": blender_bridge.BLENDER_SUBPROCESS_MODE,
                    "optional_owner": blender_bridge.BLENDER_SUBPROCESS_MODE,
                    "supported_stages": list(blender_bridge.BLENDER_STAGE_NAMES),
                    "candidate": {"source": "path", "path": str(blender_path)},
                }
            },
        }
        self.ready_context = self._build_runtime_context(host_os="linux", host_arch="aarch64", source_build=source_build)
        state = self._read_runtime_state()
        state["platform"] = "linux-aarch64"
        state["platform_policy"] = bootstrap.resolve_platform_policy("linux", "aarch64")
        state["last_verification"] = {"host": {"os": "linux", "arch": "aarch64"}, "python_version": "3.12.3"}
        state["preflight"]["host"] = {"os": "linux", "arch": "aarch64"}
        state["source_build"] = source_build
        self._write_runtime_state(state)

    def _enable_windows_x86_64_state_with_blender_candidate(self, *, blender_path: Path) -> None:
        source_build = {
            "status": "ready",
            "mode": "prebuilt",
            "dependencies": {},
            "external_blender": {
                "candidate": {"source": "path", "path": str(blender_path)},
                "classification": {"status": "external-bpy-smoke-ready", "ready": False},
            },
            "executable_boundary": {
                "extract_merge": {
                    "enabled": True,
                    "ready": True,
                    "status": "verified",
                    "proof_kind": blender_bridge.BLENDER_SUBPROCESS_MODE,
                    "optional_owner": blender_bridge.BLENDER_SUBPROCESS_MODE,
                    "supported_stages": list(blender_bridge.BLENDER_STAGE_NAMES),
                    "candidate": {"source": "path", "path": str(blender_path)},
                }
            },
        }
        self.ready_context = self._build_runtime_context(host_os="windows", host_arch="AMD64", source_build=source_build)
        state = self._read_runtime_state()
        state["platform"] = "windows-amd64"
        state["platform_policy"] = bootstrap.resolve_platform_policy("windows", "AMD64")
        state["last_verification"] = {"host": {"os": "windows", "arch": "amd64"}, "python_version": "3.12.3"}
        state["preflight"]["host"] = {"os": "windows", "arch": "amd64"}
        state["source_build"] = source_build
        self._write_runtime_state(state)

    def _run_processor_inprocess(self, payload: dict, *, timeout_seconds: float | None = None, **env_overrides: str) -> dict:
        stdout_stream = text_io.StringIO()
        stderr_stream = text_io.StringIO()
        stdin_stream = text_io.StringIO(json.dumps(payload) + "\n")
        env = os.environ.copy()
        env.update(env_overrides)

        saved_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "unirig_ext" or name.startswith("unirig_ext.")
        }
        for name in list(saved_modules):
            sys.modules.pop(name, None)

        copied_src = str(self.ext_dir / "src")
        original_sys_path = list(sys.path)
        sys.path.insert(0, copied_src)

        try:
            module_name = f"copied_processor_{id(self)}_{len(sys.modules)}"
            spec = importlib.util.spec_from_file_location(module_name, self.ext_dir / PROCESSOR.name)
            assert spec is not None and spec.loader is not None
            processor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(processor_module)
            ready_context_patch = (
                mock.patch.object(processor_module.bootstrap, "ensure_ready", return_value=self.ready_context)
                if self.ready_context is not None
                else nullcontext()
            )
            prepare_mesh_patch = nullcontext()
            if env_overrides.pop("UNIRIG_TEST_PREPARE_PASSTHROUGH", None) == "1":
                def _prepare_passthrough(staged_input: Path, run_dir: Path, context: RuntimeContext) -> Path:
                    del context
                    prepared_path = run_dir / f"prepared{staged_input.suffix.lower() or '.glb'}"
                    shutil.copy2(staged_input, prepared_path)
                    return prepared_path

                prepare_mesh_patch = mock.patch.object(processor_module.pipeline.io, "prepare_input_mesh", side_effect=_prepare_passthrough)

            with mock.patch.dict(os.environ, env, clear=True), mock.patch("sys.stdin", stdin_stream), mock.patch(
                "sys.stdout", stdout_stream
            ), mock.patch("sys.stderr", stderr_stream), ready_context_patch, prepare_mesh_patch:
                if timeout_seconds is None:
                    exit_code = processor_module.main()
                else:
                    with mock.patch.object(processor_module.pipeline, "BLENDER_SUBPROCESS_TIMEOUT_SECONDS", timeout_seconds):
                        exit_code = processor_module.main()
        finally:
            sys.path[:] = original_sys_path
            for name in list(sys.modules):
                if name == "unirig_ext" or name.startswith("unirig_ext."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

        return {
            "exit_code": exit_code,
            "stdout": stdout_stream.getvalue(),
            "stderr": stderr_stream.getvalue(),
            "messages": [json.loads(line) for line in stdout_stream.getvalue().splitlines() if line.strip()],
        }

    def _run_processor_inprocess_with_pipeline_patches(
        self,
        payload: dict,
        *,
        configure_pipeline: Callable[[object, ExitStack], None],
        configure_metadata: Callable[[object, ExitStack], None] | None = None,
        timeout_seconds: float | None = None,
        **env_overrides: str,
    ) -> dict:
        stdout_stream = text_io.StringIO()
        stderr_stream = text_io.StringIO()
        stdin_stream = text_io.StringIO(json.dumps(payload) + "\n")
        env = os.environ.copy()
        env.update(env_overrides)

        saved_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == "unirig_ext" or name.startswith("unirig_ext.")
        }
        for name in list(saved_modules):
            sys.modules.pop(name, None)

        copied_src = str(self.ext_dir / "src")
        original_sys_path = list(sys.path)
        sys.path.insert(0, copied_src)

        try:
            module_name = f"copied_processor_{id(self)}_{len(sys.modules)}_patched"
            spec = importlib.util.spec_from_file_location(module_name, self.ext_dir / PROCESSOR.name)
            assert spec is not None and spec.loader is not None
            processor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(processor_module)
            ready_context_patch = (
                mock.patch.object(processor_module.bootstrap, "ensure_ready", return_value=self.ready_context)
                if self.ready_context is not None
                else nullcontext()
            )

            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
                stack.enter_context(mock.patch("sys.stdin", stdin_stream))
                stack.enter_context(mock.patch("sys.stdout", stdout_stream))
                stack.enter_context(mock.patch("sys.stderr", stderr_stream))
                stack.enter_context(ready_context_patch)
                configure_pipeline(processor_module.pipeline, stack)
                if configure_metadata is not None:
                    configure_metadata(processor_module.metadata, stack)
                if timeout_seconds is None:
                    exit_code = processor_module.main()
                else:
                    with mock.patch.object(processor_module.pipeline, "BLENDER_SUBPROCESS_TIMEOUT_SECONDS", timeout_seconds):
                        exit_code = processor_module.main()
        finally:
            sys.path[:] = original_sys_path
            for name in list(sys.modules):
                if name == "unirig_ext" or name.startswith("unirig_ext."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

        return {
            "exit_code": exit_code,
            "stdout": stdout_stream.getvalue(),
            "stderr": stderr_stream.getvalue(),
            "messages": [json.loads(line) for line in stdout_stream.getvalue().splitlines() if line.strip()],
        }

    def _run_processor(self, payload: dict, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(env_overrides)
        return subprocess.run(
            [sys.executable, str(self.ext_dir / PROCESSOR.name)],
            input=json.dumps(payload) + "\n",
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def _latest_run_dir(self) -> Path:
        runs_dir = self.ext_dir / ".unirig-runtime" / "runs"
        return max((path for path in runs_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime_ns)

    def _trace_stages(self, trace_path: Path) -> list[str]:
        return [json.loads(line)["stage"] for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_manifest_stays_aligned_with_processor_contract(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(manifest["type"], "process")
        self.assertEqual(manifest["entry"], PROCESSOR.name)
        self.assertEqual([node["id"] for node in manifest["nodes"]], ["rig-mesh"])

    def test_processor_emits_progress_and_done_json_lines(self) -> None:
        result = self._run_processor({"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 7}})
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertTrue(messages, msg="processor emitted no protocol messages")
        self.assertEqual(messages[-1]["type"], "done")
        self.assertTrue({message["type"] for message in messages}.issubset({"progress", "log", "done", "error"}))
        self.assertTrue(any(message["type"] == "progress" for message in messages))
        done = [message for message in messages if message["type"] == "done"]
        self.assertEqual(len(done), 1)
        self.assertEqual(set(done[0]["result"].keys()), {"filePath"})
        self.assertTrue(done[0]["result"]["filePath"].endswith("mesh_unirig.glb"))

    def test_processor_runs_real_runtime_pipeline_and_writes_sidecar(self) -> None:
        original_input = self.input_mesh.read_bytes()
        result = self._run_processor({"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 11}})
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertTrue(messages, msg="processor emitted no protocol messages")
        self.assertTrue({message["type"] for message in messages}.issubset({"progress", "log", "done", "error"}))
        self.assertTrue(any(message.get("message", "") == "running skeleton stage" for message in messages if message["type"] == "log"))
        self.assertFalse(any("run.py" in message.get("message", "") for message in messages if message["type"] == "log"))
        self.assertFalse(any("src.data.extract" in message.get("message", "") for message in messages if message["type"] == "log"))
        self.assertFalse(any(".unirig-runtime" in message.get("message", "") for message in messages if message["type"] == "log"))
        self.assertFalse(any("runtime mode:" in message.get("message", "") for message in messages if message["type"] == "log"))

        done = [message for message in messages if message["type"] == "done"]
        self.assertEqual(len(done), 1)
        output_path = Path(done[0]["result"]["filePath"])
        self.assertTrue(output_path.exists())
        self.assertNotEqual(output_path.read_bytes(), original_input)

        sidecar_path = output_path.with_name(f"{output_path.stem}.rigmeta.json")
        self.assertTrue(sidecar_path.exists())
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["output_mesh"], output_path.name)
        self.assertEqual(sidecar["seed"], 11)
        self.assertEqual(sidecar["runtime"]["mode"], "real")

    def test_processor_publishes_done_result_inside_workspace_workflows_when_workspace_dir_is_provided(self) -> None:
        source_mesh = self.temp_dir / "Descargas" / "avatar.glb"
        source_mesh.parent.mkdir(parents=True, exist_ok=True)
        write_minimal_valid_glb(source_mesh)
        workspace_dir = self.temp_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        result = self._run_processor(
            {
                "input": {"filePath": str(source_mesh), "nodeId": "rig-mesh"},
                "params": {"seed": 11},
                "workspaceDir": str(workspace_dir),
            }
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(messages[-1]["type"], "done")
        output_path = Path(messages[-1]["result"]["filePath"])
        self.assertEqual(output_path, workspace_dir / "Workflows" / "avatar_unirig.glb")
        self.assertTrue(output_path.exists())
        self.assertFalse((source_mesh.parent / "avatar_unirig.glb").exists())

        sidecar_path = output_path.with_name(f"{output_path.stem}.rigmeta.json")
        self.assertTrue(sidecar_path.exists())

    def test_processor_keeps_legacy_done_result_when_workspace_dir_is_missing_on_disk(self) -> None:
        source_mesh = self.temp_dir / "Descargas" / "avatar.glb"
        source_mesh.parent.mkdir(parents=True, exist_ok=True)
        write_minimal_valid_glb(source_mesh)
        workspace_dir = self.temp_dir / "workspace-missing"

        result = self._run_processor(
            {
                "input": {"filePath": str(source_mesh), "nodeId": "rig-mesh"},
                "params": {"seed": 13},
                "workspaceDir": str(workspace_dir),
            }
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(messages[-1]["type"], "done")
        output_path = Path(messages[-1]["result"]["filePath"])
        self.assertEqual(output_path, source_mesh.parent / "avatar_unirig.glb")
        self.assertTrue(output_path.exists())

    def test_processor_keeps_legacy_done_result_when_workspace_dir_is_empty_string(self) -> None:
        source_mesh = self.temp_dir / "Descargas" / "avatar.glb"
        source_mesh.parent.mkdir(parents=True, exist_ok=True)
        write_minimal_valid_glb(source_mesh)

        result = self._run_processor(
            {
                "input": {"filePath": str(source_mesh), "nodeId": "rig-mesh"},
                "params": {"seed": 17},
                "workspaceDir": "   ",
            }
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(messages[-1]["type"], "done")
        output_path = Path(messages[-1]["result"]["filePath"])
        self.assertEqual(output_path, source_mesh.parent / "avatar_unirig.glb")
        self.assertTrue(output_path.exists())

    def test_processor_emits_error_without_done_when_workspace_publication_fails(self) -> None:
        source_mesh = self.temp_dir / "Descargas" / "avatar.glb"
        source_mesh.parent.mkdir(parents=True, exist_ok=True)
        write_minimal_valid_glb(source_mesh)
        workspace_dir = self.temp_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        def configure_pipeline(pipeline_module: object, stack: ExitStack) -> None:
            def fake_run(*, mesh_path: Path, params: dict, context: RuntimeContext, progress: Callable, log: Callable, workspace_dir: Path | None = None) -> Path:
                del mesh_path, params, context, progress, log
                raise pipeline_module.io.OutputPublicationError(
                    "Failed to publish UniRig output into the workspace Workflows directory. "
                    f"workspaceDir={workspace_dir}; target={workspace_dir / 'Workflows' / 'avatar_unirig.glb'}; error: workspace denied"
                )

            stack.enter_context(mock.patch.object(pipeline_module, "run", side_effect=fake_run))

        result = self._run_processor_inprocess_with_pipeline_patches(
            {
                "input": {"filePath": str(source_mesh), "nodeId": "rig-mesh"},
                "params": {"seed": 19},
                "workspaceDir": str(workspace_dir),
            },
            configure_pipeline=configure_pipeline,
        )

        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(any(message["type"] == "done" for message in result["messages"]))
        self.assertEqual(result["messages"][-1]["type"], "error")
        self.assertIn("Failed to publish UniRig output into the workspace Workflows directory", result["messages"][-1]["message"])

    def test_processor_emits_error_without_done_when_humanoid_contract_validation_fails(self) -> None:
        def configure_pipeline(pipeline_module: object, stack: ExitStack) -> None:
            def fake_run(*, mesh_path: Path, params: dict, context: RuntimeContext, progress: Callable, log: Callable, workspace_dir: Path | None = None) -> Path:
                del params, context, progress, log, workspace_dir
                output_path = mesh_path.with_name(f"{mesh_path.stem}_unirig.glb")
                output_path.write_bytes(b"published")
                return output_path

            stack.enter_context(mock.patch.object(pipeline_module, "run", side_effect=fake_run))

        def configure_metadata(metadata_module: object, stack: ExitStack) -> None:
            stack.enter_context(
                mock.patch.object(
                    metadata_module,
                    "write_sidecar",
                    side_effect=HumanoidContractError("Missing required humanoid role 'hips' from declared metadata."),
                )
            )

        result = self._run_processor_inprocess_with_pipeline_patches(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 29}},
            configure_pipeline=configure_pipeline,
            configure_metadata=configure_metadata,
        )

        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(any(message["type"] == "done" for message in result["messages"]))
        self.assertEqual(result["messages"][-1]["type"], "error")
        self.assertIn("Missing required humanoid role 'hips'", result["messages"][-1]["message"])

    def test_processor_keeps_workspace_result_canonical_for_linux_arm64_hosts(self) -> None:
        trace_path = self.temp_dir / "fake-blender-arm64-workspace.jsonl"
        blender_path = self._make_fake_blender_executable(trace_path=trace_path, scenario="success")
        self._enable_linux_arm64_blender_seam(blender_path=blender_path)
        source_mesh = self.temp_dir / "Descargas" / "avatar.glb"
        source_mesh.parent.mkdir(parents=True, exist_ok=True)
        write_minimal_valid_glb(source_mesh)
        workspace_dir = self.temp_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        result = self._run_processor_inprocess(
            {
                "input": {"filePath": str(source_mesh), "nodeId": "rig-mesh"},
                "params": {"seed": 23},
                "workspaceDir": str(workspace_dir),
            }
        )

        self.assertEqual(result["exit_code"], 0, msg=result["stderr"])
        messages = result["messages"]
        self.assertEqual(messages[-1]["type"], "done")
        output_path = Path(messages[-1]["result"]["filePath"])
        self.assertEqual(output_path, workspace_dir / "Workflows" / "avatar_unirig.glb")
        self.assertTrue(output_path.exists())
        self.assertEqual(source_mesh.read_bytes(), output_path.read_bytes())
        self.assertFalse((source_mesh.parent / "avatar_unirig.glb").exists())
        self.assertTrue(trace_path.exists(), msg="fake Blender seam should execute on Linux ARM64 workspace runs")

    def test_processor_rejects_missing_input_deterministically(self) -> None:
        result = self._run_processor({"input": {"filePath": str(self.temp_dir / "missing.glb"), "nodeId": "rig-mesh"}, "params": {}})
        self.assertNotEqual(result.returncode, 0)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(messages[-1]["type"], "error")
        self.assertIn("Input mesh does not exist", messages[-1]["message"])

    def test_processor_rejects_unknown_node(self) -> None:
        result = self._run_processor({"input": {"filePath": str(self.input_mesh), "nodeId": "other-node"}, "params": {}})
        self.assertNotEqual(result.returncode, 0)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(messages[-1]["type"], "error")
        self.assertIn("Unsupported nodeId", messages[-1]["message"])

    def test_processor_rejects_non_object_input_payload(self) -> None:
        result = self._run_processor({"input": "bad", "params": {}})
        self.assertNotEqual(result.returncode, 0)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(messages[-1]["type"], "error")
        self.assertIn("'input' must be a JSON object.", messages[-1]["message"])

    def test_processor_rejects_non_object_params_payload(self) -> None:
        result = self._run_processor({"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": []})
        self.assertNotEqual(result.returncode, 0)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(messages[-1]["type"], "error")
        self.assertIn("'params' must be a JSON object.", messages[-1]["message"])

    def test_processor_rejects_invalid_metadata_mode_before_runtime_publication(self) -> None:
        def configure_pipeline(pipeline_module: object, stack: ExitStack) -> None:
            stack.enter_context(mock.patch.object(pipeline_module, "run", side_effect=AssertionError("pipeline must not run")))

        result = self._run_processor_inprocess_with_pipeline_patches(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"metadata_mode": "bones"}},
            configure_pipeline=configure_pipeline,
        )

        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(any(message["type"] == "done" for message in result["messages"]))
        self.assertEqual(result["messages"][-1]["type"], "error")
        self.assertIn("metadata_mode", result["messages"][-1]["message"])
        self.assertIn("auto, legacy, humanoid", result["messages"][-1]["message"])

    def test_processor_humanoid_mode_failure_emits_error_without_done(self) -> None:
        def configure_pipeline(pipeline_module: object, stack: ExitStack) -> None:
            def fake_run(*, mesh_path: Path, params: dict, context: RuntimeContext, progress: Callable, log: Callable, workspace_dir: Path | None = None) -> Path:
                del params, context, progress, log, workspace_dir
                output_path = mesh_path.with_name(f"{mesh_path.stem}_unirig.glb")
                output_path.write_bytes(b"published")
                return output_path

            stack.enter_context(mock.patch.object(pipeline_module, "run", side_effect=fake_run))

        result = self._run_processor_inprocess_with_pipeline_patches(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 31, "metadata_mode": "humanoid"}},
            configure_pipeline=configure_pipeline,
        )

        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(any(message["type"] == "done" for message in result["messages"]))
        self.assertEqual(result["messages"][-1]["type"], "error")
        self.assertIn("metadata_mode=humanoid", result["messages"][-1]["message"])
        self.assertIn("valid humanoid", result["messages"][-1]["message"])

    def test_processor_humanoid_mode_insufficient_output_reports_contract_boundary_without_done(self) -> None:
        write_glb_json(self.input_mesh, real_unirig_40_payload())

        def configure_pipeline(pipeline_module: object, stack: ExitStack) -> None:
            def fake_run(*, mesh_path: Path, params: dict, context: RuntimeContext, progress: Callable, log: Callable, workspace_dir: Path | None = None) -> Path:
                del params, context, progress, log, workspace_dir
                output_path = mesh_path.with_name(f"{mesh_path.stem}_unirig.glb")
                write_glb_json(output_path, short_trunk_output_payload(prefix="out"))
                return output_path

            stack.enter_context(mock.patch.object(pipeline_module, "run", side_effect=fake_run))

        result = self._run_processor_inprocess_with_pipeline_patches(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 37, "metadata_mode": "humanoid"}},
            configure_pipeline=configure_pipeline,
        )

        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(any(message["type"] == "done" for message in result["messages"]))
        self.assertEqual(result["messages"][-1]["type"], "error")
        self.assertIn("humanoid_output_contract_insufficient", result["messages"][-1]["message"])
        self.assertIn("semantic_spine_missing", result["messages"][-1]["message"])
        self.assertIn("verified source-to-output transfer", result["messages"][-1]["message"])

    def test_processor_ignores_stage_override_environment_and_runs_upstream_path(self) -> None:
        trace_path = self.temp_dir / "hook-trace.jsonl"

        result = self._run_processor(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 13}},
            UNIRIG_SKELETON_COMMAND=f'{sys.executable} "{ROOT / "tests" / "fixtures" / "process_stage_hook.py"}"',
            UNIRIG_HOOK_TRACE_FILE=str(trace_path),
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertTrue(any(message.get("message", "") == "running skeleton stage" for message in messages if message["type"] == "log"))
        self.assertFalse(trace_path.exists(), msg="pipeline should ignore stage override hooks in normal runtime flow")

    def test_processor_stage_override_environment_does_not_change_public_error(self) -> None:
        trace_path = self.temp_dir / "hook-failure-trace.jsonl"

        result = self._run_processor(
            {"input": {"filePath": str(self.temp_dir / "missing.glb"), "nodeId": "rig-mesh"}, "params": {}},
            UNIRIG_SKELETON_COMMAND=f'{sys.executable} "{ROOT / "tests" / "fixtures" / "process_stage_hook.py"}"',
            UNIRIG_HOOK_TRACE_FILE=str(trace_path),
            UNIRIG_HOOK_FAIL_STAGE="skeleton",
            UNIRIG_HOOK_FAIL_CODE="23",
        )

        self.assertNotEqual(result.returncode, 0)
        messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(messages[-1]["type"], "error")
        self.assertIn("Input mesh does not exist", messages[-1]["message"])
        self.assertFalse(trace_path.exists(), msg="override hook should not execute on error paths either")

    def test_processor_transports_blender_subprocess_success_result_end_to_end(self) -> None:
        trace_path = self.temp_dir / "fake-blender-success.jsonl"
        blender_path = self._make_fake_blender_executable(trace_path=trace_path, scenario="success")
        self._enable_linux_arm64_blender_seam(blender_path=blender_path)

        result = self._run_processor_inprocess(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 101}}
        )

        self.assertEqual(result["exit_code"], 0, msg=result["stderr"])
        messages = result["messages"]
        self.assertEqual(messages[-1]["type"], "done")
        output_path = Path(messages[-1]["result"]["filePath"])
        self.assertTrue(output_path.exists())
        self.assertIn(b"::blender-subprocess::", output_path.read_bytes())
        self.assertTrue(trace_path.exists(), msg="fake Blender seam should execute on Linux ARM64 opt-in")
        self.assertEqual(self._trace_stages(trace_path), ["extract-prepare", "skeleton", "extract-skin", "skin", "merge"])

    def test_processor_transports_blender_subprocess_timeout_error_end_to_end(self) -> None:
        trace_path = self.temp_dir / "fake-blender-timeout.jsonl"
        blender_path = self._make_fake_blender_executable(trace_path=trace_path, scenario="timeout")
        self._enable_linux_arm64_blender_seam(blender_path=blender_path)

        result = self._run_processor_inprocess(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 103}},
            timeout_seconds=0.2,
            UNIRIG_FAKE_BLENDER_SLEEP_SECONDS="0.4",
        )

        self.assertNotEqual(result["exit_code"], 0)
        self.assertEqual(result["messages"][-1]["type"], "error")
        self.assertIn("UniRig merge stage failed", result["messages"][-1]["message"])
        self.assertTrue(trace_path.exists(), msg="fake Blender should have been launched before timeout")
        self.assertEqual(self._trace_stages(trace_path), ["extract-prepare", "skeleton", "extract-skin", "skin", "merge"])

    def test_processor_transports_blender_subprocess_parse_failure_error_end_to_end(self) -> None:
        linux_trace = self.temp_dir / "fake-blender-parse.jsonl"
        blender_path = self._make_fake_blender_executable(trace_path=linux_trace, scenario="parse-failure")
        self._enable_linux_arm64_blender_seam(blender_path=blender_path)

        linux_result = self._run_processor_inprocess(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 107}}
        )

        self.assertNotEqual(linux_result["exit_code"], 0)
        self.assertEqual(linux_result["messages"][-1]["type"], "error")
        self.assertIn("UniRig merge stage failed", linux_result["messages"][-1]["message"])
        self.assertTrue(linux_trace.exists(), msg="parse-failure path should still execute fake Blender")
        self.assertEqual(self._trace_stages(linux_trace), ["extract-prepare", "skeleton", "extract-skin", "skin", "merge"])
        latest_run_dir = self._latest_run_dir()
        merge_log = self.ext_dir / ".unirig-runtime" / "logs" / latest_run_dir.name / "merge.log"
        self.assertTrue(merge_log.exists())
        result_path = latest_run_dir / blender_bridge.BLENDER_RESULT_FILE_NAME
        self.assertTrue(result_path.exists())
        self.assertEqual(result_path.read_text(encoding="utf-8"), "{not-json")

    def test_processor_transports_skeleton_stage_diagnostics_end_to_end(self) -> None:
        trace_path = self.temp_dir / "fake-blender-skeleton-shape-mismatch.jsonl"
        blender_path = self._make_fake_blender_executable(trace_path=trace_path, scenario="skeleton-shape-mismatch")
        self._enable_linux_arm64_blender_seam(blender_path=blender_path)

        result = self._run_processor_inprocess(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 111}}
        )

        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(any(message["type"] == "done" for message in result["messages"]))
        self.assertEqual(result["messages"][-1]["type"], "error")
        message = result["messages"][-1]["message"]
        self.assertIn("UniRig skeleton stage failed", message)
        self.assertIn("run_id=run-", message)
        self.assertIn("stage=skeleton", message)
        self.assertIn("error_code=expected-output-missing", message)
        self.assertIn("original_input=", message)
        self.assertIn("staged_input=", message)
        self.assertIn("runtime_input=", message)
        self.assertIn("expected_output=", message)
        self.assertIn("skeleton_stage.fbx", message)
        self.assertIn("result_json=", message)
        self.assertIn("result.json", message)
        self.assertIn("stage_log=", message)
        self.assertIn("skeleton.log", message)
        self.assertIn("stdout_tail=", message)
        self.assertIn("all input arrays must have the same shape", message)
        self.assertIn("stderr_tail=(not captured)", message)
        self.assertIn("blender_returncode=0", message)

    def test_processor_keeps_windows_x86_64_blender_seam_isolated_end_to_end(self) -> None:
        windows_trace = self.temp_dir / "fake-blender-windows.jsonl"
        windows_blender = self._make_fake_blender_executable(trace_path=windows_trace, scenario="success")
        self._enable_windows_x86_64_state_with_blender_candidate(blender_path=windows_blender)

        windows_trace = self.temp_dir / "fake-blender-windows.jsonl"
        windows_result = self._run_processor_inprocess(
            {"input": {"filePath": str(self.input_mesh), "nodeId": "rig-mesh"}, "params": {"seed": 109}},
            UNIRIG_TEST_PREPARE_PASSTHROUGH="1",
        )

        self.assertEqual(windows_result["exit_code"], 0, msg=windows_result["stderr"])
        self.assertEqual(windows_result["messages"][-1]["type"], "done")
        output_path = Path(windows_result["messages"][-1]["result"]["filePath"])
        self.assertTrue(output_path.exists())
        self.assertNotIn(b"::blender-subprocess::", output_path.read_bytes())
        self.assertFalse(windows_trace.exists(), msg="Windows x86_64 must stay isolated from the Blender seam end to end")


class PipelineGuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unirig-pipeline-"))
        self.context = self._context(host_os="linux")
        self.run_dir = self.temp_dir / ".unirig-runtime" / "runs" / "run-fixed"
        self.run_dir.mkdir(parents=True, exist_ok=True)

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
            platform_tag=f"{host_os}-x86_64",
            python_version="3.12.3",
            platform_policy=bootstrap.resolve_platform_policy(host_os, "x86_64" if host_os != "windows" else "AMD64"),
            source_build={"status": "ready", "mode": "prebuilt", "dependencies": {}},
        )

    def _context_with_host(self, *, host_os: str, host_arch: str, source_build: dict | None = None) -> RuntimeContext:
        return RuntimeContext(
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
            platform_tag=f"{host_os}-{host_arch.lower()}",
            python_version="3.12.3",
            platform_policy=bootstrap.resolve_platform_policy(host_os, host_arch),
            source_build=source_build or {"status": "ready", "mode": "prebuilt", "dependencies": {}},
        )

    def _stage_log_path(self, stage_name: str) -> Path:
        return self.context.logs_dir / self.run_dir.name / f"{stage_name}.log"

    def _linux_arm64_blender_context(self) -> RuntimeContext:
        return self._context_with_host(
            host_os="linux",
            host_arch="aarch64",
            source_build={
                "status": "blocked",
                "mode": "staged-source-build",
                "host_class": "linux-arm64",
                "external_blender": {
                    "candidate": {
                        "source": "path",
                        "path": "/opt/blender/blender",
                    },
                    "classification": {"status": "external-bpy-smoke-ready", "ready": False},
                },
                "executable_boundary": {
                    "extract_merge": {
                        "enabled": True,
                        "ready": True,
                        "status": "verified",
                        "proof_kind": "blender-subprocess",
                        "optional_owner": "blender-subprocess",
                        "supported_stages": ["extract-prepare", "extract-skin", "merge"],
                        "candidate": {
                            "source": "path",
                            "path": "/opt/blender/blender",
                        },
                    }
                },
            },
        )

    def _linux_arm64_qualification_context(
        self,
        *,
        mode: str,
        stages: list[str] | None = None,
    ) -> RuntimeContext:
        context = self._linux_arm64_blender_context()
        source_build = dict(context.source_build)
        qualification = {
            "extract_merge": {
                "mode": mode,
                "stages": list(stages or blender_bridge.BLENDER_STAGE_NAMES),
            }
        }
        source_build["qualification"] = qualification
        return RuntimeContext(**{**context.__dict__, "source_build": source_build})

    def _merge_blender_stage(self, *, seed: int = 61) -> tuple[RuntimeContext, pipeline.ExecutionStage, Path]:
        context = self._linux_arm64_blender_context()
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")
        source_path = self.run_dir / "skin_stage.fbx"
        source_path.write_bytes(b"skin")
        stage = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=seed,
        )[4]
        return context, stage, prepared_path

    def _assert_blender_stage_public_error_contract(self, exc: pipeline.PipelineError, *, stage_name: str, error_code: str) -> None:
        self.assertIn(error_code, str(exc))
        message = pipeline.public_error_message(exc)
        self.assertTrue(message.startswith(f"UniRig {stage_name} stage failed. Inspect extension runtime logs for details."))
        self.assertIn(f"stage={stage_name}", message)
        self.assertIn(f"error_code={error_code}", message)
        self.assertIn("run_id=", message)
        self.assertIn("expected_output=", message)
        self.assertIn("result_json=", message)
        self.assertIn("stage_log=", message)
        self.assertIn("stdout_tail=", message)
        self.assertIn("stderr_tail=", message)
        self.assertIn("blender_returncode=", message)

    def test_public_error_message_formats_structured_stage_diagnostic(self) -> None:
        diagnostic = pipeline.StageFailureDiagnostic(
            run_id="run-synthetic",
            stage="skeleton",
            error_code="expected-output-missing",
            original_input=str(self.temp_dir / "avatar.glb"),
            staged_input=str(self.run_dir / "input.glb"),
            runtime_input=str(self.context.unirig_dir / ".modly_stage_input_run-processor.glb"),
            expected_output=str(self.run_dir / "skeleton_stage.fbx"),
            result_json=str(self.run_dir / "result.json"),
            stage_log=str(self._stage_log_path("skeleton")),
            stdout_tail="all input arrays must have the same shape",
            stderr_tail="(not captured)",
            blender_returncode=0,
        )

        message = pipeline.public_error_message(pipeline.PipelineError("UniRig skeleton stage failed", diagnostic=diagnostic))

        self.assertTrue(message.startswith("UniRig skeleton stage failed"))
        for expected in (
            "run_id=run-synthetic",
            "stage=skeleton",
            "error_code=expected-output-missing",
            "original_input=",
            "staged_input=",
            "runtime_input=",
            "expected_output=",
            "result_json=",
            "stage_log=",
            "stdout_tail=all input arrays must have the same shape",
            "stderr_tail=(not captured)",
            "blender_returncode=0",
        ):
            self.assertIn(expected, message)

    def test_public_error_message_keeps_required_keys_when_diagnostic_values_are_missing(self) -> None:
        diagnostic = pipeline.StageFailureDiagnostic(stage="skeleton", error_code="expected-output-missing")

        message = pipeline.public_error_message(pipeline.PipelineError("UniRig skeleton stage failed", diagnostic=diagnostic))

        for expected in (
            "run_id=<unavailable>",
            "stage=skeleton",
            "error_code=expected-output-missing",
            "original_input=<unavailable>",
            "staged_input=<unavailable>",
            "runtime_input=<unavailable>",
            "expected_output=<unavailable>",
            "result_json=<unavailable>",
            "stage_log=<unavailable>",
            "stdout_tail=(not captured)",
            "stderr_tail=(not captured)",
            "blender_returncode=<unavailable>",
        ):
            self.assertIn(expected, message)

    def test_public_error_message_preserves_generic_fallback_without_diagnostic(self) -> None:
        message = pipeline.public_error_message(pipeline.PipelineError("UniRig skeleton stage failed (stage-failed)."))

        self.assertEqual(message, "UniRig skeleton stage failed. Inspect extension runtime logs for details.")

    def test_bounded_stream_tail_strips_control_chars_and_limits_lines_and_bytes(self) -> None:
        lines = [f"line-{index}" for index in range(45)]
        tail = pipeline.bounded_stream_tail("\n".join(["bad\x00control", *lines]))
        byte_limited_tail = pipeline.bounded_stream_tail("x" * 9000)

        self.assertNotIn("\x00", tail)
        self.assertLessEqual(len(tail.splitlines()), 40)
        self.assertLessEqual(len(tail.encode("utf-8")), 8192)
        self.assertLessEqual(len(byte_limited_tail.encode("utf-8")), 8192)
        self.assertIn("line-44", tail)
        self.assertNotIn("line-0", tail)

    def test_run_stage_maps_skeleton_expected_output_missing_with_stdout_tail_to_diagnostic_error(self) -> None:
        context = self._linux_arm64_blender_context()
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")
        stage = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=113,
        )[1]
        result_path = blender_bridge.result_path_for_run_dir(self.run_dir)

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            result_path.write_text(
                json.dumps(
                    blender_bridge.build_stage_failed_result(
                        stage="skeleton",
                        error_code="expected-output-missing",
                        message="Expected Blender stage outputs were not created on disk.",
                        stdout_tail=["all input arrays must have the same shape"],
                    )
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=["/opt/blender/blender"],
                returncode=0,
                stdout=f"all input arrays must have the same shape\n{blender_bridge.build_result_marker_line(result_path.resolve())}\n",
                stderr="",
            )

        with mock.patch("unirig_ext.pipeline.subprocess.run", side_effect=fake_run):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        message = pipeline.public_error_message(ctx.exception)
        self.assertIn("stage=skeleton", message)
        self.assertIn("error_code=expected-output-missing", message)
        self.assertIn("expected_output=", message)
        self.assertIn("result_json=", message)
        self.assertIn("stage_log=", message)
        self.assertIn("stdout_tail=all input arrays must have the same shape", message)
        self.assertIn("blender_returncode=0", message)

    def test_run_stage_diagnostic_survives_missing_marker_result_and_log_context(self) -> None:
        context, stage, _prepared_path = self._merge_blender_stage(seed=115)

        with mock.patch(
            "unirig_ext.pipeline.subprocess.run",
            return_value=subprocess.CompletedProcess(args=["/opt/blender/blender"], returncode=0, stdout="boot only\n", stderr=""),
        ):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        message = pipeline.public_error_message(ctx.exception)
        self.assertIn("stage=merge", message)
        self.assertIn("error_code=marker-missing", message)
        self.assertIn("result_json=<unavailable>", message)
        self.assertIn("stage_log=", message)
        self.assertIn("stdout_tail=boot only", message)
        self.assertIn("stderr_tail=(not captured)", message)
        self.assertIn("blender_returncode=0", message)

    def test_run_command_fails_when_process_exits_nonzero_even_if_output_exists(self) -> None:
        output_path = self.temp_dir / "stage-output.bin"
        output_path.write_bytes(b"partial")
        result = subprocess.CompletedProcess(args=["fake"], returncode=7, stdout="", stderr="boom")

        with mock.patch("unirig_ext.pipeline.subprocess.run", return_value=result):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_command(
                    ["fake"],
                    cwd=self.temp_dir,
                    context=self.context,
                    success_path=output_path,
                    run_dir=self.run_dir,
                    stage_name="skin",
                )

        log_path = self._stage_log_path("skin")
        self.assertTrue(log_path.exists())
        log_text = log_path.read_text(encoding="utf-8")
        self.assertIn("returncode: 7", log_text)
        self.assertIn("=== stderr ===", log_text)
        self.assertIn("boom", log_text)
        self.assertIn("exit code 7", str(ctx.exception))
        self.assertIn(str(output_path), str(ctx.exception))
        self.assertIn(str(log_path), str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))

    def test_run_command_tolerates_windows_native_access_violation_for_extract_prepare_when_output_exists(self) -> None:
        output_path = self.temp_dir / "skeleton_npz" / "input" / pipeline.SKIN_DATA_NAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ready")
        result = subprocess.CompletedProcess(
            args=["fake"],
            returncode=-1073741819,
            stdout=f"save to: {output_path}\n",
            stderr="native crash after write",
        )

        with mock.patch("unirig_ext.pipeline.subprocess.run", return_value=result):
            pipeline._run_command(
                ["fake"],
                cwd=self.temp_dir,
                context=self._context(host_os="windows"),
                success_path=output_path,
                run_dir=self.run_dir,
                stage_name="extract-prepare",
                log_stage_name="extract-prepare",
            )

        log_path = self._stage_log_path("extract-prepare")
        self.assertTrue(log_path.exists())
        log_text = log_path.read_text(encoding="utf-8")
        self.assertIn("native crash after write", log_text)
        self.assertIn(f"save to: {output_path}", log_text)
        self.assertIn("tolerated_windows_native_access_violation: true", log_text)

    def test_run_command_still_fails_for_windows_native_access_violation_without_output(self) -> None:
        output_path = self.temp_dir / "skeleton_npz" / "missing" / pipeline.SKIN_DATA_NAME
        result = subprocess.CompletedProcess(args=["fake"], returncode=3221225477, stdout="", stderr="native crash before write")

        with mock.patch("unirig_ext.pipeline.subprocess.run", return_value=result):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_command(
                    ["fake"],
                    cwd=self.temp_dir,
                    context=self._context(host_os="windows"),
                    success_path=output_path,
                    run_dir=self.run_dir,
                    stage_name="extract-prepare",
                    log_stage_name="extract-prepare",
                )

        log_path = self._stage_log_path("extract-prepare")
        self.assertTrue(log_path.exists())
        self.assertIn("3221225477", str(ctx.exception))
        self.assertIn(str(output_path), str(ctx.exception))
        self.assertIn(str(log_path), str(ctx.exception))
        self.assertIn("tolerated_windows_native_access_violation: false", log_path.read_text(encoding="utf-8"))

    def test_run_command_uses_windows_runtime_dll_path_shim(self) -> None:
        context = self._context(host_os="windows")
        torch_lib = context.venv_dir / "Lib" / "site-packages" / "torch" / "lib"
        cublas_bin = context.venv_dir / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
        for path in (torch_lib, cublas_bin):
            path.mkdir(parents=True, exist_ok=True)

        output_path = self.temp_dir / "stage-output.bin"
        output_path.write_bytes(b"ready")
        result = subprocess.CompletedProcess(args=["fake"], returncode=0, stdout="ok", stderr="")

        with mock.patch("unirig_ext.bootstrap.os.name", "nt"), mock.patch(
            "unirig_ext.pipeline.subprocess.run", return_value=result
        ) as run_mock:
            pipeline._run_command(
                ["fake"],
                cwd=self.temp_dir,
                context=context,
                success_path=output_path,
                run_dir=self.run_dir,
                stage_name="skin",
            )

        env = run_mock.call_args.kwargs["env"]
        expected_dll_paths = bootstrap.windows_runtime_dll_search_paths(context.venv_dir)
        self.assertEqual(env["PATH"].split(os.pathsep)[: len(expected_dll_paths)], [str(path) for path in expected_dll_paths])
        self.assertEqual(env["PYTHONPATH"], bootstrap.runtime_pythonpath(context))

    def test_run_command_uses_distinct_extract_log_stage_names(self) -> None:
        output_path = self.temp_dir / "skeleton_npz" / "input" / pipeline.SKIN_DATA_NAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"npz")
        result = subprocess.CompletedProcess(args=["fake"], returncode=0, stdout="extract ok", stderr="")

        with mock.patch("unirig_ext.pipeline.subprocess.run", return_value=result):
            pipeline._run_command(
                ["fake"],
                cwd=self.temp_dir,
                context=self.context,
                success_path=output_path,
                run_dir=self.run_dir,
                stage_name="extract",
                log_stage_name="extract-prepare",
            )

        log_path = self._stage_log_path("extract-prepare")
        self.assertTrue(log_path.exists())
        self.assertIn("extract ok", log_path.read_text(encoding="utf-8"))

    def test_run_command_wraps_launch_oserror_with_stage_log(self) -> None:
        output_path = self.temp_dir / "missing-output.bin"

        with mock.patch("unirig_ext.pipeline.subprocess.run", side_effect=FileNotFoundError("missing python")):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_command(
                    ["python", "run.py"],
                    cwd=self.temp_dir,
                    context=self.context,
                    success_path=output_path,
                    run_dir=self.run_dir,
                    stage_name="skeleton",
                )

        log_path = self._stage_log_path("skeleton")
        self.assertTrue(log_path.exists())
        self.assertIn("could not start", str(ctx.exception))
        self.assertIn(str(log_path), str(ctx.exception))
        self.assertIn("missing python", log_path.read_text(encoding="utf-8"))

    def test_build_execution_plan_returns_deterministic_upstream_stage_specs(self) -> None:
        mesh_path = self.temp_dir / "mesh.glb"
        mesh_path.write_bytes(b"glb")
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=mesh_path,
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=self.context,
            seed=17,
        )

        self.assertEqual([stage.name for stage in plan], ["extract-prepare", "skeleton", "extract-skin", "skin", "merge"])
        self.assertEqual(plan[0].command[0:3], [str(self.context.venv_python), "-m", "src.data.extract"])
        self.assertIn(f"--config={pipeline.EXTRACT_CONFIG}", plan[0].command)
        self.assertIn(f"--output_dir={self.run_dir / 'skeleton_npz'}", plan[0].command)
        self.assertEqual(plan[1].success_path, self.run_dir / "skeleton_stage.fbx")
        self.assertIn(f"--task={pipeline.SKELETON_TASK}", plan[1].command)
        self.assertEqual(plan[2].success_path, self.run_dir / "skin_npz" / plan[2].runtime_input_path.stem / pipeline.SKIN_DATA_NAME)
        self.assertEqual(plan[3].success_path, self.run_dir / "skin_stage.fbx")
        self.assertIn(f"--data_name={pipeline.SKIN_DATA_NAME}", plan[3].command)
        self.assertEqual(plan[4].success_path, self.run_dir / "merged.glb")
        self.assertIn(f"--target={prepared_path}", plan[4].command)

    def test_build_execution_plan_uses_runtime_staging_names_for_upstream_inputs(self) -> None:
        prepared_path = self.run_dir / "prepared.obj"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.obj",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=self.context,
            seed=23,
        )

        self.assertEqual(plan[0].runtime_input_name, ".modly_stage_input_run-processor.obj")
        self.assertEqual(plan[2].runtime_input_name, ".modly_stage_skeleton_run-processor.fbx")
        self.assertIn("--time=modly_extract_run-processor", plan[0].command)
        self.assertIn("--time=modly_extract_run-processor", plan[2].command)
        self.assertTrue(plan[0].runtime_input_path.is_relative_to(self.context.unirig_dir))
        self.assertTrue(plan[2].runtime_input_path.is_relative_to(self.context.unirig_dir))

    def test_build_execution_plan_forces_extract_prepare_override_on_linux_arm64_only(self) -> None:
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        linux_arm64_context = self._context_with_host(
            host_os="linux",
            host_arch="aarch64",
            source_build={"status": "ready", "mode": "staged-source-build", "dependencies": {}},
        )
        windows_context = self._context_with_host(
            host_os="windows",
            host_arch="AMD64",
            source_build={"status": "ready", "mode": "prebuilt", "dependencies": {}},
        )

        linux_plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=linux_arm64_context,
            seed=37,
        )
        windows_plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=windows_context,
            seed=41,
        )

        self.assertIn("--force_override=true", linux_plan[0].command)
        self.assertIn("--force_override=false", windows_plan[0].command)

    def test_wrapper_owned_stage_command_uses_context_venv_python_for_extract_and_merge(self) -> None:
        extract_command = pipeline._wrapper_owned_stage_command(self.context, "-m", "src.data.extract")
        merge_command = pipeline._wrapper_owned_stage_command(self.context, "-m", "src.inference.merge")

        self.assertEqual(extract_command, [str(self.context.venv_python), "-m", "src.data.extract"])
        self.assertEqual(merge_command, [str(self.context.venv_python), "-m", "src.inference.merge"])

    def test_build_execution_plan_keeps_extract_and_merge_on_wrapper_owned_boundary(self) -> None:
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=self.context,
            seed=29,
        )

        extract_prepare = plan[0]
        extract_skin = plan[2]
        merge_stage = plan[4]

        self.assertEqual(extract_prepare.runtime_boundary_owner, pipeline.WRAPPER_RUNTIME_BOUNDARY_OWNER)
        self.assertEqual(extract_skin.runtime_boundary_owner, pipeline.WRAPPER_RUNTIME_BOUNDARY_OWNER)
        self.assertEqual(merge_stage.runtime_boundary_owner, pipeline.WRAPPER_RUNTIME_BOUNDARY_OWNER)
        self.assertEqual(extract_prepare.command[0], str(self.context.venv_python))
        self.assertEqual(extract_skin.command[0], str(self.context.venv_python))
        self.assertEqual(merge_stage.command[0], str(self.context.venv_python))
        self.assertNotIn("blender", extract_prepare.command[0].lower())
        self.assertNotIn("blender", extract_skin.command[0].lower())
        self.assertNotIn("blender", merge_stage.command[0].lower())

    def test_build_execution_plan_keeps_default_path_on_context_venv_python(self) -> None:
        custom_python = self.temp_dir / "custom-venv" / "bin" / "python"
        context = self._context_with_host(
            host_os="linux",
            host_arch="x86_64",
            source_build={"status": "ready", "mode": "prebuilt", "dependencies": {}},
        )
        context = RuntimeContext(**{**context.__dict__, "venv_python": custom_python})
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=31,
        )

        self.assertEqual(plan[0].command[0], str(custom_python))
        self.assertEqual(plan[2].command[0], str(custom_python))
        self.assertEqual(plan[4].command[0], str(custom_python))

    def test_build_execution_plan_never_selects_blender_subprocess_on_windows_x86_64(self) -> None:
        context = self._context_with_host(
            host_os="windows",
            host_arch="AMD64",
            source_build={
                "status": "ready",
                "mode": "prebuilt",
                "dependencies": {},
                "executable_boundary": {"extract_merge": {"enabled": True, "ready": True, "mode": "blender-subprocess"}},
            },
        )
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=37,
        )

        for stage in plan:
            self.assertEqual(stage.runtime_boundary_owner, pipeline.WRAPPER_RUNTIME_BOUNDARY_OWNER)
            self.assertEqual(stage.command[0], str(context.venv_python))

    def test_build_execution_plan_keeps_windows_x86_64_on_pinned_prebuilt_path_even_with_blender_candidate_metadata(self) -> None:
        context = self._context_with_host(
            host_os="windows",
            host_arch="AMD64",
            source_build={
                "status": "ready",
                "mode": "prebuilt",
                "dependencies": {},
                "external_blender": {
                    "candidate": {"source": "path", "path": "C:/Program Files/Blender/blender.exe"},
                    "classification": {"status": "external-bpy-smoke-ready", "ready": False},
                },
                "executable_boundary": {
                    "extract_merge": {
                        "enabled": True,
                        "ready": True,
                        "status": "verified",
                        "proof_kind": blender_bridge.BLENDER_SUBPROCESS_MODE,
                        "optional_owner": blender_bridge.BLENDER_SUBPROCESS_MODE,
                        "supported_stages": ["extract-prepare", "extract-skin", "merge"],
                        "candidate": {"source": "path", "path": "C:/Program Files/Blender/blender.exe"},
                    }
                },
            },
        )
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=38,
        )

        for stage in plan:
            self.assertEqual(stage.runtime_boundary_owner, pipeline.WRAPPER_RUNTIME_BOUNDARY_OWNER)
            self.assertEqual(stage.command[0], str(context.venv_python))

    def test_build_execution_plan_selects_blender_subprocess_only_for_scoped_linux_arm64_stages(self) -> None:
        context = self._context_with_host(
            host_os="linux",
            host_arch="aarch64",
            source_build={
                "status": "blocked",
                "mode": "staged-source-build",
                "host_class": "linux-arm64",
                "external_blender": {
                    "classification": {"status": "external-bpy-smoke-ready", "ready": False}
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
            },
        )
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=39,
        )

        self.assertEqual(plan[0].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[1].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[2].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[3].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[4].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)

    def test_build_execution_plan_infers_merge_blender_subprocess_from_recovered_linux_arm64_stage_subset(self) -> None:
        context = self._context_with_host(
            host_os="linux",
            host_arch="aarch64",
            source_build={
                "status": "partial",
                "mode": "staged-source-build",
                "host_class": "linux-arm64",
                "external_blender": {
                    "classification": {"status": "external-bpy-smoke-ready", "ready": False}
                },
                "executable_boundary": {
                    "extract_merge": {
                        "enabled": True,
                        "ready": True,
                        "status": "verified",
                        "proof_kind": "blender-subprocess",
                        "optional_owner": "blender-subprocess",
                        "supported_stages": ["extract-prepare", "skeleton", "extract-skin", "skin"],
                    }
                },
            },
        )
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=40,
        )

        self.assertEqual(plan[4].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)

    def test_build_execution_plan_uses_default_linux_arm64_mode_when_no_qualification_override_is_present(self) -> None:
        context = self._linux_arm64_blender_context()
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=40,
        )

        self.assertEqual(plan[0].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[1].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[2].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[3].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[4].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)

    def test_build_execution_plan_supports_forced_seam_mode_for_linux_arm64_scoped_stages(self) -> None:
        context = self._linux_arm64_qualification_context(mode="seam")
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=42,
        )

        self.assertEqual(plan[0].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[1].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[2].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[3].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)
        self.assertEqual(plan[4].runtime_boundary_owner, blender_bridge.BLENDER_SUBPROCESS_MODE)

    def test_build_execution_plan_supports_forced_fallback_mode_for_linux_arm64_scoped_stages(self) -> None:
        context = self._linux_arm64_qualification_context(mode="forced-fallback")
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=44,
        )

        for stage in plan:
            self.assertEqual(stage.runtime_boundary_owner, pipeline.WRAPPER_RUNTIME_BOUNDARY_OWNER)
            self.assertEqual(stage.command[0], str(context.venv_python))

    def test_linux_arm64_qualification_mode_rejects_unsupported_hosts(self) -> None:
        context = self._context_with_host(
            host_os="windows",
            host_arch="AMD64",
            source_build={
                "status": "ready",
                "mode": "prebuilt",
                "dependencies": {},
                "qualification": {"extract_merge": {"mode": "seam", "stages": ["extract-prepare"]}},
            },
        )

        with self.assertRaises(pipeline.PipelineError) as ctx:
            pipeline._linux_arm64_blender_subprocess_stage_names(context)

        self.assertIn("linux-aarch64", str(ctx.exception))

    def test_linux_arm64_qualification_mode_rejects_unsupported_stages(self) -> None:
        context = self._linux_arm64_qualification_context(mode="seam", stages=["extract-prepare", "predict-skin"])

        with self.assertRaises(pipeline.PipelineError) as ctx:
            pipeline._linux_arm64_blender_subprocess_stage_names(context)

        self.assertIn("predict-skin", str(ctx.exception))

    def test_build_execution_plan_keeps_linux_arm64_on_wrapper_without_explicit_seam_proof(self) -> None:
        context = self._context_with_host(
            host_os="linux",
            host_arch="aarch64",
            source_build={
                "status": "blocked",
                "mode": "staged-source-build",
                "host_class": "linux-arm64",
                "external_blender": {
                    "classification": {"status": "external-bpy-smoke-ready", "ready": False}
                },
                "executable_boundary": {"extract_merge": {"enabled": True, "ready": False}},
            },
        )
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=41,
        )

        for stage in (plan[0], plan[2], plan[4]):
            self.assertEqual(stage.runtime_boundary_owner, pipeline.WRAPPER_RUNTIME_BOUNDARY_OWNER)
            self.assertEqual(stage.command[0], str(context.venv_python))

    def test_cleanup_staged_files_raises_actionable_error(self) -> None:
        target = self.temp_dir / "staged-input.glb"

        with mock.patch("pathlib.Path.unlink", side_effect=PermissionError("denied")):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._cleanup_staged_files(target)

        self.assertIn("cleanup failed", str(ctx.exception))
        self.assertIn("denied", str(ctx.exception))

    def test_run_stage_executes_blender_subprocess_with_payload_result_and_stage_log(self) -> None:
        context = self._linux_arm64_blender_context()
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")
        source_path = self.run_dir / "skin_stage.fbx"
        source_path.write_bytes(b"skin")
        plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=47,
        )
        stage = plan[4]
        payload_path = blender_bridge.payload_path_for_run_dir(self.run_dir)
        result_path = blender_bridge.result_path_for_run_dir(self.run_dir)

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(command[0], "/opt/blender/blender")
            self.assertEqual(command[1:5], ["--background", "--factory-startup", "--python", str(ROOT / "src" / "unirig_ext" / "blender_bridge.py")])
            self.assertEqual(command[5:7], ["--", str(payload_path)])
            self.assertEqual(
                payload_path.read_text(encoding="utf-8"),
                blender_bridge.render_stage_payload_json(
                    blender_bridge.build_stage_payload(
                        stage="merge",
                        run_dir=self.run_dir,
                        source_path=source_path,
                        target_path=prepared_path,
                        output_path=stage.success_path,
                        require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
                        seed=47,
                    )
                ),
            )
            stage.success_path.write_bytes(b"merged")
            result_path.write_text(
                json.dumps(blender_bridge.build_stage_success_result(stage="merge", produced=[stage.success_path])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=f"boot\n{blender_bridge.build_result_marker_line(result_path.resolve())}\n",
                stderr="",
            )

        with mock.patch("unirig_ext.pipeline.subprocess.run", side_effect=fake_run):
            pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self.assertTrue(payload_path.exists())
        self.assertTrue(stage.success_path.exists())
        log_path = self._stage_log_path("merge")
        self.assertTrue(log_path.exists())
        self.assertIn("UNIRIG_BLENDER_STAGE_RESULT=", log_path.read_text(encoding="utf-8"))

    def test_run_stage_rejects_blender_success_result_when_declared_output_is_missing_on_disk(self) -> None:
        context = self._linux_arm64_blender_context()
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")
        source_path = self.run_dir / "skin_stage.fbx"
        source_path.write_bytes(b"skin")
        stage = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=context,
            seed=53,
        )[4]
        result_path = blender_bridge.result_path_for_run_dir(self.run_dir)

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            result_path.write_text(
                json.dumps(blender_bridge.build_stage_success_result(stage="merge", produced=[stage.success_path])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=["/opt/blender/blender"],
                returncode=0,
                stdout=f"{blender_bridge.build_result_marker_line(result_path.resolve())}\n",
                stderr="",
            )

        with mock.patch("unirig_ext.pipeline.subprocess.run", side_effect=fake_run):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self.assertIn(str(stage.success_path), str(ctx.exception))
        self.assertIn(str(self._stage_log_path("merge")), str(ctx.exception))

    def test_run_stage_keeps_wrapper_owned_execution_for_non_selected_hosts_and_stages(self) -> None:
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")
        stage = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir,
            context=self.context,
            seed=59,
        )[4]

        with mock.patch("unirig_ext.pipeline._run_command") as run_command_mock:
            pipeline._run_stage(stage, context=self.context, run_dir=self.run_dir)

        run_command_mock.assert_called_once_with(
            stage.command,
            cwd=stage.cwd,
            context=self.context,
            success_path=stage.success_path,
            run_dir=self.run_dir,
            stage_name=stage.name,
            log_stage_name=stage.log_stage_name,
        )

    def test_run_stage_maps_blender_launch_failure_to_explicit_pipeline_error(self) -> None:
        context, stage, _prepared_path = self._merge_blender_stage(seed=61)

        with mock.patch("unirig_ext.pipeline.subprocess.run", side_effect=FileNotFoundError("missing blender")):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self._assert_blender_stage_public_error_contract(ctx.exception, stage_name="merge", error_code="launch-failed")
        self.assertIn(str(self._stage_log_path("merge")), str(ctx.exception))

    def test_run_stage_maps_blender_timeout_to_explicit_pipeline_error(self) -> None:
        context, stage, _prepared_path = self._merge_blender_stage(seed=67)

        with mock.patch(
            "unirig_ext.pipeline.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["/opt/blender/blender"], timeout=123),
        ):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self._assert_blender_stage_public_error_contract(ctx.exception, stage_name="merge", error_code="timed-out")
        self.assertIn("123", str(ctx.exception))

    def test_run_stage_maps_missing_blender_marker_to_explicit_pipeline_error(self) -> None:
        context, stage, _prepared_path = self._merge_blender_stage(seed=71)

        with mock.patch(
            "unirig_ext.pipeline.subprocess.run",
            return_value=subprocess.CompletedProcess(args=["/opt/blender/blender"], returncode=0, stdout="boot only\n", stderr=""),
        ):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self._assert_blender_stage_public_error_contract(ctx.exception, stage_name="merge", error_code="marker-missing")

    def test_run_stage_maps_missing_blender_result_file_to_explicit_pipeline_error(self) -> None:
        context, stage, _prepared_path = self._merge_blender_stage(seed=73)
        result_path = blender_bridge.result_path_for_run_dir(self.run_dir)

        with mock.patch(
            "unirig_ext.pipeline.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["/opt/blender/blender"],
                returncode=0,
                stdout=f"{blender_bridge.build_result_marker_line(result_path.resolve())}\n",
                stderr="",
            ),
        ):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self._assert_blender_stage_public_error_contract(ctx.exception, stage_name="merge", error_code="result-missing")
        self.assertIn(str(result_path), str(ctx.exception))

    def test_run_stage_maps_invalid_blender_result_to_explicit_pipeline_error(self) -> None:
        context, stage, _prepared_path = self._merge_blender_stage(seed=79)
        result_path = blender_bridge.result_path_for_run_dir(self.run_dir)

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            result_path.write_text("{not-json", encoding="utf-8")
            return subprocess.CompletedProcess(
                args=["/opt/blender/blender"],
                returncode=0,
                stdout=f"{blender_bridge.build_result_marker_line(result_path.resolve())}\n",
                stderr="",
            )

        with mock.patch("unirig_ext.pipeline.subprocess.run", side_effect=fake_run):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self._assert_blender_stage_public_error_contract(ctx.exception, stage_name="merge", error_code="result-invalid")

    def test_run_stage_maps_missing_declared_output_to_explicit_pipeline_error(self) -> None:
        context, stage, _prepared_path = self._merge_blender_stage(seed=83)
        result_path = blender_bridge.result_path_for_run_dir(self.run_dir)

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            result_path.write_text(
                json.dumps(blender_bridge.build_stage_success_result(stage="merge", produced=[stage.success_path])),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=["/opt/blender/blender"],
                returncode=0,
                stdout=f"{blender_bridge.build_result_marker_line(result_path.resolve())}\n",
                stderr="",
            )

        with mock.patch("unirig_ext.pipeline.subprocess.run", side_effect=fake_run):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self._assert_blender_stage_public_error_contract(ctx.exception, stage_name="merge", error_code="expected-output-missing")
        self.assertIn(str(stage.success_path), str(ctx.exception))

    def test_run_stage_maps_stage_failed_result_to_explicit_pipeline_error(self) -> None:
        context, stage, _prepared_path = self._merge_blender_stage(seed=89)
        result_path = blender_bridge.result_path_for_run_dir(self.run_dir)

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            result_path.write_text(
                json.dumps(
                    blender_bridge.build_stage_failed_result(
                        stage="merge",
                        error_code="stage-failed",
                        message="merge script reported failure",
                    )
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=["/opt/blender/blender"],
                returncode=0,
                stdout=f"{blender_bridge.build_result_marker_line(result_path.resolve())}\n",
                stderr="",
            )

        with mock.patch("unirig_ext.pipeline.subprocess.run", side_effect=fake_run):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._run_stage(stage, context=context, run_dir=self.run_dir)

        self._assert_blender_stage_public_error_contract(ctx.exception, stage_name="merge", error_code="stage-failed")
        self.assertIn("merge script reported failure", str(ctx.exception))


class BlenderBridgeTests(unittest.TestCase):
    def test_validate_qualification_failure_code_accepts_bridge_and_comparison_codes(self) -> None:
        self.assertEqual(blender_bridge.validate_qualification_failure_code("launch-failed"), "launch-failed")
        self.assertEqual(blender_bridge.validate_qualification_failure_code("output-mismatch"), "output-mismatch")

    def test_qualification_failure_code_for_bridge_failure_preserves_machine_readable_code(self) -> None:
        self.assertTrue(blender_bridge.BLENDER_FAILURE_CODES.issubset(blender_bridge.QUALIFICATION_FAILURE_CODES))
        self.assertEqual(
            blender_bridge.qualification_failure_code_for_bridge_failure("marker-missing"),
            "marker-missing",
        )

        with self.assertRaises(ValueError) as ctx:
            blender_bridge.validate_qualification_failure_code("totally-unknown")

        self.assertIn("Unsupported Blender qualification failure code", str(ctx.exception))

    def test_render_stage_payload_json_uses_stable_sorted_content(self) -> None:
        payload = blender_bridge.build_stage_payload(
            stage="extract-prepare",
            run_dir=Path("/tmp/run-1"),
            source_path=Path("/tmp/input.glb"),
            output_dir=Path("/tmp/skeleton_npz"),
            output_path=Path("/tmp/skeleton_npz/input/raw_data.npz"),
            require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
            seed=17,
            extract_token="modly_extract_run-1",
        )

        self.assertEqual(
            blender_bridge.render_stage_payload_json(payload),
            "{\n"
            '  "config": {\n'
            '    "extract_token": "modly_extract_run-1",\n'
            f'    "require_suffix": "{pipeline.MERGE_REQUIRE_SUFFIX}",\n'
            '    "seed": 17\n'
            "  },\n"
            '  "input": {\n'
            '    "output": "/tmp/skeleton_npz/input/raw_data.npz",\n'
            '    "output_dir": "/tmp/skeleton_npz",\n'
            '    "source": "/tmp/input.glb",\n'
            '    "target": ""\n'
            "  },\n"
            '  "protocol_version": 1,\n'
            '  "run_dir": "/tmp/run-1",\n'
            '  "stage": "extract-prepare"\n'
            "}\n",
        )

    def test_build_payload_for_extract_prepare_uses_stable_contract_shape(self) -> None:
        payload = blender_bridge.build_stage_payload(
            stage="extract-prepare",
            run_dir=Path("/tmp/run-1"),
            source_path=Path("/tmp/input.glb"),
            output_dir=Path("/tmp/skeleton_npz"),
            output_path=Path("/tmp/skeleton_npz/input/raw_data.npz"),
            require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
            seed=17,
            extract_token="modly_extract_run-1",
        )

        self.assertEqual(payload["protocol_version"], 1)
        self.assertEqual(payload["stage"], "extract-prepare")
        self.assertEqual(payload["run_dir"], "/tmp/run-1")
        self.assertEqual(
            payload["input"],
            {
                "source": "/tmp/input.glb",
                "target": "",
                "output_dir": "/tmp/skeleton_npz",
                "output": "/tmp/skeleton_npz/input/raw_data.npz",
            },
        )
        self.assertEqual(
            payload["config"],
            {
                "require_suffix": pipeline.MERGE_REQUIRE_SUFFIX,
                "seed": 17,
                "extract_token": "modly_extract_run-1",
            },
        )

    def test_build_payload_for_merge_uses_same_stable_contract_shape(self) -> None:
        payload = blender_bridge.build_stage_payload(
            stage="merge",
            run_dir=Path("/tmp/run-2"),
            source_path=Path("/tmp/skin_stage.fbx"),
            target_path=Path("/tmp/prepared.glb"),
            output_path=Path("/tmp/merged.glb"),
            require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
            seed=23,
        )

        self.assertEqual(payload["protocol_version"], 1)
        self.assertEqual(payload["stage"], "merge")
        self.assertEqual(
            payload["input"],
            {
                "source": "/tmp/skin_stage.fbx",
                "target": "/tmp/prepared.glb",
                "output_dir": "",
                "output": "/tmp/merged.glb",
            },
        )
        self.assertEqual(
            payload["config"],
            {
                "require_suffix": pipeline.MERGE_REQUIRE_SUFFIX,
                "seed": 23,
                "extract_token": "",
            },
        )

    def test_build_payload_for_skeleton_uses_same_stable_contract_shape(self) -> None:
        payload = blender_bridge.build_stage_payload(
            stage="skeleton",
            run_dir=Path("/tmp/run-2"),
            source_path=Path("/tmp/prepared.glb"),
            output_dir=Path("/tmp/skeleton_npz"),
            output_path=Path("/tmp/skeleton_stage.fbx"),
            seed=29,
        )

        self.assertEqual(payload["protocol_version"], 1)
        self.assertEqual(payload["stage"], "skeleton")
        self.assertEqual(
            payload["input"],
            {
                "source": "/tmp/prepared.glb",
                "target": "",
                "output_dir": "/tmp/skeleton_npz",
                "output": "/tmp/skeleton_stage.fbx",
            },
        )

    def test_build_payload_for_skin_uses_same_stable_contract_shape(self) -> None:
        payload = blender_bridge.build_stage_payload(
            stage="skin",
            run_dir=Path("/tmp/run-3"),
            source_path=Path("/tmp/skeleton_stage.fbx"),
            output_dir=Path("/tmp/skin_npz"),
            output_path=Path("/tmp/skin_stage.fbx"),
            seed=31,
        )

        self.assertEqual(payload["protocol_version"], 1)
        self.assertEqual(payload["stage"], "skin")
        self.assertEqual(
            payload["input"],
            {
                "source": "/tmp/skeleton_stage.fbx",
                "target": "",
                "output_dir": "/tmp/skin_npz",
                "output": "/tmp/skin_stage.fbx",
            },
        )
        self.assertEqual(
            payload["config"],
            {
                "require_suffix": "",
                "seed": 31,
                "extract_token": "",
            },
        )
    def test_parse_result_marker_returns_last_absolute_marker_path(self) -> None:
        marker_path = blender_bridge.parse_result_marker(
            "boot line\n"
            "UNIRIG_BLENDER_STAGE_RESULT=/tmp/ignored.json\n"
            "other log\n"
            "UNIRIG_BLENDER_STAGE_RESULT=/tmp/final-result.json\n"
        )

        self.assertEqual(marker_path, Path("/tmp/final-result.json"))

    def test_parse_result_marker_rejects_relative_marker_path(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            blender_bridge.parse_result_marker("UNIRIG_BLENDER_STAGE_RESULT=result.json\n")

        self.assertIn("absolute", str(ctx.exception))

    def test_load_stage_result_rejects_invalid_result_payload(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            blender_bridge.load_stage_result(
                {
                    "protocol_version": 1,
                    "stage": "merge",
                    "status": "ok",
                    "produced": "/tmp/merged.glb",
                    "error_code": "",
                    "message": "",
                    "stdout_tail": [],
                    "stderr_tail": [],
                },
                expected_stage="merge",
            )

        self.assertIn("produced", str(ctx.exception))

    def test_load_stage_result_validates_expected_outputs_from_structured_result(self) -> None:
        merged_output = Path("/tmp/merged.glb")
        result = blender_bridge.load_stage_result(
            blender_bridge.build_stage_success_result(stage="merge", produced=[merged_output]),
            expected_stage="merge",
            expected_outputs=[merged_output],
        )

        self.assertEqual(result["stage"], "merge")
        self.assertEqual(result["status"], blender_bridge.BLENDER_STAGE_STATUS_OK)
        self.assertEqual(result["produced"], [str(merged_output)])

    def test_load_stage_result_rejects_missing_expected_output_without_log_text_parsing(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            blender_bridge.load_stage_result(
                blender_bridge.build_stage_success_result(
                    stage="extract-skin",
                    produced=[Path("/tmp/other-output.fbx")],
                    stdout_tail=["human log that should not matter"],
                ),
                expected_stage="extract-skin",
                expected_outputs=[Path("/tmp/skin_stage.fbx")],
            )

        self.assertIn("expected output", str(ctx.exception).lower())

    def test_actual_extract_output_for_payload_matches_absolute_source_contract(self) -> None:
        payload = blender_bridge.build_stage_payload(
            stage="extract-prepare",
            run_dir=Path("/tmp/run-1"),
            source_path=Path("/tmp/run-1/prepared.glb"),
            output_dir=Path("/tmp/skeleton_npz"),
            output_path=Path("/tmp/skeleton_npz/prepared/raw_data.npz"),
            require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
            seed=17,
            extract_token="modly_extract_run-1",
        )

        self.assertEqual(
            blender_bridge._actual_extract_output_for_payload(payload),
            Path("/tmp/run-1/prepared/raw_data.npz"),
        )

    def test_run_stage_module_copies_extract_output_into_wrapper_expected_location(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="unirig-bridge-extract-contract-"))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        source_path = temp_dir / "prepared.glb"
        source_path.write_bytes(b"mesh")
        expected_output = temp_dir / "skeleton_npz" / "prepared" / "raw_data.npz"
        actual_output = temp_dir / "prepared" / "raw_data.npz"
        payload = blender_bridge.build_stage_payload(
            stage="extract-prepare",
            run_dir=temp_dir,
            source_path=source_path,
            output_dir=temp_dir / "skeleton_npz",
            output_path=expected_output,
            require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
            seed=17,
            extract_token="modly_extract_run-1",
        )

        def fake_extract(*, module_name: str, args: list[str]) -> None:
            self.assertEqual(module_name, "src.data.extract")
            self.assertTrue(any(item.startswith("--input=") for item in args))
            self.assertIn("--force_override=true", args)
            actual_output.parent.mkdir(parents=True, exist_ok=True)
            actual_output.write_bytes(b"npz")

        with mock.patch("unirig_ext.blender_bridge._run_module_in_unirig_runtime", side_effect=fake_extract):
            produced = blender_bridge._run_stage_module(payload)

        self.assertEqual(produced, [expected_output])
        self.assertTrue(actual_output.exists())
        self.assertTrue(expected_output.exists())
        self.assertEqual(expected_output.read_bytes(), b"npz")

    def test_run_stage_module_executes_skeleton_through_blender_runtime_script(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="unirig-bridge-skeleton-contract-"))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        source_path = temp_dir / "prepared.glb"
        source_path.write_bytes(b"mesh")
        output_path = temp_dir / "skeleton_stage.fbx"
        payload = blender_bridge.build_stage_payload(
            stage="skeleton",
            run_dir=temp_dir,
            source_path=source_path,
            output_dir=temp_dir / "skeleton_npz",
            output_path=output_path,
            seed=31,
        )

        def fake_run_script(*, script_name: str, args: list[str]) -> None:
            self.assertEqual(script_name, "run.py")
            self.assertIn("--task=configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml", args)
            self.assertIn(f"--seed={payload['config']['seed']}", args)
            self.assertIn(f"--input={payload['input']['source']}", args)
            self.assertIn(f"--output={payload['input']['output']}", args)
            self.assertIn(f"--npz_dir={payload['input']['output_dir']}", args)
            output_path.write_bytes(b"skeleton")

        with mock.patch("unirig_ext.blender_bridge._run_script_in_unirig_runtime", side_effect=fake_run_script):
            produced = blender_bridge._run_stage_module(payload)

        self.assertEqual(produced, [output_path])
        self.assertTrue(output_path.exists())

    def test_run_stage_module_executes_skin_through_blender_runtime_script(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="unirig-bridge-skin-contract-"))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        source_path = temp_dir / "skeleton_stage.fbx"
        source_path.write_bytes(b"mesh")
        output_path = temp_dir / "skin_stage.fbx"
        payload = blender_bridge.build_stage_payload(
            stage="skin",
            run_dir=temp_dir,
            source_path=source_path,
            output_dir=temp_dir / "skin_npz",
            output_path=output_path,
            seed=37,
        )

        def fake_run_script(*, script_name: str, args: list[str]) -> None:
            self.assertEqual(script_name, "run.py")
            self.assertIn("--task=configs/task/quick_inference_unirig_skin.yaml", args)
            self.assertIn(f"--seed={payload['config']['seed']}", args)
            self.assertIn(f"--input={payload['input']['source']}", args)
            self.assertIn(f"--output={payload['input']['output']}", args)
            self.assertIn(f"--npz_dir={payload['input']['output_dir']}", args)
            self.assertIn("--data_name=raw_data.npz", args)
            output_path.write_bytes(b"skin")

        with mock.patch("unirig_ext.blender_bridge._run_script_in_unirig_runtime", side_effect=fake_run_script):
            produced = blender_bridge._run_stage_module(payload)

        self.assertEqual(produced, [output_path])
        self.assertTrue(output_path.exists())

    def test_run_stage_module_executes_merge_with_optional_open3d_stub_when_missing(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="unirig-bridge-merge-contract-"))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        source_path = temp_dir / "skin_stage.fbx"
        target_path = temp_dir / "prepared.glb"
        output_path = temp_dir / "merged.glb"
        source_path.write_bytes(b"skin")
        target_path.write_bytes(b"prepared")
        payload = blender_bridge.build_stage_payload(
            stage="merge",
            run_dir=temp_dir,
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
            seed=41,
        )

        observed_has_stub: list[bool] = []

        def fake_run_module(*, module_name: str, args: list[str]) -> None:
            self.assertEqual(module_name, "src.inference.merge")
            self.assertIn(f"--source={payload['input']['source']}", args)
            self.assertIn(f"--target={payload['input']['target']}", args)
            self.assertIn(f"--output={payload['input']['output']}", args)
            observed_has_stub.append("open3d" in sys.modules)
            output_path.write_bytes(b"merged")

        with mock.patch("unirig_ext.blender_bridge.importlib.util.find_spec", return_value=None), mock.patch(
            "unirig_ext.blender_bridge._run_module_in_unirig_runtime", side_effect=fake_run_module
        ):
            produced = blender_bridge._run_stage_module(payload)

        self.assertEqual(produced, [output_path])
        self.assertEqual(observed_has_stub, [True])
        self.assertNotIn("open3d", sys.modules)

    def test_build_stage_failed_result_accepts_supported_failure_codes(self) -> None:
        result = blender_bridge.build_stage_failed_result(
            stage="extract-skin",
            error_code="result-invalid",
            message="result payload is malformed",
            stdout_tail=["stdout line"],
            stderr_tail=["stderr line"],
        )

        self.assertEqual(result["protocol_version"], 1)
        self.assertEqual(result["stage"], "extract-skin")
        self.assertEqual(result["status"], "stage-failed")
        self.assertEqual(result["error_code"], "result-invalid")
        self.assertEqual(result["message"], "result payload is malformed")
        self.assertEqual(result["produced"], [])
        self.assertEqual(result["stdout_tail"], ["stdout line"])
        self.assertEqual(result["stderr_tail"], ["stderr line"])

    def test_build_stage_failed_result_rejects_unknown_failure_codes(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            blender_bridge.build_stage_failed_result(
                stage="merge",
                error_code="totally-unknown",
                message="bad code",
            )

        self.assertIn("Unsupported Blender bridge failure code", str(ctx.exception))

    def test_bridge_runtime_sys_path_entries_include_persisted_venv_site_packages(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="unirig-bridge-sys-path-"))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        extension_root = temp_dir / "extension"
        unirig_dir = extension_root / ".unirig-runtime" / "vendor" / "unirig"
        site_packages = extension_root / "venv" / "lib" / "python3.12" / "site-packages"
        dist_packages = extension_root / "venv" / "local" / "lib" / "python3.12" / "dist-packages"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        site_packages.mkdir(parents=True, exist_ok=True)
        dist_packages.mkdir(parents=True, exist_ok=True)
        (extension_root / ".unirig-runtime").mkdir(parents=True, exist_ok=True)
        (extension_root / ".unirig-runtime" / "bootstrap_state.json").write_text(
            json.dumps(
                {
                    "runtime_paths": {
                        "venv_python": str(extension_root / "venv" / "bin" / "python"),
                    }
                }
            ),
            encoding="utf-8",
        )

        with mock.patch("unirig_ext.blender_bridge._bridge_extension_root", return_value=extension_root):
            entries = blender_bridge._bridge_runtime_sys_path_entries(unirig_dir)

        self.assertEqual(entries[0], str(unirig_dir.resolve()))
        self.assertIn(str(site_packages.resolve()), entries)
        self.assertIn(str(dist_packages.resolve()), entries)

    def test_run_entry_in_unirig_runtime_registers_torch_safe_globals_before_entrypoint(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="unirig-bridge-safe-globals-"))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        unirig_dir = temp_dir / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        observed: list[str] = []

        def fake_run(entrypoint: str) -> None:
            observed.append(entrypoint)

        with mock.patch("unirig_ext.blender_bridge._bridge_unirig_dir", return_value=unirig_dir), mock.patch(
            "unirig_ext.blender_bridge._register_torch_safe_globals", side_effect=lambda: observed.append("registered")
        ):
            blender_bridge._run_entry_in_unirig_runtime(entrypoint="run.py", args=["--task=x"], run_callable=fake_run)

        self.assertEqual(observed, ["registered", "run.py"])

    def test_main_executes_merge_stage_and_emits_result_marker(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="unirig-bridge-main-"))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        unirig_dir = temp_dir / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / "merged.glb"
        payload_path = blender_bridge.payload_path_for_run_dir(temp_dir)
        payload_path.write_text(
            blender_bridge.render_stage_payload_json(
                blender_bridge.build_stage_payload(
                    stage="merge",
                    run_dir=temp_dir,
                    source_path=temp_dir / "skin_stage.fbx",
                    target_path=temp_dir / "prepared.glb",
                    output_path=output_path,
                    require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
                    seed=23,
                )
            ),
            encoding="utf-8",
        )

        def fake_run_module(module_name: str, *, run_name: str) -> dict[str, object]:
            self.assertEqual(module_name, "src.inference.merge")
            self.assertEqual(run_name, "__main__")
            self.assertIn(f"--output={output_path}", sys.argv)
            output_path.write_bytes(b"merged")
            print("merge executed")
            return {}

        stdout = text_io.StringIO()
        stderr = text_io.StringIO()
        with mock.patch("unirig_ext.blender_bridge._bridge_unirig_dir", return_value=unirig_dir), mock.patch(
            "unirig_ext.blender_bridge.runpy.run_module", side_effect=fake_run_module
        ), mock.patch("sys.stdout", new=stdout), mock.patch("sys.stderr", new=stderr):
            return_code = blender_bridge.main(["blender_bridge.py", "--", str(payload_path)])

        self.assertEqual(return_code, 0)
        self.assertIn("merge executed", stdout.getvalue())
        result_path = blender_bridge.result_path_for_run_dir(temp_dir)
        self.assertIn(blender_bridge.build_result_marker_line(result_path.resolve()), stdout.getvalue())
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], blender_bridge.BLENDER_STAGE_STATUS_OK)
        self.assertEqual(result["produced"], [str(output_path)])

    def test_main_writes_failed_result_and_marker_when_stage_execution_raises(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="unirig-bridge-fail-"))
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        unirig_dir = temp_dir / "unirig"
        unirig_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / "merged.glb"
        payload_path = blender_bridge.payload_path_for_run_dir(temp_dir)
        payload_path.write_text(
            blender_bridge.render_stage_payload_json(
                blender_bridge.build_stage_payload(
                    stage="merge",
                    run_dir=temp_dir,
                    source_path=temp_dir / "skin_stage.fbx",
                    target_path=temp_dir / "prepared.glb",
                    output_path=output_path,
                    require_suffix=pipeline.MERGE_REQUIRE_SUFFIX,
                    seed=29,
                )
            ),
            encoding="utf-8",
        )

        stdout = text_io.StringIO()
        stderr = text_io.StringIO()
        with mock.patch("unirig_ext.blender_bridge._bridge_unirig_dir", return_value=unirig_dir), mock.patch(
            "unirig_ext.blender_bridge.runpy.run_module", side_effect=RuntimeError("bridge exploded")
        ), mock.patch("sys.stdout", new=stdout), mock.patch("sys.stderr", new=stderr):
            return_code = blender_bridge.main(
                [
                    "/usr/bin/blender",
                    "--background",
                    "--factory-startup",
                    "--python",
                    "blender_bridge.py",
                    "--",
                    str(payload_path),
                ]
            )

        self.assertEqual(return_code, 0)
        result_path = blender_bridge.result_path_for_run_dir(temp_dir)
        self.assertIn(blender_bridge.build_result_marker_line(result_path.resolve()), stdout.getvalue())
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], blender_bridge.BLENDER_STAGE_STATUS_FAILED)
        self.assertEqual(result["error_code"], "stage-failed")
        self.assertIn("bridge exploded", result["message"])


if __name__ == "__main__":
    unittest.main()
