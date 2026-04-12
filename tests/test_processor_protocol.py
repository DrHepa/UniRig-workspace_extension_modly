from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PROCESSOR = ROOT / "processor.py"
MANIFEST = ROOT / "manifest.json"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext import bootstrap, pipeline
from unirig_ext.bootstrap import RuntimeContext


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
        self.input_mesh = self.temp_dir / "mesh.glb"
        self.input_mesh.write_bytes(b"glTFstub")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

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
        self.assertNotEqual(output_path.read_bytes(), self.input_mesh.read_bytes())

        sidecar_path = output_path.with_name(f"{output_path.stem}.rigmeta.json")
        self.assertTrue(sidecar_path.exists())
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["output_mesh"], output_path.name)
        self.assertEqual(sidecar["seed"], 11)
        self.assertEqual(sidecar["runtime"]["mode"], "real")

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

    def _stage_log_path(self, stage_name: str) -> Path:
        return self.context.logs_dir / self.run_dir.name / f"{stage_name}.log"

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

    def test_cleanup_staged_files_raises_actionable_error(self) -> None:
        target = self.temp_dir / "staged-input.glb"

        with mock.patch("pathlib.Path.unlink", side_effect=PermissionError("denied")):
            with self.assertRaises(pipeline.PipelineError) as ctx:
                pipeline._cleanup_staged_files(target)

        self.assertIn("cleanup failed", str(ctx.exception))
        self.assertIn("denied", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
