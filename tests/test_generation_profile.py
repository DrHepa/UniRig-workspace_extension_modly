from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext import bootstrap, pipeline
from unirig_ext.bootstrap import RuntimeContext
from unirig_ext.generation_profile import (
    GenerationProfileConfigError,
    GenerationProfileValidationError,
    normalize_generation_profile,
    resolve_generation_profile,
)


def write_vroid_source_config(unirig_dir: Path, *, omit: str | None = None) -> Path:
    config = {
        "task": {"name": "skeleton", "assign_cls": "articulationxl"},
        "system": {"skeleton_prior": "articulationxl"},
        "generate_kwargs": {"cls": "articulationxl"},
        "tokenizer": {"skeleton_order": ["hips", "spine", "head"]},
    }
    if omit == "system.skeleton_prior":
        del config["system"]["skeleton_prior"]
    elif omit is not None:
        config.pop(omit, None)
    path = unirig_dir / pipeline.SKELETON_TASK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


class GenerationProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="unirig-profile-"))
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
            platform_tag="linux-x86_64",
            python_version="3.12.3",
            platform_policy=bootstrap.resolve_platform_policy("linux", "x86_64"),
            source_build={"status": "ready", "mode": "prebuilt", "dependencies": {}},
        )
        self.run_dir = self.temp_dir / ".unirig-runtime" / "runs" / "run-fixed"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_normalize_rejects_mixamo_unknown_and_passthrough_fields(self) -> None:
        self.assertEqual(normalize_generation_profile({}).name, "articulationxl")
        self.assertEqual(normalize_generation_profile({"generation_profile": "vroid"}).status, "experimental")
        for value in ("mixamo", "unknown", 3):
            with self.subTest(value=value):
                with self.assertRaisesRegex(GenerationProfileValidationError, "generation_profile"):
                    normalize_generation_profile({"generation_profile": value})
        with self.assertRaisesRegex(GenerationProfileValidationError, "unsupported.*task"):
            normalize_generation_profile({"task": "configs/task/raw.yaml"})

    def test_articulationxl_resolves_to_existing_skeleton_task_contract(self) -> None:
        normalized = normalize_generation_profile({})

        resolved = resolve_generation_profile(normalized, context=self.context, run_dir=self.run_dir)

        self.assertEqual(resolved.name, "articulationxl")
        self.assertEqual(resolved.status, "stable")
        self.assertEqual(resolved.skeleton_prior, "articulationxl")
        self.assertEqual(resolved.skeleton_task, pipeline.SKELETON_TASK)
        self.assertEqual(resolved.profile_config_source, "upstream_task")
        self.assertIsNone(resolved.generated_config_path)

    def test_vroid_resolves_to_deterministic_run_local_config_without_vendor_mutation(self) -> None:
        vendor_config = write_vroid_source_config(self.context.unirig_dir)
        before = vendor_config.read_bytes()
        first_run = self.run_dir / "first"
        second_run = self.run_dir / "second"
        first_run.mkdir()
        second_run.mkdir()

        first = resolve_generation_profile(normalize_generation_profile({"generation_profile": "vroid"}), context=self.context, run_dir=first_run)
        second = resolve_generation_profile(normalize_generation_profile({"generation_profile": "vroid"}), context=self.context, run_dir=second_run)

        self.assertEqual(vendor_config.read_bytes(), before)
        self.assertEqual(first.status, "experimental")
        self.assertEqual(first.skeleton_prior, "vroid")
        self.assertEqual(first.profile_config_source, "generated_run_config")
        self.assertTrue(first.generated_config_path.is_relative_to(first_run))
        self.assertTrue(second.generated_config_path.is_relative_to(second_run))
        first_text = first.generated_config_path.read_text(encoding="utf-8")
        second_text = second.generated_config_path.read_text(encoding="utf-8")
        self.assertEqual(first_text, second_text)
        self.assertIn('"assign_cls": "vroid"', first_text)
        self.assertIn('"skeleton_prior": "vroid"', first_text)
        self.assertEqual(first.generated_config_sha256, hashlib.sha256(first_text.encode("utf-8")).hexdigest())
        self.assertEqual(first.generated_config_sha256, second.generated_config_sha256)

    def test_pipeline_consumes_profile_only_for_skeleton_task_selection(self) -> None:
        write_vroid_source_config(self.context.unirig_dir)
        prepared_path = self.run_dir / "prepared.glb"
        prepared_path.write_bytes(b"prepared")

        default_plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir / "default",
            context=self.context,
            seed=17,
        )
        vroid_plan = pipeline.build_execution_plan(
            mesh_path=self.temp_dir / "avatar.glb",
            prepared_path=prepared_path,
            run_dir=self.run_dir / "vroid",
            context=self.context,
            seed=17,
            generation_profile=normalize_generation_profile({"generation_profile": "vroid"}),
        )

        self.assertIn(f"--task={pipeline.SKELETON_TASK}", default_plan[1].command)
        vroid_task = next(item for item in vroid_plan[1].command if item.startswith("--task="))
        self.assertTrue(vroid_task.endswith("generation_profiles/vroid_skeleton_task.yaml"))
        self.assertNotEqual(vroid_task, f"--task={pipeline.SKELETON_TASK}")
        self.assertIn(f"--task={pipeline.SKIN_TASK}", vroid_plan[3].command)
        self.assertIn(f"--require_suffix={pipeline.MERGE_REQUIRE_SUFFIX}", vroid_plan[4].command)
        self.assertEqual(default_plan[3].runtime_boundary_owner, vroid_plan[3].runtime_boundary_owner)
        self.assertEqual(default_plan[4].runtime_boundary_owner, vroid_plan[4].runtime_boundary_owner)

    def test_vroid_missing_upstream_seam_fails_with_profile_configuration_error(self) -> None:
        missing = write_vroid_source_config(self.context.unirig_dir, omit="system.skeleton_prior")

        with self.assertRaises(GenerationProfileConfigError) as raised:
            resolve_generation_profile(normalize_generation_profile({"generation_profile": "vroid"}), context=self.context, run_dir=self.run_dir)

        self.assertEqual(raised.exception.profile, "vroid")
        self.assertEqual(raised.exception.key, "system.skeleton_prior")
        self.assertEqual(raised.exception.path, missing)
        self.assertIn("profile-configuration", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
