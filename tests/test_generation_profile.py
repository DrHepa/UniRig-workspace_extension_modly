from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


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


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - production runtime depends on PyYAML; JSON is YAML-compatible fallback.
        text = json.dumps(data, indent=2, sort_keys=True)
    else:
        text = yaml.safe_dump(data, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover
        return json.loads(path.read_text(encoding="utf-8"))
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError(f"expected object YAML at {path}")
    return loaded


def write_vroid_source_config(unirig_dir: Path, *, omit: str | None = None) -> dict[str, Path]:
    paths = {
        "task": unirig_dir / pipeline.SKELETON_TASK,
        "system": unirig_dir / "configs/system/ar_inference_articulationxl.yaml",
        "tokenizer": unirig_dir / "configs/tokenizer/tokenizer_parts_articulationxl_256.yaml",
        "skeleton": unirig_dir / "configs/skeleton/vroid.yaml",
    }
    task_config: dict[str, Any] = {
        "mode": "predict",
        "debug": False,
        "experiment_name": "quick_inference_skeleton_articulationxl_ar_256",
        "resume_from_checkpoint": "experiments/skeleton/articulation-xl_quantization_256/model.ckpt",
        "components": {
            "data": "quick_inference",
            "tokenizer": "tokenizer_parts_articulationxl_256",
            "transform": "inference_ar_transform",
            "model": "unirig_ar_350m_1024_81920_float32",
            "system": "ar_inference_articulationxl",
            "data_name": "raw_data.npz",
        },
        "writer": {"__target__": "ar", "output_dir": None, "add_num": False, "repeat": 1, "export_npz": "predict_skeleton", "export_obj": "skeleton", "export_fbx": "skeleton"},
        "trainer": {"max_epochs": 1, "num_nodes": 1, "devices": 1, "precision": "bf16-mixed", "accelerator": "gpu", "strategy": "auto"},
    }
    system_config: dict[str, Any] = {
        "__target__": "ar",
        "val_interval": 1,
        "generate_kwargs": {
            "max_new_tokens": 2048,
            "num_return_sequences": 1,
            "num_beams": 15,
            "do_sample": True,
            "top_k": 5,
            "top_p": 0.95,
            "repetition_penalty": 3.0,
            "temperature": 1.5,
            "no_cls": False,
            "assign_cls": "articulationxl",
            "use_dir_cls": False,
        },
    }
    tokenizer_config: dict[str, Any] = {
        "method": "tokenizer_part",
        "num_discrete": 256,
        "continuous_range": [-1, 1],
        "cls_token_id": {"vroid": 0, "mixamo": 1, "articulationxl": 2},
        "parts_token_id": {"body": 0, "hand": 1},
        "order_config": {"skeleton_path": {"vroid": "./configs/skeleton/vroid.yaml", "mixamo": "./configs/skeleton/mixamo.yaml"}},
    }
    skeleton_config: dict[str, Any] = {"skeleton": "vroid", "joints": ["hips", "spine", "head"]}

    if omit == "components.system":
        del task_config["components"]["system"]
    elif omit == "components.tokenizer":
        del task_config["components"]["tokenizer"]
    elif omit == "generate_kwargs.assign_cls":
        del system_config["generate_kwargs"]["assign_cls"]
    elif omit == "cls_token_id.vroid":
        del tokenizer_config["cls_token_id"]["vroid"]
    elif omit == "order_config.skeleton_path.vroid":
        del tokenizer_config["order_config"]["skeleton_path"]["vroid"]

    write_yaml(paths["task"], task_config)
    write_yaml(paths["system"], system_config)
    write_yaml(paths["tokenizer"], tokenizer_config)
    if omit != "skeleton_path_exists":
        write_yaml(paths["skeleton"], skeleton_config)
    return paths


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
        vendor_paths = write_vroid_source_config(self.context.unirig_dir)
        before = {name: path.read_bytes() for name, path in vendor_paths.items()}
        first_run = self.run_dir / "first"
        second_run = self.run_dir / "second"
        first_run.mkdir()
        second_run.mkdir()

        first = resolve_generation_profile(normalize_generation_profile({"generation_profile": "vroid"}), context=self.context, run_dir=first_run)
        second = resolve_generation_profile(normalize_generation_profile({"generation_profile": "vroid"}), context=self.context, run_dir=second_run)

        self.assertEqual({name: path.read_bytes() for name, path in vendor_paths.items()}, before)
        self.assertEqual(first.status, "experimental")
        self.assertEqual(first.skeleton_prior, "vroid")
        self.assertEqual(first.profile_config_source, "generated_run_config")
        self.assertTrue(first.generated_config_path.is_relative_to(first_run))
        self.assertTrue(second.generated_config_path.is_relative_to(second_run))
        self.assertEqual(first.generated_config_path.name, "vroid_skeleton_task.yaml")
        generated_system_path = first.generated_config_path.with_name("vroid_ar_inference_articulationxl.yaml")
        generated_tokenizer_path = first.generated_config_path.with_name("vroid_tokenizer_parts_articulationxl_256.yaml")
        self.assertTrue(generated_system_path.is_file())
        self.assertTrue(generated_tokenizer_path.is_file())
        first_task_text = first.generated_config_path.read_text(encoding="utf-8")
        second_task_text = second.generated_config_path.read_text(encoding="utf-8")
        first_system_text = generated_system_path.read_text(encoding="utf-8")
        second_system_text = second.generated_config_path.with_name("vroid_ar_inference_articulationxl.yaml").read_text(encoding="utf-8")
        first_tokenizer_text = generated_tokenizer_path.read_text(encoding="utf-8")
        second_tokenizer_text = second.generated_config_path.with_name("vroid_tokenizer_parts_articulationxl_256.yaml").read_text(encoding="utf-8")
        self.assertEqual(first_system_text, second_system_text)
        self.assertEqual(first_tokenizer_text, second_tokenizer_text)
        generated_task = read_yaml(first.generated_config_path)
        second_generated_task = read_yaml(second.generated_config_path)
        generated_system = read_yaml(generated_system_path)
        generated_tokenizer = read_yaml(generated_tokenizer_path)
        self.assertEqual(generated_system["generate_kwargs"]["assign_cls"], "vroid")
        self.assertNotEqual(generated_task["components"]["system"], "ar_inference_articulationxl")
        self.assertTrue(str(generated_task["components"]["system"]).endswith("generation_profiles/vroid_ar_inference_articulationxl"))
        self.assertNotEqual(generated_task["components"]["tokenizer"], "tokenizer_parts_articulationxl_256")
        self.assertTrue(str(generated_task["components"]["tokenizer"]).endswith("generation_profiles/vroid_tokenizer_parts_articulationxl_256"))
        generated_task_without_system = dict(generated_task)
        second_generated_task_without_system = dict(second_generated_task)
        generated_task_without_system["components"] = dict(generated_task["components"])
        second_generated_task_without_system["components"] = dict(second_generated_task["components"])
        generated_task_without_system["components"].pop("system")
        second_generated_task_without_system["components"].pop("system")
        generated_task_without_system["components"].pop("tokenizer")
        second_generated_task_without_system["components"].pop("tokenizer")
        self.assertEqual(generated_task_without_system, second_generated_task_without_system)
        self.assertEqual(generated_tokenizer["cls_token_id"]["vroid"], 0)
        self.assertNotIn("vroid", generated_tokenizer["order_config"]["skeleton_path"])
        self.assertEqual(generated_tokenizer["order_config"]["skeleton_path"], {"mixamo": "./configs/skeleton/mixamo.yaml"})
        vendor_tokenizer = read_yaml(vendor_paths["tokenizer"])
        self.assertEqual(vendor_tokenizer["order_config"]["skeleton_path"]["vroid"], "./configs/skeleton/vroid.yaml")
        self.assertEqual(first.generated_config_sha256, hashlib.sha256(first_task_text.encode("utf-8")).hexdigest())
        self.assertEqual(second.generated_config_sha256, hashlib.sha256(second_task_text.encode("utf-8")).hexdigest())

    def test_vroid_generated_tokenizer_keeps_class_token_without_exact_name_skeleton_enforcement(self) -> None:
        write_vroid_source_config(self.context.unirig_dir)

        resolved = resolve_generation_profile(normalize_generation_profile({"generation_profile": "vroid"}), context=self.context, run_dir=self.run_dir)

        generated_task = read_yaml(resolved.generated_config_path)
        generated_tokenizer_path = resolved.generated_config_path.with_name("vroid_tokenizer_parts_articulationxl_256.yaml")
        generated_tokenizer = read_yaml(generated_tokenizer_path)
        self.assertTrue(str(generated_task["components"]["tokenizer"]).endswith("generation_profiles/vroid_tokenizer_parts_articulationxl_256"))
        self.assertEqual(generated_tokenizer["cls_token_id"]["vroid"], 0)
        self.assertNotIn("vroid", generated_tokenizer["order_config"]["skeleton_path"])

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

    def test_vroid_missing_upstream_seams_fail_with_explicit_profile_configuration_keys(self) -> None:
        cases = {
            "components.system": "task",
            "components.tokenizer": "task",
            "generate_kwargs.assign_cls": "system",
            "cls_token_id.vroid": "tokenizer",
            "order_config.skeleton_path.vroid": "tokenizer",
            "skeleton_path_exists": "skeleton",
        }
        for omitted, expected_path_name in cases.items():
            with self.subTest(omitted=omitted):
                shutil.rmtree(self.context.unirig_dir, ignore_errors=True)
                paths = write_vroid_source_config(self.context.unirig_dir, omit=omitted)

                with self.assertRaises(GenerationProfileConfigError) as raised:
                    resolve_generation_profile(normalize_generation_profile({"generation_profile": "vroid"}), context=self.context, run_dir=self.run_dir)

                expected_key = "order_config.skeleton_path.vroid" if omitted == "skeleton_path_exists" else omitted
                self.assertEqual(raised.exception.profile, "vroid")
                self.assertEqual(raised.exception.key, expected_key)
                self.assertEqual(raised.exception.path, paths[expected_path_name])
                self.assertIn("profile-configuration", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
