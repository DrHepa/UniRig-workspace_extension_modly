from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PRIVATE_FIELDS = (
    "extension_type",
    "workspace_tool",
    "workspace_tool_class",
    "workspaceToolClass",
    "tool_kind",
    "privateRoute",
    "private_route",
    "uiHook",
    "ui_hook",
)
FORBIDDEN_GENERATION_PROFILE_PASSTHROUGH_PARAMS = (
    "task",
    "class",
    "cls",
    "yaml",
    "config",
    "generation_kwargs",
    "generate_kwargs",
)
REQUIRED_ROOT_ARTIFACTS = (
    "manifest.json",
    "processor.py",
    "setup.py",
    "src/unirig_ext/__init__.py",
    "src/unirig_ext/bootstrap.py",
    "src/unirig_ext/io.py",
    "src/unirig_ext/metadata.py",
    "src/unirig_ext/pipeline.py",
)


def validate_manifest_contract(manifest: dict, root: Path) -> list[str]:
    errors: list[str] = []
    if manifest.get("type") != "process":
        errors.append('manifest.type must be "process"')
    if manifest.get("entry") != "processor.py":
        errors.append('manifest.entry must be "processor.py"')
    if not manifest.get("id"):
        errors.append('manifest.id is required')
    for field in FORBIDDEN_PRIVATE_FIELDS:
        if field in manifest:
            errors.append(f'legacy/private field "{field}" is forbidden')
    if not (root / (manifest.get("entry") or "")).exists():
        errors.append("manifest entry file is missing from repository")
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 1:
        errors.append("manifest.nodes must contain exactly one MVP node")
        return errors
    node = nodes[0]
    if node.get("id") != "rig-mesh":
        errors.append('node.id must be "rig-mesh"')
    if node.get("input") != "mesh" or node.get("output") != "mesh":
        errors.append("node input/output must both be mesh")
    return errors


def validate_install_artifacts(root: Path) -> list[str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return ["manifest.json missing from repository"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = [
        error
        for error in validate_manifest_contract(manifest, root)
        if error != "manifest entry file is missing from repository"
    ]
    entry_name = str(manifest.get("entry") or "")
    entry_file = root / entry_name
    if not entry_file.exists():
        errors.append(f'manifest.json: entry file "{entry_name}" missing from repository')
    for rel_path in REQUIRED_ROOT_ARTIFACTS:
        if not (root / rel_path).exists():
            errors.append(f'required install artifact missing from repository: "{rel_path}"')
    return errors


class ManifestContractTests(unittest.TestCase):
    def test_root_artifacts_exist(self) -> None:
        for rel_path in REQUIRED_ROOT_ARTIFACTS:
            self.assertTrue((ROOT / rel_path).exists(), msg=f"missing required artifact: {rel_path}")

    def test_manifest_matches_process_contract(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_manifest_contract(manifest, ROOT), [])
        composite_id = f"{manifest['id']}/{manifest['nodes'][0]['id']}"
        self.assertEqual(composite_id, "unirig-process-extension/rig-mesh")

    def test_manifest_exposes_safe_generation_profile_enum_only(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        params = manifest["nodes"][0]["params_schema"]
        by_id = {param["id"]: param for param in params}

        generation_profile = by_id["generation_profile"]
        self.assertEqual(generation_profile["type"], "select")
        self.assertEqual(generation_profile["default"], "articulationxl")
        self.assertEqual(
            [option["value"] for option in generation_profile["options"]],
            ["articulationxl", "vroid"],
        )
        self.assertIn("Experimental", by_id["generation_profile"]["options"][1]["label"])
        for forbidden in FORBIDDEN_GENERATION_PROFILE_PASSTHROUGH_PARAMS:
            self.assertNotIn(forbidden, by_id)

    def test_manifest_rejects_private_or_invalid_shape(self) -> None:
        invalid = {
            "id": "bad",
            "extension_type": "workspace_tool",
            "nodes": [{"id": "rig-mesh", "input": "image", "output": "mesh"}],
        }
        errors = validate_manifest_contract(invalid, ROOT)
        self.assertIn('manifest.type must be "process"', errors)
        self.assertIn('manifest.entry must be "processor.py"', errors)
        self.assertIn('legacy/private field "extension_type" is forbidden', errors)

    def test_manifest_rejects_other_private_contract_fields(self) -> None:
        invalid = {
            "id": "bad",
            "type": "process",
            "entry": "processor.py",
            "workspace_tool_class": "LegacyTool",
            "privateRoute": "/legacy",
            "nodes": [{"id": "rig-mesh", "input": "mesh", "output": "mesh"}],
        }
        errors = validate_manifest_contract(invalid, ROOT)
        self.assertIn('legacy/private field "workspace_tool_class" is forbidden', errors)
        self.assertIn('legacy/private field "privateRoute" is forbidden', errors)

    def test_manifest_rejects_snake_case_private_contract_fields(self) -> None:
        invalid = {
            "id": "bad",
            "type": "process",
            "entry": "processor.py",
            "private_route": "/legacy",
            "ui_hook": "legacy-panel",
            "nodes": [{"id": "rig-mesh", "input": "mesh", "output": "mesh"}],
        }
        errors = validate_manifest_contract(invalid, ROOT)
        self.assertIn('legacy/private field "private_route" is forbidden', errors)
        self.assertIn('legacy/private field "ui_hook" is forbidden', errors)

    def test_tarball_install_contract_matches_upstream_shape_checks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-tarball-") as temp_dir:
            tarball = Path(temp_dir) / "repo.tar.gz"
            extract_dir = Path(temp_dir) / "extract"
            with tarfile.open(tarball, "w:gz") as archive:
                for rel_path in REQUIRED_ROOT_ARTIFACTS:
                    archive.add(ROOT / rel_path, arcname=rel_path)
            with tarfile.open(tarball, "r:gz") as archive:
                archive.extractall(extract_dir, filter="data")
            self.assertEqual(validate_install_artifacts(extract_dir), [])

    def test_missing_root_artifact_blocks_install_before_registration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-install-contract-") as temp_dir:
            repo_copy = Path(temp_dir) / "repo"
            shutil.copytree(ROOT, repo_copy, ignore=shutil.ignore_patterns(".git", "venv", ".unirig-runtime", "__pycache__", "*.pyc"))

            manifest_missing = repo_copy / "manifest.json"
            manifest_missing.unlink()
            self.assertEqual(validate_install_artifacts(repo_copy), ["manifest.json missing from repository"])

            shutil.copy2(ROOT / "manifest.json", repo_copy / "manifest.json")
            (repo_copy / "processor.py").unlink()
            errors = validate_install_artifacts(repo_copy)
            self.assertIn('manifest.json: entry file "processor.py" missing from repository', errors)
            self.assertIn('required install artifact missing from repository: "processor.py"', errors)


if __name__ == "__main__":
    unittest.main()
