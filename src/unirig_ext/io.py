from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .bootstrap import RuntimeContext, UniRigError, stage_environment


DIRECT_INPUT_SUFFIXES = {".obj", ".fbx", ".glb", ".vrm"}
CONVERTIBLE_INPUT_SUFFIXES = {".gltf", ".stl", ".ply"}
SUPPORTED_SUFFIXES = DIRECT_INPUT_SUFFIXES | CONVERTIBLE_INPUT_SUFFIXES
WORKSPACE_GLB_NORMALIZE_SUFFIXES = {".glb", ".gltf"}


class InputValidationError(UniRigError):
    pass


def validate_mesh_input(mesh_path: Path) -> Path:
    mesh_path = Path(mesh_path).expanduser().resolve()
    if not mesh_path.exists():
        raise InputValidationError(f"Input mesh does not exist: {mesh_path}")
    if not mesh_path.is_file():
        raise InputValidationError(f"Input mesh is not a file: {mesh_path}")
    if mesh_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InputValidationError(
            f"Unsupported mesh format '{mesh_path.suffix.lower()}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    return mesh_path


def create_run_dir(context: RuntimeContext) -> Path:
    base = context.runtime_root / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=base))


def stage_input(mesh_path: Path, run_dir: Path) -> Path:
    staged = run_dir / f"input{mesh_path.suffix.lower()}"
    shutil.copy2(mesh_path, staged)
    return staged


def prepare_input_mesh(mesh_path: Path, run_dir: Path, context: RuntimeContext) -> Path:
    suffix = mesh_path.suffix.lower()
    if _should_normalize_workspace_extract_input(suffix=suffix, context=context):
        return _export_prepared_glb(mesh_path=mesh_path, run_dir=run_dir, context=context)
    if suffix in DIRECT_INPUT_SUFFIXES:
        prepared = run_dir / f"prepared{suffix}"
        shutil.copy2(mesh_path, prepared)
        return prepared
    if suffix not in CONVERTIBLE_INPUT_SUFFIXES:
        raise InputValidationError(f"Unsupported input format: {suffix}")

    return _export_prepared_glb(mesh_path=mesh_path, run_dir=run_dir, context=context)


def _should_normalize_workspace_extract_input(*, suffix: str, context: RuntimeContext) -> bool:
    return _runtime_host_os(context) == "windows" and suffix in WORKSPACE_GLB_NORMALIZE_SUFFIXES


def _runtime_host_os(context: RuntimeContext) -> str:
    host = context.platform_policy.get("host") if isinstance(context.platform_policy, dict) else None
    return str((host or {}).get("os") or "").strip().lower()


def _export_prepared_glb(*, mesh_path: Path, run_dir: Path, context: RuntimeContext) -> Path:
    suffix = mesh_path.suffix.lower()
    if suffix not in CONVERTIBLE_INPUT_SUFFIXES and suffix not in WORKSPACE_GLB_NORMALIZE_SUFFIXES:
        raise InputValidationError(f"Unsupported input format: {suffix}")

    prepared = run_dir / "prepared.glb"
    code = (
        "from pathlib import Path\n"
        "import trimesh\n"
        "source = Path(__import__('sys').argv[1])\n"
        "destination = Path(__import__('sys').argv[2])\n"
        "load_as_scene = source.suffix.lower() in {'.glb', '.gltf'}\n"
        "loaded = trimesh.load(source, force='scene' if load_as_scene else None)\n"
        "scene = loaded.scene() if isinstance(loaded, trimesh.Trimesh) else loaded\n"
        "blob = scene.export(file_type='glb')\n"
        "destination.write_bytes(blob if isinstance(blob, (bytes, bytearray)) else bytes(blob))\n"
    )
    result = subprocess.run(
        [str(context.venv_python), "-c", code, str(mesh_path), str(prepared)],
        capture_output=True,
        text=True,
        check=False,
        cwd=context.unirig_dir,
        env=stage_environment(context=context),
    )
    if result.returncode != 0 or not prepared.exists():
        raise InputValidationError(
            "UniRig failed to prepare a convertible mesh input. "
            f"stderr: {result.stderr.strip() or '(empty)'}"
        )
    return prepared


def derive_output_path(mesh_path: Path) -> Path:
    return mesh_path.with_name(f"{mesh_path.stem}_unirig.glb")


def publish_output(source_path: Path, mesh_path: Path) -> Path:
    output_path = derive_output_path(mesh_path)
    shutil.copy2(source_path, output_path)
    return output_path


def copy_file(source_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    return output_path


def write_json(file_path: Path, payload: dict) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return file_path


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
