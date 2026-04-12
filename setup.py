from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext import bootstrap


LOGGER = logging.getLogger("unirig.setup")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYG_INDEX_URL = "https://data.pyg.org/whl/torch-2.7.0+cu128.html"
TORCH_PACKAGES = ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]
PYG_PACKAGES = ["torch_scatter==2.1.2+pt27cu128", "torch_cluster==1.6.3+pt27cu128"]
SPCONV_PACKAGE = "spconv-cu120"
WINDOWS_CUMM_PACKAGE_DEFAULT = "cumm-cu126==0.7.11"
WINDOWS_SPCONV_PACKAGE_DEFAULT = "spconv-cu126==2.3.8"
NUMPY_PIN = "numpy==1.26.4"
FLASH_ATTN_WHEEL_DEFAULT = (
    "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/"
    "flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp311-cp311-win_amd64.whl"
)
TRITON_WINDOWS_PACKAGE_DEFAULT = "triton-windows==3.3.1.post19"
WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH = Path("configs/model/unirig_ar_350m_1024_81920_float32.yaml")
WINDOWS_FLASH_ATTN_CONFIG_SOURCE = "flash_attention_2"
WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT = "eager"
WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH = Path("configs/model/unirig_skin.yaml")
WINDOWS_RUNTIME_SMOKE_CHECKS = (
    {
        "label": "flash_attn",
        "code": 'import importlib; importlib.import_module("flash_attn")',
        "repair_hint": (
            "The pinned flash_attn wheel may not match the selected Windows interpreter. "
            "Use the validated Python 3.11 path or override MODLY_UNIRIG_FLASH_ATTN_WHEEL_URL with a compatible pinned wheel."
        ),
    },
    {
        "label": "triton",
        "code": 'import importlib; importlib.import_module("triton")',
        "repair_hint": (
            "Install the validated Windows Triton runtime or override MODLY_UNIRIG_TRITON_PACKAGE with a compatible pinned package."
        ),
    },
    {
        "label": "flash_attn.layers.rotary",
        "code": "from flash_attn.layers.rotary import apply_rotary_emb",
        "repair_hint": (
            "The flash_attn wheel imported at the top level but failed on a runtime submodule UniRig uses. "
            "Replace the wheel with a Windows build compatible with the selected Python/Torch/CUDA stack."
        ),
    },
    {
        "label": "cumm.core_cc",
        "code": 'import importlib; importlib.import_module("cumm.core_cc")',
        "repair_hint": (
            "The pinned Windows cumm runtime did not load its native extension DLLs. "
            "Reinstall the validated Windows prebuilt pair cumm-cu126==0.7.11 and spconv-cu126==2.3.8 "
            "for torch==2.7.0/cu128, then rerun setup."
        ),
    },
    {
        "label": "spconv.pytorch",
        "code": 'import importlib; importlib.import_module("spconv.pytorch")',
        "repair_hint": (
            "The pinned Windows spconv runtime is present but still not importable. "
            "Verify the validated prebuilt pair cumm-cu126==0.7.11 and spconv-cu126==2.3.8 is installed "
            "alongside torch==2.7.0/cu128, then rerun setup."
        ),
    },
)


def _log(message: str) -> None:
    LOGGER.info(message)


def _load_payload(argv: list[str]) -> dict:
    if len(argv) < 2:
        return {}
    raw = argv[1].strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"setup.py expected a JSON object argument: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("setup.py expected a JSON object argument.")
    return data


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _runtime_root(ext_dir: Path) -> Path:
    return ext_dir / ".unirig-runtime"


def _runtime_vendor_dir(ext_dir: Path) -> Path:
    return _runtime_root(ext_dir) / "vendor"


def _runtime_unirig_dir(ext_dir: Path) -> Path:
    return _runtime_vendor_dir(ext_dir) / "unirig"


def _runtime_stage_manifest_path(ext_dir: Path) -> Path:
    return _runtime_vendor_dir(ext_dir) / ".stage-source.json"


def _runtime_stage_patch_report_path(ext_dir: Path) -> Path:
    return _runtime_root(ext_dir) / "logs" / "runtime-stage-patches.json"


def _is_windows_host() -> bool:
    return os.name == "nt"


def _runtime_stage_descriptor(source: str, source_ref: str) -> dict[str, str]:
    source_path = Path(source).expanduser()
    if source_path.exists() and source_path.is_dir():
        return {"vendor_source": "local-directory", "source": str(source_path.resolve()), "source_ref": source_ref}
    if source_path.exists() and source_path.is_file():
        return {"vendor_source": "local-archive", "source": str(source_path.resolve()), "source_ref": source_ref}
    return {"vendor_source": "archive", "source": source, "source_ref": source_ref}


def _read_runtime_stage_descriptor(ext_dir: Path) -> dict[str, str]:
    manifest_path = _runtime_stage_manifest_path(ext_dir)
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Runtime stage descriptor is corrupt: {manifest_path}. "
            "Delete the staged runtime or rerun setup so UniRig can restage deterministically. "
            f"Details: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Runtime stage descriptor must be a JSON object: {manifest_path}. "
            "Delete the staged runtime or rerun setup so UniRig can restage deterministically."
        )
    return {str(key): str(value) for key, value in data.items()}


def _write_runtime_stage_descriptor(ext_dir: Path, descriptor: dict[str, str]) -> None:
    manifest_path = _runtime_stage_manifest_path(ext_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8")


def _runtime_stage_matches(ext_dir: Path, descriptor: dict[str, str]) -> bool:
    if _read_runtime_stage_descriptor(ext_dir) != descriptor:
        return False
    unirig_dir = _runtime_unirig_dir(ext_dir)
    return all((unirig_dir / rel).exists() for rel in bootstrap.REQUIRED_RUNTIME_PATHS)


def _is_commit_sha(ref: str) -> bool:
    value = str(ref or "").strip()
    return len(value) == 40 and all(char in "0123456789abcdefABCDEF" for char in value)


def _archive_url_for_ref(ref: str) -> str:
    ref = (ref or bootstrap.UPSTREAM_REF_DEFAULT).strip()
    if not _is_commit_sha(ref):
        raise RuntimeError(
            "UniRig bootstrap requires an immutable upstream commit ref. "
            f"Received '{ref or '<empty>'}'. Provide a pinned commit SHA via 'unirig_ref' or MODLY_UNIRIG_REPO_REF."
        )
    return f"https://github.com/{bootstrap.UPSTREAM_REPO}/archive/{ref}.zip"


def _resolve_source(payload: dict) -> tuple[str, str]:
    source_dir = str(payload.get("unirig_source_dir") or os.environ.get("MODLY_UNIRIG_SOURCE_DIR", "")).strip()
    if source_dir:
        if not Path(source_dir).expanduser().exists():
            raise RuntimeError(
                f"Configured UniRig source directory does not exist: {source_dir}. "
                "Provide a valid local directory or remove the local-source override."
            )
        return source_dir, "directory"
    source_zip = str(payload.get("unirig_source_zip") or os.environ.get("MODLY_UNIRIG_SOURCE_ZIP", "")).strip()
    if source_zip:
        if not Path(source_zip).expanduser().exists():
            raise RuntimeError(
                f"Configured UniRig source archive does not exist: {source_zip}. "
                "Provide a valid local archive or remove the local-source override."
            )
        return source_zip, "archive"
    ref = str(payload.get("unirig_ref") or os.environ.get("MODLY_UNIRIG_REPO_REF", bootstrap.UPSTREAM_REF_DEFAULT)).strip() or bootstrap.UPSTREAM_REF_DEFAULT
    return _archive_url_for_ref(ref), ref


def _copy_upstream_tree(source_root: Path, dest_unirig_dir: Path) -> None:
    missing = []
    for rel in bootstrap.REQUIRED_RUNTIME_PATHS:
        src = source_root / rel
        dst = dest_unirig_dir / rel
        if not src.exists():
            missing.append(rel)
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    if missing:
        raise RuntimeError("Upstream UniRig source is missing required paths: " + ", ".join(missing) + f". Source root: {source_root}")


def _locate_upstream_root(unpack_dir: Path) -> Path:
    roots = [path for path in unpack_dir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Unexpected archive layout in {unpack_dir}: {[path.name for path in roots]}")
    root = roots[0]
    if not (root / "run.py").exists() and (root / "UniRig" / "run.py").exists():
        root = root / "UniRig"
    return root


def _prepare_runtime_source(ext_dir: Path, payload: dict) -> tuple[Path, str, str]:
    vendor_dir = _runtime_vendor_dir(ext_dir)
    unirig_dir = _runtime_unirig_dir(ext_dir)
    source, source_ref = _resolve_source(payload)
    stage_descriptor = _runtime_stage_descriptor(source, source_ref)
    if _runtime_stage_matches(ext_dir, stage_descriptor):
        _apply_windows_runtime_stage_patches(ext_dir, unirig_dir)
        return unirig_dir, stage_descriptor["vendor_source"], source_ref

    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    unirig_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source).expanduser()
    if source_path.exists() and source_path.is_dir():
        _copy_upstream_tree(source_path, unirig_dir)
        _apply_windows_runtime_stage_patches(ext_dir, unirig_dir)
        _write_runtime_stage_descriptor(ext_dir, stage_descriptor)
        return unirig_dir, stage_descriptor["vendor_source"], source_ref

    with tempfile.TemporaryDirectory(prefix="unirig-source-") as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / "unirig.zip"
        if source_path.exists() and source_path.is_file():
            shutil.copy2(source_path, archive_path)
        else:
            urllib.request.urlretrieve(source, str(archive_path))
        unpack_dir = temp_root / "unpack"
        shutil.unpack_archive(str(archive_path), str(unpack_dir))
        upstream_root = _locate_upstream_root(unpack_dir)
        _copy_upstream_tree(upstream_root, unirig_dir)
    _apply_windows_runtime_stage_patches(ext_dir, unirig_dir)
    _write_runtime_stage_descriptor(ext_dir, stage_descriptor)
    return unirig_dir, stage_descriptor["vendor_source"], source_ref


def _apply_windows_runtime_stage_patches(ext_dir: Path, unirig_dir: Path) -> list[dict[str, str]]:
    if not _is_windows_host():
        return []

    report = [_patch_windows_float32_attention_config(unirig_dir), _patch_windows_skin_enable_flash(unirig_dir)]
    _write_runtime_stage_patch_report(ext_dir, report)
    return report


def _patch_windows_float32_attention_config(unirig_dir: Path) -> dict[str, str]:
    config_path = unirig_dir / WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH
    entry = {
        "patch": "windows-disable-flash-attention-2-float32",
        "path": str(WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH),
        "status": "skipped",
        "reason": (
            "Windows runtime stages the upstream float32 skeleton config through Hugging Face. "
            "flash_attention_2 is incompatible with the observed Windows host evidence, so the wrapper forces eager attention only for this staged config."
        ),
    }
    if not config_path.exists():
        entry["status"] = "target-missing"
        _log(f"Windows runtime stage patch skipped; target config is missing: {config_path}")
        return entry

    original = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*_attn_implementation\s*:\s*)flash_attention_2(\s*(?:#.*)?)$", re.MULTILINE)
    patched, replacements = pattern.subn(rf"\1{WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT}\2", original, count=1)
    if replacements:
        config_path.write_text(patched, encoding="utf-8")
        entry["status"] = "applied"
        entry["from"] = WINDOWS_FLASH_ATTN_CONFIG_SOURCE
        entry["to"] = WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT
        _log(
            "Applied Windows runtime stage patch: "
            f"{WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH} now uses {WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT} instead of {WINDOWS_FLASH_ATTN_CONFIG_SOURCE}."
        )
        return entry

    if f"_attn_implementation: {WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT}" in original:
        entry["status"] = "already-patched"
        entry["from"] = WINDOWS_FLASH_ATTN_CONFIG_SOURCE
        entry["to"] = WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT
        _log(
            "Windows runtime stage patch already present: "
            f"{WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH} is pinned to {WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT}."
        )
        return entry

    entry["status"] = "no-match"
    _log(
        "Windows runtime stage patch not applied because the upstream config no longer requests flash_attention_2: "
        f"{WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH}."
    )
    return entry


def _patch_windows_skin_enable_flash(unirig_dir: Path) -> dict[str, str]:
    config_path = unirig_dir / WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH
    entry = {
        "patch": "windows-disable-skin-mesh-encoder-flash",
        "path": str(WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH),
        "status": "skipped",
        "reason": (
            "Windows reaches the skin stage, but PTv3Object still routes through FlashAttention unless "
            "mesh_encoder.enable_flash is forced off. The observed failure only supports Ampere GPUs or newer, so Windows stages pin this flag to False."
        ),
        "setting": "mesh_encoder.enable_flash",
        "to": "False",
    }
    if not config_path.exists():
        entry["status"] = "target-missing"
        _log(f"Windows runtime stage patch skipped; target config is missing: {config_path}")
        return entry

    original = config_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    mesh_encoder_index: int | None = None
    mesh_encoder_indent = ""
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)mesh_encoder\s*:\s*(?:#.*)?$", line)
        if match:
            mesh_encoder_index = index
            mesh_encoder_indent = match.group(1)
            break

    if mesh_encoder_index is None:
        entry["status"] = "no-match"
        _log(
            "Windows runtime stage patch not applied because the upstream skin config no longer exposes mesh_encoder: "
            f"{WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH}."
        )
        return entry

    block_end = len(lines)
    child_indent: str | None = None
    enable_flash_index: int | None = None
    enable_flash_value = ""
    block_indent_len = len(mesh_encoder_indent)
    for index in range(mesh_encoder_index + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent_len = len(line) - len(line.lstrip(" "))
        if indent_len <= block_indent_len:
            block_end = index
            break
        if child_indent is None:
            child_indent = line[:indent_len]
        match = re.match(r"^\s*enable_flash\s*:\s*([^#\n]+?)(\s*(?:#.*)?)?$", line)
        if match:
            enable_flash_index = index
            enable_flash_value = match.group(1).strip()
            break

    child_indent = child_indent or f"{mesh_encoder_indent}  "
    new_line = f"{child_indent}enable_flash: False\n"

    if enable_flash_index is not None:
        if enable_flash_value.lower() == "false":
            entry["status"] = "already-patched"
            _log(
                "Windows runtime stage patch already present: "
                f"{WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH} already forces mesh_encoder.enable_flash to False."
            )
            return entry
        line = lines[enable_flash_index]
        comment_match = re.match(r"^(\s*enable_flash\s*:\s*)[^#\n]+?(\s*(?:#.*)?)?$", line)
        suffix = comment_match.group(2) or "" if comment_match else ""
        line_ending = "\r\n" if line.endswith("\r\n") else "\n"
        lines[enable_flash_index] = f"{child_indent}enable_flash: False{suffix}{line_ending}"
        config_path.write_text("".join(lines), encoding="utf-8")
        entry["status"] = "applied"
        entry["from"] = enable_flash_value
        _log(
            "Applied Windows runtime stage patch: "
            f"{WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH} now forces mesh_encoder.enable_flash to False."
        )
        return entry

    insert_at = mesh_encoder_index + 1
    while insert_at < block_end and not lines[insert_at].strip():
        insert_at += 1
    lines.insert(insert_at, new_line)
    config_path.write_text("".join(lines), encoding="utf-8")
    entry["status"] = "applied"
    entry["from"] = "implicit-default"
    _log(
        "Applied Windows runtime stage patch: "
        f"{WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH} now injects mesh_encoder.enable_flash: False."
    )
    return entry


def _write_runtime_stage_patch_report(ext_dir: Path, report: list[dict[str, str]]) -> None:
    report_path = _runtime_stage_patch_report_path(ext_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return
    tail = (result.stderr or result.stdout or "").strip().splitlines()[-60:]
    raise RuntimeError(f"Command failed: {' '.join(command)}\n" + "\n".join(tail))


def _probe_python_version(python_exe: Path) -> str:
    result = subprocess.run(
        [str(python_exe), "-c", "import platform; print(platform.python_version())"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "python probe failed").strip())
    return (result.stdout or "").strip()


def _resolve_bootstrap_python(payload: dict) -> tuple[Path, Path, dict[str, object]]:
    requested_host_python = Path(payload.get("python_exe") or sys.executable)
    try:
        resolved_version = _probe_python_version(requested_host_python)
    except (OSError, RuntimeError):
        resolved_version = ""
    resolution = {
        "platform": platform.system().lower(),
        "requested_host_python": str(requested_host_python),
        "selected_python": str(requested_host_python),
        "selected_version": resolved_version,
        "selected_source": "requested-host-python",
        "resolution_order": ["requested-host-python"],
        "attempts": [
            {
                "source": "requested-host-python",
                "command": [str(requested_host_python)],
                "status": "accepted" if resolved_version else "unavailable",
                "resolved_python": str(requested_host_python),
                "resolved_version": resolved_version,
            }
        ],
    }
    return requested_host_python, requested_host_python, resolution


def _create_virtualenv(venv_dir: Path, bootstrap_python: Path) -> None:
    _run([str(bootstrap_python), "-m", "venv", str(venv_dir)])


def _windows_flash_attn_wheel_url() -> str:
    return os.environ.get("MODLY_UNIRIG_FLASH_ATTN_WHEEL_URL", "").strip() or FLASH_ATTN_WHEEL_DEFAULT


def _windows_triton_package() -> str:
    return os.environ.get("MODLY_UNIRIG_TRITON_PACKAGE", "").strip() or TRITON_WINDOWS_PACKAGE_DEFAULT


def _windows_cumm_package() -> str:
    return os.environ.get("MODLY_UNIRIG_WINDOWS_CUMM_PACKAGE", "").strip() or WINDOWS_CUMM_PACKAGE_DEFAULT


def _windows_spconv_package() -> str:
    return os.environ.get("MODLY_UNIRIG_WINDOWS_SPCONV_PACKAGE", "").strip() or WINDOWS_SPCONV_PACKAGE_DEFAULT


def _filtered_requirements_path(unirig_dir: Path, runtime_root: Path) -> Path:
    requirements = unirig_dir / "requirements.txt"
    if not requirements.exists():
        raise RuntimeError(f"Upstream runtime requirements.txt is missing: {requirements}")
    if os.name != "nt":
        return requirements

    filtered = runtime_root / "requirements.upstream.windows.txt"
    lines: list[str] = []
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        normalized = raw.strip().replace("-", "_").lower()
        if normalized.startswith("flash_attn"):
            continue
        lines.append(raw)
    filtered.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filtered


def _install_windows_triton(pip: list[str], unirig_dir: Path) -> str:
    package = _windows_triton_package()
    if not package:
        raise RuntimeError(
            "Windows bootstrap requires a pinned Triton runtime package, but none is configured. "
            "Set MODLY_UNIRIG_TRITON_PACKAGE to a compatible pinned package and rerun setup."
        )
    try:
        _run(pip + ["install", package], cwd=unirig_dir)
    except RuntimeError as exc:
        raise RuntimeError(
            "Windows bootstrap could not install the pinned Triton runtime required by flash_attn. "
            f"Package: {package}. Override MODLY_UNIRIG_TRITON_PACKAGE with a compatible pinned package if needed. "
            f"Details: {exc}"
        ) from exc
    return package


def _install_windows_flash_attn(pip: list[str], runtime_root: Path, unirig_dir: Path) -> Path:
    wheel_url = _windows_flash_attn_wheel_url()
    if not wheel_url:
        raise RuntimeError(
            "Windows bootstrap requires a pinned flash_attn wheel URL, but none is configured. "
            "Set MODLY_UNIRIG_FLASH_ATTN_WHEEL_URL to a compatible pinned wheel and rerun setup."
        )
    wheel_name = wheel_url.rsplit("/", 1)[-1].split("?", 1)[0]
    wheel_path = runtime_root / "cache" / wheel_name
    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not wheel_path.exists():
            urllib.request.urlretrieve(wheel_url, str(wheel_path))
    except Exception as exc:
        raise RuntimeError(
            "Windows bootstrap could not download the pinned flash_attn wheel required by UniRig. "
            f"URL: {wheel_url}. Override with MODLY_UNIRIG_FLASH_ATTN_WHEEL_URL if you need a different pinned wheel. Details: {exc}"
        ) from exc
    try:
        _run(pip + ["install", str(wheel_path)], cwd=unirig_dir)
    except RuntimeError as exc:
        raise RuntimeError(
            "Windows bootstrap could not install the pinned flash_attn wheel required by UniRig. "
            f"Wheel: {wheel_path}. The default wheel targets the validated Windows Python 3.11 path; "
            "use that interpreter or override MODLY_UNIRIG_FLASH_ATTN_WHEEL_URL with a compatible pinned wheel. "
            f"Details: {exc}"
        ) from exc
    return wheel_path


def _install_windows_spconv_stack(pip: list[str], unirig_dir: Path) -> tuple[str, str]:
    cumm_package = _windows_cumm_package()
    spconv_package = _windows_spconv_package()
    missing = [package for package in (cumm_package, spconv_package) if not package]
    if missing:
        raise RuntimeError(
            "Windows bootstrap requires explicit pinned cumm/spconv prebuilt packages, but one or more are missing. "
            "Set MODLY_UNIRIG_WINDOWS_CUMM_PACKAGE and MODLY_UNIRIG_WINDOWS_SPCONV_PACKAGE to compatible pinned wheels and rerun setup."
        )
    try:
        _run(pip + ["install", "--no-cache-dir", cumm_package, spconv_package], cwd=unirig_dir)
    except RuntimeError as exc:
        raise RuntimeError(
            "Windows bootstrap could not install the validated cumm/spconv prebuilt pair required by UniRig. "
            f"Packages: {cumm_package}, {spconv_package}. Override MODLY_UNIRIG_WINDOWS_CUMM_PACKAGE or "
            f"MODLY_UNIRIG_WINDOWS_SPCONV_PACKAGE only if you have a better validated replacement. Details: {exc}"
        ) from exc
    return cumm_package, spconv_package


def _install_windows_sitecustomize(venv_dir: Path) -> list[Path]:
    return bootstrap.install_windows_dll_sitecustomize(venv_dir)


def _install_runtime_packages(ext_dir: Path, payload: dict) -> dict[str, object]:
    del payload
    if os.environ.get("UNIRIG_SETUP_SKIP_INSTALL") == "1":
        _log("skipping dependency installation because UNIRIG_SETUP_SKIP_INSTALL=1")
        return {"status": "skipped", "profile": "pinned-upstream-wrapper", "steps": []}

    runtime_root = _runtime_root(ext_dir)
    unirig_dir = _runtime_unirig_dir(ext_dir)
    requirements = _filtered_requirements_path(unirig_dir, runtime_root)
    python_exe = _venv_python(ext_dir / "venv")
    pip = [str(python_exe), "-m", "pip"]
    _run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    _run(pip + ["install", "--index-url", TORCH_INDEX_URL, *TORCH_PACKAGES], cwd=unirig_dir)
    _run(pip + ["install", "-r", str(requirements)], cwd=unirig_dir)
    wrapper_extras = ["install", "-f", PYG_INDEX_URL, "--no-cache-dir", *PYG_PACKAGES, NUMPY_PIN, "pygltflib>=1.15.0"]
    if os.name != "nt":
        wrapper_extras.append(SPCONV_PACKAGE)
    _run(pip + wrapper_extras, cwd=unirig_dir)
    windows_steps: list[str] = []
    if os.name == "nt":
        _install_windows_spconv_stack(pip, unirig_dir)
        _install_windows_triton(pip, unirig_dir)
        _install_windows_flash_attn(pip, runtime_root, unirig_dir)
        _install_windows_sitecustomize(ext_dir / "venv")
        windows_steps.extend(["windows-cumm-spconv", "windows-triton", "windows-flash-attn", "windows-sitecustomize"])
    return {
        "status": "ready",
        "profile": "pinned-upstream-wrapper",
        "steps": ["bootstrap-python", "torch", "upstream-requirements", "wrapper-extras", *windows_steps],
    }


def _run_post_setup_smoke_checks(ext_dir: Path) -> dict[str, object]:
    if os.name != "nt":
        return {"platform": platform.system().lower(), "status": "skipped", "checks": []}
    python_exe = _venv_python(ext_dir / "venv")
    smoke_env = bootstrap.runtime_environment(venv_dir=ext_dir / "venv")
    report = {"platform": "windows", "status": "ready", "checks": []}
    for check in WINDOWS_RUNTIME_SMOKE_CHECKS:
        result = subprocess.run(
            [str(python_exe), "-c", str(check["code"])],
            capture_output=True,
            text=True,
            check=False,
            env=smoke_env,
        )
        entry = {
            "label": str(check["label"]),
            "returncode": int(result.returncode),
            "status": "ready" if result.returncode == 0 else "error",
        }
        report["checks"].append(entry)
        if result.returncode == 0:
            continue
        report["status"] = "error"
        tail = (result.stderr or result.stdout or f"{check['label']} smoke import failed").strip().splitlines()[-20:]
        raise RuntimeError(
            "Windows bootstrap installed dependencies but a critical runtime import still fails. "
            f"Check: {check['label']}. Interpreter: {python_exe}. {check['repair_hint']} Details: "
            + " | ".join(tail)
        )
    return report


def _host_platform_tag() -> str:
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def _preflight_check_summary(
    python_exe: Path,
    payload: dict,
    requested_host_python: Path | None = None,
    bootstrap_resolution: dict[str, object] | None = None,
) -> dict:
    del payload, bootstrap_resolution
    host = {
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "platform_tag": _host_platform_tag(),
    }
    blocked: list[str] = []
    python_version = ""
    if not python_exe.exists():
        blocked.append(f"Missing bootstrap python: {python_exe}")
    else:
        try:
            python_version = _probe_python_version(python_exe)
        except (OSError, RuntimeError) as exc:
            blocked.append(f"Unable to execute bootstrap python: {python_exe}. Details: {exc}")

    checks = [
        {
            "id": "bootstrap-python",
            "label": "Bootstrap Python",
            "required": "an executable interpreter path",
            "observed": str(python_exe),
            "status": "pass" if not blocked else "fail",
            "message": "Bootstrap interpreter is available." if not blocked else "Bootstrap interpreter is missing or not executable.",
        }
    ]
    return {
        "status": "ready" if not blocked else "blocked",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "observed": {
            "python_exe": str(python_exe),
            "python_version": python_version,
            "requested_host_python": str(requested_host_python or python_exe),
        },
        "checks": checks,
        "blocked": blocked,
        "repeatability": {
            "checklist_file": "logs/bootstrap-preflight-checklist.txt",
            "report_file": "logs/bootstrap-preflight.json",
        },
    }


def _preflight_failure_message(preflight: dict) -> str:
    blocked = preflight.get("blocked") or []
    if blocked:
        return "Bootstrap preflight blocked: " + " | ".join(str(item) for item in blocked)
    return "Bootstrap preflight blocked for an unknown reason. Inspect .unirig-runtime/logs/bootstrap-preflight.json."


def _write_preflight_artifacts(ext_dir: Path, preflight: dict) -> None:
    logs_dir = _runtime_root(ext_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "bootstrap-preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8")
    checklist = "\n".join(bootstrap.preflight_checklist_lines(preflight)) + "\n"
    (logs_dir / "bootstrap-preflight-checklist.txt").write_text(checklist, encoding="utf-8")


def _write_state(
    ext_dir: Path,
    source_ref: str,
    preflight: dict,
    *,
    vendor_source: str,
    requested_host_python: Path,
    bootstrap_resolution: dict[str, object],
) -> None:
    runtime_root = _runtime_root(ext_dir)
    bootstrap.save_state(
        {
            "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
            "install_state": "ready",
            "source_ref": source_ref,
            "runtime_root": str(runtime_root),
            "logs_dir": str(runtime_root / "logs"),
            "runtime_vendor_dir": str(_runtime_vendor_dir(ext_dir)),
            "unirig_dir": str(_runtime_unirig_dir(ext_dir)),
            "hf_home": str(runtime_root / "hf-home"),
            "venv_python": str(_venv_python(ext_dir / "venv")),
            "python_version": _probe_python_version(_venv_python(ext_dir / "venv")),
            "vendor_source": vendor_source,
            "requested_host_python": str(requested_host_python),
            "bootstrap_resolution": bootstrap_resolution,
            "preflight": preflight,
        },
        extension_root=ext_dir,
    )


def _write_error_state(
    ext_dir: Path,
    message: str,
    preflight: dict,
    *,
    vendor_source: str = "",
    requested_host_python: Path | None = None,
    bootstrap_resolution: dict[str, object] | None = None,
) -> None:
    runtime_root = _runtime_root(ext_dir)
    bootstrap.save_state(
        {
            "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
            "install_state": "blocked" if preflight.get("status") == "blocked" else "error",
            "runtime_root": str(runtime_root),
            "logs_dir": str(runtime_root / "logs"),
            "runtime_vendor_dir": str(_runtime_vendor_dir(ext_dir)),
            "unirig_dir": str(_runtime_unirig_dir(ext_dir)),
            "hf_home": str(runtime_root / "hf-home"),
            "venv_python": str(_venv_python(ext_dir / "venv")),
            "last_error": message,
            "vendor_source": vendor_source,
            "requested_host_python": str(requested_host_python) if requested_host_python else "",
            "bootstrap_resolution": bootstrap_resolution or {},
            "preflight": preflight,
        },
        extension_root=ext_dir,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    payload = _load_payload(argv)
    ext_dir = Path(payload.get("ext_dir") or ROOT)

    if os.environ.get("UNIRIG_SETUP_FAIL"):
        raise SystemExit("UniRig bootstrap failed intentionally for verification. Clear UNIRIG_SETUP_FAIL and retry setup.")

    ext_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = _runtime_root(ext_dir)
    for directory in (runtime_root, runtime_root / "cache", runtime_root / "assets", runtime_root / "logs", runtime_root / "hf-home"):
        directory.mkdir(parents=True, exist_ok=True)

    requested_host_python, bootstrap_python, bootstrap_resolution = _resolve_bootstrap_python(payload)
    preflight = _preflight_check_summary(
        bootstrap_python,
        payload,
        requested_host_python=requested_host_python,
        bootstrap_resolution=bootstrap_resolution,
    )
    _write_preflight_artifacts(ext_dir, preflight)
    for line in bootstrap.preflight_checklist_lines(preflight):
        _log(line)

    if preflight.get("status") != "ready":
        message = _preflight_failure_message(preflight)
        _write_error_state(
            ext_dir,
            message,
            preflight,
            requested_host_python=requested_host_python,
            bootstrap_resolution=bootstrap_resolution,
        )
        raise SystemExit(message + " Repair the host prerequisites, then rerun setup.py before provisioning the runtime.")

    try:
        vendor_source = ""
        venv_dir = ext_dir / "venv"
        if not venv_dir.exists():
            _log(f"creating venv at {venv_dir}")
            _create_virtualenv(venv_dir, bootstrap_python)
        else:
            _log(f"reusing existing venv at {venv_dir}")

        _log("preparing UniRig runtime source")
        _, vendor_source, source_ref = _prepare_runtime_source(ext_dir, payload)
        _log("installing runtime dependencies")
        _install_runtime_packages(ext_dir, payload)
        _run_post_setup_smoke_checks(ext_dir)
        _write_state(
            ext_dir,
            source_ref,
            preflight,
            vendor_source=vendor_source,
            requested_host_python=requested_host_python,
            bootstrap_resolution=bootstrap_resolution,
        )
        _log("UniRig real runtime ready")
        return 0
    except Exception as exc:
        _write_error_state(
            ext_dir,
            str(exc),
            preflight,
            vendor_source=vendor_source,
            requested_host_python=requested_host_python,
            bootstrap_resolution=bootstrap_resolution,
        )
        raise SystemExit(
            "UniRig bootstrap failed while provisioning the real runtime. "
            f"Details: {exc}. Repair the extension after fixing the environment/dependency issue."
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
