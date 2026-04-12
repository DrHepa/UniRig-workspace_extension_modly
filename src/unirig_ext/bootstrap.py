from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BOOTSTRAP_VERSION = 4
EXTENSION_ID = "unirig-process-extension"
UPSTREAM_REPO = "VAST-AI-Research/UniRig"
UPSTREAM_REF_DEFAULT = "a6a4e2d6c23b88eb79b4396c0bae558aaad4744b"
SOURCE_BUILD_MODE_PREBUILT = "prebuilt"
SOURCE_BUILD_MODE_SOURCE = "source-build"
REQUIRED_RUNTIME_PATHS = ("run.py", "src", "configs", "requirements.txt")
REAL_RUNTIME_MODE = "real"
RUNTIME_ENV_DEFAULTS = {
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
}
RUNTIME_ENV_PASSTHROUGH_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "COMSPEC",
    "PATHEXT",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
)
WINDOWS_RUNTIME_DLL_RELATIVE_GLOBS = ("torch/lib", "nvidia/*/bin")
FORBIDDEN_KEYS = {
    "workspace_tool",
    "workspace_tool_class",
    "workspaceToolClass",
    "privateRoute",
    "private_route",
    "uiHook",
    "ui_hook",
    "tool_kind",
}


class UniRigError(RuntimeError):
    pass


class BootstrapError(UniRigError):
    pass


class ProtocolError(UniRigError):
    pass


@dataclass(frozen=True)
class RuntimeContext:
    extension_root: Path
    runtime_root: Path
    cache_dir: Path
    assets_dir: Path
    logs_dir: Path
    state_path: Path
    venv_dir: Path
    venv_python: Path
    runtime_vendor_dir: Path
    unirig_dir: Path
    hf_home: Path
    extension_id: str
    runtime_mode: str
    allow_local_stub_runtime: bool
    bootstrap_version: int
    vendor_source: str
    source_ref: str
    host_python: str
    platform_tag: str
    python_version: str
    platform_policy: dict[str, Any]
    source_build: dict[str, Any]
    install_state: str = "ready"
    last_verification: dict[str, Any] | None = None


def _host_details(host_os: str | None = None, host_arch: str | None = None) -> dict[str, str]:
    resolved_os = str(host_os or platform.system()).strip().lower()
    resolved_arch = str(host_arch or platform.machine()).strip().lower()
    return {
        "os": resolved_os,
        "arch": resolved_arch,
        "platform_tag": f"{resolved_os}-{resolved_arch}",
    }


def resolve_platform_policy(host_os: str | None = None, host_arch: str | None = None) -> dict[str, Any]:
    host = _host_details(host_os, host_arch)
    return {
        "host": host,
        "selected": {
            "key": host["platform_tag"],
            "status": "unvalidated",
            "notes": "Runtime state records host facts only. Support claims live in docs and verification artifacts.",
        },
    }


def arm64_prerequisite_manifest() -> dict[str, Any]:
    return {
        "kind": "reference-only",
        "target": {"os": "linux", "architectures": ["aarch64", "arm64"]},
        "upstream_repo": UPSTREAM_REPO,
        "source_ref": UPSTREAM_REF_DEFAULT,
        "notes": "Compatibility shim retained for one transition release; setup/bootstrap no longer own an ARM64 policy engine.",
    }


def windows_x64_prebuilt_manifest() -> dict[str, Any]:
    return {
        "kind": "reference-only",
        "target": {"os": "windows", "architectures": ["amd64", "x86_64"]},
        "upstream_repo": UPSTREAM_REPO,
        "source_ref": UPSTREAM_REF_DEFAULT,
        "notes": "Compatibility shim retained for one transition release; setup/bootstrap no longer own a Windows policy matrix.",
    }


def preflight_checklist_lines(preflight: dict[str, Any]) -> list[str]:
    host = dict(preflight.get("host") or {})
    observed = dict(preflight.get("observed") or {})
    lines = [f"[{'x' if preflight.get('status') == 'ready' else '!'}] preflight status: {preflight.get('status', 'unknown')}"]
    if host:
        lines.append(f"[x] host platform: {host.get('os', '?')} / {host.get('arch', '?')}")
    if observed.get("python_version"):
        lines.append(f"[x] bootstrap python: {observed['python_version']}")
    for message in preflight.get("blocked") or []:
        lines.append(f"[!] blocked: {message}")
    return lines


def _extension_root_override_from_env() -> Path | None:
    override = str(os.environ.get("UNIRIG_EXTENSION_ROOT") or "").strip()
    if not override:
        return None
    return Path(override).expanduser().resolve()


def resolve_extension_root(extension_root: Path | None = None, *, allow_env_override: bool = False) -> Path:
    if extension_root is not None:
        return extension_root.resolve()
    if allow_env_override:
        override = _extension_root_override_from_env()
        if override is not None:
            return override
    return Path(__file__).resolve().parents[2]


def state_path_for(extension_root: Path | None = None) -> Path:
    root = resolve_extension_root(extension_root)
    return root / ".unirig-runtime" / "bootstrap_state.json"


def _resolve_runtime_path(root: Path, value: Any, default: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def runtime_vendor_dir_for(root: Path, state: dict[str, Any]) -> Path:
    return _resolve_runtime_path(root, state.get("runtime_vendor_dir"), root / ".unirig-runtime" / "vendor")


def unirig_dir_for(root: Path, state: dict[str, Any]) -> Path:
    return _resolve_runtime_path(root, state.get("unirig_dir"), runtime_vendor_dir_for(root, state) / "unirig")


def load_state(extension_root: Path | None = None) -> dict[str, Any]:
    path = state_path_for(extension_root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BootstrapError(
            f"UniRig runtime state file is corrupt: {path}. Run setup.py again to repair the extension. Details: {exc}"
        ) from exc
    return normalize_state(raw, resolve_extension_root(extension_root))


def save_state(state: dict[str, Any], extension_root: Path | None = None) -> None:
    path = state_path_for(extension_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_state(state, resolve_extension_root(extension_root))
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")


def _default_runtime_paths(root: Path, state: dict[str, Any] | None = None) -> dict[str, str]:
    state = dict(state or {})
    runtime_root = _resolve_runtime_path(root, state.get("runtime_root"), root / ".unirig-runtime")
    return {
        "runtime_root": str(runtime_root),
        "logs_dir": str(_resolve_runtime_path(root, state.get("logs_dir"), runtime_root / "logs")),
        "runtime_vendor_dir": str(runtime_vendor_dir_for(root, state)),
        "unirig_dir": str(unirig_dir_for(root, state)),
        "hf_home": str(_resolve_runtime_path(root, state.get("hf_home"), runtime_root / "hf-home")),
        "venv_python": str(
            _resolve_runtime_path(
                root,
                state.get("venv_python"),
                _resolve_runtime_path(root, state.get("venv_dir"), root / "venv")
                / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
            )
        ),
    }


def _normalize_runtime_paths(root: Path, state: dict[str, Any], defaults: dict[str, str]) -> dict[str, str]:
    normalized = dict(defaults)
    for key, value in dict(state.get("runtime_paths") or {}).items():
        if key not in defaults:
            normalized[str(key)] = str(value)
            continue
        normalized[str(key)] = str(_resolve_runtime_path(root, value, Path(defaults[key])))
    return normalized


def _host_details_from_legacy_state(state: dict[str, Any]) -> dict[str, str]:
    host = ((state.get("last_verification") or {}).get("host") or {}).copy()
    if host.get("os") and host.get("arch"):
        return {"os": str(host["os"]), "arch": str(host["arch"])}

    preflight_host = (state.get("preflight") or {}).get("host") or {}
    if preflight_host.get("os") and preflight_host.get("arch"):
        return {"os": str(preflight_host["os"]), "arch": str(preflight_host["arch"])}

    platform_policy = state.get("platform_policy") or {}
    policy_host = platform_policy.get("host") or {}
    if policy_host.get("os") and policy_host.get("arch"):
        return {"os": str(policy_host["os"]), "arch": str(policy_host["arch"])}

    platform_tag = str(state.get("platform") or "").strip()
    if "-" in platform_tag:
        host_os, host_arch = platform_tag.split("-", 1)
        if host_os and host_arch:
            return {"os": host_os, "arch": host_arch}

    host = _host_details()
    return {"os": host["os"], "arch": host["arch"]}


def _verification_errors(state: dict[str, Any]) -> list[str]:
    verification = state.get("last_verification") or {}
    errors = verification.get("errors")
    if isinstance(errors, list) and errors:
        return [str(item) for item in errors if str(item).strip()]

    blocked = (state.get("preflight") or {}).get("blocked")
    if isinstance(blocked, list) and blocked:
        return [str(item) for item in blocked if str(item).strip()]

    last_error = str(state.get("last_error") or "").strip()
    if last_error:
        return [last_error]
    return []


def normalize_state(state: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise BootstrapError(
            "UniRig runtime state payload is invalid. Run setup.py again to repair the extension."
        )

    root = resolve_extension_root(root)
    default_runtime_paths = _default_runtime_paths(root, state)
    runtime_paths = _normalize_runtime_paths(root, state, default_runtime_paths)

    last_verification = dict(state.get("last_verification") or {})
    host = _host_details_from_legacy_state(state)
    install_state = str(state.get("install_state") or "unknown")
    status = str(last_verification.get("status") or (state.get("preflight") or {}).get("status") or install_state or "unknown")
    normalized = {
        "bootstrap_version": int(state.get("bootstrap_version") or 0),
        "install_state": install_state,
        "source_ref": str(state.get("source_ref") or UPSTREAM_REF_DEFAULT),
        "vendor_source": str(state.get("vendor_source") or ""),
        "requested_host_python": str(state.get("requested_host_python") or ""),
        "bootstrap_resolution": dict(state.get("bootstrap_resolution") or {}),
        "runtime_paths": runtime_paths,
        "last_verification": {
            "status": status,
            "runtime_ready": bool(last_verification.get("runtime_ready") if "runtime_ready" in last_verification else install_state == "ready"),
            "python_version": str(last_verification.get("python_version") or state.get("python_version") or ""),
            "host": host,
            "errors": _verification_errors(state),
        },
    }
    checked_at = last_verification.get("checked_at") or (state.get("preflight") or {}).get("checked_at")
    if checked_at:
        normalized["last_verification"]["checked_at"] = str(checked_at)
    return normalized


def _readiness_failure_message(state: dict[str, Any]) -> str:
    verification = dict(state.get("last_verification") or {})
    install_state = str(state.get("install_state") or "unknown")
    errors = [str(item) for item in verification.get("errors") or [] if str(item).strip()]
    if errors:
        return f"UniRig runtime is {install_state}: " + " | ".join(errors) + ". Run setup.py again to repair the extension."
    return f"UniRig runtime state is '{install_state}'. Run setup.py again to repair the extension."


def runtime_pythonpath(context: RuntimeContext) -> str:
    parts = [str(context.runtime_vendor_dir), str(context.unirig_dir)]
    return os.pathsep.join(parts)


def _site_packages_dirs(venv_dir: Path) -> list[Path]:
    roots = [venv_dir / "Lib" / "site-packages", venv_dir / "lib" / "site-packages"]
    roots.extend(sorted((venv_dir / "lib").glob("python*/site-packages")))
    found: list[Path] = []
    seen: set[str] = set()
    for path in roots:
        normalized = os.path.normcase(os.path.normpath(str(path)))
        if normalized in seen or not path.exists():
            continue
        seen.add(normalized)
        found.append(path)
    return found


def _normalized_path_key(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _windows_runtime_dll_paths_for_site_packages(site_packages: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in WINDOWS_RUNTIME_DLL_RELATIVE_GLOBS:
        for candidate in sorted(site_packages.glob(pattern)):
            normalized = _normalized_path_key(candidate)
            if normalized in seen or not candidate.is_dir():
                continue
            seen.add(normalized)
            paths.append(candidate)
    return paths


def windows_runtime_dll_search_paths(venv_dir: Path) -> list[Path]:
    if os.name != "nt":
        return []

    paths: list[Path] = []
    seen: set[str] = set()
    for site_packages in _site_packages_dirs(venv_dir):
        for candidate in _windows_runtime_dll_paths_for_site_packages(site_packages):
            normalized = _normalized_path_key(candidate)
            if normalized in seen or not candidate.is_dir():
                continue
            seen.add(normalized)
            paths.append(candidate)
    return paths


def render_windows_dll_sitecustomize() -> str:
    patterns = ", ".join(repr(pattern) for pattern in WINDOWS_RUNTIME_DLL_RELATIVE_GLOBS)
    return f'''import os
from pathlib import Path

_MODLY_UNIRIG_DLL_HANDLES = []
_MODLY_UNIRIG_BOOTSTRAP_LOG = Path(__file__).with_name("sitecustomize-unirig.log")


def _record_bootstrap_note(message):
    try:
        _MODLY_UNIRIG_BOOTSTRAP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _MODLY_UNIRIG_BOOTSTRAP_LOG.open("a", encoding="utf-8") as handle:
            handle.write(str(message) + "\\n")
    except Exception:
        return


def _iter_dll_dirs():
    root = Path(__file__).resolve().parent
    seen = set()
    for pattern in ({patterns},):
        for path in sorted(root.glob(pattern)):
            try:
                if not path.is_dir():
                    continue
            except OSError:
                continue
            key = os.path.normcase(os.path.normpath(str(path)))
            if key in seen:
                continue
            seen.add(key)
            yield path


if os.name == "nt":
    if callable(getattr(os, "add_dll_directory", None)):
        for path in _iter_dll_dirs():
            try:
                handle = os.add_dll_directory(str(path))
            except OSError as exc:
                _record_bootstrap_note(f"add_dll_directory failed for {{path}}: {{exc}}")
                continue
            if handle is not None:
                _MODLY_UNIRIG_DLL_HANDLES.append(handle)
    else:
        _record_bootstrap_note("os.add_dll_directory is unavailable on this interpreter")


def _register_torch_safe_globals():
    try:
        import torch
        from box.box import Box
    except Exception as exc:
        _record_bootstrap_note(f"torch/Box import skipped: {{exc}}")
        return
    try:
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    except Exception as exc:
        _record_bootstrap_note(f"torch.serialization lookup failed: {{exc}}")
        return
    if not callable(add_safe_globals):
        _record_bootstrap_note("torch.serialization.add_safe_globals is unavailable")
        return
    try:
        add_safe_globals([Box])
    except Exception as exc:
        _record_bootstrap_note(f"torch safe globals registration failed: {{exc}}")
        return
    _record_bootstrap_note("registered torch safe globals for box.box.Box")


_register_torch_safe_globals()
'''


def install_windows_dll_sitecustomize(venv_dir: Path) -> list[Path]:
    if os.name != "nt":
        return []

    targets = _site_packages_dirs(venv_dir)
    if not targets:
        targets = [venv_dir / "Lib" / "site-packages"]

    content = render_windows_dll_sitecustomize()
    written: list[Path] = []
    for site_packages in targets:
        site_packages.mkdir(parents=True, exist_ok=True)
        target = site_packages / "sitecustomize.py"
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


def _prepend_path_entries(current: str, extra_paths: list[Path]) -> str:
    ordered = [str(path) for path in extra_paths]
    if current:
        ordered.extend(part for part in current.split(os.pathsep) if part)

    deduped: list[str] = []
    seen: set[str] = set()
    for part in ordered:
        normalized = _normalized_path_key(part)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(part))
    return os.pathsep.join(deduped)


def runtime_environment(
    *,
    context: RuntimeContext | None = None,
    venv_dir: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in RUNTIME_ENV_PASSTHROUGH_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.update(RUNTIME_ENV_DEFAULTS)

    resolved_venv_dir = venv_dir
    if context is not None:
        env["HF_HOME"] = str(context.hf_home)
        env["TRANSFORMERS_CACHE"] = str(context.hf_home / "hub")
        env["PYTHONPATH"] = runtime_pythonpath(context)
        resolved_venv_dir = context.venv_dir

    if resolved_venv_dir is not None:
        dll_paths = windows_runtime_dll_search_paths(resolved_venv_dir)
        if dll_paths:
            env["PATH"] = _prepend_path_entries(env.get("PATH", ""), dll_paths)

    if extra:
        env.update({key: str(value) for key, value in extra.items()})
    return env


def runtime_layout_errors(unirig_dir: Path, venv_python: Path) -> list[str]:
    errors = [f"missing runtime dependency: {unirig_dir / rel}" for rel in REQUIRED_RUNTIME_PATHS if not (unirig_dir / rel).exists()]
    if not venv_python.exists():
        errors.append(f"missing runtime python: {venv_python}")
    return errors


def ensure_ready(extension_root: Path | None = None) -> RuntimeContext:
    root = resolve_extension_root(extension_root)
    state = load_state(root)
    if not state:
        raise BootstrapError("UniRig runtime is not prepared. Run setup.py from Modly install/repair before executing rig-mesh.")
    if int(state.get("bootstrap_version") or 0) < BOOTSTRAP_VERSION:
        raise BootstrapError("UniRig runtime bootstrap is outdated for the real public pipeline. Run setup.py again to repair the extension.")
    if state.get("install_state") != "ready":
        raise BootstrapError(_readiness_failure_message(state))

    runtime_paths = dict(state.get("runtime_paths") or _default_runtime_paths(root, state))
    runtime_root = Path(runtime_paths["runtime_root"])
    cache_dir = runtime_root / "cache"
    assets_dir = runtime_root / "assets"
    logs_dir = Path(runtime_paths["logs_dir"])
    venv_python = Path(runtime_paths["venv_python"])
    venv_dir = venv_python.parent.parent if venv_python.parent.name in {"bin", "Scripts"} else venv_python.parent
    runtime_vendor_dir = Path(runtime_paths["runtime_vendor_dir"])
    unirig_dir = Path(runtime_paths["unirig_dir"])
    hf_home = Path(runtime_paths["hf_home"])
    verification = dict(state.get("last_verification") or {})
    host = dict(verification.get("host") or _host_details_from_legacy_state(state))

    for path in (runtime_root, cache_dir, assets_dir, logs_dir, runtime_vendor_dir, hf_home):
        path.mkdir(parents=True, exist_ok=True)

    context = RuntimeContext(
        extension_root=root,
        runtime_root=runtime_root,
        cache_dir=cache_dir,
        assets_dir=assets_dir,
        logs_dir=logs_dir,
        state_path=state_path_for(root),
        venv_dir=venv_dir,
        venv_python=venv_python,
        runtime_vendor_dir=runtime_vendor_dir,
        unirig_dir=unirig_dir,
        hf_home=hf_home,
        extension_id=EXTENSION_ID,
        runtime_mode=REAL_RUNTIME_MODE,
        allow_local_stub_runtime=False,
        bootstrap_version=int(state.get("bootstrap_version") or BOOTSTRAP_VERSION),
        vendor_source=str(state.get("vendor_source") or ""),
        source_ref=str(state.get("source_ref") or UPSTREAM_REF_DEFAULT),
        host_python=str(state.get("requested_host_python") or ""),
        platform_tag=f"{host.get('os', platform.system().lower())}-{host.get('arch', platform.machine().lower())}",
        python_version=str(verification.get("python_version") or state.get("python_version") or ""),
        platform_policy=resolve_platform_policy(host.get("os"), host.get("arch")),
        source_build={},
        install_state=str(state.get("install_state") or "unknown"),
        last_verification=verification,
    )

    errors = runtime_layout_errors(context.unirig_dir, context.venv_python)
    if errors:
        raise BootstrapError(
            "UniRig real runtime is marked ready but is incomplete. Run setup.py again to repair the extension. "
            + " | ".join(errors)
        )
    return context


def reject_private_contracts(payload: Any, params: dict[str, Any] | None = None) -> None:
    _scan_forbidden(payload)
    _scan_forbidden(params or {})


def _scan_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                raise ProtocolError(f"Private contract '{key}' is not supported by the public UniRig process extension.")
            _scan_forbidden(item)
    elif isinstance(value, list):
        for item in value:
            _scan_forbidden(item)


def stage_environment(extra: dict[str, str] | None = None, context: RuntimeContext | None = None) -> dict[str, str]:
    return runtime_environment(context=context, extra=extra)
