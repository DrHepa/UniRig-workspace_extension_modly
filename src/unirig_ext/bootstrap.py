from __future__ import annotations

import json
import os
import platform
import subprocess
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
LINUX_ARM64_QUALIFICATION_EXECUTION_MODES = ("wrapper", "seam", "forced-fallback")
LINUX_ARM64_QUALIFICATION_VERDICTS = (
    "not-ready",
    "candidate-with-known-risks",
    "ready-for-separate-defaulting-change",
)
LINUX_ARM64_PARTIAL_RUNTIME_STAGE_NAMES = ("extract-prepare", "extract-skin", "merge")
LINUX_ARM64_PERSISTED_STAGE_PROOF_NAMES = ("extract-prepare", "skeleton", "extract-skin", "skin")
LINUX_ARM64_RECOVERED_STAGE_PROOF_NAMES = (*LINUX_ARM64_PERSISTED_STAGE_PROOF_NAMES, "merge")
LINUX_ARM64_LIVE_IMPORT_MODULES = ("torch_scatter", "torch_cluster", "spconv.pytorch")


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
    return normalize_state(raw, resolve_extension_root(extension_root), include_runtime_fields=True)


def save_state(state: dict[str, Any], extension_root: Path | None = None) -> None:
    path = state_path_for(extension_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_state(state, resolve_extension_root(extension_root), include_runtime_fields=False)
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

    blocked_reasons = (state.get("source_build") or {}).get("blocked_reasons")
    if isinstance(blocked_reasons, list) and blocked_reasons:
        return [str(item) for item in blocked_reasons if str(item).strip()]

    last_error = str(state.get("last_error") or "").strip()
    if last_error:
        return [last_error]
    return []


def _copy_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    return value


def _normalized_platform_policy(state: dict[str, Any], host: dict[str, str]) -> dict[str, Any]:
    planner = dict(state.get("planner") or {})
    preflight = dict(state.get("preflight") or {})
    install_summary = dict((state.get("install_plan") or {}).get("summary") or {})
    source_build = dict(state.get("source_build") or {})
    existing = _copy_json_value(state.get("platform_policy") or {})
    selected = dict(existing.get("selected") or {})
    selected_key = str(
        planner.get("host_class")
        or preflight.get("host_class")
        or install_summary.get("host_class")
        or source_build.get("host_class")
        or selected.get("key")
        or f"{host.get('os', '?')}-{host.get('arch', '?')}"
    )
    support_posture = str(
        planner.get("support_posture")
        or preflight.get("support_posture")
        or source_build.get("support_posture")
        or selected.get("status")
        or "unvalidated"
    )
    install_mode = str(
        planner.get("install_mode")
        or install_summary.get("install_mode")
        or source_build.get("mode")
        or selected.get("install_mode")
        or ""
    )
    notes = str(
        selected.get("notes")
        or "Runtime state records host facts and planner posture for readiness diagnostics. Support claims live in docs and verification artifacts."
    )
    normalized = {
        "host": {
            "os": str(host.get("os") or ""),
            "arch": str(host.get("arch") or ""),
            "platform_tag": f"{host.get('os', '')}-{host.get('arch', '')}",
        },
        "selected": {
            "key": selected_key,
            "status": support_posture,
            "notes": notes,
        },
    }
    if install_mode:
        normalized["selected"]["install_mode"] = install_mode
    return normalized


def _linux_arm64_legacy_spconv_stage_fallback(source_build: dict[str, Any]) -> dict[str, Any]:
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return source_build

    normalized = dict(source_build)
    stages = _copy_json_value(normalized.get("stages") or {})
    if not isinstance(stages, dict):
        stages = {}

    spconv_stage = stages.get("spconv")
    if not isinstance(spconv_stage, dict):
        message = (
            "legacy Linux ARM64 state is missing spconv stage evidence, so bootstrap keeps readiness blocked "
            "until setup.py records guarded import-smoke results."
        )
        stages["spconv"] = {
            "status": "deferred",
            "ready": False,
            "mode": "source-build",
            "verification": "import-smoke",
            "packages": ["cumm", "spconv"],
            "checks": [],
            "blockers": [
                {
                    "category": "state",
                    "code": "spconv-state-missing",
                    "dependency": "spconv",
                    "message": message,
                    "action": "stop",
                    "boundary": "wrapper",
                    "owner": "wrapper",
                }
            ],
            "blocker_codes": ["spconv-state-missing"],
            "boundary": "wrapper",
        }
        normalized["stages"] = stages
        return normalized

    spconv_stage = dict(spconv_stage)
    spconv_stage.setdefault("mode", "source-build")
    spconv_stage.setdefault("verification", "import-smoke")
    spconv_stage.setdefault("packages", ["cumm", "spconv"])
    spconv_stage.setdefault("checks", [])
    spconv_stage.setdefault("blockers", [])
    spconv_stage.setdefault("blocker_codes", [])
    stages["spconv"] = spconv_stage
    normalized["stages"] = stages
    return normalized


def _normalized_source_build(state: dict[str, Any]) -> dict[str, Any]:
    planner = dict(state.get("planner") or {})
    preflight = dict(state.get("preflight") or {})
    install_summary = dict((state.get("install_plan") or {}).get("summary") or {})
    existing = _linux_arm64_legacy_spconv_stage_fallback(_copy_json_value(state.get("source_build") or {}))
    blockers = _copy_json_value(preflight.get("blockers") or existing.get("blockers") or [])
    if not blockers:
        blockers = _linux_arm64_stage_blockers(existing)
    deferred_work = [str(item) for item in state.get("deferred_work") or planner.get("deferred") or existing.get("deferred_work") or []]
    normalized = dict(existing)
    normalized["status"] = str(preflight.get("status") or install_summary.get("status") or existing.get("status") or state.get("install_state") or "unknown")
    normalized["mode"] = str(planner.get("install_mode") or install_summary.get("install_mode") or existing.get("mode") or "")
    host_class = str(planner.get("host_class") or preflight.get("host_class") or install_summary.get("host_class") or existing.get("host_class") or "")
    if host_class:
        normalized["host_class"] = host_class
    support_posture = str(planner.get("support_posture") or preflight.get("support_posture") or existing.get("support_posture") or "")
    if support_posture:
        normalized["support_posture"] = support_posture
    normalized_stages = _copy_json_value(normalized.get("stages") or {})
    if not isinstance(normalized_stages, dict):
        normalized_stages = {}
    bpy_stage = _linux_arm64_normalized_bpy_stage(normalized)
    if bpy_stage:
        normalized_stages["bpy"] = bpy_stage
        normalized["stages"] = normalized_stages
        normalized["bpy_evidence_class"] = str(bpy_stage.get("status") or normalized.get("bpy_evidence_class") or "")
        external_blender = _copy_json_value(normalized.get("external_blender") or {})
        if not isinstance(external_blender, dict):
            external_blender = {}
        external_blender.setdefault("classification", _copy_json_value(bpy_stage))
        normalized["external_blender"] = external_blender
    executable_boundary = _linux_arm64_normalized_executable_boundary(normalized)
    if executable_boundary:
        normalized["executable_boundary"] = executable_boundary
    qualification = _linux_arm64_normalized_qualification(normalized)
    if qualification:
        normalized["qualification"] = qualification
    external_bpy_blocker = _linux_arm64_external_bpy_blocker(normalized)
    if external_bpy_blocker is not None:
        existing_codes = {
            str(item.get("code") or "").strip()
            for item in blockers
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        }
        blocker_code = str(external_bpy_blocker.get("code") or "").strip()
        if not blocker_code or blocker_code not in existing_codes:
            blockers = list(blockers) + [external_bpy_blocker]
    if not blockers:
        deferred_bpy = _linux_arm64_deferred_bpy_blocker(normalized)
        if deferred_bpy is not None:
            blockers = [deferred_bpy]
    normalized["blockers"] = blockers
    if not normalized.get("blocked_reasons"):
        normalized["blocked_reasons"] = [
            str(item.get("message") or "").strip()
            for item in blockers
            if isinstance(item, dict) and str(item.get("message") or "").strip()
        ]
    normalized["deferred_work"] = deferred_work
    return normalized


def _linux_arm64_deferred_bpy_blocker(source_build: dict[str, Any]) -> dict[str, Any] | None:
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return None

    stages = dict(source_build.get("stages") or {})
    bpy_stage = dict(stages.get("bpy") or {})
    deferred_work = {str(item).strip() for item in source_build.get("deferred_work") or [] if str(item).strip()}
    is_bpy_deferred = str(bpy_stage.get("status") or "").strip() == "deferred" or "bpy-portability" in deferred_work
    if not is_bpy_deferred:
        return None

    message = str(bpy_stage.get("message") or "").strip()
    if not message:
        blocked_reasons = [str(item).strip() for item in source_build.get("blocked_reasons") or [] if str(item).strip()]
        bpy_messages = [item for item in blocked_reasons if "bpy" in item.lower()]
        message = bpy_messages[0] if bpy_messages else "bpy remains deferred on Linux ARM64 until upstream portability evidence exists."

    return {
        "category": "portability",
        "code": str(bpy_stage.get("reason_code") or "bpy-portability-risk"),
        "dependency": "bpy",
        "message": message,
        "action": "stop",
        "boundary": "upstream",
        "owner": "upstream",
    }


def _linux_arm64_normalized_bpy_stage(source_build: dict[str, Any]) -> dict[str, Any]:
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return {}

    stages = dict(source_build.get("stages") or {})
    if isinstance(stages.get("bpy"), dict):
        bpy_stage = _copy_json_value(stages.get("bpy") or {})
        bpy_stage["ready"] = False
        return bpy_stage

    external_blender = source_build.get("external_blender") or {}
    classification = (
        _copy_json_value(external_blender.get("classification") or {})
        if isinstance(external_blender, dict) and isinstance(external_blender.get("classification"), dict)
        else {}
    )
    status = str(classification.get("status") or source_build.get("bpy_evidence_class") or "").strip()
    if not status:
        return {}

    if not classification:
        classification = {
            "status": status,
            "evidence_kind": "external-blender",
        }
    classification["status"] = status
    classification["ready"] = False
    classification.setdefault("checks", [])
    classification.setdefault("blockers", [])
    classification.setdefault("blocker_codes", [])
    if isinstance(external_blender, dict):
        candidate = _copy_json_value(external_blender.get("candidate") or {}) if isinstance(external_blender.get("candidate"), dict) else {}
        probe = _copy_json_value(external_blender.get("probe") or {}) if isinstance(external_blender.get("probe"), dict) else {}
        if candidate:
            classification.setdefault("candidate", candidate)
        blender_version = str(probe.get("blender_version") or "").strip()
        python_version = str(probe.get("python_version") or "").strip()
        if (blender_version or python_version) and "blender" not in classification:
            classification["blender"] = {
                "version": blender_version,
                "python_version": python_version,
            }
        if probe:
            classification.setdefault("verification", "blender-background-bpy-smoke")
    return classification


def _linux_arm64_normalized_executable_boundary(source_build: dict[str, Any]) -> dict[str, Any]:
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return {}

    existing = _copy_json_value(source_build.get("executable_boundary") or {})
    if not isinstance(existing, dict):
        existing = {}

    extract_merge = _copy_json_value(existing.get("extract_merge") or {})
    if not isinstance(extract_merge, dict):
        extract_merge = {}

    external_blender = source_build.get("external_blender") if isinstance(source_build.get("external_blender"), dict) else {}
    classification = (
        _copy_json_value(external_blender.get("classification") or {})
        if isinstance(external_blender.get("classification"), dict)
        else {}
    )
    candidate = _copy_json_value(external_blender.get("candidate") or {}) if isinstance(external_blender.get("candidate"), dict) else {}
    if not candidate:
        candidate = _copy_json_value(classification.get("candidate") or {}) if isinstance(classification.get("candidate"), dict) else {}

    extract_merge.setdefault("enabled", False)
    extract_merge["ready"] = bool(extract_merge.get("ready", False))
    extract_merge.setdefault("status", "verified" if extract_merge["ready"] else "missing")
    extract_merge.setdefault("default_owner", "context.venv_python")
    extract_merge.setdefault("optional_owner", "blender-subprocess")
    extract_merge.setdefault("supported_stages", ["extract-prepare", "extract-skin", "merge"])
    extract_merge.setdefault("requires_explicit_gate", True)
    extract_merge.setdefault("requires_executable_boundary_proof", True)
    if candidate:
        extract_merge.setdefault("candidate", candidate)
    evidence_kind = str(classification.get("evidence_kind") or "external-blender").strip()
    if evidence_kind:
        extract_merge.setdefault("evidence_kind", evidence_kind)
    external_blender_status = str(classification.get("status") or "").strip()
    if external_blender_status:
        extract_merge.setdefault("external_blender_status", external_blender_status)
    existing["extract_merge"] = extract_merge
    return existing


def _linux_arm64_qualification_summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "by_failure_code": {},
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "").strip()
        if status not in {"passed", "failed"}:
            continue
        summary["total"] += 1
        summary[status] += 1
        if status != "failed":
            continue
        failure_code = str(record.get("failure_code") or "").strip()
        if not failure_code:
            continue
        summary["by_failure_code"][failure_code] = summary["by_failure_code"].get(failure_code, 0) + 1
    return summary


def _linux_arm64_normalized_qualification_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = _copy_json_value(record if isinstance(record, dict) else {})
    normalized.setdefault("fixture_id", "")
    normalized.setdefault("fixture_class", "")
    normalized.setdefault("stage", "")
    normalized.setdefault("run_label", "")
    selected_mode = str(normalized.get("selected_mode") or "").strip()
    if selected_mode not in LINUX_ARM64_QUALIFICATION_EXECUTION_MODES:
        selected_mode = ""
    normalized["selected_mode"] = selected_mode
    status = str(normalized.get("status") or "").strip()
    if status not in {"passed", "failed"}:
        status = "failed" if str(normalized.get("failure_code") or "").strip() else "passed"
    normalized["status"] = status
    failure_code = str(normalized.get("failure_code") or "").strip()
    normalized["failure_code"] = failure_code if status == "failed" else ""
    normalized["host"] = _copy_json_value(normalized.get("host") or {}) if isinstance(normalized.get("host"), dict) else {}
    normalized["blender"] = _copy_json_value(normalized.get("blender") or {}) if isinstance(normalized.get("blender"), dict) else {}
    normalized["outputs"] = _copy_json_value(normalized.get("outputs") or {}) if isinstance(normalized.get("outputs"), dict) else {}
    normalized["logs"] = _copy_json_value(normalized.get("logs") or {}) if isinstance(normalized.get("logs"), dict) else {}
    return normalized


def _linux_arm64_normalized_qualification_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    normalized = _copy_json_value(fixture if isinstance(fixture, dict) else {})
    normalized.setdefault("fixture_id", "")
    normalized.setdefault("fixture_class", "")
    normalized.setdefault("stage", "")
    execution_modes = normalized.get("execution_modes") if isinstance(normalized.get("execution_modes"), list) else []
    normalized["execution_modes"] = [
        str(item).strip()
        for item in execution_modes
        if str(item).strip() in LINUX_ARM64_QUALIFICATION_EXECUTION_MODES
    ]
    runs = normalized.get("runs") if isinstance(normalized.get("runs"), list) else []
    normalized["runs"] = [_linux_arm64_normalized_qualification_record(item) for item in runs if isinstance(item, dict)]
    comparison = _copy_json_value(normalized.get("comparison") or {}) if isinstance(normalized.get("comparison"), dict) else {}
    comparison.setdefault("fixture_id", str(normalized.get("fixture_id") or "").strip())
    comparison.setdefault("stage", str(normalized.get("stage") or "").strip())
    for key, mode in (("wrapper_vs_seam", "seam"), ("wrapper_vs_forced_fallback", "forced-fallback")):
        entry = _copy_json_value(comparison.get(key) or {}) if isinstance(comparison.get(key), dict) else {}
        status = str(entry.get("status") or "skipped").strip()
        if status not in {"passed", "failed", "skipped"}:
            status = "skipped"
        entry["status"] = status
        entry["failure_code"] = str(entry.get("failure_code") or "").strip() if status == "failed" else ""
        entry.setdefault("compared_modes", ["wrapper", mode])
        comparison[key] = entry
    normalized["comparison"] = comparison
    return normalized


def _linux_arm64_normalized_qualification_extract_merge(source_build: dict[str, Any]) -> dict[str, Any]:
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return {}

    qualification = _copy_json_value(source_build.get("qualification") or {})
    if not isinstance(qualification, dict):
        return {}
    extract_merge = _copy_json_value(qualification.get("extract_merge") or {})
    if not isinstance(extract_merge, dict) or not extract_merge:
        return {}

    normalized = dict(extract_merge)
    normalized["schema_version"] = int(normalized.get("schema_version") or 1)
    normalized.setdefault("default_owner", "context.venv_python")
    normalized.setdefault("optional_owner", "blender-subprocess")
    normalized["host"] = _copy_json_value(normalized.get("host") or {}) if isinstance(normalized.get("host"), dict) else {}
    normalized["blender"] = _copy_json_value(normalized.get("blender") or {}) if isinstance(normalized.get("blender"), dict) else {}
    fixtures = normalized.get("fixtures") if isinstance(normalized.get("fixtures"), list) else []
    records = normalized.get("records") if isinstance(normalized.get("records"), list) else []
    normalized["fixtures"] = [_linux_arm64_normalized_qualification_fixture(item) for item in fixtures if isinstance(item, dict)]
    normalized["records"] = [_linux_arm64_normalized_qualification_record(item) for item in records if isinstance(item, dict)]
    summary = _copy_json_value(normalized.get("summary") or {}) if isinstance(normalized.get("summary"), dict) else {}
    derived_summary = _linux_arm64_qualification_summary_from_records(normalized["records"])
    normalized["summary"] = {
        "total": int(summary.get("total", derived_summary["total"]) or 0),
        "passed": int(summary.get("passed", derived_summary["passed"]) or 0),
        "failed": int(summary.get("failed", derived_summary["failed"]) or 0),
        "by_failure_code": _copy_json_value(summary.get("by_failure_code") or derived_summary["by_failure_code"]),
    }
    verdict = str(normalized.get("verdict") or "").strip()
    normalized["verdict"] = verdict if verdict in LINUX_ARM64_QUALIFICATION_VERDICTS else "not-ready"
    windows_non_regression = _copy_json_value(normalized.get("windows_non_regression") or {})
    if not isinstance(windows_non_regression, dict):
        windows_non_regression = {}
    windows_non_regression.setdefault("host", "windows-x86_64")
    windows_non_regression["seam_selected"] = bool(windows_non_regression.get("seam_selected", False))
    windows_non_regression.setdefault("status", "unknown")
    normalized["windows_non_regression"] = windows_non_regression
    return normalized


def _linux_arm64_normalized_qualification(source_build: dict[str, Any]) -> dict[str, Any]:
    extract_merge = _linux_arm64_normalized_qualification_extract_merge(source_build)
    if not extract_merge:
        return {}
    return {"extract_merge": extract_merge}


def _linux_arm64_external_bpy_blocker(source_build: dict[str, Any]) -> dict[str, Any] | None:
    bpy_stage = _linux_arm64_normalized_bpy_stage(source_build)
    status = str(bpy_stage.get("status") or "").strip()
    if status not in {"external-bpy-smoke-ready", "discovered-incompatible", "missing", "error"}:
        return None

    blockers = [item for item in bpy_stage.get("blockers") or [] if isinstance(item, dict)]
    if blockers:
        return _copy_json_value(blockers[0])

    if status == "external-bpy-smoke-ready":
        return {
            "category": "runtime-boundary",
            "code": "external-bpy-evidence-only",
            "dependency": "bpy",
            "message": "Linux ARM64 external Blender bpy smoke evidence is preserved, but full wrapper runtime remains blocked until bpy is supported inside the wrapper-owned runtime boundary.",
            "action": "stop",
            "boundary": "wrapper",
            "owner": "wrapper",
        }

    return None


def _linux_arm64_stage_blockers(source_build: dict[str, Any]) -> list[dict[str, Any]]:
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return []

    blockers: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for stage_state in dict(source_build.get("stages") or {}).values():
        if not isinstance(stage_state, dict):
            continue
        for blocker in stage_state.get("blockers") or []:
            if not isinstance(blocker, dict):
                continue
            normalized = _copy_json_value(blocker)
            code = str(normalized.get("code") or "").strip()
            if code:
                if code in seen_codes:
                    continue
                seen_codes.add(code)
            blockers.append(normalized)
    return blockers


def _linux_arm64_requires_full_runtime_block(source_build: dict[str, Any]) -> bool:
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return False

    if _linux_arm64_external_bpy_blocker(source_build) is not None:
        return True

    blocker_codes = [str((item or {}).get("code") or "").strip() for item in source_build.get("blockers") or [] if isinstance(item, dict)]
    if any(code for code in blocker_codes):
        return True

    if _linux_arm64_stage_blockers(source_build):
        return True

    if any(str(item).strip() for item in source_build.get("blocked_reasons") or []):
        return True

    return _linux_arm64_deferred_bpy_blocker(source_build) is not None


def _linux_arm64_has_useful_partial_runtime(source_build: dict[str, Any]) -> bool:
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return False

    if not bool(source_build.get("non_blender_runtime_ready")):
        return False

    bpy_stage = _linux_arm64_normalized_bpy_stage(source_build)
    if str(bpy_stage.get("status") or "").strip() != "external-bpy-smoke-ready":
        return False

    boundary = source_build.get("executable_boundary") if isinstance(source_build.get("executable_boundary"), dict) else {}
    extract_merge = boundary.get("extract_merge") if isinstance(boundary.get("extract_merge"), dict) else {}
    if not extract_merge:
        return False
    if not bool(extract_merge.get("enabled")) or not bool(extract_merge.get("ready")):
        return False

    proof_kind = str(extract_merge.get("proof_kind") or extract_merge.get("mode") or "").strip()
    optional_owner = str(extract_merge.get("optional_owner") or "").strip()
    if proof_kind != "blender-subprocess" and optional_owner != "blender-subprocess":
        return False

    supported_stages = extract_merge.get("supported_stages")
    if not isinstance(supported_stages, list):
        supported_stages = list(LINUX_ARM64_PARTIAL_RUNTIME_STAGE_NAMES)
    normalized_stages = {str(stage).strip() for stage in supported_stages if str(stage).strip()}
    return set(LINUX_ARM64_PARTIAL_RUNTIME_STAGE_NAMES).issubset(normalized_stages) or set(
        LINUX_ARM64_PERSISTED_STAGE_PROOF_NAMES
    ).issubset(normalized_stages)


def _linux_arm64_can_recover_partial_runtime(state: dict[str, Any]) -> bool:
    source_build = state.get("source_build") if isinstance(state.get("source_build"), dict) else {}
    if str(source_build.get("host_class") or "").strip() != "linux-arm64":
        return False
    if _linux_arm64_has_useful_partial_runtime(source_build):
        return False
    bpy_stage = _linux_arm64_normalized_bpy_stage(source_build)
    return str(bpy_stage.get("status") or "").strip() == "external-bpy-smoke-ready"


def _linux_arm64_stage_result_payloads(runtime_root: Path) -> list[dict[str, Any]]:
    runs_root = runtime_root / "runs"
    if not runs_root.exists():
        return []

    payloads: list[dict[str, Any]] = []
    for result_path in sorted(runs_root.glob("run-*/result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payloads.append(payload)
    return payloads


def _linux_arm64_persisted_stage_proofs(root: Path, runtime_root: Path) -> dict[str, Any]:
    proofs: dict[str, dict[str, Any]] = {}
    for payload in _linux_arm64_stage_result_payloads(runtime_root):
        stage = str(payload.get("stage") or "").strip()
        status = str(payload.get("status") or "").strip()
        produced = payload.get("produced") if isinstance(payload.get("produced"), list) else []
        if stage not in {"extract-prepare", "extract-skin", "skin"}:
            continue
        if status != "ok":
            continue
        existing_outputs = [str(item) for item in produced if str(item).strip() and Path(str(item)).exists()]
        if not existing_outputs:
            continue
        proofs[stage] = {"status": status, "outputs": existing_outputs}

    logs_root = runtime_root / "logs"
    runs_root = runtime_root / "runs"
    for log_path in sorted(logs_root.glob("run-*/skeleton.log")):
        run_name = log_path.parent.name
        output_path = runs_root / run_name / "skeleton_stage.fbx"
        if not output_path.exists():
            continue
        proofs["skeleton"] = {"status": "ok", "outputs": [str(output_path)]}
        break

    supported_stages = [stage for stage in LINUX_ARM64_PERSISTED_STAGE_PROOF_NAMES if stage in proofs]
    return {
        "ready": set(LINUX_ARM64_PERSISTED_STAGE_PROOF_NAMES).issubset(supported_stages),
        "supported_stages": supported_stages,
        "stages": proofs,
        "source": str(root / ".unirig-runtime" / "runs"),
    }


def _linux_arm64_live_non_blender_runtime_probe(venv_python: Path) -> dict[str, Any]:
    if not venv_python.exists():
        return {
            "ready": False,
            "checks": [],
            "message": f"missing runtime python: {venv_python}",
        }

    checks: list[dict[str, Any]] = []
    for module_name in LINUX_ARM64_LIVE_IMPORT_MODULES:
        command = [str(venv_python), "-c", f"import importlib; importlib.import_module({module_name!r})"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "ready": False,
                "checks": checks,
                "message": f"live import probe failed for {module_name}: {exc}",
            }
        checks.append(
            {
                "module": module_name,
                "returncode": int(result.returncode),
                "stdout_tail": str(result.stdout).splitlines()[-5:],
                "stderr_tail": str(result.stderr).splitlines()[-5:],
            }
        )
        if result.returncode != 0:
            return {
                "ready": False,
                "checks": checks,
                "message": f"live import probe failed for {module_name}",
            }

    return {"ready": True, "checks": checks, "message": ""}


def _recover_linux_arm64_partial_runtime_state(state: dict[str, Any], root: Path) -> dict[str, Any] | None:
    if not _linux_arm64_can_recover_partial_runtime(state):
        return None

    runtime_paths = dict(state.get("runtime_paths") or _default_runtime_paths(root, state))
    runtime_root = Path(runtime_paths["runtime_root"])
    venv_python = Path(runtime_paths["venv_python"])
    persisted_proofs = _linux_arm64_persisted_stage_proofs(root, runtime_root)
    if not persisted_proofs["ready"]:
        return None

    live_probe = _linux_arm64_live_non_blender_runtime_probe(venv_python)
    if not live_probe["ready"]:
        return None

    recovered = _copy_json_value(state)
    source_build = _copy_json_value(recovered.get("source_build") or {}) if isinstance(recovered.get("source_build"), dict) else {}
    stages = _copy_json_value(source_build.get("stages") or {}) if isinstance(source_build.get("stages"), dict) else {}
    stages.setdefault("baseline", {"status": "ready", "ready": True, "blockers": [], "blocker_codes": []})
    stages["pyg"] = {
        "status": "ready",
        "ready": True,
        "verification": "live-import-smoke",
        "blockers": [],
        "blocker_codes": [],
        "checks": [check for check in live_probe["checks"] if check.get("module") in {"torch_scatter", "torch_cluster"}],
    }
    stages["spconv"] = {
        "status": "ready",
        "ready": True,
        "verification": "live-import-smoke",
        "blockers": [],
        "blocker_codes": [],
        "checks": [check for check in live_probe["checks"] if check.get("module") == "spconv.pytorch"],
    }
    source_build["stages"] = stages
    source_build["non_blender_runtime_ready"] = True
    source_build["status"] = "partial"

    executable_boundary = _copy_json_value(source_build.get("executable_boundary") or {}) if isinstance(source_build.get("executable_boundary"), dict) else {}
    extract_merge = _copy_json_value(executable_boundary.get("extract_merge") or {}) if isinstance(executable_boundary.get("extract_merge"), dict) else {}
    extract_merge["enabled"] = True
    extract_merge["ready"] = True
    extract_merge["status"] = "verified"
    extract_merge["proof_kind"] = "blender-subprocess"
    extract_merge.setdefault("default_owner", "context.venv_python")
    extract_merge.setdefault("optional_owner", "blender-subprocess")
    extract_merge["supported_stages"] = list(LINUX_ARM64_RECOVERED_STAGE_PROOF_NAMES)
    extract_merge["recovered_from_persisted_stage_proofs"] = True
    extract_merge["recovered_proof_source"] = persisted_proofs["source"]
    executable_boundary["extract_merge"] = extract_merge
    source_build["executable_boundary"] = executable_boundary
    source_build["blocked_reasons"] = []
    source_build["blockers"] = []
    source_build["deferred_work"] = [
        str(item).strip() for item in source_build.get("deferred_work") or [] if str(item).strip() != "bpy-portability"
    ]
    recovered["source_build"] = source_build
    recovered["install_state"] = "partial"
    return recovered


def _should_preserve_source_build_for_storage(state: dict[str, Any], source_build: dict[str, Any]) -> bool:
    if not source_build:
        return False
    host_class = str(
        source_build.get("host_class")
        or (state.get("planner") or {}).get("host_class")
        or (state.get("preflight") or {}).get("host_class")
        or ""
    ).strip()
    if host_class != "linux-arm64":
        return False
    return any(
        key in source_build
        for key in (
            "baseline",
            "stages",
            "blocked_reasons",
            "non_blender_runtime_ready",
            "external_blender",
            "executable_boundary",
            "qualification",
        )
    )


def normalize_state(state: dict[str, Any], root: Path | None = None, *, include_runtime_fields: bool = True) -> dict[str, Any]:
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
    source_build_state = (
        _linux_arm64_legacy_spconv_stage_fallback(_copy_json_value(state.get("source_build") or {}))
        if isinstance(state.get("source_build"), dict)
        else {}
    )
    if _linux_arm64_has_useful_partial_runtime(source_build_state):
        install_state = "partial"
    elif install_state in {"ready", "partial"} and _linux_arm64_requires_full_runtime_block(source_build_state):
        install_state = "blocked"
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
    if install_state != "ready":
        normalized["last_verification"]["runtime_ready"] = False
    checked_at = last_verification.get("checked_at") or (state.get("preflight") or {}).get("checked_at")
    if checked_at:
        normalized["last_verification"]["checked_at"] = str(checked_at)
    planner = state.get("planner")
    if isinstance(planner, dict):
        normalized["planner"] = dict(planner)

    preflight = state.get("preflight")
    if isinstance(preflight, dict):
        normalized["preflight"] = dict(preflight)

    install_plan = state.get("install_plan")
    if isinstance(install_plan, dict):
        normalized["install_plan"] = dict(install_plan)

    deferred_work = state.get("deferred_work")
    if isinstance(deferred_work, list):
        normalized["deferred_work"] = [str(item) for item in deferred_work]
    source_build = state.get("source_build")
    if isinstance(source_build, dict) and _should_preserve_source_build_for_storage(state, source_build):
        normalized["source_build"] = _copy_json_value(source_build)
    if include_runtime_fields:
        normalized["platform_policy"] = _normalized_platform_policy(normalized, host)
        normalized["source_build"] = _normalized_source_build(normalized)
    return normalized


def _readiness_failure_message(state: dict[str, Any]) -> str:
    verification = dict(state.get("last_verification") or {})
    install_state = str(state.get("install_state") or "unknown")
    errors = [str(item) for item in verification.get("errors") or [] if str(item).strip()]
    source_build = dict(state.get("source_build") or {})
    platform_policy = dict(state.get("platform_policy") or {})
    details: list[str] = []

    host_class = str(
        source_build.get("host_class")
        or ((platform_policy.get("selected") or {}).get("key") or "")
    ).strip()
    if host_class:
        details.append(f"host={host_class}")

    install_mode = str(
        source_build.get("mode")
        or ((platform_policy.get("selected") or {}).get("install_mode") or "")
    ).strip()
    if install_mode:
        details.append(f"mode={install_mode}")

    source_build_status = str(source_build.get("status") or "").strip()
    if source_build_status and source_build_status != install_state:
        details.append(f"staged={source_build_status}")

    blocker_details = []
    for blocker in source_build.get("blockers") or []:
        if not isinstance(blocker, dict):
            continue
        code = str(blocker.get("code") or "unknown-blocker").strip()
        category = str(blocker.get("category") or "unknown").strip()
        dependency = str(blocker.get("dependency") or "").strip()
        action = str(blocker.get("action") or "").strip()
        message = str(blocker.get("message") or "").strip()
        summary = f"{code} [{category}]"
        if dependency:
            summary += f" dependency={dependency}"
        if action:
            summary += f" action={action}"
        if message:
            summary += f": {message}"
        blocker_details.append(summary)

    deferred_work = [str(item).strip() for item in source_build.get("deferred_work") or [] if str(item).strip()]
    if errors:
        message = f"UniRig runtime is {install_state}"
        if details:
            message += f" ({', '.join(details)})"
        message += ": " + " | ".join(errors)
        if blocker_details:
            message += " | blockers: " + " ; ".join(blocker_details)
        if deferred_work:
            message += " | deferred work: " + ", ".join(deferred_work)
        return message + ". Run setup.py again to repair the extension."
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
    if state.get("install_state") not in {"ready", "partial"}:
        recovered = _recover_linux_arm64_partial_runtime_state(state, root)
        if recovered is not None:
            save_state(recovered, root)
            state = load_state(root)
    if state.get("install_state") not in {"ready", "partial"}:
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
        platform_policy=_copy_json_value(state.get("platform_policy") or resolve_platform_policy(host.get("os"), host.get("arch"))),
        source_build=_copy_json_value(state.get("source_build") or {}),
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
