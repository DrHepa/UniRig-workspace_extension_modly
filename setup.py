from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext import blender_bridge, bootstrap


LOGGER = logging.getLogger("unirig.setup")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYG_INDEX_URL = "https://data.pyg.org/whl/torch-2.7.0+cu128.html"
TORCH_PACKAGES = ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]
PYG_PACKAGES = ["torch_scatter==2.1.2+pt27cu128", "torch_cluster==1.6.3+pt27cu128"]
LINUX_ARM64_PYG_STAGE_PACKAGES = ["torch_scatter", "torch_cluster"]
LINUX_ARM64_SPCONV_STAGE_PACKAGES = ["cumm", "spconv"]
LINUX_ARM64_PCCM_PACKAGE = "pccm==0.4.16"
LINUX_ARM64_SPCONV_BUILD_PREREQUISITES = ["ccimport", "pybind11", "fire"]
LINUX_ARM64_CUMM_SOURCE_URL = "https://github.com/FindDefinition/cumm/archive/refs/tags/v0.7.11.tar.gz"
LINUX_ARM64_SPCONV_SOURCE_URL = "https://github.com/traveller59/spconv/archive/refs/tags/v2.3.8.tar.gz"
LINUX_ARM64_CUMM_MAX_KNOWN_CUDA_ARCH = (9, 0)
LINUX_ARM64_CUMM_FALLBACK_CUDA_ARCH_LIST = "9.0+PTX"
LINUX_ARM64_CUMM_PATCH_MARKER = "# UNIRIG linux-arm64 cumm cuda discovery patch"
LINUX_ARM64_PREMATURE_REQUIREMENTS_PREFIXES = ("bpy", "flash_attn", "open3d")
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
RUNTIME_MODEL_PARSE_RELATIVE_PATH = Path("src/model/parse.py")
RUNTIME_MODEL_PARSE_LAZY_IMPORT_PATCH_MARKER = "# UNIRIG runtime patch: lazy model imports"
RUNTIME_UNIRIG_SKIN_RELATIVE_PATH = Path("src/model/unirig_skin.py")
RUNTIME_UNIRIG_SKIN_FLASH_ATTN_PATCH_MARKER = "# UNIRIG runtime patch: flash_attn fallback"
LINUX_ARM64_CUDA_12_8_HOME = Path("/usr/local/cuda-12.8")
LINUX_ARM64_SPCONV_STRATEGY = "linux-arm64-guarded-source-build"
LINUX_ARM64_SPCONV_REASON_CODE = "spconv-guarded-source-build"
LINUX_ARM64_SPCONV_ALLOWED_STATUSES = ["blocked", "deferred", "build-ready", "ready"]
LINUX_ARM64_BLENDER_OVERRIDE_KEY = "blender_exe"
LINUX_ARM64_BLENDER_OVERRIDE_ENV = "MODLY_UNIRIG_BLENDER_EXE"
LINUX_ARM64_BLENDER_DEFAULT_NAMES = ("blender",)
LINUX_ARM64_BLENDER_PROBE_MARKER = "UNIRIG_BLENDER_PROBE_RESULT="
LINUX_ARM64_BLENDER_PROBE_TAIL_LINES = 20
LINUX_ARM64_EXTRACT_MERGE_BOUNDARY_STAGES = ("extract-prepare", "extract-skin", "merge")
LINUX_ARM64_PERSISTED_STAGE_PROOF_STAGES = ("extract-prepare", "extract-skin", "skin")
LINUX_ARM64_QUALIFICATION_EXECUTION_MODES = ("wrapper", "seam", "forced-fallback")
LINUX_ARM64_QUALIFICATION_FIXTURE_CLASSES = (
    "known-good",
    "normalization-sensitive",
    "realistic",
    "intentionally-bad",
)
LINUX_ARM64_QUALIFICATION_VERDICTS = (
    "not-ready",
    "candidate-with-known-risks",
    "ready-for-separate-defaulting-change",
)
LINUX_ARM64_BPY_EVIDENCE_CLASSES = (
    "missing",
    "discovered-incompatible",
    "external-bpy-smoke-ready",
    "error",
)
LINUX_ARM64_BLENDER_PROBE_PYTHON_EXPR = (
    "import json, platform; "
    "result = {'blender_version': getattr(__import__('bpy').app, 'version_string', ''), "
    "'python_version': platform.python_version(), 'smoke_result': 'passed'}; "
    f"print({LINUX_ARM64_BLENDER_PROBE_MARKER!r} + json.dumps(result, sort_keys=True))"
)
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
LINUX_ARM64_PYG_IMPORT_SMOKE_CHECKS = (
    {
        "label": "torch_scatter",
        "code": 'import importlib; importlib.import_module("torch_scatter")',
    },
    {
        "label": "torch_cluster",
        "code": 'import importlib; importlib.import_module("torch_cluster")',
    },
)
LINUX_ARM64_SPCONV_IMPORT_SMOKE_CHECK = {
    "label": "spconv.pytorch",
    "code": 'import importlib; importlib.import_module("spconv.pytorch")',
}


def _linux_arm64_spconv_guarded_bringup_message() -> str:
    return (
        "spconv on Linux ARM64 remains a guarded bringup path with explicit cumm coupling risk; "
        "the wrapper may only record blocked, deferred, build-ready, or ready evidence, and the platform stays experimental and unvalidated."
    )


def _linux_arm64_spconv_missing_distribution_message() -> str:
    return (
        "spconv has no validated Linux ARM64 distribution in the current wrapper contract, so setup stops before any guarded cumm/spconv bringup attempt and keeps the platform experimental and unvalidated."
    )


def _resolve_linux_arm64_blender_candidate(payload: dict) -> dict[str, object]:
    override = str(payload.get(LINUX_ARM64_BLENDER_OVERRIDE_KEY) or os.environ.get(LINUX_ARM64_BLENDER_OVERRIDE_ENV, "")).strip()
    if override:
        override_path = Path(override).expanduser()
        if not override_path.exists():
            raise RuntimeError(
                f"Configured Linux ARM64 Blender executable does not exist: {override}. "
                f"Provide a valid {LINUX_ARM64_BLENDER_OVERRIDE_KEY} override or unset {LINUX_ARM64_BLENDER_OVERRIDE_ENV}."
            )
        return {
            "source": "override",
            "path": str(override_path.resolve()),
            "selected_because": (
                f"Selected explicit Blender override from {LINUX_ARM64_BLENDER_OVERRIDE_KEY}/{LINUX_ARM64_BLENDER_OVERRIDE_ENV} "
                "before PATH/default-name fallback."
            ),
        }

    for name in LINUX_ARM64_BLENDER_DEFAULT_NAMES:
        resolved = shutil.which(name)
        if not resolved:
            continue
        return {
            "source": "path",
            "path": resolved,
            "selected_because": f"Selected first PATH-visible Blender candidate from default executable name '{name}'.",
        }

    return {
        "source": "missing",
        "path": "",
        "selected_because": (
            "No explicit Blender override was configured and no PATH-visible Blender candidate matched the default executable names."
        ),
    }


def _probe_linux_arm64_blender_bpy(blender_executable: Path) -> dict[str, object]:
    command = [
        str(blender_executable),
        "--background",
        "--factory-startup",
        "--python-expr",
        LINUX_ARM64_BLENDER_PROBE_PYTHON_EXPR,
    ]

    def tail_lines(text: str | None) -> list[str]:
        return [line for line in (text or "").strip().splitlines()[-LINUX_ARM64_BLENDER_PROBE_TAIL_LINES:] if line]

    def parse_probe_payload(stdout_tail: list[str]) -> dict[str, str]:
        for line in reversed(stdout_tail):
            if not line.startswith(LINUX_ARM64_BLENDER_PROBE_MARKER):
                continue
            raw_payload = line[len(LINUX_ARM64_BLENDER_PROBE_MARKER) :].strip()
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                return {}
            if not isinstance(payload, dict):
                return {}
            return {
                "blender_version": str(payload.get("blender_version") or "").strip(),
                "python_version": str(payload.get("python_version") or "").strip(),
                "smoke_result": str(payload.get("smoke_result") or "").strip(),
            }
        return {}

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {
            "status": "error",
            "command": command,
            "blender_version": "",
            "python_version": "",
            "smoke_result": "launch-failed",
            "returncode": None,
            "stdout_tail": [],
            "stderr_tail": [str(exc)],
        }

    stdout_tail = tail_lines(result.stdout)
    stderr_tail = tail_lines(result.stderr)
    parsed_payload = parse_probe_payload(stdout_tail)
    smoke_result = parsed_payload.get("smoke_result") or ("passed" if result.returncode == 0 else "failed")
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "command": command,
        "blender_version": parsed_payload.get("blender_version", ""),
        "python_version": parsed_payload.get("python_version", ""),
        "smoke_result": smoke_result,
        "returncode": int(result.returncode),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def _classify_linux_arm64_bpy_viability(
    candidate: dict[str, object],
    probe: dict[str, object] | None,
    *,
    wrapper_python_version: str | None = None,
) -> dict[str, object]:
    resolved_candidate = candidate if isinstance(candidate, dict) else {}
    resolved_probe = probe if isinstance(probe, dict) else {}
    required_python = str(wrapper_python_version or "").strip()
    required_python_family = _linux_arm64_python_family(required_python)
    candidate_source = str(resolved_candidate.get("source") or "").strip()
    candidate_path = str(resolved_candidate.get("path") or "").strip()
    blender_version = str(resolved_probe.get("blender_version") or "").strip()
    blender_python_version = str(resolved_probe.get("python_version") or "").strip()

    payload = {
        "status": "error",
        "ready": False,
        "evidence_kind": "external-blender",
        "verification": "blender-background-bpy-smoke",
        "candidate": {
            "source": candidate_source,
            "path": candidate_path,
            "selected_because": str(resolved_candidate.get("selected_because") or "").strip(),
        },
        "blender": {
            "version": blender_version,
            "python_version": blender_python_version,
        },
        "probe": {
            "status": str(resolved_probe.get("status") or "").strip(),
            "command": list(resolved_probe.get("command") or []),
            "smoke_result": str(resolved_probe.get("smoke_result") or "").strip(),
            "returncode": resolved_probe.get("returncode"),
            "stdout_tail": list(resolved_probe.get("stdout_tail") or []),
            "stderr_tail": list(resolved_probe.get("stderr_tail") or []),
        },
        "checks": [],
        "blockers": [],
        "blocker_codes": [],
        "boundary": "wrapper",
        "owner": "wrapper",
    }

    def finalize(status: str, *, boundary: str, owner: str, blocker: dict[str, object] | None = None) -> dict[str, object]:
        payload["status"] = status
        payload["boundary"] = boundary
        payload["owner"] = owner
        payload["checks"] = [
            {
                "label": "bpy",
                "status": status,
                "verification": "blender-background-bpy-smoke",
                "candidate_source": candidate_source,
            }
        ]
        if blocker is None:
            payload["blockers"] = []
            payload["blocker_codes"] = []
            return payload
        payload["blockers"] = [blocker]
        payload["blocker_codes"] = [str(blocker["code"])]
        return payload

    if candidate_source == "missing" or not candidate_path:
        blocker = _linux_arm64_blocker(
            "discovery",
            "blender-executable-missing",
            "Linux ARM64 could not discover an external Blender executable, so bpy viability remains missing external evidence rather than wrapper readiness.",
            boundary="environment",
            owner="environment",
            dependency="bpy",
            candidate=payload["candidate"],
        )
        return finalize("missing", boundary="environment", owner="environment", blocker=blocker)

    if str(resolved_probe.get("status") or "").strip() != "ok":
        detail_lines = payload["probe"]["stderr_tail"] or payload["probe"]["stdout_tail"] or [
            "Linux ARM64 Blender bpy smoke probe failed before compatibility could be classified."
        ]
        blocker = _linux_arm64_blocker(
            "probe",
            "blender-bpy-smoke-error",
            "Linux ARM64 Blender bpy smoke evidence could not be collected, so wrapper readiness stays blocked and external evidence is inconclusive.",
            boundary="wrapper",
            owner="wrapper",
            dependency="bpy",
            candidate=payload["candidate"],
            details=detail_lines,
        )
        return finalize("error", boundary="wrapper", owner="wrapper", blocker=blocker)

    if not blender_python_version.startswith(f"{required_python_family}."):
        blocker = _linux_arm64_blocker(
            "compatibility",
            "blender-python-abi-mismatch",
            "Linux ARM64 discovered Blender external bpy evidence, but Blender Python "
            f"{blender_python_version or 'unknown'} does not match the wrapper Python {required_python_family} expectation.",
            boundary="upstream",
            owner="upstream-package",
            dependency="bpy",
            candidate=payload["candidate"],
            observed={
                "blender_version": blender_version,
                "blender_python_version": blender_python_version,
                "wrapper_python_version": required_python,
            },
        )
        return finalize("discovered-incompatible", boundary="upstream", owner="upstream-package", blocker=blocker)

    return finalize("external-bpy-smoke-ready", boundary="wrapper", owner="wrapper")


def _linux_arm64_bpy_evidence(
    payload: dict[str, object] | None,
    *,
    wrapper_python_version: str | None = None,
) -> dict[str, object]:
    candidate = _resolve_linux_arm64_blender_candidate(dict(payload or {}))
    probe: dict[str, object] = {}
    candidate_path = str(candidate.get("path") or "").strip()
    if candidate_path:
        probe = _probe_linux_arm64_blender_bpy(Path(candidate_path))
    classification = _classify_linux_arm64_bpy_viability(
        candidate,
        probe if probe else None,
        wrapper_python_version=wrapper_python_version,
    )
    return {
        "candidate": _copy_jsonish_dict(candidate),
        "probe": _copy_jsonish_dict(probe),
        "classification": _copy_jsonish_dict(classification),
    }


def _linux_arm64_bpy_stage_payload(evidence: dict[str, object] | None) -> dict[str, object]:
    classification = _copy_jsonish_dict((evidence or {}).get("classification") if isinstance((evidence or {}).get("classification"), dict) else {})
    if classification:
        classification["ready"] = False
        return classification
    return {}


def _linux_arm64_extract_merge_boundary_payload(source_build: dict[str, object] | None) -> dict[str, object]:
    payload = _copy_jsonish_dict(source_build)
    boundary = _copy_jsonish_dict(payload.get("executable_boundary") if isinstance(payload.get("executable_boundary"), dict) else {})
    extract_merge = _copy_jsonish_dict(boundary.get("extract_merge") if isinstance(boundary.get("extract_merge"), dict) else {})
    external_blender = _copy_jsonish_dict(payload.get("external_blender") if isinstance(payload.get("external_blender"), dict) else {})
    classification = _copy_jsonish_dict(
        external_blender.get("classification") if isinstance(external_blender.get("classification"), dict) else {}
    )

    candidate = _copy_jsonish_dict(external_blender.get("candidate") if isinstance(external_blender.get("candidate"), dict) else {})
    if not candidate:
        candidate = _copy_jsonish_dict(classification.get("candidate") if isinstance(classification.get("candidate"), dict) else {})

    extract_merge["enabled"] = bool(extract_merge.get("enabled", False))
    extract_merge["ready"] = bool(extract_merge.get("ready", False))
    extract_merge.setdefault("status", "verified" if extract_merge["ready"] else "missing")
    extract_merge.setdefault("default_owner", "context.venv_python")
    extract_merge.setdefault("optional_owner", "blender-subprocess")
    extract_merge.setdefault("supported_stages", list(LINUX_ARM64_EXTRACT_MERGE_BOUNDARY_STAGES))
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

    boundary["extract_merge"] = extract_merge
    return boundary


def _linux_arm64_runtime_stage_result_payloads(runtime_root: Path) -> list[tuple[Path, dict[str, object]]]:
    payloads: list[tuple[Path, dict[str, object]]] = []
    runs_root = runtime_root / "runs"
    if not runs_root.exists():
        return payloads

    for result_path in sorted(runs_root.glob("run-*/result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payloads.append((result_path, payload))
    return payloads


def _linux_arm64_runtime_has_persisted_partial_proofs(runtime_root: Path) -> bool:
    stages: set[str] = set()
    for _, payload in _linux_arm64_runtime_stage_result_payloads(runtime_root):
        stage = str(payload.get("stage") or "").strip()
        status = str(payload.get("status") or "").strip()
        if stage in LINUX_ARM64_PERSISTED_STAGE_PROOF_STAGES and status == "ok":
            stages.add(stage)
    if not set(LINUX_ARM64_PERSISTED_STAGE_PROOF_STAGES).issubset(stages):
        return False

    runs_root = runtime_root / "runs"
    logs_root = runtime_root / "logs"
    has_skeleton_output = any((runs_root / path.name / "skeleton_stage.fbx").exists() for path in logs_root.glob("run-*"))
    has_skeleton_log = any(path.exists() for path in logs_root.glob("run-*/skeleton.log"))
    return has_skeleton_output and has_skeleton_log


def _linux_arm64_seed_partial_runtime_proofs(ext_dir: Path) -> dict[str, object]:
    runtime_root = _runtime_root(ext_dir)
    if _linux_arm64_runtime_has_persisted_partial_proofs(runtime_root):
        return {"seeded": False, "source": "current-runtime", "reason": "proofs-already-present"}

    backup_roots = sorted(
        [path for path in ext_dir.glob(".unirig-runtime.backup-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for backup_root in backup_roots:
        if not _linux_arm64_runtime_has_persisted_partial_proofs(backup_root):
            continue

        copied_runs: list[str] = []
        copied_logs: list[str] = []
        for result_path, payload in _linux_arm64_runtime_stage_result_payloads(backup_root):
            stage = str(payload.get("stage") or "").strip()
            status = str(payload.get("status") or "").strip()
            if stage not in LINUX_ARM64_PERSISTED_STAGE_PROOF_STAGES or status != "ok":
                continue

            run_name = result_path.parent.name
            source_run_dir = result_path.parent
            target_run_dir = runtime_root / "runs" / run_name
            if not target_run_dir.exists():
                shutil.copytree(source_run_dir, target_run_dir)
                copied_runs.append(run_name)

            source_log_dir = backup_root / "logs" / run_name
            target_log_dir = runtime_root / "logs" / run_name
            skeleton_log = source_log_dir / "skeleton.log"
            if skeleton_log.exists() and not (target_log_dir / "skeleton.log").exists():
                target_log_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(skeleton_log, target_log_dir / "skeleton.log")
                copied_logs.append(run_name)

        if _linux_arm64_runtime_has_persisted_partial_proofs(runtime_root):
            return {
                "seeded": True,
                "source": str(backup_root),
                "runs": copied_runs,
                "logs": copied_logs,
            }

    return {"seeded": False, "source": "", "reason": "no-compatible-backup-proofs"}


def _linux_arm64_qualification_fixture_declarations() -> list[dict[str, object]]:
    fixtures: list[dict[str, object]] = []
    for stage in LINUX_ARM64_EXTRACT_MERGE_BOUNDARY_STAGES:
        validated_stage = blender_bridge.validate_stage_name(stage)
        for fixture_class in LINUX_ARM64_QUALIFICATION_FIXTURE_CLASSES:
            fixtures.append(
                {
                    "fixture_id": f"{validated_stage}-{fixture_class}",
                    "stage": validated_stage,
                    "fixture_class": fixture_class,
                    "execution_modes": list(LINUX_ARM64_QUALIFICATION_EXECUTION_MODES),
                    "comparison_expected": True,
                }
            )
    return fixtures


def _linux_arm64_qualification_evidence_record(
    *,
    fixture: dict[str, object],
    run_label: str,
    selected_mode: str,
    status: str,
    failure_code: str = "",
    host_facts: dict[str, object] | None = None,
    blender_facts: dict[str, object] | None = None,
    outputs: dict[str, object] | None = None,
    logs: dict[str, object] | None = None,
) -> dict[str, object]:
    resolved_fixture = _copy_jsonish_dict(fixture)
    stage = blender_bridge.validate_stage_name(str(resolved_fixture.get("stage") or ""))
    mode = str(selected_mode).strip()
    if mode not in LINUX_ARM64_QUALIFICATION_EXECUTION_MODES:
        raise ValueError(
            "Unsupported Linux ARM64 qualification mode "
            f"{selected_mode!r}. Expected one of: {', '.join(LINUX_ARM64_QUALIFICATION_EXECUTION_MODES)}."
        )
    normalized_status = str(status).strip()
    if normalized_status not in {"passed", "failed"}:
        raise ValueError(f"Linux ARM64 qualification status must be 'passed' or 'failed', got {status!r}.")

    normalized_failure_code = ""
    if normalized_status == "failed":
        normalized_failure_code = blender_bridge.validate_qualification_failure_code(failure_code)
    elif str(failure_code).strip():
        raise ValueError("Linux ARM64 passed qualification records must not declare a failure_code.")

    return {
        "fixture_id": str(resolved_fixture.get("fixture_id") or "").strip(),
        "fixture_class": str(resolved_fixture.get("fixture_class") or "").strip(),
        "stage": stage,
        "run_label": str(run_label).strip(),
        "selected_mode": mode,
        "status": normalized_status,
        "failure_code": normalized_failure_code,
        "host": _copy_jsonish_dict(host_facts),
        "blender": _copy_jsonish_dict(blender_facts),
        "outputs": _copy_jsonish_dict(outputs),
        "logs": _copy_jsonish_dict(logs),
    }


def _linux_arm64_qualification_comparison_summary(records: list[dict[str, object]] | None) -> dict[str, object]:
    summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "by_failure_code": {},
        "by_mode": {mode: {"passed": 0, "failed": 0} for mode in LINUX_ARM64_QUALIFICATION_EXECUTION_MODES},
        "required_stage_coverage": {stage: False for stage in LINUX_ARM64_EXTRACT_MERGE_BOUNDARY_STAGES},
        "seam_failures": 0,
        "risk_codes": [],
    }
    risk_codes: set[str] = set()

    for record in records or []:
        if not isinstance(record, dict):
            continue
        stage = blender_bridge.validate_stage_name(str(record.get("stage") or ""))
        mode = str(record.get("selected_mode") or "").strip()
        if mode not in LINUX_ARM64_QUALIFICATION_EXECUTION_MODES:
            raise ValueError(
                "Unsupported Linux ARM64 qualification mode "
                f"{mode!r}. Expected one of: {', '.join(LINUX_ARM64_QUALIFICATION_EXECUTION_MODES)}."
            )
        status = str(record.get("status") or "").strip()
        if status not in {"passed", "failed"}:
            raise ValueError(f"Linux ARM64 qualification status must be 'passed' or 'failed', got {status!r}.")

        summary["total"] += 1
        summary[status] += 1
        summary["by_mode"][mode][status] += 1
        if mode == "seam" and status == "passed":
            summary["required_stage_coverage"][stage] = True

        failure_code = str(record.get("failure_code") or "").strip()
        if status == "failed":
            normalized_failure_code = blender_bridge.validate_qualification_failure_code(failure_code)
            summary["by_failure_code"][normalized_failure_code] = summary["by_failure_code"].get(normalized_failure_code, 0) + 1
            if mode == "seam":
                summary["seam_failures"] += 1
            if normalized_failure_code in blender_bridge.QUALIFICATION_COMPARISON_FAILURE_CODES:
                risk_codes.add(normalized_failure_code)
        elif failure_code:
            raise ValueError("Linux ARM64 passed qualification records must not declare a failure_code.")

    summary["risk_codes"] = sorted(risk_codes)
    return summary


def _reduce_linux_arm64_qualification_verdict(summary: dict[str, object] | None) -> str:
    resolved = _copy_jsonish_dict(summary)
    required_stage_coverage = {
        stage: bool((resolved.get("required_stage_coverage") or {}).get(stage, False))
        for stage in LINUX_ARM64_EXTRACT_MERGE_BOUNDARY_STAGES
    }
    if int(resolved.get("seam_failures", 0) or 0) > 0:
        return "not-ready"
    if not all(required_stage_coverage.values()):
        return "not-ready"

    failure_codes = {
        str(code).strip()
        for code in (resolved.get("by_failure_code") or {}).keys()
        if str(code).strip()
    }
    blocking_failure_codes = failure_codes - blender_bridge.QUALIFICATION_COMPARISON_FAILURE_CODES
    if blocking_failure_codes:
        return "not-ready"

    risk_codes = [str(item).strip() for item in resolved.get("risk_codes", []) if str(item).strip()]
    if int(resolved.get("failed", 0) or 0) > 0 or risk_codes:
        return "candidate-with-known-risks"
    return "ready-for-separate-defaulting-change"


def _linux_arm64_qualification_host_facts(host_facts: dict[str, object] | None = None) -> dict[str, object]:
    resolved = _copy_jsonish_dict(host_facts)
    host_os = _normalize_host_os(str(resolved.get("os") or platform.system()))
    host_arch = _normalize_host_arch(str(resolved.get("arch") or platform.machine()))
    facts = {
        "os": host_os,
        "arch": host_arch,
        "platform_tag": str(resolved.get("platform_tag") or f"{host_os}-{host_arch}"),
        "host_class": _classify_host(host_os, host_arch),
    }
    for key, value in resolved.items():
        if key in facts:
            continue
        facts[str(key)] = value
    return facts


def _default_linux_arm64_qualification_fixture_comparison(
    *, baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    baseline_outputs = _copy_jsonish_dict(baseline.get("outputs") if isinstance(baseline.get("outputs"), dict) else {})
    candidate_outputs = _copy_jsonish_dict(candidate.get("outputs") if isinstance(candidate.get("outputs"), dict) else {})
    if not baseline_outputs or not candidate_outputs:
        return {
            "status": "failed",
            "failure_code": "expected-output-missing",
            "details": ["Qualification comparison requires wrapper and candidate outputs for the same fixture."],
        }
    if baseline_outputs == candidate_outputs:
        return {"status": "passed", "failure_code": ""}
    return {
        "status": "failed",
        "failure_code": "output-mismatch",
        "details": ["Qualification comparison observed different outputs between wrapper and optional seam runs."],
    }


def _linux_arm64_qualification_fixture_comparison(
    *,
    fixture: dict[str, object],
    records: list[dict[str, object]],
    compare_fixture_runs=None,
) -> dict[str, object]:
    by_mode = {
        str(record.get("selected_mode") or "").strip(): _copy_jsonish_dict(record)
        for record in records
        if isinstance(record, dict) and str(record.get("selected_mode") or "").strip()
    }
    comparison = {
        "fixture_id": str(fixture.get("fixture_id") or "").strip(),
        "stage": blender_bridge.validate_stage_name(str(fixture.get("stage") or "")),
    }
    baseline = by_mode.get("wrapper")
    for mode, key in (("seam", "wrapper_vs_seam"), ("forced-fallback", "wrapper_vs_forced_fallback")):
        candidate = by_mode.get(mode)
        if baseline is None or candidate is None:
            comparison[key] = {"status": "skipped", "failure_code": "", "compared_modes": ["wrapper", mode]}
            continue
        raw = (
            compare_fixture_runs(fixture=_copy_jsonish_dict(fixture), baseline=_copy_jsonish_dict(baseline), candidate=_copy_jsonish_dict(candidate))
            if compare_fixture_runs is not None
            else _default_linux_arm64_qualification_fixture_comparison(baseline=baseline, candidate=candidate)
        )
        payload = _copy_jsonish_dict(raw if isinstance(raw, dict) else {})
        status = str(payload.get("status") or "passed").strip()
        if status not in {"passed", "failed", "skipped"}:
            raise ValueError(f"Linux ARM64 qualification comparison status must be passed, failed, or skipped, got {status!r}.")
        failure_code = str(payload.get("failure_code") or "").strip()
        if status == "failed":
            failure_code = blender_bridge.validate_qualification_failure_code(failure_code or "output-mismatch")
        else:
            failure_code = ""
        payload["status"] = status
        payload["failure_code"] = failure_code
        payload.setdefault("compared_modes", ["wrapper", mode])
        comparison[key] = payload
    return comparison


def _coordinate_linux_arm64_qualification_runs(
    *,
    fixture_declarations: list[dict[str, object]] | None = None,
    host_facts: dict[str, object] | None = None,
    blender_facts: dict[str, object] | None = None,
    run_fixture=None,
    compare_fixture_runs=None,
) -> dict[str, object]:
    resolved_host = _linux_arm64_qualification_host_facts(host_facts)
    if str(resolved_host.get("host_class") or "") != "linux-arm64":
        return {}
    if run_fixture is None:
        raise ValueError("Linux ARM64 qualification coordination requires a run_fixture callable.")

    resolved_blender = _copy_jsonish_dict(blender_facts)
    declared_fixtures = [
        _copy_jsonish_dict(item)
        for item in (fixture_declarations or _linux_arm64_qualification_fixture_declarations())
        if isinstance(item, dict)
    ]
    records: list[dict[str, object]] = []
    fixture_reports: list[dict[str, object]] = []

    for fixture in declared_fixtures:
        stage = blender_bridge.validate_stage_name(str(fixture.get("stage") or ""))
        execution_modes = fixture.get("execution_modes")
        if not isinstance(execution_modes, list) or not execution_modes:
            execution_modes = list(LINUX_ARM64_QUALIFICATION_EXECUTION_MODES)

        fixture_runs: list[dict[str, object]] = []
        for selected_mode in execution_modes:
            mode = str(selected_mode).strip()
            if mode not in LINUX_ARM64_QUALIFICATION_EXECUTION_MODES:
                raise ValueError(
                    "Unsupported Linux ARM64 qualification mode "
                    f"{selected_mode!r}. Expected one of: {', '.join(LINUX_ARM64_QUALIFICATION_EXECUTION_MODES)}."
                )
            raw_result = run_fixture(
                fixture=_copy_jsonish_dict(fixture),
                selected_mode=mode,
                host_facts=_copy_jsonish_dict(resolved_host),
                blender_facts=_copy_jsonish_dict(resolved_blender),
            )
            run_payload = _copy_jsonish_dict(raw_result if isinstance(raw_result, dict) else {})
            record = _linux_arm64_qualification_evidence_record(
                fixture={
                    "fixture_id": str(fixture.get("fixture_id") or "").strip(),
                    "fixture_class": str(fixture.get("fixture_class") or "").strip(),
                    "stage": stage,
                },
                run_label=str(run_payload.get("run_label") or f"{fixture.get('fixture_id')}-{mode}").strip(),
                selected_mode=mode,
                status=str(run_payload.get("status") or "failed").strip(),
                failure_code=str(run_payload.get("failure_code") or "").strip(),
                host_facts=resolved_host,
                blender_facts=resolved_blender,
                outputs=_copy_jsonish_dict(run_payload.get("outputs") if isinstance(run_payload.get("outputs"), dict) else {}),
                logs=_copy_jsonish_dict(run_payload.get("logs") if isinstance(run_payload.get("logs"), dict) else {}),
            )
            fixture_runs.append(record)
            records.append(record)

        comparison = _linux_arm64_qualification_fixture_comparison(
            fixture={"fixture_id": str(fixture.get("fixture_id") or "").strip(), "stage": stage},
            records=fixture_runs,
            compare_fixture_runs=compare_fixture_runs,
        )
        fixture_report = {
            "fixture_id": str(fixture.get("fixture_id") or "").strip(),
            "fixture_class": str(fixture.get("fixture_class") or "").strip(),
            "stage": stage,
            "execution_modes": [str(mode).strip() for mode in execution_modes if str(mode).strip()],
            "runs": fixture_runs,
            "comparison": comparison,
        }
        fixture_reports.append(fixture_report)

    summary = _linux_arm64_qualification_comparison_summary(records)
    return {
        "schema_version": 1,
        "default_owner": "context.venv_python",
        "optional_owner": "blender-subprocess",
        "host": resolved_host,
        "blender": resolved_blender,
        "fixtures": fixture_reports,
        "records": records,
        "summary": summary,
        "verdict": _reduce_linux_arm64_qualification_verdict(summary),
    }


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
        _patch_runtime_model_parse_lazy_imports(unirig_dir)
        _patch_linux_arm64_float32_attention_config(unirig_dir)
        _patch_linux_arm64_skin_enable_flash(unirig_dir)
        _patch_linux_arm64_skin_runtime_flash_attn_fallback(unirig_dir)
        _apply_windows_runtime_stage_patches(ext_dir, unirig_dir)
        return unirig_dir, stage_descriptor["vendor_source"], source_ref

    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    unirig_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source).expanduser()
    if source_path.exists() and source_path.is_dir():
        _copy_upstream_tree(source_path, unirig_dir)
        _patch_runtime_model_parse_lazy_imports(unirig_dir)
        _patch_linux_arm64_float32_attention_config(unirig_dir)
        _patch_linux_arm64_skin_enable_flash(unirig_dir)
        _patch_linux_arm64_skin_runtime_flash_attn_fallback(unirig_dir)
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
    _patch_runtime_model_parse_lazy_imports(unirig_dir)
    _patch_linux_arm64_float32_attention_config(unirig_dir)
    _patch_linux_arm64_skin_enable_flash(unirig_dir)
    _patch_linux_arm64_skin_runtime_flash_attn_fallback(unirig_dir)
    _apply_windows_runtime_stage_patches(ext_dir, unirig_dir)
    _write_runtime_stage_descriptor(ext_dir, stage_descriptor)
    return unirig_dir, stage_descriptor["vendor_source"], source_ref


def _patch_runtime_model_parse_lazy_imports(unirig_dir: Path) -> dict[str, str]:
    parse_path = unirig_dir / RUNTIME_MODEL_PARSE_RELATIVE_PATH
    entry = {
        "patch": "runtime-model-parse-lazy-imports",
        "path": str(RUNTIME_MODEL_PARSE_RELATIVE_PATH),
        "status": "skipped",
        "reason": (
            "UniRig skeleton prediction imports src.model.parse through run.py even when the skin model is unused. "
            "The upstream parse module eagerly imports unirig_skin, which hard-requires flash_attn and blocks the Linux ARM64 Blender seam before the skeleton model is selected."
        ),
    }
    if not parse_path.exists():
        entry["status"] = "target-missing"
        return entry

    original = parse_path.read_text(encoding="utf-8")
    if RUNTIME_MODEL_PARSE_LAZY_IMPORT_PATCH_MARKER in original:
        entry["status"] = "already-patched"
        return entry

    expected = (
        "from .unirig_ar import UniRigAR\n"
        "from .unirig_skin import UniRigSkin\n\n"
        "from .spec import ModelSpec\n\n"
        "def get_model(**kwargs) -> ModelSpec:\n"
        "    MAP = {\n"
        "        'unirig_ar': UniRigAR,\n"
        "        'unirig_skin': UniRigSkin,\n"
        "    }\n"
        "    __target__ = kwargs['__target__']\n"
        "    del kwargs['__target__']\n"
        "    assert __target__ in MAP, f\"expect: [{','.join(MAP.keys())}], found: {__target__}\"\n"
        "    return MAP[__target__](**kwargs)\n"
    )
    replacement = (
        "from .spec import ModelSpec\n\n"
        f"{RUNTIME_MODEL_PARSE_LAZY_IMPORT_PATCH_MARKER}\n"
        "def _model_class(__target__: str) -> type[ModelSpec]:\n"
        "    if __target__ == 'unirig_ar':\n"
        "        from .unirig_ar import UniRigAR\n\n"
        "        return UniRigAR\n"
        "    if __target__ == 'unirig_skin':\n"
        "        from .unirig_skin import UniRigSkin\n\n"
        "        return UniRigSkin\n"
        "    expected = 'unirig_ar,unirig_skin'\n"
        "    raise AssertionError(f\"expect: [{expected}], found: {__target__}\")\n\n"
        "def get_model(**kwargs) -> ModelSpec:\n"
        "    __target__ = kwargs['__target__']\n"
        "    del kwargs['__target__']\n"
        "    return _model_class(__target__)(**kwargs)\n"
    )
    if original != expected:
        entry["status"] = "no-match"
        return entry

    parse_path.write_text(replacement, encoding="utf-8")
    entry["status"] = "applied"
    return entry


def _patch_linux_arm64_float32_attention_config(unirig_dir: Path) -> dict[str, str]:
    entry = {
        "patch": "linux-arm64-disable-flash-attention-2-float32",
        "path": str(WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH),
        "status": "skipped",
        "reason": (
            "Linux ARM64 Blender seam execution reaches the skeleton model without a validated flash_attn package path. "
            "The staged float32 AR config must use eager attention so skeleton inference can proceed conservatively without claiming full ARM64 support."
        ),
    }
    if _classify_host() != "linux-arm64":
        return entry

    config_path = unirig_dir / WINDOWS_FLASH_ATTN_CONFIG_RELATIVE_PATH
    if not config_path.exists():
        entry["status"] = "target-missing"
        return entry

    original = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*_attn_implementation\s*:\s*)flash_attention_2(\s*(?:#.*)?)$", re.MULTILINE)
    patched, replacements = pattern.subn(rf"\1{WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT}\2", original, count=1)
    if replacements:
        config_path.write_text(patched, encoding="utf-8")
        entry["status"] = "applied"
        entry["from"] = WINDOWS_FLASH_ATTN_CONFIG_SOURCE
        entry["to"] = WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT
        return entry

    if f"_attn_implementation: {WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT}" in original:
        entry["status"] = "already-patched"
        entry["from"] = WINDOWS_FLASH_ATTN_CONFIG_SOURCE
        entry["to"] = WINDOWS_FLASH_ATTN_CONFIG_REPLACEMENT
        return entry

    entry["status"] = "no-match"
    return entry


def _patch_linux_arm64_skin_enable_flash(unirig_dir: Path) -> dict[str, str]:
    entry = {
        "patch": "linux-arm64-disable-skin-mesh-encoder-flash",
        "path": str(WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH),
        "status": "skipped",
        "reason": (
            "Linux ARM64 skin staging reaches PointTransformerV3Object without a validated flash_attn package path. "
            "The staged skin config must force mesh_encoder.enable_flash to False so the seam can use the existing non-flash attention path conservatively."
        ),
        "setting": "mesh_encoder.enable_flash",
        "to": "False",
    }
    if _classify_host() != "linux-arm64":
        return entry

    config_path = unirig_dir / WINDOWS_SKIN_ENABLE_FLASH_CONFIG_RELATIVE_PATH
    if not config_path.exists():
        entry["status"] = "target-missing"
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
            return entry
        line = lines[enable_flash_index]
        comment_match = re.match(r"^(\s*enable_flash\s*:\s*)[^#\n]+?(\s*(?:#.*)?)?$", line)
        suffix = comment_match.group(2) or "" if comment_match else ""
        line_ending = "\r\n" if line.endswith("\r\n") else "\n"
        lines[enable_flash_index] = f"{child_indent}enable_flash: False{suffix}{line_ending}"
        config_path.write_text("".join(lines), encoding="utf-8")
        entry["status"] = "applied"
        entry["from"] = enable_flash_value
        return entry

    insert_at = mesh_encoder_index + 1
    while insert_at < block_end and not lines[insert_at].strip():
        insert_at += 1
    lines.insert(insert_at, new_line)
    config_path.write_text("".join(lines), encoding="utf-8")
    entry["status"] = "applied"
    entry["from"] = "implicit-default"
    return entry


def _patch_linux_arm64_skin_runtime_flash_attn_fallback(unirig_dir: Path) -> dict[str, str]:
    entry = {
        "patch": "linux-arm64-skin-flash-attn-fallback",
        "path": str(RUNTIME_UNIRIG_SKIN_RELATIVE_PATH),
        "status": "skipped",
        "reason": (
            "Linux ARM64 skin staging imports unirig_skin before the mesh encoder config can disable flash attention. "
            "The staged runtime must tolerate a missing flash_attn package and fall back to torch.nn.MultiheadAttention for the cross-attention block."
        ),
    }
    if _classify_host() != "linux-arm64":
        return entry

    skin_path = unirig_dir / RUNTIME_UNIRIG_SKIN_RELATIVE_PATH
    if not skin_path.exists():
        entry["status"] = "target-missing"
        return entry

    original = skin_path.read_text(encoding="utf-8")
    if RUNTIME_UNIRIG_SKIN_FLASH_ATTN_PATCH_MARKER in original:
        entry["status"] = "already-patched"
        return entry

    import_line = "from flash_attn.modules.mha import MHA\n"
    fallback_block = (
        "try:\n"
        "    from flash_attn.modules.mha import MHA\n"
        "except ImportError:\n"
        "    MHA = None\n\n"
        f"{RUNTIME_UNIRIG_SKIN_FLASH_ATTN_PATCH_MARKER}\n"
        "class _FallbackCrossAttention(nn.Module):\n"
        "    def __init__(self, embed_dim: int, num_heads: int):\n"
        "        super().__init__()\n"
        "        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'\n"
        "        self.num_heads = num_heads\n"
        "        self.head_dim = embed_dim // num_heads\n"
        "        self.Wq = nn.Linear(embed_dim, embed_dim)\n"
        "        self.Wkv = nn.Linear(embed_dim, embed_dim * 2)\n"
        "        self.out_proj = nn.Linear(embed_dim, embed_dim)\n\n"
        "    def forward(self, q: Tensor, x_kv: Tensor) -> Tensor:\n"
        "        batch_size, query_tokens, _ = q.shape\n"
        "        key_tokens = x_kv.shape[1]\n"
        "        q_proj = self.Wq(q).reshape(batch_size, query_tokens, self.num_heads, self.head_dim).transpose(1, 2)\n"
        "        kv_proj = self.Wkv(x_kv).reshape(batch_size, key_tokens, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)\n"
        "        k_proj, v_proj = kv_proj[0], kv_proj[1]\n"
        "        attn_output = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, dropout_p=0.0, is_causal=False)\n"
        "        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_tokens, -1)\n"
        "        return self.out_proj(attn_output)\n\n"
        "def _build_cross_attention(*, embed_dim: int, num_heads: int):\n"
        "    if MHA is not None:\n"
        "        return MHA(embed_dim=embed_dim, num_heads=num_heads, cross_attn=True)\n"
        "    return _FallbackCrossAttention(embed_dim=embed_dim, num_heads=num_heads)\n"
    )
    patched = original.replace(import_line, fallback_block, 1)
    if patched == original:
        entry["status"] = "no-match"
        return entry

    attention_line = "        self.attention = MHA(embed_dim=feat_dim, num_heads=num_heads, cross_attn=True)\n"
    replacement_line = "        self.attention = _build_cross_attention(embed_dim=feat_dim, num_heads=num_heads)\n"
    patched = patched.replace(attention_line, replacement_line, 1)
    if patched == original or attention_line in patched:
        entry["status"] = "no-match"
        return entry

    skin_path.write_text(patched, encoding="utf-8")
    entry["status"] = "applied"
    return entry


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


def _dedupe_string_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _resolve_linux_arm64_cuda_home() -> Path | None:
    candidate_roots = [
        Path(os.environ[var_name]).expanduser()
        for var_name in ("CUDA_HOME", "CUDA_PATH")
        if str(os.environ.get(var_name) or "").strip()
    ]
    candidate_roots.extend([LINUX_ARM64_CUDA_12_8_HOME, Path("/usr/local/cuda")])
    for candidate in candidate_roots:
        nvcc_path = candidate / "bin" / "nvcc"
        include_dir = candidate / "include"
        if nvcc_path.exists() and include_dir.exists():
            return candidate
    return None


def _linux_arm64_cuda_include_directories(cuda_home: Path) -> list[Path]:
    return [
        path
        for path in [
            cuda_home / "include",
            cuda_home / "targets" / "sbsa-linux" / "include",
            cuda_home / "targets" / "aarch64-linux" / "include",
        ]
        if path.exists()
    ]


def _linux_arm64_cuda_library_directories(cuda_home: Path) -> list[Path]:
    return [
        path
        for path in [
            cuda_home / "lib64",
            cuda_home / "targets" / "sbsa-linux" / "lib",
            cuda_home / "targets" / "aarch64-linux" / "lib",
            cuda_home / "lib",
        ]
        if path.exists()
    ]


def _linux_arm64_source_build_environment(python_exe: Path | None = None) -> dict[str, str] | None:
    if "cu128" not in TORCH_INDEX_URL:
        return None

    cuda_home = _resolve_linux_arm64_cuda_home()
    if cuda_home is None:
        return None
    nvcc_path = cuda_home / "bin" / "nvcc"
    include_dirs = _linux_arm64_cuda_include_directories(cuda_home)
    lib_dirs = _linux_arm64_cuda_library_directories(cuda_home)
    if not nvcc_path.exists() or not include_dirs:
        return None

    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env["CUDACXX"] = str(nvcc_path)
    env["CUDA_BIN_PATH"] = str(cuda_home)
    env["CUMM_CUDA_VERSION"] = "12.8"
    env["CUMM_DISABLE_JIT"] = "1"
    env["SPCONV_DISABLE_JIT"] = "1"
    env["CUDAFLAGS"] = " ".join(_dedupe_string_items([env.get("CUDAFLAGS", ""), "-allow-unsupported-compiler"]))
    env["CMAKE_CUDA_FLAGS"] = " ".join(
        _dedupe_string_items([env.get("CMAKE_CUDA_FLAGS", ""), "-allow-unsupported-compiler"])
    )

    path_entries = []
    if python_exe is not None:
        python_bin_dir = python_exe.expanduser().parent
        path_entries.append(str(python_bin_dir))
    path_entries.append(str(cuda_home / "bin"))
    if env.get("PATH"):
        path_entries.append(env["PATH"])
    env["PATH"] = os.pathsep.join(path_entries)

    include_entries = [str(path) for path in include_dirs]
    library_entries = [str(path) for path in lib_dirs]
    for env_name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        env[env_name] = os.pathsep.join(_dedupe_string_items([*include_entries, env.get(env_name, "")]))
    for env_name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        env[env_name] = os.pathsep.join(_dedupe_string_items([*library_entries, env.get(env_name, "")]))

    if not str(env.get("CUMM_CUDA_ARCH_LIST") or "").strip():
        detected_capability = _probe_linux_arm64_gpu_compute_capability(python_exe)
        if detected_capability is not None and detected_capability > LINUX_ARM64_CUMM_MAX_KNOWN_CUDA_ARCH:
            env["CUMM_CUDA_ARCH_LIST"] = LINUX_ARM64_CUMM_FALLBACK_CUDA_ARCH_LIST

    return env


def _parse_cuda_compute_capability(raw_value: str | None) -> tuple[int, int] | None:
    value = str(raw_value or "").strip()
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?", value)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or "0")
    return (major, minor)


def _probe_linux_arm64_gpu_compute_capability(python_exe: Path | None) -> tuple[int, int] | None:
    if python_exe is None:
        return None

    try:
        result = subprocess.run(
            [
                str(python_exe),
                "-c",
                (
                    "import torch; "
                    "available = torch.cuda.is_available(); "
                    "capability = torch.cuda.get_device_capability(0) if available else None; "
                    "print('' if capability is None else f'{capability[0]}.{capability[1]}')"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    return _parse_cuda_compute_capability((result.stdout or "").strip())


def _linux_arm64_spconv_source_requirement(package_name: str) -> str:
    normalized = str(package_name or "").strip()
    if normalized == "cumm":
        return LINUX_ARM64_CUMM_SOURCE_URL
    if normalized == "spconv":
        return LINUX_ARM64_SPCONV_SOURCE_URL
    raise RuntimeError(f"Unsupported Linux ARM64 guarded spconv package '{package_name}'.")


def _patched_linux_arm64_cumm_common_source(source: str) -> str:
    if LINUX_ARM64_CUMM_PATCH_MARKER in source:
        return source

    target = """        else:\n            try:\n                nvcc_path = subprocess.check_output([\"which\", \"nvcc\"\n                                                    ]).decode(\"utf-8\").strip()\n                lib = Path(nvcc_path).parent.parent / \"lib\"\n                include = Path(nvcc_path).parent.parent / \"targets/x86_64-linux/include\"\n                if lib.exists() and include.exists():\n                    if (lib / \"libcudart.so\").exists() and (include / \"cuda.h\").exists():\n                        # should be nvidia conda package\n                        _CACHED_CUDA_INCLUDE_LIB = ([include], lib)\n                        return _CACHED_CUDA_INCLUDE_LIB\n            except:\n                pass \n\n            linux_cuda_root = Path(\"/usr/local/cuda\")\n            include = linux_cuda_root / f\"include\"\n            lib64 = linux_cuda_root / f\"lib64\"\n            assert linux_cuda_root.exists(), f\"can't find cuda in {linux_cuda_root} install via cuda installer or conda first.\"\n        _CACHED_CUDA_INCLUDE_LIB = ([include], lib64)\n"""
    replacement = """        else:\n            cuda_home_env = os.getenv(\"CUDA_HOME\", \"\").strip() or os.getenv(\"CUDA_PATH\", \"\").strip()\n            candidate_roots = []\n            if cuda_home_env:\n                candidate_roots.append(Path(cuda_home_env))\n            try:\n                nvcc_path = subprocess.check_output([\"which\", \"nvcc\"\n                                                    ]).decode(\"utf-8\").strip()\n                if nvcc_path:\n                    candidate_roots.append(Path(nvcc_path).parent.parent)\n            except:\n                pass \n\n            candidate_roots.append(Path(\"/usr/local/cuda\"))\n            seen_roots = set()\n            for linux_cuda_root in candidate_roots:\n                resolved_root = Path(linux_cuda_root)\n                root_key = str(resolved_root)\n                if not root_key or root_key in seen_roots:\n                    continue\n                seen_roots.add(root_key)\n                include_candidates = [\n                    resolved_root / \"targets\" / \"sbsa-linux\" / \"include\",\n                    resolved_root / \"targets\" / \"aarch64-linux\" / \"include\",\n                    resolved_root / \"include\",\n                ]\n                lib_candidates = [\n                    resolved_root / \"targets\" / \"sbsa-linux\" / \"lib\",\n                    resolved_root / \"targets\" / \"aarch64-linux\" / \"lib\",\n                    resolved_root / \"lib64\",\n                    resolved_root / \"lib\",\n                ]\n                valid_includes = [path for path in include_candidates if path.exists() and (path / \"cuda.h\").exists()]\n                valid_libs = [path for path in lib_candidates if path.exists() and (path / \"libcudart.so\").exists()]\n                if valid_includes and valid_libs:\n                    _CACHED_CUDA_INCLUDE_LIB = (valid_includes, valid_libs[0])\n                    return _CACHED_CUDA_INCLUDE_LIB\n            linux_cuda_root = Path(cuda_home_env) if cuda_home_env else Path(\"/usr/local/cuda\")\n            include = linux_cuda_root / f\"include\"\n            lib64 = linux_cuda_root / f\"lib64\"\n            assert linux_cuda_root.exists(), f\"can't find cuda in {linux_cuda_root} install via cuda installer or conda first.\"\n        """ + LINUX_ARM64_CUMM_PATCH_MARKER + """\n        _CACHED_CUDA_INCLUDE_LIB = ([include], lib64)\n"""
    if target not in source:
        raise RuntimeError(
            "Installed cumm/common.py no longer matches the expected Linux branch layout for UniRig's guarded ARM64 patch. "
            "Inspect the installed cumm version before rerunning setup."
        )
    return source.replace(target, replacement, 1)


def _locate_installed_linux_arm64_cumm_common(python_exe: Path) -> Path:
    result = subprocess.run(
        [str(python_exe), "-c", "import cumm.common; print(cumm.common.__file__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Linux ARM64 guarded bringup installed cumm but could not locate cumm.common for the wrapper patch. "
            f"Details: {(result.stderr or result.stdout or 'cumm.common probe failed').strip()}"
        )
    common_path = Path((result.stdout or "").strip())
    if not common_path.exists():
        raise RuntimeError(
            "Linux ARM64 guarded bringup located a non-existent cumm.common path. "
            f"Reported path: {common_path}"
        )
    return common_path


def _patch_linux_arm64_installed_cumm_common(python_exe: Path) -> Path:
    common_path = _locate_installed_linux_arm64_cumm_common(python_exe)
    original_source = common_path.read_text(encoding="utf-8")
    patched_source = _patched_linux_arm64_cumm_common_source(original_source)
    if patched_source != original_source:
        common_path.write_text(patched_source, encoding="utf-8")
    return common_path


def _normalize_host_os(host_os: str) -> str:
    return str(host_os or "").strip().lower()


def _normalize_host_arch(host_arch: str) -> str:
    normalized = str(host_arch or "").strip().lower()
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "aarch64",
    }
    return aliases.get(normalized, normalized)


def _classify_host(host_os: str | None = None, host_arch: str | None = None) -> str:
    normalized_os = _normalize_host_os(platform.system() if host_os is None else host_os)
    normalized_arch = _normalize_host_arch(platform.machine() if host_arch is None else host_arch)

    if normalized_os == "windows" and normalized_arch == "x86_64":
        return "windows-x86_64"
    if normalized_os == "linux" and normalized_arch == "x86_64":
        return "linux-x86_64"
    if normalized_os == "linux" and normalized_arch == "aarch64":
        return "linux-arm64"
    return "unsupported"


def _windows_validated_runtime_contract() -> dict[str, object]:
    return {
        "install_mode": "pinned-prebuilt",
        "profile": "pinned-upstream-wrapper",
        "cumm_package": _windows_cumm_package(),
        "spconv_package": _windows_spconv_package(),
        "triton_package": _windows_triton_package(),
        "flash_attn_wheel": _windows_flash_attn_wheel_url(),
        "smoke_check_policy": "windows-runtime-imports",
        "smoke_checks": [dict(check) for check in WINDOWS_RUNTIME_SMOKE_CHECKS],
        "dependency_entries": [
            {"name": "cumm", "strategy": "windows-pinned-prebuilt", "package": _windows_cumm_package()},
            {"name": "spconv", "strategy": "windows-pinned-prebuilt", "package": _windows_spconv_package()},
            {"name": "triton", "strategy": "windows-pinned-package", "package": _windows_triton_package()},
            {"name": "flash_attn", "strategy": "windows-pinned-wheel", "wheel": _windows_flash_attn_wheel_url()},
            {"name": "sitecustomize", "strategy": "windows-dll-shim"},
        ],
    }


def _build_host_install_policy(host_os: str, host_arch: str) -> dict[str, object]:
    normalized_os = _normalize_host_os(host_os)
    normalized_arch = _normalize_host_arch(host_arch)
    host_class = _classify_host(normalized_os, normalized_arch)
    windows_contract = _windows_validated_runtime_contract()

    if host_class == "windows-x86_64":
        return {
            "host_class": host_class,
            "support_posture": "validated",
            "install_mode": str(windows_contract["install_mode"]),
            "profile": str(windows_contract["profile"]),
            "stages": [],
            "cumm_package": str(windows_contract["cumm_package"]),
            "spconv_package": str(windows_contract["spconv_package"]),
            "triton_package": str(windows_contract["triton_package"]),
            "flash_attn_wheel": str(windows_contract["flash_attn_wheel"]),
            "smoke_check_policy": str(windows_contract["smoke_check_policy"]),
            "smoke_checks": list(windows_contract["smoke_checks"]),
        }

    if host_class == "linux-x86_64":
        return {
            "host_class": host_class,
            "support_posture": "supported",
            "install_mode": "prebuilt",
            "profile": "pinned-upstream-wrapper",
            "stages": [],
            "smoke_check_policy": "standard",
            "smoke_checks": [],
        }

    if host_class == "linux-arm64":
        return {
            "host_class": host_class,
            "support_posture": "experimental-unvalidated",
            "install_mode": "staged-source-build",
            "profile": "linux-arm64-runtime-bringup",
            "stages": ["baseline", "pyg", "spconv", "bpy-deferred"],
            "smoke_check_policy": "blocked",
            "smoke_checks": [],
        }

    return {
        "host_class": host_class,
        "support_posture": "unsupported",
        "install_mode": "prebuilt",
        "profile": "pinned-upstream-wrapper",
        "stages": [],
        "smoke_check_policy": "standard",
        "smoke_checks": [],
    }


def build_install_plan(host_os: str | None = None, host_arch: str | None = None) -> dict[str, object]:
    normalized_os = _normalize_host_os(platform.system() if host_os is None else host_os)
    normalized_arch = _normalize_host_arch(platform.machine() if host_arch is None else host_arch)
    policy = _build_host_install_policy(normalized_os, normalized_arch)
    host_class = str(policy["host_class"])

    plan = {
        "host_class": host_class,
        "support_posture": str(policy["support_posture"]),
        "install_mode": str(policy["install_mode"]),
        "profile": str(policy["profile"]),
        "stages": [str(stage) for stage in policy.get("stages", [])],
        "dependencies": [
            {
                "name": "torch",
                "strategy": "torch-index-cu128",
                "index_url": TORCH_INDEX_URL,
                "packages": list(TORCH_PACKAGES),
            },
            {
                "name": "upstream-requirements",
                "strategy": "windows-filtered-requirements" if host_class == "windows-x86_64" else "upstream-requirements-file",
            },
            {
                "name": "pyg",
                "strategy": "pyg-wheel-index",
                "index_url": PYG_INDEX_URL,
                "packages": list(PYG_PACKAGES) + [NUMPY_PIN, "pygltflib>=1.15.0"],
            },
        ],
        "deferred": [],
    }

    if host_class == "windows-x86_64":
        plan["dependencies"].extend(_windows_validated_runtime_contract()["dependency_entries"])
        return plan

    if host_class == "linux-x86_64":
        plan["dependencies"].append({"name": "spconv", "strategy": "generic-prebuilt-package", "package": SPCONV_PACKAGE})
        return plan

    if host_class == "linux-arm64":
        plan["dependencies"] = [
            plan["dependencies"][0],
            plan["dependencies"][1],
            {
                "name": "pyg",
                "strategy": "linux-arm64-source-build-only",
                "reason_code": "pyg-source-build",
                "stage": "pyg",
                "packages": list(LINUX_ARM64_PYG_STAGE_PACKAGES),
                "verification": "deferred",
            },
            {
                "name": "spconv",
                "strategy": LINUX_ARM64_SPCONV_STRATEGY,
                "reason_code": LINUX_ARM64_SPCONV_REASON_CODE,
                "stage": "spconv",
                "verification": "import-smoke",
                "allowed_statuses": list(LINUX_ARM64_SPCONV_ALLOWED_STATUSES),
            },
            {"name": "bpy", "strategy": "deferred-portability-review", "reason_code": "bpy-portability"},
        ]
        plan["deferred"] = ["bpy-portability"]
        return plan

    plan["dependencies"].append({"name": "spconv", "strategy": "unsupported-host"})
    plan["deferred"] = ["unsupported-host"]
    return plan


def _probe_python_header_candidates(python_exe: Path) -> list[Path]:
    def _dedupe(paths: list[str]) -> list[Path]:
        seen: set[str] = set()
        ordered: list[Path] = []
        for raw in paths:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(Path(value))
        return ordered

    try:
        if python_exe.resolve() == Path(sys.executable).resolve():
            return _dedupe(
                [
                    str(sysconfig.get_path("include") or ""),
                    str(sysconfig.get_path("platinclude") or ""),
                    str(sysconfig.get_config_var("INCLUDEPY") or ""),
                    str(sysconfig.get_config_var("CONFINCLUDEPY") or ""),
                ]
            )
    except OSError:
        return []

    result = subprocess.run(
        [
            str(python_exe),
            "-c",
            (
                "import json, sysconfig; "
                "paths=[sysconfig.get_path('include'), sysconfig.get_path('platinclude'), "
                "sysconfig.get_config_var('INCLUDEPY'), sysconfig.get_config_var('CONFINCLUDEPY')]; "
                "print(json.dumps([item for item in paths if item]))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    try:
        payload = json.loads((result.stdout or "[]").strip() or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return _dedupe([str(item) for item in payload])


def _probe_command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _normalize_linux_arm64_build_environment(raw: dict[str, object] | None) -> dict[str, dict[str, object]]:
    source = dict(raw or {})
    gpu = source.get("gpu") if isinstance(source.get("gpu"), dict) else {}
    nvcc = source.get("nvcc") if isinstance(source.get("nvcc"), dict) else {}
    cuda = source.get("cuda") if isinstance(source.get("cuda"), dict) else {}
    compiler = source.get("compiler") if isinstance(source.get("compiler"), dict) else {}
    python_headers = source.get("python_headers") if isinstance(source.get("python_headers"), dict) else {}

    legacy_nvcc_path = str(source.get("nvcc_path") or "").strip()
    legacy_compiler = str(source.get("cxx_compiler") or "").strip()
    legacy_cuda_home = str(source.get("cuda_home") or "").strip()
    legacy_python_headers_ready = bool(source.get("python_headers_ready"))
    legacy_python_header = str(source.get("python_header") or "").strip()

    return {
        "gpu": {
            "present": bool(gpu.get("present", False)),
            "vendor": str(gpu.get("vendor") or "").strip(),
            "model": str(gpu.get("model") or "").strip(),
            "nvidia_smi_path": str(gpu.get("nvidia_smi_path") or "").strip(),
        },
        "nvcc": {
            "path": str(nvcc.get("path") or legacy_nvcc_path).strip(),
            "version": str(nvcc.get("version") or "").strip(),
            "is_real": bool(nvcc.get("is_real", bool(legacy_nvcc_path))),
        },
        "cuda": {
            "home": str(cuda.get("home") or legacy_cuda_home).strip(),
            "version": str(cuda.get("version") or "").strip(),
            "facts_ready": bool(cuda.get("facts_ready", bool(legacy_cuda_home and legacy_nvcc_path))),
        },
        "compiler": {
            "path": str(compiler.get("path") or legacy_compiler).strip(),
            "kind": str(compiler.get("kind") or Path(legacy_compiler).name).strip(),
        },
        "python_headers": {
            "ready": bool(python_headers.get("ready", legacy_python_headers_ready)),
            "header": str(python_headers.get("header") or legacy_python_header).strip(),
        },
    }


def _probe_linux_arm64_build_environment(python_exe: Path) -> dict[str, object]:
    nvidia_smi_path = shutil.which("nvidia-smi") or ""
    gpu_vendor = ""
    gpu_model = ""
    gpu_present = False
    if nvidia_smi_path:
        gpu_output = _probe_command_output([nvidia_smi_path, "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if gpu_output:
            first_line = gpu_output.splitlines()[0].strip()
            fields = [part.strip() for part in first_line.split(",") if part.strip()]
            gpu_model = fields[0] if fields else ""
            gpu_vendor = "NVIDIA"
            gpu_present = True

    nvcc_path = shutil.which("nvcc") or ""
    nvcc_version = _probe_command_output([nvcc_path, "--version"]) if nvcc_path else ""
    nvcc_is_real = bool(nvcc_path and "Cuda compilation tools" in nvcc_version)

    cxx_compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++") or ""
    cuda_home = str(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "").strip()
    if not cuda_home and nvcc_path:
        cuda_home = str(Path(nvcc_path).resolve().parents[1])

    cuda_version = ""
    if nvcc_version:
        match = re.search(r"release\s+([^,\s]+)", nvcc_version)
        if match:
            cuda_version = match.group(1)
    cuda_facts_ready = bool(cuda_home and nvcc_is_real and cuda_version)

    python_header = ""
    for candidate in _probe_python_header_candidates(python_exe):
        header_path = candidate / "Python.h"
        if header_path.exists():
            python_header = str(header_path)
            break

    return {
        "gpu": {
            "present": gpu_present,
            "vendor": gpu_vendor,
            "model": gpu_model,
            "nvidia_smi_path": nvidia_smi_path,
        },
        "nvcc": {
            "path": nvcc_path,
            "version": nvcc_version,
            "is_real": nvcc_is_real,
        },
        "cuda": {
            "home": cuda_home,
            "version": cuda_version,
            "facts_ready": cuda_facts_ready,
        },
        "compiler": {
            "path": cxx_compiler,
            "kind": Path(cxx_compiler).name if cxx_compiler else "",
        },
        "python_headers": {
            "ready": bool(python_header),
            "header": python_header,
        },
    }


def _linux_arm64_blocker(
    category: str,
    code: str,
    message: str,
    *,
    action: str = "stop",
    boundary: str = "wrapper",
    owner: str | None = None,
    repair_hint: str | None = None,
    **extra: object,
) -> dict[str, object]:
    blocker = {
        "category": category,
        "code": code,
        "message": message,
        "action": action,
        "boundary": boundary,
        "owner": owner or boundary,
    }
    if repair_hint:
        blocker["repair_hint"] = repair_hint
    blocker.update(extra)
    return blocker


def _linux_arm64_python_family(python_version: str | None) -> str:
    value = str(python_version or "").strip()
    parts = [part for part in value.split(".") if part]
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return value or "host-selected"


def _collect_linux_arm64_baseline_prerequisites(
    python_exe: Path, *, python_version: str | None = None
) -> dict[str, object]:
    environment = _normalize_linux_arm64_build_environment(_probe_linux_arm64_build_environment(python_exe))
    resolved_python_version = str(python_version or "").strip()
    if not resolved_python_version:
        try:
            resolved_python_version = _probe_python_version(python_exe)
        except (OSError, RuntimeError):
            resolved_python_version = ""

    blockers: list[dict[str, object]] = []
    blocked: list[str] = []

    def add_failure(code: str, message: str, *, category: str, repair_hint: str, observed: dict[str, object]) -> None:
        blockers.append(
            _linux_arm64_blocker(
                category,
                code,
                message,
                boundary="environment",
                owner="environment",
                repair_hint=repair_hint,
                observed=observed,
            )
        )
        blocked.append(message)

    gpu = environment["gpu"]
    if not bool(gpu.get("present")):
        add_failure(
            "missing-nvidia-gpu",
            "Linux ARM64 bringup requires an NVIDIA GPU with driver visibility on the host before source-build preparation can proceed.",
            category="gpu",
            repair_hint="Run this staged bringup on a host with an NVIDIA GPU and working nvidia-smi visibility.",
            observed=gpu,
        )

    nvcc = environment["nvcc"]
    nvcc_ready = bool(nvcc.get("path")) and bool(nvcc.get("is_real"))
    if not nvcc_ready:
        add_failure(
            "missing-real-nvcc",
            "Linux ARM64 bringup requires the real NVIDIA nvcc compiler; wrapper shims or missing executables are not sufficient.",
            category="toolchain",
            repair_hint="Install the NVIDIA CUDA toolkit that provides a real nvcc binary and ensure it is first on PATH.",
            observed=nvcc,
        )

    cuda = environment["cuda"]
    if not bool(cuda.get("facts_ready")):
        add_failure(
            "missing-system-cuda-facts",
            "Linux ARM64 bringup requires system CUDA facts (CUDA_HOME plus a detectable nvcc release) so the wrapper can report the host toolchain honestly.",
            category="cuda",
            repair_hint="Export CUDA_HOME/CUDA_PATH if needed and install a CUDA toolkit whose nvcc reports a release version.",
            observed=cuda,
        )

    compiler = environment["compiler"]
    if not bool(compiler.get("path")):
        add_failure(
            "missing-cxx-compiler",
            "Linux ARM64 bringup requires a C++ compiler (g++, clang++, or c++) for source-build preparation.",
            category="toolchain",
            repair_hint="Install g++ or clang++ on the Linux ARM64 host before rerunning setup.py.",
            observed=compiler,
        )

    python_headers = environment["python_headers"]
    if not bool(python_headers.get("ready")):
        add_failure(
            "missing-python-headers",
            "Linux ARM64 bringup requires Python development headers so Python.h is available for extension builds.",
            category="headers",
            repair_hint="Install the matching python3-dev/python-devel package for the selected interpreter.",
            observed=python_headers,
        )

    return {
        "ready": not blockers,
        "python": {
            "required": _linux_arm64_python_family(resolved_python_version),
            "version": resolved_python_version,
            "executable": str(python_exe),
        },
        "gpu": gpu,
        "nvcc": nvcc,
        "cuda": cuda,
        "compiler": compiler,
        "python_headers": python_headers,
        "blockers": blockers,
        "blocked": blocked,
    }


def _linux_arm64_dependency_blocker(dependency: dict[str, object]) -> dict[str, object] | None:
    name = str(dependency.get("name") or "").strip()
    strategy = str(dependency.get("strategy") or "").strip()
    reason_code = str(dependency.get("reason_code") or "").strip()

    if name == "spconv" and strategy == LINUX_ARM64_SPCONV_STRATEGY:
        return _linux_arm64_blocker(
            "source-build",
            LINUX_ARM64_SPCONV_REASON_CODE,
            _linux_arm64_spconv_guarded_bringup_message(),
            dependency=name,
            strategy=strategy,
            reason_code=reason_code,
            boundary="upstream-package",
            owner="upstream-package",
        )
    if name == "spconv" and strategy == "blocked-missing-distribution":
        return _linux_arm64_blocker(
            "distribution",
            "spconv-missing-distribution",
            _linux_arm64_spconv_missing_distribution_message(),
            dependency=name,
            strategy=strategy,
            reason_code=reason_code,
        )
    if name == "bpy" and strategy == "deferred-portability-review":
        return _linux_arm64_blocker(
            "portability",
            "bpy-portability-risk",
            "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists; this change only records the blocker.",
            dependency=name,
            strategy=strategy,
            reason_code=reason_code,
        )
    return None


def _linux_arm64_bpy_stage_allows_staged_testing(stage: dict[str, object] | None) -> bool:
    if not isinstance(stage, dict):
        return False
    return str(stage.get("status") or "").strip() == "external-bpy-smoke-ready"


def _linux_arm64_preflight_checks(
    python_exe: Path, *, python_version: str | None = None
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str], dict[str, object]]:
    plan = build_install_plan(host_os="linux", host_arch="aarch64")
    baseline = _collect_linux_arm64_baseline_prerequisites(python_exe, python_version=python_version)
    checks: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    blocked: list[str] = []

    def add_check(
        check_id: str,
        label: str,
        passed: bool,
        message: str,
        *,
        category: str,
        code: str,
        action: str = "stop",
        boundary: str = "wrapper",
        owner: str | None = None,
        observed: dict[str, object] | None = None,
        repair_hint: str | None = None,
    ) -> None:
        entry = {
            "id": check_id,
            "label": label,
            "required": "linux-arm64 diagnostic preflight",
            "status": "pass" if passed else "fail",
            "message": message,
            "category": category,
            "code": code,
            "action": action,
            "boundary": boundary,
            "owner": owner or boundary,
        }
        if repair_hint:
            entry["repair_hint"] = repair_hint
        if observed:
            entry["observed"] = observed
        checks.append(entry)
        if not passed:
            blocker = _linux_arm64_blocker(
                category,
                code,
                message,
                action=action,
                boundary=boundary,
                owner=owner,
                repair_hint=repair_hint,
            )
            if observed:
                blocker["observed"] = observed
            blockers.append(blocker)
            blocked.append(message)

    dependency_map = {
        str(item.get("name")): dict(item)
        for item in plan.get("dependencies", [])
        if isinstance(item, dict) and item.get("name")
    }
    deferred = {str(item) for item in plan.get("deferred", [])}

    add_check(
        "linux-arm64-missing-distributions",
        "Linux ARM64 distribution assumptions",
        False,
        "Linux ARM64 has no validated prebuilt distribution path for this wrapper; do not assume the Windows-pinned package route exists on this host.",
        category="distribution",
        code="missing-distribution",
        boundary="wrapper",
        observed={"host_class": plan.get("host_class"), "install_mode": plan.get("install_mode")},
    )

    pyg_dependency = dependency_map.get("pyg", {})
    add_check(
        "linux-arm64-pyg-source-build",
        "PyG source-build pressure",
        str(pyg_dependency.get("strategy")) == "linux-arm64-source-build-only",
        "PyG on Linux ARM64 is restricted to source builds for torch_scatter and torch_cluster; prerequisite-only readiness is not runtime verification.",
        category="source-build",
        code="pyg-source-build",
        boundary="wrapper",
        observed={
            "strategy": pyg_dependency.get("strategy"),
            "reason_code": pyg_dependency.get("reason_code"),
            "packages": pyg_dependency.get("packages"),
            "stage": pyg_dependency.get("stage"),
            "verification": pyg_dependency.get("verification"),
        },
    )

    spconv_dependency = dependency_map.get("spconv", {})
    add_check(
        "linux-arm64-spconv-guarded-source-build",
        "spconv guarded source-build contract",
        str(spconv_dependency.get("strategy")) == LINUX_ARM64_SPCONV_STRATEGY,
        _linux_arm64_spconv_guarded_bringup_message(),
        category="source-build",
        code=LINUX_ARM64_SPCONV_REASON_CODE,
        boundary="wrapper",
        observed={
            "strategy": spconv_dependency.get("strategy"),
            "reason_code": spconv_dependency.get("reason_code"),
            "stage": spconv_dependency.get("stage"),
            "verification": spconv_dependency.get("verification"),
            "allowed_statuses": spconv_dependency.get("allowed_statuses"),
        },
    )

    baseline_blockers = {
        str(item.get("code") or ""): dict(item)
        for item in baseline.get("blockers", [])
        if isinstance(item, dict) and item.get("code")
    }
    nvcc = dict(baseline.get("nvcc") or {})
    compiler = dict(baseline.get("compiler") or {})
    cuda = dict(baseline.get("cuda") or {})
    python_headers = dict(baseline.get("python_headers") or {})
    gpu = dict(baseline.get("gpu") or {})
    python = dict(baseline.get("python") or {})

    add_check(
        "linux-arm64-bootstrap-python",
        "Bootstrap Python baseline",
        True,
        f"Linux ARM64 staged bringup will use the selected bootstrap interpreter '{python.get('version') or 'unknown'}'; compatibility remains unvalidated until dependency bringup proves otherwise.",
        category="python",
        code="bootstrap-python-selected",
        boundary="environment",
        owner="environment",
        observed=python,
    )

    add_check(
        "linux-arm64-nvidia-gpu",
        "NVIDIA GPU visibility",
        "missing-nvidia-gpu" not in baseline_blockers,
        "Linux ARM64 bringup requires an NVIDIA GPU with driver visibility on the host before source-build preparation can proceed.",
        category="gpu",
        code="missing-nvidia-gpu" if "missing-nvidia-gpu" in baseline_blockers else "nvidia-gpu-ready",
        boundary="environment",
        owner="environment",
        observed=gpu,
        repair_hint="Run this staged bringup on a host with an NVIDIA GPU and working nvidia-smi visibility.",
    )

    add_check(
        "linux-arm64-real-nvcc",
        "Real nvcc compiler",
        "missing-real-nvcc" not in baseline_blockers,
        "Linux ARM64 bringup requires the real NVIDIA nvcc compiler; wrapper shims or missing executables are not sufficient.",
        category="toolchain",
        code="missing-real-nvcc" if "missing-real-nvcc" in baseline_blockers else "real-nvcc-ready",
        boundary="environment",
        owner="environment",
        observed=nvcc,
        repair_hint="Install the NVIDIA CUDA toolkit that provides a real nvcc binary and ensure it is first on PATH.",
    )

    add_check(
        "linux-arm64-system-cuda-facts",
        "System CUDA facts",
        "missing-system-cuda-facts" not in baseline_blockers,
        "Linux ARM64 bringup requires system CUDA facts (CUDA_HOME plus a detectable nvcc release) so the wrapper can report the host toolchain honestly.",
        category="cuda",
        code="missing-system-cuda-facts" if "missing-system-cuda-facts" in baseline_blockers else "system-cuda-facts-ready",
        boundary="environment",
        owner="environment",
        observed=cuda,
        repair_hint="Export CUDA_HOME/CUDA_PATH if needed and install a CUDA toolkit whose nvcc reports a release version.",
    )

    add_check(
        "linux-arm64-cxx-compiler",
        "C++ compiler",
        "missing-cxx-compiler" not in baseline_blockers,
        "Linux ARM64 bringup requires a C++ compiler (g++, clang++, or c++) for source-build preparation.",
        category="toolchain",
        code="missing-cxx-compiler" if "missing-cxx-compiler" in baseline_blockers else "cxx-compiler-ready",
        boundary="environment",
        owner="environment",
        observed=compiler,
        repair_hint="Install g++ or clang++ on the Linux ARM64 host before rerunning setup.py.",
    )

    add_check(
        "linux-arm64-python-headers-baseline",
        "Python headers baseline",
        "missing-python-headers" not in baseline_blockers,
        "Linux ARM64 bringup requires Python development headers so Python.h is available for extension builds.",
        category="headers",
        code="missing-python-headers" if "missing-python-headers" in baseline_blockers else "python-headers-ready",
        boundary="environment",
        owner="environment",
        observed=python_headers,
        repair_hint="Install the matching python3-dev/python-devel package for the selected interpreter.",
    )

    nvcc_path = str(nvcc.get("path") or "").strip()
    cxx_compiler = str(compiler.get("path") or "").strip()
    cuda_home = str(cuda.get("home") or "").strip()
    toolchain_ready = bool(nvcc_path and cxx_compiler)
    toolchain_message = (
        f"nvcc/toolchain probe ready: nvcc={nvcc_path}, compiler={cxx_compiler}, cuda_home={cuda_home or 'unknown'}."
        if toolchain_ready
        else "nvcc or a C++ compiler is not ready on Linux ARM64. Install the CUDA toolkit (including nvcc) plus g++/clang++ before attempting any source-build investigation."
    )
    add_check(
        "linux-arm64-nvcc-toolchain",
        "CUDA nvcc and toolchain readiness",
        toolchain_ready,
        toolchain_message,
        category="toolchain",
        code="missing-nvcc-toolchain" if not toolchain_ready else "nvcc-toolchain-ready",
        boundary="environment",
        owner="environment",
        observed={"nvcc_path": nvcc_path, "cxx_compiler": cxx_compiler, "cuda_home": cuda_home},
    )

    python_headers_ready = bool(python_headers.get("ready"))
    python_header = str(python_headers.get("header") or "").strip()
    headers_message = (
        f"Python development headers detected at {python_header}."
        if python_headers_ready
        else "Python development headers are not ready for Linux ARM64 source builds. Install the matching python3-dev/python3-devel package so Python.h is available."
    )
    add_check(
        "linux-arm64-python-headers",
        "Python development headers",
        python_headers_ready,
        headers_message,
        category="headers",
        code="missing-python-headers" if not python_headers_ready else "python-headers-ready",
        boundary="environment",
        owner="environment",
        observed={"python_header": python_header},
    )

    bpy_dependency = dependency_map.get("bpy", {})
    bpy_evidence = _linux_arm64_bpy_evidence(
        {},
        wrapper_python_version=str((baseline.get("python") or {}).get("version") or "").strip() or None,
    )
    bpy_stage = _linux_arm64_bpy_stage_payload(bpy_evidence)
    bpy_stage_allows_staged_testing = _linux_arm64_bpy_stage_allows_staged_testing(bpy_stage)
    add_check(
        "linux-arm64-bpy-portability",
        "bpy portability risk",
        "bpy-portability" not in deferred or bpy_stage_allows_staged_testing,
        (
            "Linux ARM64 preserved external Blender bpy smoke evidence for staged testing, but full wrapper runtime remains blocked until bpy is supported inside the wrapper-owned runtime boundary."
            if bpy_stage_allows_staged_testing
            else "bpy remains a likely Linux ARM64 portability risk and is deferred until upstream portability evidence exists; this change only records the blocker."
        ),
        category="portability",
        code="bpy-portability-risk",
        boundary="upstream",
        observed={
            "strategy": bpy_dependency.get("strategy"),
            "deferred": sorted(deferred),
            "bpy_stage_status": str(bpy_stage.get("status") or "missing"),
        },
    )

    return checks, blockers, blocked, baseline


def _verify_linux_arm64_pyg_import_smoke(python_exe: Path) -> dict[str, object]:
    def classify_failure(detail_lines: list[str]) -> tuple[str, str]:
        detail_blob = "\n".join(detail_lines).lower()
        environment_markers = (
            "cannot open shared object file",
            "libcuda",
            "libcudart",
            "libcusparse",
            "libcublas",
            "undefined symbol",
            "no kernel image is available",
            "driver version is insufficient",
        )
        if any(marker in detail_blob for marker in environment_markers):
            return "environment", "environment"
        return "upstream", "upstream"

    checks: list[dict[str, object]] = []
    for check in LINUX_ARM64_PYG_IMPORT_SMOKE_CHECKS:
        try:
            result = subprocess.run(
                [str(python_exe), "-c", str(check["code"])],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            blocker = _linux_arm64_blocker(
                "distribution",
                "pyg-import-smoke-failed",
                "Linux ARM64 PyG import/smoke verification could not be executed by the wrapper after baseline prerequisites were satisfied.",
                boundary="wrapper",
                owner="wrapper",
                dependency="pyg",
                verification="import-smoke",
                failed_check=str(check["label"]),
                details=[str(exc)],
            )
            return {
                "status": "blocked",
                "ready": False,
                "verification": "import-smoke",
                "checks": checks,
                "blockers": [blocker],
                "blocker_codes": [str(blocker["code"])],
            }
        entry = {
            "label": str(check["label"]),
            "returncode": int(result.returncode),
            "status": "ready" if result.returncode == 0 else "error",
        }
        checks.append(entry)
        if result.returncode == 0:
            continue
        tail = (result.stderr or result.stdout or f"{check['label']} import smoke failed").strip().splitlines()[-20:]
        boundary, owner = classify_failure(tail)
        blocker = _linux_arm64_blocker(
            "distribution",
            "pyg-import-smoke-failed",
            "Linux ARM64 PyG import/smoke verification failed after baseline prerequisites were satisfied; prerequisite-only readiness is not enough to claim stage readiness.",
            boundary=boundary,
            owner=owner,
            dependency="pyg",
            verification="import-smoke",
            failed_check=str(check["label"]),
            details=tail,
        )
        return {
            "status": "blocked",
            "ready": False,
            "verification": "import-smoke",
            "checks": checks,
            "blockers": [blocker],
            "blocker_codes": [str(blocker["code"])],
        }

    return {
        "status": "ready",
        "ready": True,
        "verification": "import-smoke",
        "checks": checks,
        "blockers": [],
        "blocker_codes": [],
    }


def _classify_linux_arm64_spconv_failure(detail_lines: list[str]) -> tuple[str, str]:
    detail_blob = "\n".join(detail_lines).lower()
    environment_markers = (
        "cannot open shared object file",
        "libcuda",
        "libcudart",
        "libcusparse",
        "libcublas",
        "undefined symbol",
        "no kernel image is available",
        "driver version is insufficient",
    )
    if any(marker in detail_blob for marker in environment_markers):
        return "environment", "environment"
    return "upstream-package", "upstream-package"


def _verify_linux_arm64_spconv_import_smoke(python_exe: Path) -> dict[str, object]:
    check = dict(LINUX_ARM64_SPCONV_IMPORT_SMOKE_CHECK)
    try:
        result = subprocess.run(
            [str(python_exe), "-c", str(check["code"])],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        blocker = _linux_arm64_blocker(
            "source-build",
            "spconv-import-smoke-failed",
            "Linux ARM64 spconv import/smoke verification could not be executed by the wrapper after guarded cumm/spconv orchestration completed.",
            boundary="wrapper",
            owner="wrapper",
            dependency="spconv",
            verification="import-smoke",
            failed_check=str(check["label"]),
            details=[str(exc)],
        )
        return {
            "status": "blocked",
            "ready": False,
            "verification": "import-smoke",
            "checks": [],
            "blockers": [blocker],
            "blocker_codes": [str(blocker["code"])],
            "boundary": "wrapper",
        }

    entry = {
        "label": str(check["label"]),
        "returncode": int(result.returncode),
        "status": "ready" if result.returncode == 0 else "error",
    }
    if result.returncode == 0:
        return {
            "status": "ready",
            "ready": True,
            "verification": "import-smoke",
            "checks": [entry],
            "blockers": [],
            "blocker_codes": [],
            "boundary": "wrapper",
        }

    tail = (result.stderr or result.stdout or f"{check['label']} import smoke failed").strip().splitlines()[-20:]
    boundary, owner = _classify_linux_arm64_spconv_failure(tail)
    blocker = _linux_arm64_blocker(
        "source-build",
        "spconv-import-smoke-failed",
        "Linux ARM64 spconv import/smoke verification failed after guarded cumm/spconv orchestration; build artifacts alone do not count as stage readiness.",
        boundary=boundary,
        owner=owner,
        dependency="spconv",
        verification="import-smoke",
        failed_check=str(check["label"]),
        details=tail,
    )
    return {
        "status": "blocked",
        "ready": False,
        "verification": "import-smoke",
        "checks": [entry],
        "blockers": [blocker],
        "blocker_codes": [str(blocker["code"])],
        "boundary": boundary,
    }


def _run_linux_arm64_spconv_guarded_bringup(
    pip: list[str],
    python_exe: Path,
    unirig_dir: Path,
    packages: list[str] | None = None,
) -> dict[str, object]:
    stage_packages = [str(item).strip() for item in (packages or LINUX_ARM64_SPCONV_STAGE_PACKAGES) if str(item).strip()]
    if stage_packages != LINUX_ARM64_SPCONV_STAGE_PACKAGES:
        raise RuntimeError(
            "Linux ARM64 guarded spconv bringup requires the explicit cumm -> spconv package sequence."
        )

    checks: list[dict[str, object]] = []
    source_build_env = _linux_arm64_source_build_environment(python_exe=python_exe)
    try:
        subprocess.run(
            pip + ["uninstall", "-y", "cumm", "cumm-cu128", "spconv", "spconv-cu128"],
            cwd=unirig_dir,
            env=source_build_env,
            capture_output=True,
            text=True,
            check=False,
        )
        prerequisite_result = subprocess.run(
            pip + ["install", "--no-cache-dir", LINUX_ARM64_PCCM_PACKAGE, *LINUX_ARM64_SPCONV_BUILD_PREREQUISITES],
            cwd=unirig_dir,
            env=source_build_env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        blocker = _linux_arm64_blocker(
            "source-build",
            "spconv-guarded-bringup-failed",
            "Linux ARM64 guarded cumm/spconv orchestration could not be executed by the wrapper.",
            boundary="wrapper",
            owner="wrapper",
            dependency="spconv",
            verification="import-smoke",
            failed_check="cumm",
            details=[str(exc)],
        )
        return {
            "status": "blocked",
            "ready": False,
            "mode": "source-build",
            "verification": "import-smoke",
            "packages": stage_packages,
            "checks": checks,
            "blockers": [blocker],
            "blocker_codes": [str(blocker["code"])],
            "boundary": "wrapper",
        }

    if prerequisite_result.returncode != 0:
        tail = (
            prerequisite_result.stderr
            or prerequisite_result.stdout
            or f"{LINUX_ARM64_PCCM_PACKAGE} install failed"
        ).strip().splitlines()[-20:]
        boundary, owner = _classify_linux_arm64_spconv_failure(tail)
        blocker = _linux_arm64_blocker(
            "source-build",
            "spconv-guarded-bringup-failed",
            "Linux ARM64 guarded cumm/spconv orchestration failed before import verification could prove runtime readiness.",
            boundary=boundary,
            owner=owner,
            dependency="spconv",
            verification="import-smoke",
            failed_check="cumm",
            details=tail,
        )
        return {
            "status": "blocked",
            "ready": False,
            "mode": "source-build",
            "verification": "import-smoke",
            "packages": stage_packages,
            "checks": checks,
            "blockers": [blocker],
            "blocker_codes": [str(blocker["code"])],
            "boundary": boundary,
        }

    for package_name in stage_packages:
        package_list = ",".join(stage_packages)
        requirement = _linux_arm64_spconv_source_requirement(package_name)
        try:
            result = subprocess.run(
                pip + ["install", "--no-cache-dir", "--no-build-isolation", f"--no-binary={package_list}", requirement],
                cwd=unirig_dir,
                env=source_build_env,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            blocker = _linux_arm64_blocker(
                "source-build",
                "spconv-guarded-bringup-failed",
                "Linux ARM64 guarded cumm/spconv orchestration could not be executed by the wrapper.",
                boundary="wrapper",
                owner="wrapper",
                dependency="spconv",
                verification="import-smoke",
                failed_check=package_name,
                details=[str(exc)],
            )
            return {
                "status": "blocked",
                "ready": False,
                "mode": "source-build",
                "verification": "import-smoke",
                "packages": stage_packages,
                "checks": checks,
                "blockers": [blocker],
                "blocker_codes": [str(blocker["code"])],
                "boundary": "wrapper",
            }

        entry = {
            "label": package_name,
            "returncode": int(result.returncode),
            "status": "ready" if result.returncode == 0 else "error",
        }
        checks.append(entry)
        if result.returncode == 0 and package_name == "cumm":
            try:
                patched_common_path = _patch_linux_arm64_installed_cumm_common(python_exe)
                _log(f"Applied Linux ARM64 cumm CUDA discovery patch at {patched_common_path}")
            except RuntimeError as exc:
                blocker = _linux_arm64_blocker(
                    "source-build",
                    "spconv-guarded-bringup-failed",
                    "Linux ARM64 guarded cumm/spconv orchestration failed before import verification could prove runtime readiness.",
                    boundary="wrapper",
                    owner="wrapper",
                    dependency="spconv",
                    verification="import-smoke",
                    failed_check="cumm",
                    details=[str(exc)],
                )
                return {
                    "status": "blocked",
                    "ready": False,
                    "mode": "source-build",
                    "verification": "import-smoke",
                    "packages": stage_packages,
                    "checks": checks,
                    "blockers": [blocker],
                    "blocker_codes": [str(blocker["code"])],
                    "boundary": "wrapper",
                }
        if result.returncode == 0:
            continue

        tail = (result.stderr or result.stdout or f"{package_name} install failed").strip().splitlines()[-20:]
        boundary, owner = _classify_linux_arm64_spconv_failure(tail)
        blocker = _linux_arm64_blocker(
            "source-build",
            "spconv-guarded-bringup-failed",
            "Linux ARM64 guarded cumm/spconv orchestration failed before import verification could prove runtime readiness.",
            boundary=boundary,
            owner=owner,
            dependency="spconv",
            verification="import-smoke",
            failed_check=package_name,
            details=tail,
        )
        return {
            "status": "blocked",
            "ready": False,
            "mode": "source-build",
            "verification": "import-smoke",
            "packages": stage_packages,
            "checks": checks,
            "blockers": [blocker],
            "blocker_codes": [str(blocker["code"])],
            "boundary": boundary,
        }

    verification = _verify_linux_arm64_spconv_import_smoke(python_exe)
    combined_checks = checks + [dict(item) for item in verification.get("checks", []) if isinstance(item, dict)]
    result = {
        "status": str(verification.get("status") or "blocked"),
        "ready": bool(verification.get("ready")),
        "mode": "source-build",
        "verification": "import-smoke",
        "packages": stage_packages,
        "checks": combined_checks,
        "blockers": [dict(item) for item in verification.get("blockers", []) if isinstance(item, dict)],
        "blocker_codes": [str(item) for item in verification.get("blocker_codes", []) if str(item).strip()],
        "boundary": str(verification.get("boundary") or "wrapper"),
    }
    return result


def _probe_linux_arm64_spconv_preparation(
    *,
    baseline_blockers: list[dict[str, object]],
    pyg_stage: dict[str, object],
    spconv_dependency: dict[str, object],
) -> dict[str, object]:
    verification = str(spconv_dependency.get("verification") or "import-smoke")
    packages = [
        str(item).strip()
        for item in spconv_dependency.get("packages", LINUX_ARM64_SPCONV_STAGE_PACKAGES)
        if str(item).strip()
    ]
    if not packages:
        packages = list(LINUX_ARM64_SPCONV_STAGE_PACKAGES)
    blocker_codes = [
        str(item.get("code") or "")
        for item in baseline_blockers
        if isinstance(item, dict) and str(item.get("code") or "").strip()
    ]
    if baseline_blockers:
        return {
            "status": "blocked",
            "ready": False,
            "verification": verification,
            "packages": packages,
            "checks": [],
            "blockers": [dict(item) for item in baseline_blockers if isinstance(item, dict)],
            "blocker_codes": blocker_codes,
            "boundary": "environment",
            "blocked_by_stage": "baseline",
        }

    if str(pyg_stage.get("status") or "") != "ready":
        return {
            "status": "deferred",
            "ready": False,
            "verification": verification,
            "packages": packages,
            "checks": [
                {
                    "label": "spconv-preparation-probe",
                    "status": "deferred",
                    "reason": "waiting-for-pyg-ready",
                    "message": "spconv preparation stays deferred until the preceding PyG stage reaches verified import/smoke readiness on Linux ARM64.",
                }
            ],
            "blockers": [],
            "blocker_codes": [],
            "boundary": str(pyg_stage.get("boundary") or "wrapper"),
            "blocked_by_stage": "pyg",
        }

    return {
        "status": "build-ready",
        "ready": False,
        "verification": verification,
        "packages": packages,
        "checks": [
            {
                "label": "spconv-guarded-source-build",
                "status": "build-ready",
                "message": "Linux ARM64 baseline and PyG verification are ready; spconv is build-ready for guarded source-build bringup, but import-smoke success is still required before it can be marked ready.",
            }
        ],
        "blockers": [],
        "blocker_codes": [],
        "boundary": "wrapper",
    }


def _linux_arm64_staged_preflight_state(
    *,
    report_status: str,
    plan: dict[str, object],
    baseline: dict[str, object],
    python_exe: Path,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    baseline_blockers = [
        dict(item)
        for item in baseline.get("blockers", [])
        if isinstance(item, dict)
    ]
    baseline_ready = bool(baseline.get("ready"))
    pyg_dependency = {}
    spconv_dependency = {}
    for dependency in plan.get("dependencies", []):
        if isinstance(dependency, dict) and str(dependency.get("name") or "") == "pyg":
            pyg_dependency = dict(dependency)
        if isinstance(dependency, dict) and str(dependency.get("name") or "") == "spconv":
            spconv_dependency = dict(dependency)

    pyg_verification = {
        "status": "blocked",
        "ready": False,
        "verification": str(pyg_dependency.get("verification") or "deferred"),
        "checks": [],
        "blockers": [],
        "blocker_codes": [],
    }
    if baseline_ready:
        pyg_verification = _verify_linux_arm64_pyg_import_smoke(python_exe)

    pyg_blockers = baseline_blockers if not baseline_ready else list(pyg_verification.get("blockers", []))
    pyg_stage = {
        "status": str(pyg_verification.get("status") or "source-build-only") if baseline_ready else "blocked",
        "ready": bool(pyg_verification.get("ready")) if baseline_ready else False,
        "mode": "source-build-only",
        "verification": str(pyg_verification.get("verification") or pyg_dependency.get("verification") or "deferred"),
        "packages": [str(item) for item in pyg_dependency.get("packages", LINUX_ARM64_PYG_STAGE_PACKAGES)],
        "blocker_codes": [str(item.get("code") or "") for item in pyg_blockers if str(item.get("code") or "").strip()],
        "blockers": pyg_blockers,
        "checks": [dict(item) for item in pyg_verification.get("checks", []) if isinstance(item, dict)],
        "boundary": (
            "environment"
            if not baseline_ready and pyg_blockers
            else str(pyg_blockers[0].get("boundary") or "wrapper") if pyg_blockers else "wrapper"
        ),
    }
    if not baseline_ready:
        pyg_stage["blocked_by_stage"] = "baseline"

    spconv_probe = _probe_linux_arm64_spconv_preparation(
        baseline_blockers=baseline_blockers,
        pyg_stage=pyg_stage,
        spconv_dependency=spconv_dependency,
    )
    spconv_stage = {
        "status": str(spconv_probe.get("status") or "deferred"),
        "ready": False,
        "mode": "source-build",
        "verification": str(spconv_probe.get("verification") or spconv_dependency.get("verification") or "import-smoke"),
        "allowed_statuses": [
            str(item)
            for item in spconv_dependency.get("allowed_statuses", ["blocked", "deferred", "build-ready", "ready"])
        ],
        "packages": [str(item) for item in spconv_probe.get("packages", LINUX_ARM64_SPCONV_STAGE_PACKAGES) if str(item).strip()],
        "blocker_codes": [str(item) for item in spconv_probe.get("blocker_codes", []) if str(item).strip()],
        "blockers": [dict(item) for item in spconv_probe.get("blockers", []) if isinstance(item, dict)],
        "checks": [dict(item) for item in spconv_probe.get("checks", []) if isinstance(item, dict)],
        "boundary": str(spconv_probe.get("boundary") or "wrapper"),
    }
    blocked_by_stage = str(spconv_probe.get("blocked_by_stage") or "").strip()
    if blocked_by_stage:
        spconv_stage["blocked_by_stage"] = blocked_by_stage

    bpy_evidence = _linux_arm64_bpy_evidence(
        payload,
        wrapper_python_version=str((baseline.get("python") or {}).get("version") or "").strip() or None,
    )
    bpy_stage = _linux_arm64_bpy_stage_payload(bpy_evidence)

    state = {
        "status": report_status,
        "non_blender_runtime_ready": False,
        "bpy_evidence_class": str((bpy_stage or {}).get("status") or "missing"),
        "external_blender": bpy_evidence,
        "stages": {
            "baseline": {
                "status": "ready" if baseline_ready else "blocked",
                "ready": baseline_ready,
                "blocker_codes": [str(item.get("code") or "") for item in baseline_blockers if str(item.get("code") or "").strip()],
                "blockers": baseline_blockers,
            },
            "pyg": pyg_stage,
            "spconv": spconv_stage,
            "bpy": bpy_stage,
        },
    }
    state["executable_boundary"] = _linux_arm64_extract_merge_boundary_payload(state)
    if not baseline_ready:
        state["current_stage"] = "baseline"
    elif str(pyg_stage.get("status") or "") != "ready":
        state["current_stage"] = "pyg"
    else:
        state["current_stage"] = "spconv"
    return state


def _executor_host_os() -> str:
    if os.name == "nt":
        return "windows"
    return platform.system()


def _plan_dependency(plan: dict[str, object], name: str, expected_strategy: str | None = None) -> dict[str, object]:
    for dependency in plan.get("dependencies", []):
        if not isinstance(dependency, dict):
            continue
        if str(dependency.get("name")) != name:
            continue
        if expected_strategy is not None and str(dependency.get("strategy")) != expected_strategy:
            raise RuntimeError(
                f"Windows install plan dependency '{name}' must use strategy '{expected_strategy}', "
                f"got '{dependency.get('strategy')}'."
            )
        return dependency
    raise RuntimeError(f"Windows install plan is missing required dependency entry: {name}.")


def _resolve_windows_executor_packages(plan: dict[str, object]) -> dict[str, str]:
    if str(plan.get("host_class")) != "windows-x86_64":
        raise RuntimeError(
            "Windows install execution requires the validated windows-x86_64 plan, "
            f"got '{plan.get('host_class')}'."
        )
    if str(plan.get("install_mode")) != "pinned-prebuilt":
        raise RuntimeError(
            "Windows install execution requires the validated pinned-prebuilt plan, "
            f"got '{plan.get('install_mode')}'."
        )

    cumm_dependency = _plan_dependency(plan, "cumm", "windows-pinned-prebuilt")
    spconv_dependency = _plan_dependency(plan, "spconv", "windows-pinned-prebuilt")
    triton_dependency = _plan_dependency(plan, "triton", "windows-pinned-package")
    flash_attn_dependency = _plan_dependency(plan, "flash_attn", "windows-pinned-wheel")
    _plan_dependency(plan, "sitecustomize", "windows-dll-shim")

    return {
        "cumm_package": str(cumm_dependency.get("package") or "").strip(),
        "spconv_package": str(spconv_dependency.get("package") or "").strip(),
        "triton_package": str(triton_dependency.get("package") or "").strip(),
        "flash_attn_wheel": str(flash_attn_dependency.get("wheel") or "").strip(),
    }


def _diagnostic_only_install_result(plan: dict[str, object]) -> dict[str, object]:
    blocked: list[dict[str, str]] = []
    for dependency in plan.get("dependencies", []):
        if not isinstance(dependency, dict):
            continue
        blocker = _linux_arm64_dependency_blocker(dependency)
        if blocker is None:
            continue
        blocked.append(blocker)

    return {
        "status": "diagnostic-only",
        "profile": "linux-arm64-prep",
        "host_class": str(plan.get("host_class") or "unknown"),
        "steps": ["diagnostic-stop"],
        "deferred_work": [str(item) for item in plan.get("deferred", [])],
        "blocked": blocked,
    }


def _filtered_requirements_path(unirig_dir: Path, runtime_root: Path, *, host_class: str) -> Path:
    requirements = unirig_dir / "requirements.txt"
    if not requirements.exists():
        raise RuntimeError(f"Upstream runtime requirements.txt is missing: {requirements}")
    if host_class not in {"windows-x86_64", "linux-arm64"}:
        return requirements

    filtered_name = "requirements.upstream.windows.txt" if host_class == "windows-x86_64" else "requirements.upstream.linux-arm64.txt"
    filtered = runtime_root / filtered_name
    lines: list[str] = []
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        normalized = raw.strip().replace("-", "_").lower()
        if host_class == "windows-x86_64" and normalized.startswith("flash_attn"):
            continue
        if host_class == "linux-arm64" and normalized.startswith(LINUX_ARM64_PREMATURE_REQUIREMENTS_PREFIXES):
            continue
        lines.append(raw)
    filtered.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filtered


def _install_windows_triton(pip: list[str], unirig_dir: Path, package: str | None = None) -> str:
    package = (package or _windows_triton_package()).strip()
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


def _install_windows_flash_attn(pip: list[str], runtime_root: Path, unirig_dir: Path, wheel_url: str | None = None) -> Path:
    wheel_url = (wheel_url or _windows_flash_attn_wheel_url()).strip()
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


def _install_windows_spconv_stack(
    pip: list[str], unirig_dir: Path, cumm_package: str | None = None, spconv_package: str | None = None
) -> tuple[str, str]:
    cumm_package = (cumm_package or _windows_cumm_package()).strip()
    spconv_package = (spconv_package or _windows_spconv_package()).strip()
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


def _install_linux_arm64_pyg_source_build(
    pip: list[str],
    unirig_dir: Path,
    pyg_dependency: dict[str, object],
) -> list[str]:
    packages = [str(item).strip() for item in pyg_dependency.get("packages", LINUX_ARM64_PYG_STAGE_PACKAGES) if str(item).strip()]
    if not packages:
        raise RuntimeError(
            "Linux ARM64 staged PyG bringup requires explicit source-build package names for torch_scatter and torch_cluster."
        )
    package_list = ",".join(packages)
    _run(
        pip + ["install", "--no-cache-dir", "--no-build-isolation", f"--no-binary={package_list}", *packages],
        cwd=unirig_dir,
        env=_linux_arm64_source_build_environment(),
    )
    return packages


def _install_linux_arm64_spconv_stage(plan: dict[str, object]) -> dict[str, object] | None:
    spconv_dependency = next(
        (
            dict(dependency)
            for dependency in plan.get("dependencies", [])
            if isinstance(dependency, dict) and str(dependency.get("name") or "") == "spconv"
        ),
        {},
    )
    if not spconv_dependency:
        raise RuntimeError("Linux ARM64 staged install requires an spconv dependency entry in the install plan.")
    return _linux_arm64_dependency_blocker(spconv_dependency)


def _linux_arm64_install_source_build_state(
    *,
    plan: dict[str, object],
    python_exe: Path,
    pip: list[str],
    unirig_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline_stage = {"status": "ready", "ready": True, "blocker_codes": [], "blockers": []}
    pyg_dependency = next(
        (
            dict(dependency)
            for dependency in plan.get("dependencies", [])
            if isinstance(dependency, dict) and str(dependency.get("name") or "") == "pyg"
        ),
        {},
    )
    spconv_dependency = next(
        (
            dict(dependency)
            for dependency in plan.get("dependencies", [])
            if isinstance(dependency, dict) and str(dependency.get("name") or "") == "spconv"
        ),
        {},
    )

    pyg_verification = _verify_linux_arm64_pyg_import_smoke(python_exe)
    pyg_blockers = [dict(item) for item in pyg_verification.get("blockers", []) if isinstance(item, dict)]
    pyg_stage = {
        "status": str(pyg_verification.get("status") or "source-build-only"),
        "ready": bool(pyg_verification.get("ready")),
        "mode": "source-build-only",
        "verification": str(pyg_verification.get("verification") or pyg_dependency.get("verification") or "deferred"),
        "packages": [str(item) for item in pyg_dependency.get("packages", LINUX_ARM64_PYG_STAGE_PACKAGES)],
        "blocker_codes": [str(item.get("code") or "") for item in pyg_blockers if str(item.get("code") or "").strip()],
        "blockers": pyg_blockers,
        "checks": [dict(item) for item in pyg_verification.get("checks", []) if isinstance(item, dict)],
        "boundary": str(pyg_verification.get("boundary") or (pyg_blockers[0].get("boundary") if pyg_blockers else "wrapper") or "wrapper"),
    }

    spconv_probe = _probe_linux_arm64_spconv_preparation(
        baseline_blockers=[],
        pyg_stage=pyg_stage,
        spconv_dependency=spconv_dependency,
    )
    if str(pyg_stage.get("status") or "") == "ready":
        if str(spconv_dependency.get("strategy") or "") == LINUX_ARM64_SPCONV_STRATEGY:
            spconv_probe = _run_linux_arm64_spconv_guarded_bringup(
                pip,
                python_exe,
                unirig_dir,
                packages=[str(item) for item in spconv_dependency.get("packages", LINUX_ARM64_SPCONV_STAGE_PACKAGES)],
            )
        else:
            blocker = _linux_arm64_dependency_blocker(spconv_dependency)
            if blocker is not None:
                spconv_probe = {
                    "status": "blocked",
                    "ready": False,
                    "mode": "source-build",
                    "verification": str(spconv_dependency.get("verification") or "import-smoke"),
                    "packages": [
                        str(item).strip()
                        for item in spconv_dependency.get("packages", LINUX_ARM64_SPCONV_STAGE_PACKAGES)
                        if str(item).strip()
                    ] or list(LINUX_ARM64_SPCONV_STAGE_PACKAGES),
                    "checks": [dict(item) for item in spconv_probe.get("checks", []) if isinstance(item, dict)],
                    "blockers": [blocker],
                    "blocker_codes": [str(blocker.get("code") or "")],
                    "boundary": str(blocker.get("boundary") or "wrapper"),
                }

    spconv_stage = {
        "status": str(spconv_probe.get("status") or "deferred"),
        "ready": bool(spconv_probe.get("ready")),
        "mode": str(spconv_probe.get("mode") or "source-build"),
        "verification": str(spconv_probe.get("verification") or spconv_dependency.get("verification") or "import-smoke"),
        "allowed_statuses": [
            str(item)
            for item in spconv_dependency.get("allowed_statuses", ["blocked", "deferred", "build-ready", "ready"])
        ],
        "packages": [str(item) for item in spconv_probe.get("packages", LINUX_ARM64_SPCONV_STAGE_PACKAGES) if str(item).strip()],
        "blocker_codes": [str(item) for item in spconv_probe.get("blocker_codes", []) if str(item).strip()],
        "blockers": [dict(item) for item in spconv_probe.get("blockers", []) if isinstance(item, dict)],
        "checks": [dict(item) for item in spconv_probe.get("checks", []) if isinstance(item, dict)],
        "boundary": str(spconv_probe.get("boundary") or "wrapper"),
    }
    blocked_by_stage = str(spconv_probe.get("blocked_by_stage") or "").strip()
    if blocked_by_stage:
        spconv_stage["blocked_by_stage"] = blocked_by_stage

    source_build = {
        "status": "blocked",
        "current_stage": "spconv" if str(pyg_stage.get("status") or "") == "ready" else "pyg",
        "non_blender_runtime_ready": False,
        "stages": {
            "baseline": baseline_stage,
            "pyg": pyg_stage,
            "spconv": spconv_stage,
        },
    }

    blocked = [*pyg_blockers, *spconv_stage["blockers"]]
    return source_build, blocked


def _install_runtime_packages(ext_dir: Path, payload: dict) -> dict[str, object]:
    if os.environ.get("UNIRIG_SETUP_SKIP_INSTALL") == "1":
        _log("skipping dependency installation because UNIRIG_SETUP_SKIP_INSTALL=1")
        return {"status": "skipped", "profile": "pinned-upstream-wrapper", "steps": []}

    plan = build_install_plan(host_os=_executor_host_os(), host_arch=platform.machine())
    runtime_root = _runtime_root(ext_dir)
    unirig_dir = _runtime_unirig_dir(ext_dir)
    requirements = _filtered_requirements_path(unirig_dir, runtime_root, host_class=str(plan.get("host_class") or "unknown"))
    python_exe = _venv_python(ext_dir / "venv")
    pip = [str(python_exe), "-m", "pip"]
    if str(plan.get("install_mode")) == "diagnostic-only":
        _log(
            "Skipping risky dependency installation because the current install plan is diagnostic-only: "
            f"host_class={plan.get('host_class')} deferred={plan.get('deferred', [])}"
        )
        return _diagnostic_only_install_result(plan)
    if str(plan.get("host_class")) == "linux-arm64":
        _run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"])
        _run(pip + ["install", "--index-url", TORCH_INDEX_URL, *TORCH_PACKAGES], cwd=unirig_dir)
        _run(pip + ["install", "-r", str(requirements)], cwd=unirig_dir)
        pyg_dependency = next(
            (
                dict(dependency)
                for dependency in plan.get("dependencies", [])
                if isinstance(dependency, dict) and str(dependency.get("name") or "") == "pyg"
            ),
            {},
        )
        if str(pyg_dependency.get("strategy") or "") != "linux-arm64-source-build-only":
            raise RuntimeError(
                "Linux ARM64 staged install requires a PyG dependency entry with strategy 'linux-arm64-source-build-only'."
            )
        installed_packages = _install_linux_arm64_pyg_source_build(pip, unirig_dir, pyg_dependency)
        source_build, blocked = _linux_arm64_install_source_build_state(
            plan=plan,
            python_exe=python_exe,
            pip=pip,
            unirig_dir=unirig_dir,
        )
        wrapper_python_version = None
        try:
            wrapper_python_version = _probe_python_version(python_exe)
        except (OSError, RuntimeError):
            wrapper_python_version = None
        bpy_evidence = _linux_arm64_bpy_evidence(
            payload,
            wrapper_python_version=wrapper_python_version,
        )
        bpy_stage = _linux_arm64_bpy_stage_payload(bpy_evidence)
        source_build["external_blender"] = bpy_evidence
        source_build["bpy_evidence_class"] = str((bpy_stage or {}).get("status") or "missing")
        source_build.setdefault("stages", {})
        source_build["stages"]["bpy"] = bpy_stage
        for dependency in plan.get("dependencies", []):
            if not isinstance(dependency, dict) or str(dependency.get("name") or "") in {"pyg", "spconv"}:
                continue
            if str(dependency.get("name") or "") == "bpy" and _linux_arm64_bpy_stage_allows_staged_testing(bpy_stage):
                continue
            blocker = _linux_arm64_dependency_blocker(dependency)
            if blocker is not None:
                blocked.append(blocker)
        result_status = "blocked"
        if not blocked and _linux_arm64_bpy_stage_allows_staged_testing(bpy_stage):
            result_status = "partial"
            source_build["status"] = "partial"
        elif not blocked:
            result_status = "ready"
            source_build["status"] = "ready"
        return {
            "status": result_status,
            "profile": "linux-arm64-runtime-bringup",
            "host_class": "linux-arm64",
            "steps": [
                "bootstrap-python",
                "torch",
                "upstream-requirements",
                "linux-arm64-pyg-source-build",
                *(["linux-arm64-spconv-guarded-source-build"] if str(source_build["stages"]["pyg"].get("status") or "") == "ready" else []),
            ],
            "installed": {"pyg": installed_packages},
            "source_build": source_build,
            "deferred_work": [str(item) for item in plan.get("deferred", [])],
            "blocked": blocked,
        }
    windows_packages: dict[str, str] | None = None
    if os.name == "nt":
        windows_packages = _resolve_windows_executor_packages(plan)
    _run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    _run(pip + ["install", "--index-url", TORCH_INDEX_URL, *TORCH_PACKAGES], cwd=unirig_dir)
    _run(pip + ["install", "-r", str(requirements)], cwd=unirig_dir)
    wrapper_extras = ["install", "-f", PYG_INDEX_URL, "--no-cache-dir", *PYG_PACKAGES, NUMPY_PIN, "pygltflib>=1.15.0"]
    if os.name != "nt":
        wrapper_extras.append(SPCONV_PACKAGE)
    _run(pip + wrapper_extras, cwd=unirig_dir)
    windows_steps: list[str] = []
    if os.name == "nt":
        assert windows_packages is not None
        _install_windows_spconv_stack(
            pip,
            unirig_dir,
            cumm_package=windows_packages["cumm_package"],
            spconv_package=windows_packages["spconv_package"],
        )
        _install_windows_triton(pip, unirig_dir, package=windows_packages["triton_package"])
        _install_windows_flash_attn(pip, runtime_root, unirig_dir, wheel_url=windows_packages["flash_attn_wheel"])
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
    del bootstrap_resolution
    host_class = _classify_host()
    arm64_baseline: dict[str, object] | None = None
    host = {
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "platform_tag": _host_platform_tag(),
    }
    blockers: list[dict[str, object]] = []
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
    if host_class == "linux-arm64" and not blocked:
        arm64_checks, arm64_blockers, arm64_blocked, arm64_baseline = _linux_arm64_preflight_checks(
            python_exe, python_version=python_version
        )
        checks.extend(arm64_checks)
        blockers.extend(arm64_blockers)
        blocked.extend(arm64_blocked)
    report = {
        "status": "ready" if not blocked else "blocked",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "observed": {
            "python_exe": str(python_exe),
            "python_version": python_version,
            "requested_host_python": str(requested_host_python or python_exe),
        },
        "checks": checks,
        "blockers": blockers,
        "blocked": blocked,
        "repeatability": {
            "checklist_file": "logs/bootstrap-preflight-checklist.txt",
            "report_file": "logs/bootstrap-preflight.json",
        },
    }
    if host_class == "linux-arm64" and arm64_baseline is not None:
        report["baseline"] = arm64_baseline
        report["source_build"] = _linux_arm64_staged_preflight_state(
            report_status=str(report["status"]),
            plan=build_install_plan(host_os="linux", host_arch="aarch64"),
            baseline=arm64_baseline,
            python_exe=python_exe,
            payload=payload,
        )
    return report


def _preflight_failure_message(preflight: dict) -> str:
    blocked = preflight.get("blocked") or []
    if blocked:
        return "Bootstrap preflight blocked: " + " | ".join(str(item) for item in blocked)
    return "Bootstrap preflight blocked for an unknown reason. Inspect .unirig-runtime/logs/bootstrap-preflight.json."


def _linux_arm64_preflight_allows_staged_provisioning(
    preflight: dict[str, object], planner: dict[str, object]
) -> bool:
    if str(planner.get("host_class") or "") != "linux-arm64":
        return False
    if str(planner.get("install_mode") or "") != "staged-source-build":
        return False
    if str(preflight.get("status") or "") != "blocked":
        return False

    source_build = preflight.get("source_build")
    if not isinstance(source_build, dict):
        return False
    stages = source_build.get("stages")
    if not isinstance(stages, dict):
        return False
    baseline = stages.get("baseline")
    if not isinstance(baseline, dict):
        return False
    return bool(baseline.get("ready"))


def _install_result_blocked_message(install_result: dict[str, object]) -> str:
    blocked = [
        str(item.get("message") or "").strip()
        for item in install_result.get("blocked", [])
        if isinstance(item, dict) and str(item.get("message") or "").strip()
    ]
    if blocked:
        return "Linux ARM64 staged provisioning completed its reachable PyG tranche, but full runtime remains blocked by remaining blockers: " + " | ".join(blocked)
    return "Linux ARM64 staged provisioning completed its reachable PyG tranche, but full runtime remains blocked by remaining blockers."


def _install_result_partial_message(install_result: dict[str, object]) -> str:
    source_build = install_result.get("source_build") if isinstance(install_result.get("source_build"), dict) else {}
    bpy_stage = source_build.get("stages", {}).get("bpy") if isinstance(source_build.get("stages"), dict) else {}
    if _linux_arm64_bpy_stage_allows_staged_testing(bpy_stage if isinstance(bpy_stage, dict) else None):
        return (
            "Linux ARM64 staged provisioning completed through the current reachable dependency tranche and preserved external Blender bpy smoke evidence for staged testing, "
            "but full wrapper runtime remains blocked until bpy is supported inside the wrapper-owned runtime boundary."
        )
    return "Linux ARM64 staged provisioning completed partially, but wrapper-owned runtime readiness remains blocked."


def _copy_jsonish_dict(value: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return json.loads(json.dumps(value))


def _copy_jsonish_list(values: object) -> list[object]:
    if not isinstance(values, list):
        return []
    return json.loads(json.dumps(values))


def _preflight_state_payload(preflight: dict, planner: dict[str, object]) -> dict[str, object]:
    payload = _copy_jsonish_dict(preflight)
    payload["host_class"] = str(planner.get("host_class") or payload.get("host_class") or "unknown")
    payload["support_posture"] = str(planner.get("support_posture") or payload.get("support_posture") or "unknown")
    payload["blockers"] = _copy_jsonish_list(payload.get("blockers"))
    return payload


def _linux_arm64_bpy_deferred_stage_payload(
    planner: dict[str, object],
    *,
    deferred_work: list[str],
    blockers: list[object],
    existing_stage: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if str(planner.get("host_class") or "") != "linux-arm64":
        return None

    deferred_codes = {str(item).strip() for item in deferred_work if str(item).strip()}
    if "bpy-portability" not in deferred_codes:
        return None

    bpy_dependency = next(
        (
            dict(dependency)
            for dependency in planner.get("dependencies", [])
            if isinstance(dependency, dict) and str(dependency.get("name") or "") == "bpy"
        ),
        {},
    )
    bpy_blocker = next(
        (
            dict(item)
            for item in blockers
            if isinstance(item, dict)
            and str(item.get("dependency") or "").strip() == "bpy"
        ),
        _linux_arm64_dependency_blocker(bpy_dependency) or {},
    )
    stage = _copy_jsonish_dict(existing_stage)
    if str(stage.get("status") or "") in LINUX_ARM64_BPY_EVIDENCE_CLASSES:
        stage["ready"] = False
        return stage
    stage.setdefault("status", "deferred")
    stage.setdefault("ready", False)
    stage.setdefault("reason_code", str(bpy_dependency.get("reason_code") or bpy_blocker.get("code") or "bpy-portability"))
    if bpy_blocker:
        stage.setdefault("message", str(bpy_blocker.get("message") or "").strip())
        stage.setdefault("blockers", [bpy_blocker])
        stage.setdefault("blocker_codes", [str(bpy_blocker.get("code") or "").strip()])
        stage.setdefault("boundary", str(bpy_blocker.get("boundary") or "upstream"))
    return stage


def _linux_arm64_state_source_build_payload(
    preflight: dict,
    planner: dict[str, object],
    *,
    deferred_work: list[str],
    install_result: dict[str, object] | None = None,
) -> dict[str, object]:
    if str(planner.get("host_class") or "") != "linux-arm64":
        return {}

    staged = _copy_jsonish_dict(preflight.get("source_build") if isinstance(preflight.get("source_build"), dict) else {})
    install_source_build = _copy_jsonish_dict(
        install_result.get("source_build") if isinstance((install_result or {}).get("source_build"), dict) else {}
    )
    if install_source_build:
        staged.update({
            key: value for key, value in install_source_build.items() if key != "stages"
        })
        install_stages = install_source_build.get("stages")
        if isinstance(install_stages, dict):
            staged_stages = _copy_jsonish_dict(staged.get("stages") if isinstance(staged.get("stages"), dict) else {})
            for stage_name, stage_payload in install_stages.items():
                if not isinstance(stage_payload, dict):
                    continue
                merged_stage = _copy_jsonish_dict(staged_stages.get(stage_name) if isinstance(staged_stages.get(stage_name), dict) else {})
                merged_stage.update(_copy_jsonish_dict(stage_payload))
                staged_stages[str(stage_name)] = merged_stage
            staged["stages"] = staged_stages
    payload = dict(staged)
    payload["status"] = str(staged.get("status") or preflight.get("status") or "unknown")
    payload["mode"] = str(planner.get("install_mode") or staged.get("mode") or "unknown")
    payload["host_class"] = "linux-arm64"
    payload["support_posture"] = str(planner.get("support_posture") or staged.get("support_posture") or "unknown")
    payload["baseline"] = _copy_jsonish_dict(preflight.get("baseline") if isinstance(preflight.get("baseline"), dict) else {})
    payload["stages"] = _copy_jsonish_dict(staged.get("stages") if isinstance(staged.get("stages"), dict) else {})
    payload["blockers"] = _copy_jsonish_list((install_result or {}).get("blocked") if isinstance((install_result or {}).get("blocked"), list) else preflight.get("blockers"))
    payload["blocked_reasons"] = [
        str(item.get("message") or "") for item in payload["blockers"] if isinstance(item, dict) and str(item.get("message") or "").strip()
    ] or [str(item) for item in preflight.get("blocked", [])]
    payload["deferred_work"] = [str(item) for item in deferred_work]
    payload["non_blender_runtime_ready"] = bool(staged.get("non_blender_runtime_ready", False))
    if isinstance(staged.get("external_blender"), dict):
        payload["external_blender"] = _copy_jsonish_dict(staged.get("external_blender"))
    bpy_stage = _linux_arm64_bpy_deferred_stage_payload(
        planner,
        deferred_work=payload["deferred_work"],
        blockers=payload["blockers"],
        existing_stage=payload["stages"].get("bpy") if isinstance(payload["stages"].get("bpy"), dict) else None,
    )
    if bpy_stage is not None:
        payload["stages"]["bpy"] = bpy_stage
        payload["bpy_evidence_class"] = str(bpy_stage.get("status") or payload.get("bpy_evidence_class") or "")
        payload.setdefault("external_blender", {})
        if isinstance(payload["external_blender"], dict):
            payload["external_blender"].setdefault("classification", _copy_jsonish_dict(bpy_stage))
    payload["executable_boundary"] = _linux_arm64_extract_merge_boundary_payload(payload)
    return payload


def _install_plan_summary(
    planner: dict[str, object],
    *,
    install_state: str,
    install_result: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = {
        "host_class": str(planner.get("host_class") or "unknown"),
        "support_posture": str(planner.get("support_posture") or "unknown"),
        "install_mode": str(planner.get("install_mode") or "unknown"),
        "status": install_state,
    }
    profile = str((install_result or {}).get("profile") or planner.get("profile") or "").strip()
    if profile:
        summary["profile"] = profile
    return summary


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
    planner: dict[str, object],
    install_state: str = "ready",
    install_result: dict[str, object] | None = None,
) -> None:
    runtime_root = _runtime_root(ext_dir)
    deferred_work = [str(item) for item in (install_result or {}).get("deferred_work", planner.get("deferred", []))]
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
            "planner": _copy_jsonish_dict(planner),
            "preflight": _preflight_state_payload(preflight, planner),
            "source_build": _linux_arm64_state_source_build_payload(preflight, planner, deferred_work=deferred_work, install_result=install_result),
            "install_plan": {"summary": _install_plan_summary(planner, install_state=install_state, install_result=install_result)},
            "deferred_work": deferred_work,
            "install_state": install_state,
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
    planner: dict[str, object] | None = None,
    install_result: dict[str, object] | None = None,
) -> None:
    runtime_root = _runtime_root(ext_dir)
    planner = dict(planner or build_install_plan())
    install_state = "blocked" if preflight.get("status") == "blocked" else "error"
    deferred_work = [str(item) for item in (install_result or {}).get("deferred_work", planner.get("deferred", []))]
    bootstrap.save_state(
        {
            "bootstrap_version": bootstrap.BOOTSTRAP_VERSION,
            "install_state": install_state,
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
            "planner": _copy_jsonish_dict(planner),
            "preflight": _preflight_state_payload(preflight, planner),
            "source_build": _linux_arm64_state_source_build_payload(preflight, planner, deferred_work=deferred_work, install_result=install_result),
            "install_plan": {"summary": _install_plan_summary(planner, install_state=install_state, install_result=install_result)},
            "deferred_work": deferred_work,
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
    planner = build_install_plan()
    preflight = _preflight_check_summary(
        bootstrap_python,
        payload,
        requested_host_python=requested_host_python,
        bootstrap_resolution=bootstrap_resolution,
    )
    _write_preflight_artifacts(ext_dir, preflight)
    for line in bootstrap.preflight_checklist_lines(preflight):
        _log(line)

    allow_linux_arm64_staged_provisioning = _linux_arm64_preflight_allows_staged_provisioning(preflight, planner)
    if preflight.get("status") != "ready" and not allow_linux_arm64_staged_provisioning:
        message = _preflight_failure_message(preflight)
        _write_error_state(
            ext_dir,
            message,
            preflight,
            requested_host_python=requested_host_python,
            bootstrap_resolution=bootstrap_resolution,
            planner=planner,
        )
        raise SystemExit(message + " Repair the host prerequisites, then rerun setup.py before provisioning the runtime.")
    if allow_linux_arm64_staged_provisioning:
        _log(
            "Linux ARM64 preflight remains blocked for full runtime readiness, but baseline prerequisites are ready so staged non-Blender provisioning can continue."
        )

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
        install_result = _install_runtime_packages(ext_dir, payload)
        if str(install_result.get("status") or "") == "blocked":
            message = _install_result_blocked_message(install_result)
            _write_error_state(
                ext_dir,
                message,
                preflight,
                vendor_source=vendor_source,
                requested_host_python=requested_host_python,
                bootstrap_resolution=bootstrap_resolution,
                planner=planner,
                install_result=install_result,
            )
            raise SystemExit(message)
        if str(install_result.get("status") or "") == "partial":
            _write_state(
                ext_dir,
                source_ref,
                preflight,
                vendor_source=vendor_source,
                requested_host_python=requested_host_python,
                bootstrap_resolution=bootstrap_resolution,
                planner=planner,
                install_state="partial",
                install_result=install_result,
            )
            _log(_install_result_partial_message(install_result))
            return 0
        _run_post_setup_smoke_checks(ext_dir)
        _write_state(
            ext_dir,
            source_ref,
            preflight,
            vendor_source=vendor_source,
            requested_host_python=requested_host_python,
            bootstrap_resolution=bootstrap_resolution,
            planner=planner,
            install_state="ready",
            install_result=install_result,
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
            planner=planner,
        )
        raise SystemExit(
            "UniRig bootstrap failed while provisioning the real runtime. "
            f"Details: {exc}. Repair the extension after fixing the environment/dependency issue."
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
