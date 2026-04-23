from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import blender_bridge, io
from .bootstrap import LINUX_ARM64_PERSISTED_STAGE_PROOF_NAMES, REAL_RUNTIME_MODE, RuntimeContext, UniRigError, stage_environment


class PipelineError(UniRigError):
    pass


ProgressFn = Callable[..., None]
LogFn = Callable[..., None]
CANONICAL_STAGE_NAMES = ("extract-prepare", "skeleton", "extract-skin", "skin", "merge")
SKELETON_TASK = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
SKIN_TASK = "configs/task/quick_inference_unirig_skin.yaml"
SKIN_DATA_NAME = "raw_data.npz"
MERGE_REQUIRE_SUFFIX = "obj,fbx,FBX,dae,glb,gltf,vrm"
EXTRACT_CONFIG = "configs/data/quick_inference.yaml"
EXTRACT_FACES_TARGET_COUNT = 50000
PIPELINE_PUBLIC_ERROR = "UniRig processing failed. Inspect extension runtime logs for details."
WINDOWS_NATIVE_ACCESS_VIOLATION_CODES = {3221225477, -1073741819}
RUNTIME_STAGE_TOKEN = "run-processor"
WRAPPER_RUNTIME_BOUNDARY_OWNER = "wrapper-venv-python"
BLENDER_SUBPROCESS_TIMEOUT_SECONDS = 1800
HEARTBEAT_INTERVAL_SECONDS = 10
HEARTBEAT_POLL_SECONDS = 1.0
_MONOTONIC = time.monotonic
_SLEEP = time.sleep


@dataclass(frozen=True)
class ExecutionStage:
    name: str
    command: list[str]
    cwd: Path
    success_path: Path
    runtime_boundary_owner: str = WRAPPER_RUNTIME_BOUNDARY_OWNER
    runtime_input_name: str = ""
    runtime_input_path: Path | None = None
    log_stage_name: str | None = None
    payload_seed: int = 12345


def run(
    mesh_path: Path,
    params: dict,
    context: RuntimeContext,
    progress: ProgressFn | None = None,
    log: LogFn | None = None,
    workspace_dir: Path | None = None,
) -> Path:
    progress = progress or (lambda percent, label, **meta: None)
    log = log or (lambda message, **meta: None)

    mesh_path = io.validate_mesh_input(mesh_path)
    run_dir = io.create_run_dir(context)
    staged_input = io.stage_input(mesh_path, run_dir)
    progress(12, "input validated")
    log("input staged for processing")

    if context.runtime_mode != REAL_RUNTIME_MODE:
        raise PipelineError(
            "UniRig runtime is not configured for deterministic upstream execution. "
            "Run setup.py to provision the real runtime."
        )

    published = _run_real_pipeline(
        mesh_path=mesh_path,
        staged_input=staged_input,
        params=params,
        context=context,
        run_dir=run_dir,
        progress=progress,
        log=log,
        workspace_dir=workspace_dir,
    )
    progress(98, "output published")
    return published


def _run_real_pipeline(
    mesh_path: Path,
    staged_input: Path,
    params: dict,
    context: RuntimeContext,
    run_dir: Path,
    progress: ProgressFn,
    log: LogFn,
    workspace_dir: Path | None,
) -> Path:
    seed = int(params.get("seed", 12345))
    prepared = io.prepare_input_mesh(staged_input, run_dir, context)
    progress(20, "input prepared", stage="extract-prepare", status="complete")

    plan = build_execution_plan(
        mesh_path=mesh_path,
        prepared_path=prepared,
        run_dir=run_dir,
        context=context,
        seed=seed,
    )
    staged_files = [stage.runtime_input_path for stage in plan if stage.runtime_input_path is not None]

    try:
        shutil.copy2(prepared, _require_runtime_input_path(plan[0]))
        _run_stage(plan[0], context=context, run_dir=run_dir, log=log)
        progress(35, "prepare/extract complete", stage="extract-prepare", status="complete")

        _run_stage(plan[1], context=context, run_dir=run_dir, log=log)
        progress(55, "skeleton stage complete", stage="skeleton", status="complete")

        shutil.copy2(plan[1].success_path, _require_runtime_input_path(plan[2]))
        _run_stage(plan[2], context=context, run_dir=run_dir, log=log)
        _run_stage(plan[3], context=context, run_dir=run_dir, log=log)
        progress(78, "skin stage complete", stage="skin", status="complete")

        _run_stage(plan[4], context=context, run_dir=run_dir, log=log)
        progress(92, "merge stage complete", stage="merge", status="complete")
        return io.publish_output(plan[4].success_path, mesh_path, context=context, workspace_dir=workspace_dir)
    finally:
        _cleanup_staged_files(*staged_files)


def build_execution_plan(
    *,
    mesh_path: Path,
    prepared_path: Path,
    run_dir: Path,
    context: RuntimeContext,
    seed: int,
) -> list[ExecutionStage]:
    del mesh_path

    stage_token = RUNTIME_STAGE_TOKEN
    blender_subprocess_stages = _linux_arm64_blender_subprocess_stage_names(context)
    extract_prepare_force_override = _should_force_extract_prepare_override(context)
    staged_prepared = context.unirig_dir / f".modly_stage_input_{stage_token}{prepared_path.suffix.lower()}"
    staged_skeleton = context.unirig_dir / f".modly_stage_skeleton_{stage_token}.fbx"
    skeleton_output = run_dir / "skeleton_stage.fbx"
    skin_output = run_dir / "skin_stage.fbx"
    merged_output = run_dir / "merged.glb"
    skeleton_npz_dir = run_dir / "skeleton_npz"
    skin_npz_dir = run_dir / "skin_npz"

    return [
        ExecutionStage(
            name="extract-prepare",
            command=_extract_command(
                input_name=staged_prepared.name,
                output_dir=skeleton_npz_dir,
                force_override=extract_prepare_force_override,
                context=context,
                extract_token=f"modly_extract_{stage_token}",
            ),
            cwd=context.unirig_dir,
            success_path=skeleton_npz_dir / staged_prepared.stem / SKIN_DATA_NAME,
            runtime_boundary_owner=_runtime_boundary_owner_for_stage("extract-prepare", blender_subprocess_stages),
            runtime_input_name=staged_prepared.name,
            runtime_input_path=staged_prepared,
            log_stage_name="extract-prepare",
            payload_seed=seed,
        ),
        ExecutionStage(
            name="skeleton",
            command=_prediction_command(
                task=SKELETON_TASK,
                input_name=staged_prepared.name,
                output_path=skeleton_output,
                npz_dir=skeleton_npz_dir,
                seed=seed,
                include_data_name=False,
                context=context,
            ),
            cwd=context.unirig_dir,
            success_path=skeleton_output,
            runtime_boundary_owner=_runtime_boundary_owner_for_stage("skeleton", blender_subprocess_stages),
            runtime_input_name=staged_prepared.name,
            runtime_input_path=staged_prepared,
            log_stage_name="skeleton",
            payload_seed=seed,
        ),
        ExecutionStage(
            name="extract-skin",
            command=_extract_command(
                input_name=staged_skeleton.name,
                output_dir=skin_npz_dir,
                force_override=True,
                context=context,
                extract_token=f"modly_extract_{stage_token}",
            ),
            cwd=context.unirig_dir,
            success_path=skin_npz_dir / staged_skeleton.stem / SKIN_DATA_NAME,
            runtime_boundary_owner=_runtime_boundary_owner_for_stage("extract-skin", blender_subprocess_stages),
            runtime_input_name=staged_skeleton.name,
            runtime_input_path=staged_skeleton,
            log_stage_name="extract-skin",
            payload_seed=seed,
        ),
        ExecutionStage(
            name="skin",
            command=_prediction_command(
                task=SKIN_TASK,
                input_name=staged_skeleton.name,
                output_path=skin_output,
                npz_dir=skin_npz_dir,
                seed=seed,
                include_data_name=True,
                context=context,
            ),
            cwd=context.unirig_dir,
            success_path=skin_output,
            runtime_boundary_owner=_runtime_boundary_owner_for_stage("skin", blender_subprocess_stages),
            runtime_input_name=staged_skeleton.name,
            runtime_input_path=staged_skeleton,
            log_stage_name="skin",
        ),
        ExecutionStage(
            name="merge",
            command=_merge_command(input_path=skin_output, prepared_path=prepared_path, output_path=merged_output, context=context),
            cwd=context.unirig_dir,
            success_path=merged_output,
            runtime_boundary_owner=_runtime_boundary_owner_for_stage("merge", blender_subprocess_stages),
            payload_seed=seed,
        ),
    ]


def _runtime_boundary_owner_for_stage(stage_name: str, blender_subprocess_stages: set[str]) -> str:
    if stage_name in blender_subprocess_stages:
        return blender_bridge.BLENDER_SUBPROCESS_MODE
    return WRAPPER_RUNTIME_BOUNDARY_OWNER


def _linux_arm64_blender_subprocess_stage_names(context: RuntimeContext) -> set[str]:
    qualification_override = _linux_arm64_qualification_stage_names(context)
    if qualification_override is not None:
        return qualification_override

    if _host_platform_tag(context) != "linux-aarch64":
        return set()

    source_build = context.source_build if isinstance(context.source_build, dict) else {}
    boundary = source_build.get("executable_boundary") if isinstance(source_build.get("executable_boundary"), dict) else {}
    extract_merge = boundary.get("extract_merge") if isinstance(boundary.get("extract_merge"), dict) else {}
    if not extract_merge:
        return set()
    if not bool(extract_merge.get("enabled")):
        return set()
    if not bool(extract_merge.get("ready")):
        return set()

    proof_kind = str(extract_merge.get("proof_kind") or extract_merge.get("mode") or "").strip()
    optional_owner = str(extract_merge.get("optional_owner") or "").strip()
    if proof_kind != blender_bridge.BLENDER_SUBPROCESS_MODE and optional_owner != blender_bridge.BLENDER_SUBPROCESS_MODE:
        return set()

    supported_stages = extract_merge.get("supported_stages")
    if not isinstance(supported_stages, list):
        supported_stages = list(blender_bridge.BLENDER_STAGE_NAMES)

    allowed = {str(stage).strip() for stage in supported_stages if str(stage).strip() in blender_bridge.BLENDER_STAGE_NAMES}
    if {"extract-prepare", "extract-skin", "merge"}.issubset(allowed):
        allowed.add("skeleton")
    if "extract-skin" in allowed:
        allowed.add("skin")
    if set(LINUX_ARM64_PERSISTED_STAGE_PROOF_NAMES).issubset(allowed):
        allowed.add("merge")
    if not allowed:
        return set()
    return allowed


def _linux_arm64_qualification_stage_names(context: RuntimeContext) -> set[str] | None:
    source_build = context.source_build if isinstance(context.source_build, dict) else {}
    qualification = source_build.get("qualification") if isinstance(source_build.get("qualification"), dict) else {}
    extract_merge = qualification.get("extract_merge") if isinstance(qualification.get("extract_merge"), dict) else {}
    mode = str(extract_merge.get("mode") or "").strip().lower()
    if not mode:
        return None

    if _host_platform_tag(context) != "linux-aarch64":
        raise PipelineError(
            "UniRig qualification extract/merge mode overrides are supported only on linux-aarch64 hosts. "
            f"Observed host: {_host_platform_tag(context)}"
        )

    requested_stages = extract_merge.get("stages")
    if requested_stages is None:
        requested_stages = list(blender_bridge.BLENDER_STAGE_NAMES)
    if not isinstance(requested_stages, list):
        raise PipelineError("UniRig qualification extract/merge stages must be provided as a list when mode overrides are used.")

    normalized_stages = [str(stage).strip() for stage in requested_stages if str(stage).strip()]
    unsupported_stages = [stage for stage in normalized_stages if stage not in blender_bridge.BLENDER_STAGE_NAMES]
    if unsupported_stages:
        supported = ", ".join(sorted(blender_bridge.BLENDER_STAGE_NAMES))
        invalid = ", ".join(unsupported_stages)
        raise PipelineError(
            "UniRig qualification extract/merge mode overrides received unsupported stages: "
            f"{invalid}. Supported stages: {supported}."
        )

    if mode == "seam":
        return set(normalized_stages)
    if mode in {"forced-fallback", "wrapper"}:
        return set()
    raise PipelineError(
        "UniRig qualification extract/merge mode override must be one of: seam, forced-fallback, wrapper. "
        f"Received: {mode or '<empty>'}"
    )


def _host_platform_tag(context: RuntimeContext) -> str:
    host = context.platform_policy.get("host") if isinstance(context.platform_policy, dict) else {}
    host_os = str((host or {}).get("os") or "").strip().lower()
    host_arch = str((host or {}).get("arch") or "").strip().lower()
    return f"{host_os}-{host_arch}"


def _extract_command(*, input_name: str, output_dir: Path, force_override: bool, context: RuntimeContext, extract_token: str) -> list[str]:
    return _wrapper_owned_stage_command(
        context,
        "-m",
        "src.data.extract",
        f"--config={EXTRACT_CONFIG}",
        f"--require_suffix={MERGE_REQUIRE_SUFFIX}",
        f"--force_override={str(force_override).lower()}",
        "--num_runs=1",
        "--id=0",
        f"--time={extract_token}",
        f"--faces_target_count={EXTRACT_FACES_TARGET_COUNT}",
        f"--input={input_name}",
        f"--output_dir={output_dir}",
    )


def _should_force_extract_prepare_override(context: RuntimeContext) -> bool:
    return _host_platform_tag(context) == "linux-aarch64"


def _prediction_command(
    *,
    task: str,
    input_name: str,
    output_path: Path,
    npz_dir: Path,
    seed: int,
    include_data_name: bool,
    context: RuntimeContext,
) -> list[str]:
    return _wrapper_owned_stage_command(
        context,
        "run.py",
        f"--task={task}",
        f"--seed={seed}",
        f"--input={input_name}",
        f"--output={output_path}",
        f"--npz_dir={npz_dir}",
        *([f"--data_name={SKIN_DATA_NAME}"] if include_data_name else []),
    )


def _merge_command(*, input_path: Path, prepared_path: Path, output_path: Path, context: RuntimeContext) -> list[str]:
    return _wrapper_owned_stage_command(
        context,
        "-m",
        "src.inference.merge",
        f"--require_suffix={MERGE_REQUIRE_SUFFIX}",
        "--num_runs=1",
        "--id=0",
        f"--source={input_path}",
        f"--target={prepared_path}",
        f"--output={output_path}",
    )


def _wrapper_owned_stage_command(context: RuntimeContext, *args: str) -> list[str]:
    return [str(context.venv_python), *args]


def _require_runtime_input_path(stage: ExecutionStage) -> Path:
    if stage.runtime_input_path is None:
        raise PipelineError(f"UniRig {stage.name} stage is missing its deterministic runtime input path.")
    stage.runtime_input_path.parent.mkdir(parents=True, exist_ok=True)
    return stage.runtime_input_path


def _emit_stage_start(stage: str, log: LogFn) -> None:
    log(f"running {stage} stage", stage=stage, kind="stage-start", status="running")


def _emit_stage_heartbeat(stage: str, elapsed_seconds: int, log: LogFn) -> None:
    log(
        f"{stage} stage still running ({elapsed_seconds}s elapsed)",
        stage=stage,
        kind="heartbeat",
        status="running",
        elapsedSeconds=elapsed_seconds,
    )


def _heartbeat_elapsed_seconds(*, started_at: float, current_time: float) -> int:
    elapsed = int(current_time - started_at)
    if elapsed < 0:
        return 0
    return elapsed


def _sleep_until_next_heartbeat_check(*, started_at: float, next_heartbeat_at: int) -> None:
    now = _MONOTONIC()
    elapsed = _heartbeat_elapsed_seconds(started_at=started_at, current_time=now)
    remaining = next_heartbeat_at - elapsed
    if remaining <= 0:
        _SLEEP(0)
        return
    _SLEEP(min(HEARTBEAT_POLL_SECONDS, float(remaining)))


def _run_stage(stage: ExecutionStage, *, context: RuntimeContext, run_dir: Path, log: LogFn | None = None) -> None:
    log = log or (lambda message, **meta: None)
    _emit_stage_start(stage.name, log)
    _run_stage_with_heartbeat(stage=stage, context=context, run_dir=run_dir, log=log)


def _run_stage_with_heartbeat(*, stage: ExecutionStage, context: RuntimeContext, run_dir: Path, log: LogFn) -> None:
    started_at = _MONOTONIC()
    next_heartbeat_at = HEARTBEAT_INTERVAL_SECONDS

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_stage_blocking, stage=stage, context=context, run_dir=run_dir)
        while not future.done():
            elapsed = _heartbeat_elapsed_seconds(started_at=started_at, current_time=_MONOTONIC())
            while elapsed >= next_heartbeat_at:
                _SLEEP(0)
                if future.done():
                    break
                _emit_stage_heartbeat(stage.name, next_heartbeat_at, log)
                next_heartbeat_at += HEARTBEAT_INTERVAL_SECONDS
            if future.done():
                break
            if future.done():
                break
            _sleep_until_next_heartbeat_check(started_at=started_at, next_heartbeat_at=next_heartbeat_at)
        future.result()


def _run_stage_blocking(*, stage: ExecutionStage, context: RuntimeContext, run_dir: Path) -> None:
    if stage.runtime_boundary_owner == blender_bridge.BLENDER_SUBPROCESS_MODE:
        _run_blender_subprocess_stage(stage=stage, context=context, run_dir=run_dir)
        return
    _run_command(
        stage.command,
        cwd=stage.cwd,
        context=context,
        success_path=stage.success_path,
        run_dir=run_dir,
        stage_name=stage.name,
        log_stage_name=stage.log_stage_name,
    )


def _run_blender_subprocess_stage(*, stage: ExecutionStage, context: RuntimeContext, run_dir: Path) -> None:
    payload_path = blender_bridge.payload_path_for_run_dir(run_dir)
    result_path = blender_bridge.result_path_for_run_dir(run_dir)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)

    payload = _build_blender_stage_payload(stage=stage, run_dir=run_dir)
    payload_path.write_text(blender_bridge.render_stage_payload_json(payload), encoding="utf-8")
    command = _blender_subprocess_command(context=context, payload_path=payload_path)

    try:
        result = subprocess.run(
            command,
            cwd=stage.cwd,
            env=stage_environment(context=context),
            capture_output=True,
            text=True,
            check=False,
            timeout=BLENDER_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        log_path = _write_stage_log(
            command=command,
            cwd=stage.cwd,
            result=subprocess.CompletedProcess(command, returncode=-1, stdout="", stderr=str(exc)),
            run_dir=run_dir,
            stage_name=stage.log_stage_name or stage.name,
            success_path=stage.success_path,
            tolerated_windows_crash=False,
        )
        _raise_blender_stage_error(
            error_code="launch-failed",
            stage=stage,
            command=command,
            log_path=log_path,
            details=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        log_path = _write_stage_log(
            command=command,
            cwd=stage.cwd,
            result=subprocess.CompletedProcess(
                command,
                returncode=-1,
                stdout=_subprocess_stream_text(exc.output),
                stderr=_subprocess_stream_text(exc.stderr),
            ),
            run_dir=run_dir,
            stage_name=stage.log_stage_name or stage.name,
            success_path=stage.success_path,
            tolerated_windows_crash=False,
        )
        _raise_blender_stage_error(
            error_code="timed-out",
            stage=stage,
            command=command,
            log_path=log_path,
            details=f"Timed out after {exc.timeout} seconds.",
        )

    log_path = _write_stage_log(
        command=command,
        cwd=stage.cwd,
        result=result,
        run_dir=run_dir,
        stage_name=stage.log_stage_name or stage.name,
        success_path=stage.success_path,
        tolerated_windows_crash=False,
    )

    if result.returncode != 0:
        raise PipelineError(
            f"UniRig {stage.name} stage failed with exit code {result.returncode}. Command: {' '.join(command)}\n"
            f"Expected output: {stage.success_path}\n"
            f"Stage log: {log_path}\n"
            f"Tail: {_tail_summary(result)}"
        )

    try:
        marker_path = blender_bridge.parse_result_marker(result.stdout)
    except ValueError as exc:
        _raise_blender_stage_error(
            error_code="result-invalid",
            stage=stage,
            command=command,
            log_path=log_path,
            details=str(exc),
        )
    if marker_path is None:
        _raise_blender_stage_error(
            error_code="marker-missing",
            stage=stage,
            command=command,
            log_path=log_path,
        )
    if not marker_path.exists():
        _raise_blender_stage_error(
            error_code="result-missing",
            stage=stage,
            command=command,
            log_path=log_path,
            details=f"Missing result file: {marker_path}",
        )

    try:
        result_payload = json.loads(marker_path.read_text(encoding="utf-8"))
        normalized = blender_bridge.load_stage_result(
            result_payload,
            expected_stage=stage.name,
            expected_outputs=[stage.success_path],
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _raise_blender_stage_error(
            error_code="result-invalid",
            stage=stage,
            command=command,
            log_path=log_path,
            details=str(exc),
        )

    if normalized["status"] != blender_bridge.BLENDER_STAGE_STATUS_OK:
        _raise_blender_stage_error(
            error_code=normalized["error_code"] or blender_bridge.BLENDER_STAGE_STATUS_FAILED,
            stage=stage,
            command=command,
            log_path=log_path,
            details=normalized["message"] or normalized["error_code"] or blender_bridge.BLENDER_STAGE_STATUS_FAILED,
        )
    if not stage.success_path.exists():
        _raise_blender_stage_error(
            error_code="expected-output-missing",
            stage=stage,
            command=command,
            log_path=log_path,
            details=f"Expected output missing on disk: {stage.success_path}",
        )


def _raise_blender_stage_error(
    *,
    error_code: str,
    stage: ExecutionStage,
    command: list[str],
    log_path: Path,
    details: str = "",
) -> None:
    normalized_code = blender_bridge.validate_failure_code(error_code)
    detail_suffix = f"\nDetails: {details}" if details else ""
    raise PipelineError(
        f"UniRig {stage.name} stage failed ({normalized_code}). Command: {' '.join(command)}\n"
        f"Expected output: {stage.success_path}\n"
        f"Stage log: {log_path}"
        f"{detail_suffix}"
    )


def _subprocess_stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _build_blender_stage_payload(*, stage: ExecutionStage, run_dir: Path) -> dict:
    if stage.name in {"extract-prepare", "extract-skin"}:
        return blender_bridge.build_stage_payload(
            stage=stage.name,
            run_dir=run_dir,
            source_path=_require_runtime_input_path(stage),
            output_dir=Path(_require_command_value(stage.command, "--output_dir")),
            output_path=stage.success_path,
            require_suffix=_require_command_value(stage.command, "--require_suffix"),
            seed=stage.payload_seed,
            extract_token=_require_command_value(stage.command, "--time"),
        )
    if stage.name == "skeleton":
        return blender_bridge.build_stage_payload(
            stage=stage.name,
            run_dir=run_dir,
            source_path=_require_runtime_input_path(stage),
            output_dir=Path(_require_command_value(stage.command, "--npz_dir")),
            output_path=Path(_require_command_value(stage.command, "--output")),
            seed=stage.payload_seed,
        )
    if stage.name == "skin":
        return blender_bridge.build_stage_payload(
            stage=stage.name,
            run_dir=run_dir,
            source_path=_require_runtime_input_path(stage),
            output_dir=Path(_require_command_value(stage.command, "--npz_dir")),
            output_path=Path(_require_command_value(stage.command, "--output")),
            seed=stage.payload_seed,
        )
    if stage.name == "merge":
        return blender_bridge.build_stage_payload(
            stage=stage.name,
            run_dir=run_dir,
            source_path=Path(_require_command_value(stage.command, "--source")),
            target_path=Path(_require_command_value(stage.command, "--target")),
            output_path=Path(_require_command_value(stage.command, "--output")),
            require_suffix=_require_command_value(stage.command, "--require_suffix"),
            seed=stage.payload_seed,
        )
    raise PipelineError(f"UniRig {stage.name} stage is not supported by the Blender subprocess bridge.")


def _blender_subprocess_command(*, context: RuntimeContext, payload_path: Path) -> list[str]:
    blender_executable = _require_blender_subprocess_executable(context)
    bridge_script = Path(blender_bridge.__file__).resolve()
    return [
        blender_executable,
        "--background",
        "--factory-startup",
        "--python",
        str(bridge_script),
        "--",
        str(payload_path),
    ]


def _require_blender_subprocess_executable(context: RuntimeContext) -> str:
    source_build = context.source_build if isinstance(context.source_build, dict) else {}
    boundary = source_build.get("executable_boundary") if isinstance(source_build.get("executable_boundary"), dict) else {}
    extract_merge = boundary.get("extract_merge") if isinstance(boundary.get("extract_merge"), dict) else {}
    candidate = extract_merge.get("candidate") if isinstance(extract_merge.get("candidate"), dict) else {}
    if not candidate:
        external_blender = source_build.get("external_blender") if isinstance(source_build.get("external_blender"), dict) else {}
        candidate = external_blender.get("candidate") if isinstance(external_blender.get("candidate"), dict) else {}

    blender_path = str(candidate.get("path") or "").strip()
    if blender_path:
        return blender_path
    raise PipelineError(
        "UniRig Blender subprocess stage is selected but no Blender executable candidate path was persisted in source_build."
    )


def _require_command_value(command: list[str], option_name: str) -> str:
    prefix = f"{option_name}="
    for item in command:
        if item.startswith(prefix):
            return item[len(prefix) :]
    raise PipelineError(f"UniRig stage command is missing required option {option_name} for Blender subprocess payload generation.")


def _cleanup_staged_files(*staged_files: Path) -> None:
    cleanup_errors: list[str] = []
    for staged_file in staged_files:
        try:
            staged_file.unlink(missing_ok=True)
        except Exception as exc:
            cleanup_errors.append(f"{staged_file}: {exc}")
    if not cleanup_errors:
        return

    message = "UniRig staged-file cleanup failed:\n" + "\n".join(cleanup_errors)
    active_exception = sys.exc_info()[1]
    if active_exception is not None:
        active_exception.add_note(message)
        return
    raise PipelineError(message)


def _run_command(
    command: list[str],
    cwd: Path,
    context: RuntimeContext | None,
    success_path: Path,
    run_dir: Path,
    stage_name: str,
    log_stage_name: str | None = None,
) -> None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=stage_environment(context=context),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        log_path = _write_stage_log(
            command=command,
            cwd=cwd,
            result=subprocess.CompletedProcess(command, returncode=-1, stdout="", stderr=str(exc)),
            run_dir=run_dir,
            stage_name=log_stage_name or stage_name,
            success_path=success_path,
            tolerated_windows_crash=False,
        )
        raise PipelineError(
            f"UniRig {stage_name} stage could not start. Command: {' '.join(command)}\n"
            f"Expected output: {success_path}\n"
            f"Stage log: {log_path}\n"
            f"Details: {exc}"
        ) from exc
    tolerated_windows_crash = _should_tolerate_windows_native_access_violation(
        result=result,
        context=context,
        success_path=success_path,
    )
    log_path = _write_stage_log(
        command=command,
        cwd=cwd,
        result=result,
        run_dir=run_dir,
        stage_name=log_stage_name or stage_name,
        success_path=success_path,
        tolerated_windows_crash=tolerated_windows_crash,
    )
    if (result.returncode == 0 or tolerated_windows_crash) and success_path.exists():
        return
    detail = _tail_summary(result)
    raise PipelineError(
        f"UniRig {stage_name} stage failed with exit code {result.returncode}. Command: {' '.join(command)}\n"
        f"Expected output: {success_path}\n"
        f"Stage log: {log_path}\n"
        f"Tail: {detail}"
    )


def _write_stage_log(
    *,
    command: list[str],
    cwd: Path,
    result: subprocess.CompletedProcess[str],
    run_dir: Path,
    stage_name: str,
    success_path: Path,
    tolerated_windows_crash: bool,
) -> Path:
    log_path = _stage_log_path(run_dir=run_dir, stage_name=stage_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"stage: {stage_name}",
                f"command: {' '.join(command)}",
                f"cwd: {cwd}",
                f"returncode: {result.returncode}",
                f"success_path: {success_path}",
                f"tolerated_windows_native_access_violation: {'true' if tolerated_windows_crash else 'false'}",
                "=== stdout ===",
                result.stdout or "",
                "=== stderr ===",
                result.stderr or "",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return log_path


def _should_tolerate_windows_native_access_violation(
    *,
    result: subprocess.CompletedProcess[str],
    context: RuntimeContext | None,
    success_path: Path,
) -> bool:
    if context is None:
        return False
    host = context.platform_policy.get("host") if isinstance(context.platform_policy, dict) else None
    host_os = str((host or {}).get("os") or "").strip().lower()
    return host_os == "windows" and result.returncode in WINDOWS_NATIVE_ACCESS_VIOLATION_CODES and success_path.exists()


def _stage_log_path(*, run_dir: Path, stage_name: str) -> Path:
    return run_dir.parents[1] / "logs" / run_dir.name / f"{stage_name}.log"


def _tail_summary(result: subprocess.CompletedProcess[str], max_lines: int = 12) -> str:
    combined = []
    if result.stdout and result.stdout.strip():
        combined.append("[stdout]")
        combined.extend(result.stdout.strip().splitlines())
    if result.stderr and result.stderr.strip():
        combined.append("[stderr]")
        combined.extend(result.stderr.strip().splitlines())
    if not combined:
        return "(empty)"
    return "\n".join(combined[-max_lines:])


def public_error_message(exc: PipelineError) -> str:
    message = str(exc)
    cleanup_prefix = "UniRig staged-file cleanup failed"
    if message.startswith(cleanup_prefix):
        return "UniRig cleanup failed after processing. Inspect extension runtime logs for details."

    match = re.search(r"UniRig\s+([\w-]+)\s+(stage|hook)\s+failed", message)
    if match:
        return f"UniRig {match.group(1)} stage failed. Inspect extension runtime logs for details."

    return PIPELINE_PUBLIC_ERROR
