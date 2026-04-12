from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import io
from .bootstrap import REAL_RUNTIME_MODE, RuntimeContext, UniRigError, stage_environment


class PipelineError(UniRigError):
    pass


ProgressFn = Callable[[int, str], None]
LogFn = Callable[[str], None]
SKELETON_TASK = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
SKIN_TASK = "configs/task/quick_inference_unirig_skin.yaml"
SKIN_DATA_NAME = "raw_data.npz"
MERGE_REQUIRE_SUFFIX = "obj,fbx,FBX,dae,glb,gltf,vrm"
EXTRACT_CONFIG = "configs/data/quick_inference.yaml"
EXTRACT_FACES_TARGET_COUNT = 50000
PIPELINE_PUBLIC_ERROR = "UniRig processing failed. Inspect extension runtime logs for details."
WINDOWS_NATIVE_ACCESS_VIOLATION_CODES = {3221225477, -1073741819}
RUNTIME_STAGE_TOKEN = "run-processor"


@dataclass(frozen=True)
class ExecutionStage:
    name: str
    command: list[str]
    cwd: Path
    success_path: Path
    runtime_input_name: str = ""
    runtime_input_path: Path | None = None
    log_stage_name: str | None = None


def run(
    mesh_path: Path,
    params: dict,
    context: RuntimeContext,
    progress: ProgressFn | None = None,
    log: LogFn | None = None,
) -> Path:
    progress = progress or (lambda percent, label: None)
    log = log or (lambda message: None)

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
) -> Path:
    seed = int(params.get("seed", 12345))
    prepared = io.prepare_input_mesh(staged_input, run_dir, context)
    progress(20, "input prepared")

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
        _run_stage(plan[0], context=context, run_dir=run_dir)
        progress(35, "prepare/extract complete")

        log("running skeleton stage")
        _run_stage(plan[1], context=context, run_dir=run_dir)
        progress(55, "skeleton stage complete")

        shutil.copy2(plan[1].success_path, _require_runtime_input_path(plan[2]))
        _run_stage(plan[2], context=context, run_dir=run_dir)
        log("running skin stage")
        _run_stage(plan[3], context=context, run_dir=run_dir)
        progress(78, "skin stage complete")

        log("running merge stage")
        _run_stage(plan[4], context=context, run_dir=run_dir)
        progress(92, "merge stage complete")
        return io.publish_output(plan[4].success_path, mesh_path)
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
                force_override=False,
                context=context,
                extract_token=f"modly_extract_{stage_token}",
            ),
            cwd=context.unirig_dir,
            success_path=skeleton_npz_dir / staged_prepared.stem / SKIN_DATA_NAME,
            runtime_input_name=staged_prepared.name,
            runtime_input_path=staged_prepared,
            log_stage_name="extract-prepare",
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
            runtime_input_name=staged_prepared.name,
            runtime_input_path=staged_prepared,
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
            runtime_input_name=staged_skeleton.name,
            runtime_input_path=staged_skeleton,
            log_stage_name="extract-skin",
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
            runtime_input_name=staged_skeleton.name,
            runtime_input_path=staged_skeleton,
        ),
        ExecutionStage(
            name="merge",
            command=_merge_command(input_path=skin_output, prepared_path=prepared_path, output_path=merged_output, context=context),
            cwd=context.unirig_dir,
            success_path=merged_output,
        ),
    ]


def _extract_command(*, input_name: str, output_dir: Path, force_override: bool, context: RuntimeContext, extract_token: str) -> list[str]:
    return [
        str(context.venv_python),
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
    ]


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
    return [
        str(context.venv_python),
        "run.py",
        f"--task={task}",
        f"--seed={seed}",
        f"--input={input_name}",
        f"--output={output_path}",
        f"--npz_dir={npz_dir}",
        *([f"--data_name={SKIN_DATA_NAME}"] if include_data_name else []),
    ]


def _merge_command(*, input_path: Path, prepared_path: Path, output_path: Path, context: RuntimeContext) -> list[str]:
    return [
        str(context.venv_python),
        "-m",
        "src.inference.merge",
        f"--require_suffix={MERGE_REQUIRE_SUFFIX}",
        "--num_runs=1",
        "--id=0",
        f"--source={input_path}",
        f"--target={prepared_path}",
        f"--output={output_path}",
    ]


def _require_runtime_input_path(stage: ExecutionStage) -> Path:
    if stage.runtime_input_path is None:
        raise PipelineError(f"UniRig {stage.name} stage is missing its deterministic runtime input path.")
    stage.runtime_input_path.parent.mkdir(parents=True, exist_ok=True)
    return stage.runtime_input_path


def _run_stage(stage: ExecutionStage, *, context: RuntimeContext, run_dir: Path) -> None:
    _run_command(
        stage.command,
        cwd=stage.cwd,
        context=context,
        success_path=stage.success_path,
        run_dir=run_dir,
        stage_name=stage.name,
        log_stage_name=stage.log_stage_name,
    )


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
