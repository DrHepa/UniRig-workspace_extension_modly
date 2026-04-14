from __future__ import annotations

import io
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import traceback
import types
from contextlib import contextmanager, redirect_stderr, redirect_stdout


BLENDER_SUBPROCESS_MODE = "blender-subprocess"
BLENDER_PROTOCOL_VERSION = 1
BLENDER_PAYLOAD_FILE_NAME = "payload.json"
BLENDER_RESULT_FILE_NAME = "result.json"
BLENDER_RESULT_MARKER_PREFIX = "UNIRIG_BLENDER_STAGE_RESULT="
BLENDER_STAGE_STATUS_OK = "ok"
BLENDER_STAGE_STATUS_FAILED = "stage-failed"
BLENDER_STAGE_NAMES = ("extract-prepare", "skeleton", "extract-skin", "skin", "merge")
BLENDER_FAILURE_CODES = frozenset(
    {
        "launch-failed",
        "timed-out",
        "marker-missing",
        "result-missing",
        "result-invalid",
        "expected-output-missing",
        "stage-failed",
    }
)
QUALIFICATION_COMPARISON_FAILURE_CODES = frozenset(
    {
        "missing-blender",
        "output-mismatch",
        "downstream-incompatibility",
        "environment-failure",
        "upstream-package-mismatch",
    }
)
QUALIFICATION_FAILURE_CODES = frozenset(BLENDER_FAILURE_CODES | QUALIFICATION_COMPARISON_FAILURE_CODES)
EXTRACT_CONFIG = "configs/data/quick_inference.yaml"
EXTRACT_FACES_TARGET_COUNT = 50000
MERGE_REQUIRE_SUFFIX = "obj,fbx,FBX,dae,glb,gltf,vrm"


def validate_stage_name(stage: str) -> str:
    normalized = str(stage).strip()
    if normalized not in BLENDER_STAGE_NAMES:
        raise ValueError(
            "Unsupported Blender bridge stage "
            f"{stage!r}. Expected one of: {', '.join(BLENDER_STAGE_NAMES)}."
        )
    return normalized


def validate_failure_code(error_code: str) -> str:
    normalized = str(error_code).strip()
    if normalized not in BLENDER_FAILURE_CODES:
        raise ValueError(
            "Unsupported Blender bridge failure code "
            f"{error_code!r}. Expected one of: {', '.join(sorted(BLENDER_FAILURE_CODES))}."
        )
    return normalized


def validate_qualification_failure_code(error_code: str) -> str:
    normalized = str(error_code).strip()
    if normalized not in QUALIFICATION_FAILURE_CODES:
        raise ValueError(
            "Unsupported Blender qualification failure code "
            f"{error_code!r}. Expected one of: {', '.join(sorted(QUALIFICATION_FAILURE_CODES))}."
        )
    return normalized


def qualification_failure_code_for_bridge_failure(error_code: str) -> str:
    return validate_failure_code(error_code)


def payload_path_for_run_dir(run_dir: Path) -> Path:
    return Path(run_dir) / BLENDER_PAYLOAD_FILE_NAME


def result_path_for_run_dir(run_dir: Path) -> Path:
    return Path(run_dir) / BLENDER_RESULT_FILE_NAME


def build_result_marker_line(result_path: Path) -> str:
    result_file = Path(result_path)
    if not result_file.is_absolute():
        raise ValueError(f"Blender result marker path must be absolute, got: {result_file}")
    return f"{BLENDER_RESULT_MARKER_PREFIX}{result_file}"


def parse_result_marker(stdout_text: str) -> Path | None:
    marker_path: Path | None = None
    for line in str(stdout_text).splitlines():
        if not line.startswith(BLENDER_RESULT_MARKER_PREFIX):
            continue
        candidate = Path(line[len(BLENDER_RESULT_MARKER_PREFIX) :].strip())
        if not candidate.is_absolute():
            raise ValueError(f"Blender result marker path must be absolute, got: {candidate}")
        marker_path = candidate
    return marker_path


def build_stage_payload(
    *,
    stage: str,
    run_dir: Path,
    source_path: Path | None = None,
    target_path: Path | None = None,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    require_suffix: str = "",
    seed: int = 12345,
    extract_token: str = "",
) -> dict:
    return {
        "protocol_version": BLENDER_PROTOCOL_VERSION,
        "stage": validate_stage_name(stage),
        "run_dir": str(Path(run_dir)),
        "input": {
            "source": _path_text(source_path),
            "target": _path_text(target_path),
            "output_dir": _path_text(output_dir),
            "output": _path_text(output_path),
        },
        "config": {
            "require_suffix": str(require_suffix),
            "seed": int(seed),
            "extract_token": str(extract_token),
        },
    }


def build_stage_success_result(
    *,
    stage: str,
    produced: list[Path | str],
    stdout_tail: list[str] | None = None,
    stderr_tail: list[str] | None = None,
) -> dict:
    return {
        "protocol_version": BLENDER_PROTOCOL_VERSION,
        "stage": validate_stage_name(stage),
        "status": BLENDER_STAGE_STATUS_OK,
        "produced": [str(Path(item)) for item in produced],
        "error_code": "",
        "message": "",
        "stdout_tail": list(stdout_tail or []),
        "stderr_tail": list(stderr_tail or []),
    }


def build_stage_failed_result(
    *,
    stage: str,
    error_code: str,
    message: str,
    produced: list[Path | str] | None = None,
    stdout_tail: list[str] | None = None,
    stderr_tail: list[str] | None = None,
) -> dict:
    return {
        "protocol_version": BLENDER_PROTOCOL_VERSION,
        "stage": validate_stage_name(stage),
        "status": BLENDER_STAGE_STATUS_FAILED,
        "produced": [str(Path(item)) for item in (produced or [])],
        "error_code": validate_failure_code(error_code),
        "message": str(message),
        "stdout_tail": list(stdout_tail or []),
        "stderr_tail": list(stderr_tail or []),
    }


def render_stage_payload_json(payload: dict) -> str:
    normalized = load_stage_payload(payload)
    return json.dumps(normalized, indent=2, sort_keys=True) + "\n"


def load_stage_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Blender stage payload must be a JSON object.")

    stage = validate_stage_name(payload.get("stage", ""))
    protocol_version = _require_protocol_version(payload)
    run_dir = _require_string(payload, "run_dir")
    input_payload = _require_object(payload, "input")
    config_payload = _require_object(payload, "config")
    return {
        "protocol_version": protocol_version,
        "stage": stage,
        "run_dir": run_dir,
        "input": {
            "source": _optional_string(input_payload, "source"),
            "target": _optional_string(input_payload, "target"),
            "output_dir": _optional_string(input_payload, "output_dir"),
            "output": _optional_string(input_payload, "output"),
        },
        "config": {
            "require_suffix": _optional_string(config_payload, "require_suffix"),
            "seed": _require_int(config_payload, "seed"),
            "extract_token": _optional_string(config_payload, "extract_token"),
        },
    }


def load_stage_result(result: dict, *, expected_stage: str, expected_outputs: list[Path | str] | None = None) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Blender stage result must be a JSON object.")

    normalized_stage = validate_stage_name(expected_stage)
    protocol_version = _require_protocol_version(result)
    stage = validate_stage_name(_require_string(result, "stage"))
    if stage != normalized_stage:
        raise ValueError(f"Blender stage result stage mismatch: expected {normalized_stage!r}, got {stage!r}.")

    status = _require_string(result, "status")
    if status not in {BLENDER_STAGE_STATUS_OK, BLENDER_STAGE_STATUS_FAILED}:
        raise ValueError(
            "Blender stage result status must be 'ok' or 'stage-failed', "
            f"got {status!r}."
        )

    produced = _require_string_list(result, "produced")
    error_code = _optional_string(result, "error_code")
    if status == BLENDER_STAGE_STATUS_OK:
        if error_code:
            raise ValueError("Blender stage success result must not declare an error_code.")
    else:
        validate_failure_code(error_code)

    normalized = {
        "protocol_version": protocol_version,
        "stage": stage,
        "status": status,
        "produced": produced,
        "error_code": error_code,
        "message": _optional_string(result, "message"),
        "stdout_tail": _require_string_list(result, "stdout_tail"),
        "stderr_tail": _require_string_list(result, "stderr_tail"),
    }

    if status == BLENDER_STAGE_STATUS_OK:
        _validate_expected_outputs(normalized, expected_outputs or [])
    return normalized


def _validate_expected_outputs(result: dict, expected_outputs: list[Path | str]) -> None:
    if not expected_outputs:
        return
    produced = {str(Path(item)) for item in result["produced"]}
    missing = [str(Path(item)) for item in expected_outputs if str(Path(item)) not in produced]
    if missing:
        raise ValueError(
            "Blender stage result is missing expected output declarations: " + ", ".join(missing)
        )


def _path_text(value: Path | None) -> str:
    if value is None:
        return ""
    return str(Path(value))


def _optional_string(payload: dict, key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Blender bridge field {key!r} must be a string.")
    return value


def _require_string(payload: dict, key: str) -> str:
    if key not in payload:
        raise ValueError(f"Blender bridge field {key!r} is required.")
    return _optional_string(payload, key)


def _require_object(payload: dict, key: str) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Blender bridge field {key!r} must be a JSON object.")
    return value


def _require_int(payload: dict, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Blender bridge field {key!r} must be an integer.")
    return value


def _require_protocol_version(payload: dict) -> int:
    protocol_version = payload.get("protocol_version")
    if protocol_version != BLENDER_PROTOCOL_VERSION:
        raise ValueError(
            "Unsupported Blender bridge protocol version "
            f"{protocol_version!r}. Expected {BLENDER_PROTOCOL_VERSION}."
        )
    return protocol_version


def _require_string_list(payload: dict, key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Blender bridge field {key!r} must be a list of strings.")
    return list(value)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    payload_path = _payload_path_from_argv(argv)
    result_path = result_path_for_run_dir(payload_path.parent).resolve()

    try:
        payload = load_stage_payload(json.loads(payload_path.read_text(encoding="utf-8")))
        result = _execute_stage_payload(payload)
        exit_code = 0
    except Exception as exc:
        stage = _stage_name_from_payload_path(payload_path)
        if not stage:
            raise
        result = build_stage_failed_result(
            stage=stage,
            error_code="stage-failed",
            message=str(exc),
            stderr_tail=_tail_lines(traceback.format_exc()),
        )
        exit_code = 0

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(build_result_marker_line(result_path), flush=True)
    return exit_code


def _payload_path_from_argv(argv: list[str]) -> Path:
    if "--" in argv:
        marker_index = argv.index("--")
        if marker_index + 1 < len(argv):
            return Path(argv[marker_index + 1]).expanduser().resolve()
    if len(argv) >= 2:
        return Path(argv[-1]).expanduser().resolve()
    raise ValueError("Blender bridge expected a payload path argument after '--'.")


def _stage_name_from_payload_path(payload_path: Path) -> str:
    try:
        raw = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    stage = str(raw.get("stage") or "").strip()
    if stage not in BLENDER_STAGE_NAMES:
        return ""
    return stage


def _execute_stage_payload(payload: dict) -> dict:
    stage = payload["stage"]
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            produced = _run_stage_module(payload)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0 if exc.code in (None, "") else 1
            if code not in (0, None):
                raise RuntimeError(f"Stage exited with code {code}.") from exc
            produced = _expected_outputs_for_payload(payload)

    captured_stdout = stdout_buffer.getvalue()
    captured_stderr = stderr_buffer.getvalue()
    if captured_stdout:
        sys.stdout.write(captured_stdout)
    if captured_stderr:
        sys.stderr.write(captured_stderr)

    missing_outputs = [path for path in produced if not Path(path).exists()]
    if missing_outputs:
        return build_stage_failed_result(
            stage=stage,
            error_code="expected-output-missing",
            message="Expected Blender stage outputs were not created on disk.",
            produced=produced,
            stdout_tail=_tail_lines(captured_stdout),
            stderr_tail=_tail_lines(captured_stderr),
        )

    return build_stage_success_result(
        stage=stage,
        produced=produced,
        stdout_tail=_tail_lines(captured_stdout),
        stderr_tail=_tail_lines(captured_stderr),
    )


def _run_stage_module(payload: dict) -> list[Path]:
    stage = payload["stage"]
    input_payload = payload["input"]
    config_payload = payload["config"]

    if stage in {"extract-prepare", "extract-skin"}:
        force_override = "true"
        _run_module_in_unirig_runtime(
            module_name="src.data.extract",
            args=[
                f"--config={EXTRACT_CONFIG}",
                f"--require_suffix={config_payload['require_suffix'] or MERGE_REQUIRE_SUFFIX}",
                f"--force_override={force_override}",
                "--num_runs=1",
                "--id=0",
                f"--time={config_payload['extract_token']}",
                f"--faces_target_count={EXTRACT_FACES_TARGET_COUNT}",
                f"--input={input_payload['source']}",
                f"--output_dir={input_payload['output_dir']}",
            ],
        )
        _sync_extract_output_contract(payload)
        return _expected_outputs_for_payload(payload)

    if stage == "merge":
        with _optional_merge_open3d_stub():
            _run_module_in_unirig_runtime(
                module_name="src.inference.merge",
                args=[
                    f"--require_suffix={config_payload['require_suffix'] or MERGE_REQUIRE_SUFFIX}",
                    "--num_runs=1",
                    "--id=0",
                    f"--source={input_payload['source']}",
                    f"--target={input_payload['target']}",
                    f"--output={input_payload['output']}",
                ],
            )
        return _expected_outputs_for_payload(payload)

    if stage == "skeleton":
        _run_script_in_unirig_runtime(
            script_name="run.py",
            args=[
                "--task=configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml",
                f"--seed={config_payload['seed']}",
                f"--input={input_payload['source']}",
                f"--output={input_payload['output']}",
                f"--npz_dir={input_payload['output_dir']}",
            ],
        )
        return _expected_outputs_for_payload(payload)

    if stage == "skin":
        _run_script_in_unirig_runtime(
            script_name="run.py",
            args=[
                "--task=configs/task/quick_inference_unirig_skin.yaml",
                f"--seed={config_payload['seed']}",
                f"--input={input_payload['source']}",
                f"--output={input_payload['output']}",
                f"--npz_dir={input_payload['output_dir']}",
                "--data_name=raw_data.npz",
            ],
        )
        return _expected_outputs_for_payload(payload)

    raise ValueError(f"Unsupported Blender bridge stage {stage!r}.")


def _run_module_in_unirig_runtime(*, module_name: str, args: list[str]) -> None:
    _run_entry_in_unirig_runtime(entrypoint=module_name, args=args, run_callable=lambda entrypoint: runpy.run_module(entrypoint, run_name="__main__"))


@contextmanager
def _optional_merge_open3d_stub():
    if importlib.util.find_spec("open3d") is not None:
        yield
        return

    previous = sys.modules.get("open3d")
    sys.modules["open3d"] = types.ModuleType("open3d")
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("open3d", None)
        else:
            sys.modules["open3d"] = previous


def _run_script_in_unirig_runtime(*, script_name: str, args: list[str]) -> None:
    _run_entry_in_unirig_runtime(
        entrypoint=script_name,
        args=args,
        run_callable=lambda entrypoint: runpy.run_path(str(_bridge_unirig_dir() / entrypoint), run_name="__main__"),
    )


def _run_entry_in_unirig_runtime(*, entrypoint: str, args: list[str], run_callable) -> None:
    unirig_dir = _bridge_unirig_dir()
    previous_cwd = Path.cwd()
    previous_argv = list(sys.argv)
    previous_sys_path = list(sys.path)
    try:
        os.chdir(unirig_dir)
        sys.argv = [entrypoint, *args]
        for entry in reversed(_bridge_runtime_sys_path_entries(unirig_dir)):
            sys.path.insert(0, entry)
        _register_torch_safe_globals()
        run_callable(entrypoint)
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path


def _register_torch_safe_globals() -> None:
    try:
        import torch
        from box.box import Box
    except Exception:
        return

    try:
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    except Exception:
        return

    if not callable(add_safe_globals):
        return

    try:
        add_safe_globals([Box])
    except Exception:
        return


def _bridge_runtime_sys_path_entries(unirig_dir: Path) -> list[str]:
    entries = [str(Path(unirig_dir).resolve())]
    seen = {os.path.normcase(os.path.normpath(entries[0]))}
    for site_packages in _bridge_venv_site_packages_dirs():
        resolved = str(site_packages.resolve())
        normalized = os.path.normcase(os.path.normpath(resolved))
        if normalized in seen:
            continue
        seen.add(normalized)
        entries.append(resolved)
    return entries


def _bridge_unirig_dir() -> Path:
    extension_root = _bridge_extension_root()
    return (extension_root / ".unirig-runtime" / "vendor" / "unirig").resolve()


def _bridge_extension_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bridge_venv_dir() -> Path:
    extension_root = _bridge_extension_root()
    state_path = extension_root / ".unirig-runtime" / "bootstrap_state.json"
    if not state_path.exists():
        return (extension_root / "venv").resolve()

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (extension_root / "venv").resolve()

    runtime_paths = state.get("runtime_paths") if isinstance(state.get("runtime_paths"), dict) else {}
    venv_python = str(runtime_paths.get("venv_python") or state.get("venv_python") or "").strip()
    if not venv_python:
        return (extension_root / "venv").resolve()

    python_path = Path(venv_python).expanduser()
    if not python_path.is_absolute():
        python_path = (extension_root / python_path).resolve()
    if python_path.parent.name in {"bin", "Scripts"}:
        return python_path.parent.parent.resolve()
    return python_path.parent.resolve()


def _bridge_venv_site_packages_dirs() -> list[Path]:
    venv_dir = _bridge_venv_dir()
    candidates = [
        venv_dir / "Lib" / "site-packages",
        venv_dir / "Lib" / "dist-packages",
        venv_dir / "lib" / "site-packages",
        venv_dir / "lib" / "dist-packages",
    ]
    candidates.extend(sorted((venv_dir / "lib").glob("python*/site-packages")))
    candidates.extend(sorted((venv_dir / "lib").glob("python*/dist-packages")))
    candidates.extend(sorted((venv_dir / "local" / "lib").glob("python*/site-packages")))
    candidates.extend(sorted((venv_dir / "local" / "lib").glob("python*/dist-packages")))

    found: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.normpath(str(candidate)))
        if normalized in seen or not candidate.exists():
            continue
        seen.add(normalized)
        found.append(candidate)
    return found


def _expected_outputs_for_payload(payload: dict) -> list[Path]:
    output_path = Path(payload["input"].get("output") or "").expanduser()
    if not output_path:
        raise ValueError("Blender bridge payload is missing the expected output path.")
    return [output_path]


def _sync_extract_output_contract(payload: dict) -> None:
    stage = payload["stage"]
    if stage not in {"extract-prepare", "extract-skin"}:
        return

    expected_output = _expected_outputs_for_payload(payload)[0]
    if expected_output.exists():
        return

    actual_output = _actual_extract_output_for_payload(payload)
    if actual_output == expected_output or not actual_output.exists():
        return

    expected_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(actual_output, expected_output)


def _actual_extract_output_for_payload(payload: dict) -> Path:
    input_payload = payload["input"]
    source_path = Path(input_payload.get("source") or "").expanduser()
    output_dir = Path(input_payload.get("output_dir") or "").expanduser()
    if not source_path:
        raise ValueError("Blender bridge payload is missing the extract source path.")
    if not output_dir:
        raise ValueError("Blender bridge payload is missing the extract output directory.")

    source_stem = source_path.with_suffix("")
    if source_path.is_absolute():
        return source_stem / "raw_data.npz"
    return output_dir / source_stem / "raw_data.npz"


def _tail_lines(text: str, limit: int = 20) -> list[str]:
    return [line for line in str(text).splitlines()[-limit:] if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
