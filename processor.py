# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext import bootstrap, io, metadata, pipeline  # noqa: E402


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _public_optional_metadata(**meta: object) -> dict[str, object]:
    allowed_fields = ("stage", "kind", "status", "elapsedSeconds")
    public_meta: dict[str, object] = {}
    for field in allowed_fields:
        value = meta.get(field)
        if value is None:
            continue
        if field == "elapsedSeconds":
            public_meta[field] = int(value)
            continue
        public_meta[field] = value
    return public_meta


def _progress(percent: int, label: str, **meta: object) -> None:
    _send(
        {
            "type": "progress",
            "percent": int(percent),
            "label": label,
            **_public_optional_metadata(**meta),
        }
    )


def _log(message: str, **meta: object) -> None:
    _send({"type": "log", "message": message, **_public_optional_metadata(**meta)})


def _read_payload() -> dict[str, object]:
    raw = sys.stdin.readline()
    if not raw:
        raise ValueError("Processor expected one JSON line on stdin.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Processor stdin is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Processor payload must be a JSON object.")
    return {"raw": raw, "parsed": payload}


def _capture_request_payload(raw_payload: str) -> None:
    if os.environ.get("UNIRIG_CAPTURE_REQUEST") != "1":
        return

    debug_dir = ROOT / ".unirig-runtime" / "debug"
    capture_path = debug_dir / f"request-capture-{time.time_ns()}.json"
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        capture_path.write_text(raw_payload, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            "Failed to capture processor request payload. "
            f"Set UNIRIG_CAPTURE_REQUEST=1 only when '{debug_dir}' is writable; "
            f"capture target: {capture_path}; error: {exc}"
        ) from exc


def _require_object(payload: dict, key: str) -> dict:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise bootstrap.ProtocolError(f"'{key}' must be a JSON object.")
    return value


def _public_error_message(exc: Exception) -> str:
    if isinstance(exc, pipeline.PipelineError):
        return pipeline.public_error_message(exc)
    return str(exc)


def _resolve_workspace_dir(payload: dict[str, object]) -> Path | None:
    workspace_dir = payload.get("workspaceDir")
    if workspace_dir is None:
        return None
    if not isinstance(workspace_dir, str):
        raise bootstrap.ProtocolError("'workspaceDir' must be a string when provided.")

    candidate = workspace_dir.strip()
    if not candidate:
        return None

    resolved = Path(candidate).expanduser()
    if not resolved.is_absolute():
        raise bootstrap.ProtocolError("'workspaceDir' must be an absolute path when provided.")
    if resolved.exists() and not resolved.is_dir():
        raise bootstrap.ProtocolError("'workspaceDir' must point to a directory when provided.")
    if not resolved.exists():
        return None
    return resolved.resolve()


def main() -> int:
    try:
        request_payload = _read_payload()
        _capture_request_payload(request_payload["raw"])
        payload = request_payload["parsed"]
        input_payload = _require_object(payload, "input")
        params = _require_object(payload, "params")
        workspace_dir = _resolve_workspace_dir(payload)
        node_id = input_payload.get("nodeId") or payload.get("nodeId") or ""

        if node_id != "rig-mesh":
            raise bootstrap.ProtocolError(
                f"Unsupported nodeId '{node_id}'. This MVP only exposes 'rig-mesh'."
            )

        file_path = input_payload.get("filePath")
        if not file_path:
            raise bootstrap.ProtocolError("rig-mesh requires input.filePath.")

        mesh_path = io.validate_mesh_input(Path(file_path))
        bootstrap.reject_private_contracts(payload, params)
        context = bootstrap.ensure_ready()

        _progress(5, "bootstrap ready")
        _log(f"Running {context.extension_id}/{node_id}")

        output_path = pipeline.run(
            mesh_path=mesh_path,
            params=params,
            context=context,
            progress=_progress,
            log=_log,
            workspace_dir=workspace_dir,
        )

        metadata.write_sidecar(
            output_path=output_path,
            input_path=mesh_path,
            seed=int(params.get("seed", 12345)),
            context=context,
        )
        _send({"type": "done", "result": {"filePath": str(output_path)}})
        return 0
    except Exception as exc:  # pragma: no cover - exercised through protocol tests
        _send({"type": "error", "message": _public_error_message(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
