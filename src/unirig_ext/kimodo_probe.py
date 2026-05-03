from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FAILURE_LAYERS = (
    "contract_invalid",
    "mapping_incompatible",
    "retarget_failed",
    "export_invalid",
    "structural_validation_failed",
    "visual_quality_unknown",
    "ambiguous",
)


@dataclass(frozen=True)
class ProbeResult:
    status: str
    primary_failure_layer: str | None
    code: str | None
    message: str
    diagnostics: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class KimodoProbeBackend:
    """Optional Kimodo seam adapter.

    The default implementation is deliberately conservative. It reports
    unavailable instead of importing local Kimodo source implicitly, because this
    public extension must not depend on a developer-machine checkout to perform
    base candidate analysis.
    """

    def __init__(self, kimodo_root: str | Path | None = None) -> None:
        self.kimodo_root = Path(kimodo_root).resolve() if kimodo_root is not None else None

    def available(self) -> bool:
        if self.kimodo_root is None or not self.kimodo_root.is_dir():
            return False
        required = [
            self.kimodo_root / "rig_inspector.py",
            self.kimodo_root / "retargeting.py",
            self.kimodo_root / "kimodo_bvh.py",
            self.kimodo_root / "retarget_errors.py",
            self._source_bvh_path(),
        ]
        return all(path.is_file() for path in required)

    def unavailable_result(self) -> ProbeResult:
        message = "Kimodo probe backend is unavailable; candidate analysis completed without retarget compatibility proof."
        if self.kimodo_root is not None:
            message = f"Kimodo probe backend is unavailable at {self.kimodo_root}; candidate analysis completed without retarget compatibility proof."
        return ProbeResult(
            status="unavailable",
            primary_failure_layer=None,
            code="probe_unavailable",
            message=message,
            diagnostics=[{"code": "probe_unavailable", "message": message}],
        )

    def probe(self, copied_glb: Path, sidecar_payload: dict[str, Any], *, probe_retarget: bool = False) -> ProbeResult:
        _ = sidecar_payload
        if not self.available():
            return self.unavailable_result()
        with _kimodo_import_context(self.kimodo_root):
            try:
                kimodo_bvh = importlib.import_module("kimodo_bvh")
                rig_inspector = importlib.import_module("rig_inspector")
                retargeting = importlib.import_module("retargeting")
                motion = kimodo_bvh.parse_bvh(self._source_bvh_path())
                rig = rig_inspector.inspect_rig(Path(copied_glb), source_kind="unirig_humanoid_mapping_candidate")
                mapping = retargeting.build_joint_mapping(motion, rig)
            except Exception as exc:
                code = str(getattr(exc, "code", "") or _exception_code(exc) or "ambiguous")
                diagnostics = getattr(exc, "diagnostics", None)
                return classify_probe_failure(code, str(exc), diagnostics=_normalize_diagnostics(diagnostics))
        diagnostics = [
            {
                "code": "kimodo_probe_accepted",
                "message": "Kimodo inspect_rig and build_joint_mapping completed on a disposable GLB copy.",
                "stages": ["inspect_rig", "build_joint_mapping"],
                "mapping_confidence": str(getattr(mapping, "confidence", "unknown")),
                "mapping_warnings": list(getattr(mapping, "warnings", ()) or ()),
            }
        ]
        status = "accepted_with_unknown_visual_quality" if probe_retarget else "accepted"
        primary = "visual_quality_unknown" if probe_retarget else None
        code = "visual_quality_unknown" if probe_retarget else None
        message = "Kimodo seam accepted the candidate sidecar; visual retarget quality was not validated."
        return ProbeResult(status=status, primary_failure_layer=primary, code=code, message=message, diagnostics=diagnostics)

    def _source_bvh_path(self) -> Path:
        assert self.kimodo_root is not None
        return self.kimodo_root / "tests" / "fixtures" / "synthetic_walk.bvh"


def unavailable_probe_result(message: str | None = None) -> ProbeResult:
    text = message or "Kimodo probe backend is unavailable; base candidate analysis still succeeded."
    return ProbeResult(
        status="unavailable",
        primary_failure_layer=None,
        code="probe_unavailable",
        message=text,
        diagnostics=[{"code": "probe_unavailable", "message": text}],
    )


def classify_probe_failure(code: str | None, message: str, *, diagnostics: list[dict[str, Any]] | None = None) -> ProbeResult:
    normalized = (code or "ambiguous").strip()
    layer = _failure_layer_for_code(normalized)
    status = "accepted_with_unknown_visual_quality" if layer == "visual_quality_unknown" else "rejected"
    if normalized.lower() == "accepted":
        return ProbeResult(status="accepted", primary_failure_layer=None, code=None, message=message, diagnostics=diagnostics or [])
    return ProbeResult(
        status=status,
        primary_failure_layer=layer,
        code=normalized,
        message=message,
        diagnostics=diagnostics or [{"code": normalized, "message": message}],
    )


def probe_result_to_dict(result: ProbeResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, ProbeResult):
        return result.as_dict()
    return dict(result)


def _failure_layer_for_code(code: str) -> str:
    lowered = code.casefold()
    if "contract" in lowered or "schema" in lowered or "invalid_candidate" in lowered:
        return "contract_invalid"
    if "mapping_incompatible" in lowered or "calibration_unavailable" in lowered or "mapping" in lowered:
        return "mapping_incompatible"
    if "export_invalid" in lowered or "invalid_output" in lowered:
        return "export_invalid"
    if "validator" in lowered or "structural" in lowered:
        return "structural_validation_failed"
    if "visual" in lowered:
        return "visual_quality_unknown"
    if "retarget" in lowered:
        return "retarget_failed"
    if "ambiguous" in lowered:
        return "ambiguous"
    return "ambiguous"


class _kimodo_import_context:
    MODULES = ("kimodo_bvh", "rig_inspector", "retargeting", "retarget_errors")

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self.previous_modules: dict[str, Any] = {}

    def __enter__(self) -> None:
        if self.root is None:
            return None
        sys.path.insert(0, str(self.root))
        for name in self.MODULES:
            if name in sys.modules:
                self.previous_modules[name] = sys.modules.pop(name)
        return None

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self.root is not None:
            try:
                sys.path.remove(str(self.root))
            except ValueError:
                pass
        for name in self.MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(self.previous_modules)


def _exception_code(exc: Exception) -> str | None:
    text = str(exc)
    if ":" in text:
        candidate = text.split(":", 1)[0].strip()
        return candidate or None
    return None


def _normalize_diagnostics(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return [{"code": str(key), "message": str(item)} for key, item in sorted(value.items())]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return [{"code": "kimodo_probe_diagnostic", "message": str(value)}]
