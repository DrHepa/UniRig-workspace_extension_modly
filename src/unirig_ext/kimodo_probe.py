from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FAILURE_LAYERS = (
    "contract_invalid",
    "source_calibration_incomplete",
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
    source_coverage: dict[str, Any] | None = None
    calibration_status: str | None = None
    mapping_confidence: str | None = None
    chain_coverage: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["probe_status"] = self.status
        if payload["failure"] is None and self.primary_failure_layer is not None:
            payload["failure"] = {"layer": self.primary_failure_layer, "code": self.code, "message": self.message}
        return payload


class KimodoProbeConfigError(ValueError):
    pass


class KimodoProbeBackend:
    """Optional Kimodo seam adapter.

    The default implementation is deliberately conservative. It reports
    unavailable instead of importing local Kimodo source implicitly, because this
    public extension must not depend on a developer-machine checkout to perform
    base candidate analysis.
    """

    def __init__(self, kimodo_root: str | Path | None = None, source_bvh: str | Path | None = None, probe_output_root: str | Path | None = None) -> None:
        self.kimodo_root = Path(kimodo_root).resolve() if kimodo_root is not None else None
        self.source_bvh = Path(source_bvh).expanduser().resolve() if source_bvh is not None else None
        self.probe_output_root = Path(probe_output_root).expanduser().resolve() if probe_output_root is not None else None
        self._validate_configured_paths()

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
                stages: list[str] = []
                source_path = self._source_bvh_path()
                source_coverage = source_coverage_for_bvh(source_path)
                motion = kimodo_bvh.parse_bvh(source_path)
                stages.append("parse_bvh")
                rig = rig_inspector.inspect_rig(Path(copied_glb), source_kind="unirig_humanoid_mapping_candidate")
                stages.append("inspect_rig")
                mapping = retargeting.build_joint_mapping(motion, rig)
                stages.append("build_joint_mapping")
                clip = None
                if probe_retarget:
                    clip = retargeting.build_retarget_clip(motion, rig)
                    stages.append("build_retarget_clip")
            except Exception as exc:
                code = str(getattr(exc, "code", "") or _exception_code(exc) or "ambiguous")
                diagnostics = getattr(exc, "diagnostics", None)
                normalized_diagnostics = _normalize_diagnostics(diagnostics) or []
                if not normalized_diagnostics:
                    normalized_diagnostics.append({"code": code, "message": str(exc)})
                if "stages" in locals():
                    normalized_diagnostics.append({"code": "kimodo_probe_stages", "message": ",".join(stages), "stages": list(stages)})
                return classify_probe_failure(
                    code,
                    str(exc),
                    diagnostics=normalized_diagnostics or None,
                    source_coverage=source_coverage if "source_coverage" in locals() else None,
                )
        chain_coverage = _chain_coverage_from(mapping, clip)
        mapping_confidence = str(getattr(mapping, "confidence", "unknown"))
        calibration_status = "incomplete" if source_coverage.get("missing_roles") else "complete"
        diagnostics = [
            {
                "code": "kimodo_probe_accepted",
                "message": "Kimodo parse_bvh, inspect_rig, build_joint_mapping, and requested retarget stages completed on a disposable GLB copy.",
                "stages": stages,
                "mapping_confidence": mapping_confidence,
                "mapping_warnings": list(getattr(mapping, "warnings", ()) or ()),
                "chain_coverage": chain_coverage,
                "calibration_status": calibration_status,
            }
        ]
        status = "accepted_with_unknown_visual_quality" if probe_retarget else "accepted"
        primary = "visual_quality_unknown" if probe_retarget else None
        code = "visual_quality_unknown" if probe_retarget else None
        message = "Kimodo seam accepted the candidate sidecar; visual retarget quality was not validated."
        return ProbeResult(
            status=status,
            primary_failure_layer=primary,
            code=code,
            message=message,
            diagnostics=diagnostics,
            source_coverage=source_coverage,
            calibration_status=calibration_status,
            mapping_confidence=mapping_confidence,
            chain_coverage=chain_coverage,
        )

    def _source_bvh_path(self) -> Path:
        assert self.kimodo_root is not None
        return self.source_bvh if self.source_bvh is not None else self.kimodo_root / "tests" / "fixtures" / "synthetic_walk.bvh"

    def _validate_configured_paths(self) -> None:
        if self.source_bvh is not None:
            if not self.source_bvh.exists():
                raise KimodoProbeConfigError(f"source BVH path does not exist: {self.source_bvh}")
            if not self.source_bvh.is_file():
                raise KimodoProbeConfigError(f"source BVH path is not a file: {self.source_bvh}")
        if self.probe_output_root is not None:
            if not self.probe_output_root.exists():
                raise KimodoProbeConfigError(f"probe output root does not exist: {self.probe_output_root}")
            if not self.probe_output_root.is_dir():
                raise KimodoProbeConfigError(f"probe output root is not a directory: {self.probe_output_root}")


def unavailable_probe_result(message: str | None = None) -> ProbeResult:
    text = message or "Kimodo probe backend is unavailable; base candidate analysis still succeeded."
    return ProbeResult(
        status="unavailable",
        primary_failure_layer=None,
        code="probe_unavailable",
        message=text,
        diagnostics=[{"code": "probe_unavailable", "message": text}],
    )


def classify_probe_failure(code: str | None, message: str, *, diagnostics: list[dict[str, Any]] | None = None, source_coverage: dict[str, Any] | None = None) -> ProbeResult:
    normalized = (code or "ambiguous").strip()
    layer = _failure_layer_for_code(normalized, message=message, diagnostics=diagnostics)
    status = "accepted_with_unknown_visual_quality" if layer == "visual_quality_unknown" else "rejected"
    if normalized.lower() == "accepted":
        return ProbeResult(status="accepted", primary_failure_layer=None, code=None, message=message, diagnostics=diagnostics or [])
    calibration_status = "incomplete" if layer == "source_calibration_incomplete" else None
    return ProbeResult(
        status=status,
        primary_failure_layer=layer,
        code=normalized,
        message=message,
        diagnostics=diagnostics or [{"code": normalized, "message": message}],
        source_coverage=source_coverage,
        calibration_status=calibration_status,
        failure={"layer": layer, "code": normalized, "message": message},
    )


def probe_result_to_dict(result: ProbeResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, ProbeResult):
        return result.as_dict()
    return dict(result)


def _failure_layer_for_code(code: str, *, message: str = "", diagnostics: list[dict[str, Any]] | None = None) -> str:
    lowered = code.casefold()
    if "calibration_unavailable" in lowered and _has_missing_source_basis_or_role(message, diagnostics):
        return "source_calibration_incomplete"
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
    MODULES = ("kimodo_bvh", "rig_inspector", "retargeting", "retarget_errors", "probe_calls")

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


REQUIRED_SOURCE_ROLES = (
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
)


BVH_JOINT_ROLE_ALIASES = {
    "hips": "hips",
    "spine": "spine",
    "chest": "chest",
    "neck": "neck",
    "head": "head",
    "leftupperarm": "left_upper_arm",
    "leftforearm": "left_forearm",
    "lefthand": "left_hand",
    "rightupperarm": "right_upper_arm",
    "rightforearm": "right_forearm",
    "righthand": "right_hand",
    "leftupperleg": "left_upper_leg",
    "leftlowerleg": "left_lower_leg",
    "leftfoot": "left_foot",
    "rightupperleg": "right_upper_leg",
    "rightlowerleg": "right_lower_leg",
    "rightfoot": "right_foot",
}


def source_coverage_for_bvh(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    present: set[str] = set()
    for raw in text.replace("{", " ").replace("}", " ").split():
        role = BVH_JOINT_ROLE_ALIASES.get(raw.strip().casefold())
        if role is not None:
            present.add(role)
    missing = [role for role in REQUIRED_SOURCE_ROLES if role not in present]
    return {"status": "complete" if not missing else "incomplete", "present_roles": sorted(present), "missing_roles": missing}


def _has_missing_source_basis_or_role(message: str, diagnostics: list[dict[str, Any]] | None) -> bool:
    fragments = [message]
    for item in diagnostics or []:
        fragments.append(str(item.get("code", "")))
        fragments.append(str(item.get("message", "")))
    joined = "\n".join(fragments).casefold()
    return "missing_source_role:" in joined or "missing_source_basis:" in joined


def _chain_coverage_from(mapping: Any, clip: Any) -> dict[str, Any]:
    clip_calibration = getattr(clip, "calibration", None)
    if isinstance(clip_calibration, dict) and isinstance(clip_calibration.get("chain_coverage"), dict):
        return {str(key): value for key, value in sorted(clip_calibration["chain_coverage"].items())}
    coverage = getattr(mapping, "coverage", None)
    if isinstance(coverage, dict):
        return {str(key): value for key, value in sorted(coverage.items())}
    return {}
