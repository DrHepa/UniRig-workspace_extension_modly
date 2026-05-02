from __future__ import annotations

import glob
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from .gltf_skin_analysis import GltfSkinAnalysisError, read_glb_container, summarize_joint_weights
from .humanoid_contract import HumanoidContractError, build_contract_from_declared_data
from .humanoid_quality_gate import HumanoidQualityGateError, run_humanoid_quality_gate
from .semantic_body_graph import build_semantic_body_report
from .semantic_humanoid_resolver import SemanticHumanoidResolutionError, extract_joint_graph, resolve_humanoid


SCHEMA_VERSION = "unirig.humanoid_corpus.v1"
FAMILY_PRECEDENCE = [
    "malformed_unparseable",
    "missing_invalid_skin_evidence",
    "resolver_ambiguity",
    "shallow_minimal_rig_output",
    "high_region_contamination",
    "passive_accessory_sleeve_contamination",
    "non_local_weight_spread",
    "contract_ready",
    "other_unknown",
]
REASON_CODE_ORDER = [
    "malformed_unparseable",
    "skin_weight_data_unavailable",
    "semantic_skin_missing",
    "semantic_skin_malformed",
    "semantic_nodes_missing",
    "semantic_graph_disconnected",
    "semantic_symmetry_ambiguous",
    "semantic_leg_symmetry_ambiguous",
    "semantic_spine_missing",
    "semantic_required_roles_missing",
    "high_region_weighted_by_torso_or_arm",
    "sleeve_branch_under_arm",
    "non_anatomical_leaf_under_arm",
    "semantic_passive_noncontract_subtree",
    "non_local_weight_spread",
    "anatomical_role_not_separable",
    "contract_ready",
    "other_unknown",
]
PASSIVE_CODES = {"sleeve_branch_under_arm", "non_anatomical_leaf_under_arm", "semantic_passive_noncontract_subtree"}
AMBIGUITY_CODES = {"semantic_symmetry_ambiguous", "semantic_leg_symmetry_ambiguous", "semantic_graph_ambiguous"}
MISSING_SKIN_CODES = {"semantic_skin_missing", "semantic_skin_malformed", "skin_weight_data_unavailable"}
ProgressCallback = Callable[[int, int, str, str], None]


class CorpusInputError(ValueError):
    pass


class CorpusOutputError(ValueError):
    pass


def select_glb_inputs(inputs: Iterable[str | Path], *, limit: int | None = None) -> list[Path]:
    if limit is not None and limit < 1:
        raise CorpusInputError(f"limit must be a positive integer; got {limit}")
    selected: list[Path] = []
    raw_inputs = list(inputs)
    if not raw_inputs:
        raise CorpusInputError("no input paths were provided; pass a directory, glob, or explicit .glb file list")
    for raw in raw_inputs:
        text = str(raw)
        if _has_glob_magic(text):
            matches = [Path(match) for match in glob.glob(text)]
            glbs = [path for path in matches if path.is_file() and path.suffix.lower() == ".glb"]
            if not glbs:
                raise CorpusInputError(f"glob input selected no .glb files: {text}")
            selected.extend(glbs)
            continue
        path = Path(raw)
        if not path.exists():
            raise CorpusInputError(f"input path does not exist: {path}")
        if path.is_dir():
            glbs = [child for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".glb"]
            if not glbs:
                raise CorpusInputError(f"directory input contains no .glb files: {path}")
            selected.extend(glbs)
            continue
        if not path.is_file():
            raise CorpusInputError(f"input is not a regular file: {path}")
        if path.suffix.lower() != ".glb":
            raise CorpusInputError(f"explicit input is not a .glb file: {path}")
        selected.append(path)
    unique = {path.resolve(): path for path in selected}
    if not unique:
        raise CorpusInputError("selection produced no .glb files")
    paths = sorted(unique.values(), key=_path_sort_key)
    for path in paths:
        _validate_readable_glb(path)
    if limit is not None:
        return paths[:limit]
    return paths


def validate_output_parent(path: str | Path) -> Path:
    output = Path(path)
    parent = output.parent if output.parent != Path("") else Path(".")
    if not parent.exists():
        raise CorpusOutputError(f"output parent does not exist: {parent}")
    if not parent.is_dir():
        raise CorpusOutputError(f"output parent is not a directory: {parent}")
    return output


def build_corpus_report(
    inputs: Iterable[str | Path],
    *,
    include_hash: bool = False,
    limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
    json_refresh_path: str | Path | None = None,
) -> dict[str, Any]:
    output = validate_output_parent(json_refresh_path) if json_refresh_path is not None else None
    paths = select_glb_inputs(inputs, limit=limit)
    base = _common_parent(paths)
    rows: list[dict[str, Any]] = []
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        display_path = _display_path(path, base)
        if progress_callback is not None:
            progress_callback(index, total, display_path, "STARTED")
        try:
            row = _profile_asset(path, base=base, include_hash=include_hash)
        except Exception as exc:
            row = _failed_profile_row(path, base=base, include_hash=include_hash, exc=exc)
        rows.append(row)
        terminal_status = str(row.get("profile_status", "OK"))
        if progress_callback is not None:
            progress_callback(index, total, display_path, terminal_status)
        if output is not None:
            write_json_report_atomic(build_corpus_report_from_rows(rows, assets_selected=total, is_limited=limit is not None), output)
    return build_corpus_report_from_rows(rows, assets_selected=total, is_limited=limit is not None)


def build_corpus_report_from_rows(rows: list[dict[str, Any]], *, assets_selected: int, is_limited: bool) -> dict[str, Any]:
    family_counts = {family: 0 for family in FAMILY_PRECEDENCE}
    resolver_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    contract_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    assets_completed = 0
    assets_failed = 0
    for row in rows:
        family_counts[row["primary_family"]] += 1
        if row.get("profile_status") == "FAILED":
            assets_failed += 1
        else:
            assets_completed += 1
        _increment(resolver_counts, str(row["resolver"]["status"]))
        _increment(quality_counts, str(row["quality_gate"]["status"]))
        _increment(contract_counts, str(row["contract_readiness"]["status"]))
        for code in row.get("secondary_reason_codes", []):
            _increment(reason_counts, str(code))
    is_partial = assets_completed + assets_failed < assets_selected
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_status": _report_status(is_partial=is_partial, is_limited=is_limited, assets_failed=assets_failed),
        "is_partial": is_partial,
        "is_limited": is_limited,
        "assets_selected": assets_selected,
        "assets_completed": assets_completed,
        "assets_failed": assets_failed,
        "corpus_summary": {
            "total_assets": len(rows),
            "parseable_assets": sum(1 for row in rows if row.get("parse_status") == "parsed"),
            "resolver_status_counts": dict(sorted(resolver_counts.items())),
            "quality_status_counts": dict(sorted(quality_counts.items())),
            "contract_status_counts": dict(sorted(contract_counts.items())),
            "family_counts": family_counts,
            "reason_code_counts": _sort_reason_counts(reason_counts),
            "publication_evidence": False,
        },
        "per_asset_rows": rows,
        "family_summaries": _build_family_summaries(rows, family_counts),
        "reason_codes": [{"code": code, "count": reason_counts[code]} for code in _sorted_reason_codes(reason_counts)],
    }
    return report


def _report_status(*, is_partial: bool, is_limited: bool, assets_failed: int) -> str:
    if is_partial:
        return "partial"
    if is_limited and assets_failed:
        return "limited_with_failures"
    if is_limited:
        return "limited"
    if assets_failed:
        return "complete_with_failures"
    return "complete"


def classify_asset_family(row: dict[str, Any]) -> dict[str, Any]:
    codes = _collect_reason_codes(row)
    resolver_failure = _resolver_failure_code(row)
    skin_count = int(row.get("skin_count") or 0)
    root_count = int(row.get("root_count") or 0)
    skin_joint_count = int(row.get("skin_joint_count") or 0)
    weighted_joint_count = int(row.get("weighted_joint_count") or 0)
    if row.get("parse_status") == "unparseable":
        primary = "malformed_unparseable"
        codes.append("malformed_unparseable")
    elif skin_count != 1 or root_count > 1 or resolver_failure in MISSING_SKIN_CODES or any(code in MISSING_SKIN_CODES for code in codes):
        primary = "missing_invalid_skin_evidence"
    elif resolver_failure in AMBIGUITY_CODES:
        primary = "resolver_ambiguity"
    elif resolver_failure == "semantic_spine_missing" or (not _is_contract_ready(row) and (0 < skin_joint_count < 8 or 0 < weighted_joint_count < 8)):
        primary = "shallow_minimal_rig_output"
    elif "high_region_weighted_by_torso_or_arm" in codes:
        primary = "high_region_contamination"
    elif any(code in PASSIVE_CODES for code in codes):
        primary = "passive_accessory_sleeve_contamination"
    elif "non_local_weight_spread" in codes:
        primary = "non_local_weight_spread"
    elif _is_contract_ready(row):
        primary = "contract_ready"
        codes.append("contract_ready")
    else:
        primary = "other_unknown"
        codes.append("other_unknown")
    return {"primary_family": primary, "secondary_reason_codes": _dedupe_reason_codes(codes)}


def dumps_canonical_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False, separators=(",", ": ")) + "\n"


def write_json_report(report: dict[str, Any], path: str | Path) -> Path:
    output = validate_output_parent(path)
    output.write_text(dumps_canonical_json(report), encoding="utf-8")
    return output


def write_json_report_atomic(report: dict[str, Any], path: str | Path) -> Path:
    output = validate_output_parent(path)
    parent = output.parent if output.parent != Path("") else Path(".")
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(dumps_canonical_json(report))
            handle.flush()
        temp_path.replace(output)
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise
    return output


def write_markdown_report_from_json(report: dict[str, Any], path: str | Path) -> Path:
    output = validate_output_parent(path)
    output.write_text(render_markdown_from_report_json(report), encoding="utf-8")
    return output


def render_markdown_from_report_json(report: dict[str, Any]) -> str:
    summary = report.get("corpus_summary", {}) if isinstance(report.get("corpus_summary"), dict) else {}
    family_summaries = report.get("family_summaries", {}) if isinstance(report.get("family_summaries"), dict) else {}
    reason_codes = report.get("reason_codes", []) if isinstance(report.get("reason_codes"), list) else []
    lines = [
        "# UniRig Humanoid Corpus Profile",
        "",
        "> Diagnostic only: this report is not publication evidence and does not relax resolver, contract, or quality gates.",
        "",
    ]
    if report.get("is_partial") or report.get("is_limited"):
        lines.extend(
            [
                f"> WARNING: report_status={report.get('report_status')} selected={report.get('assets_selected', 0)} completed={report.get('assets_completed', 0)} failed={report.get('assets_failed', 0)}.",
                "> This is a partial or limited selection report; do not treat it as complete corpus evidence.",
                "",
            ]
        )
        if report.get("is_limited"):
            lines.extend(["> WARNING: limited selection was requested; unselected assets are absent from this report.", ""])
    lines.extend(
        [
            "## Corpus Summary",
            f"- Total assets: {summary.get('total_assets', 0)}",
            f"- Parseable assets: {summary.get('parseable_assets', 0)}",
            f"- Publication evidence: {summary.get('publication_evidence', False)}",
            "",
            "## Family Counts",
        ]
    )
    for family in FAMILY_PRECEDENCE:
        item = family_summaries.get(family, {}) if isinstance(family_summaries.get(family), dict) else {}
        lines.append(f"- {family}: {item.get('count', 0)}")
    lines.extend(["", "## Reason Codes"])
    if reason_codes:
        for item in reason_codes:
            if isinstance(item, dict):
                lines.append(f"- {item.get('code')}: {item.get('count', 0)}")
    else:
        lines.append("- none")
    lines.extend(["", "## Assets"])
    rows = report.get("per_asset_rows", []) if isinstance(report.get("per_asset_rows"), list) else []
    for row in rows:
        if isinstance(row, dict):
            lines.append(f"- {row.get('path')}: {row.get('primary_family')}")
    return "\n".join(lines) + "\n"


def _profile_asset(path: Path, *, base: Path, include_hash: bool) -> dict[str, Any]:
    digest = _sha256_file(path)
    row = _base_row(path, base=base, sha256=digest if include_hash else None)
    weight_analysis: tuple[dict[str, Any], dict[str, Any]] | None = None
    try:
        container = read_glb_container(path)
        row["profile_status"] = "OK"
        row["parse_status"] = "parsed"
        gltf = container.json
        row["node_count"] = len(gltf.get("nodes")) if isinstance(gltf.get("nodes"), list) else 0
        skins = gltf.get("skins") if isinstance(gltf.get("skins"), list) else []
        row["skin_count"] = len(skins)
        if len(skins) == 1 and isinstance(skins[0], dict) and isinstance(skins[0].get("joints"), list):
            row["skin_joint_count"] = len(skins[0]["joints"])
        try:
            graph = extract_joint_graph(gltf)
            row["root_count"] = len(graph.roots)
        except Exception as exc:  # graph errors are represented through resolver failure below
            row["diagnostics"].append(_diagnostic_from_exception("joint_graph", exc))
        try:
            weight_analysis = summarize_joint_weights(container)
            summaries, _weight_summary = weight_analysis
            row["weighted_joint_count"] = len(summaries)
        except Exception as exc:
            row["diagnostics"].append(_diagnostic_from_exception("weight_summary", exc))
        declared = resolve_humanoid(gltf)
        row["resolver"] = {"status": "success", "source_kind": "semantic_humanoid_resolver", "failure_code": None}
        contract = build_contract_from_declared_data(declared, source_hash=digest, output_hash=digest)
        row["contract_readiness"] = {"status": "ready", "ready": True, "failure_code": None}
        try:
            semantic_report = build_semantic_body_report(container, declared, weight_analysis=weight_analysis)
            quality = run_humanoid_quality_gate(container, declared, semantic_report=semantic_report, weight_analysis=weight_analysis)
            row["quality_gate"] = {"status": quality.status, "reasons": [], "diagnostic": quality.diagnostic}
        except HumanoidQualityGateError as exc:
            row["quality_gate"] = {"status": "failed", "reasons": _reason_dicts(exc.diagnostic), "diagnostic": exc.diagnostic}
        row["contract_readiness"]["schema"] = contract.get("schema")
    except GltfSkinAnalysisError as exc:
        _mark_unparseable(row, exc)
    except SemanticHumanoidResolutionError as exc:
        code = _exception_code(exc)
        row["resolver"] = {"status": "failed", "source_kind": None, "failure_code": code}
        row["quality_gate"] = {"status": "not_run", "reasons": []}
        row["contract_readiness"] = {"status": "not_ready", "ready": False, "failure_code": code}
        row["diagnostics"].extend(_resolver_diagnostics(exc))
    except HumanoidContractError as exc:
        code = _exception_code(exc)
        row["contract_readiness"] = {"status": "not_ready", "ready": False, "failure_code": code}
        row["diagnostics"].append(_diagnostic_from_exception("contract", exc))
    except Exception as exc:
        _mark_unparseable(row, exc)
    classification = classify_asset_family(row)
    if row.get("parse_status") == "unparseable":
        row["profile_status"] = "FAILED"
        row["failure"] = _failure_payload("profile_failed", "row_profile_error", _failure_message_from_row(row))
    row.update(classification)
    return row


def _base_row(path: Path, *, base: Path, sha256: str | None) -> dict[str, Any]:
    return {
        "path": _display_path(path, base),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
        "profile_status": "OK",
        "parse_status": "not_run",
        "node_count": 0,
        "skin_count": 0,
        "root_count": 0,
        "skin_joint_count": 0,
        "weighted_joint_count": 0,
        "resolver": {"status": "not_run", "source_kind": None, "failure_code": None},
        "quality_gate": {"status": "not_run", "reasons": []},
        "contract_readiness": {"status": "not_ready", "ready": False, "failure_code": None},
        "primary_family": "other_unknown",
        "secondary_reason_codes": [],
        "diagnostics": [],
        "publication_evidence": False,
    }


def _failed_profile_row(path: Path, *, base: Path, include_hash: bool, exc: Exception) -> dict[str, Any]:
    row = _base_row(path, base=base, sha256=_sha256_file(path) if include_hash else None)
    row["profile_status"] = "FAILED"
    row["parse_status"] = "unparseable"
    row["failure"] = _failure_payload("profile_failed", "row_profile_error", str(exc))
    row["diagnostics"].append({"code": "profile_failed", "message": str(exc)})
    row.update(classify_asset_family(row))
    return row


def _failure_payload(code: str, category: str, message: str) -> dict[str, str]:
    return {"code": code, "category": category, "message": message or "asset profiling failed"}


def _failure_message_from_row(row: dict[str, Any]) -> str:
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), list) else []
    for diagnostic in diagnostics:
        if isinstance(diagnostic, dict) and diagnostic.get("message"):
            return str(diagnostic["message"])
    return "asset could not be parsed or profiled"


def _mark_unparseable(row: dict[str, Any], exc: Exception) -> None:
    code = _exception_code(exc) or "malformed_unparseable"
    row["parse_status"] = "unparseable"
    row["resolver"] = {"status": "not_run", "source_kind": None, "failure_code": None}
    row["quality_gate"] = {"status": "not_run", "reasons": []}
    row["contract_readiness"] = {"status": "not_ready", "ready": False, "failure_code": code}
    row["diagnostics"].append({"code": code, "message": str(exc)})


def _collect_reason_codes(row: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for reason in row.get("quality_gate", {}).get("reasons", []):
        if isinstance(reason, dict) and reason.get("code"):
            codes.append(str(reason["code"]))
    for diagnostic in row.get("diagnostics", []):
        if isinstance(diagnostic, dict) and diagnostic.get("code"):
            codes.append(str(diagnostic["code"]))
    failure = _resolver_failure_code(row)
    if failure:
        codes.append(failure)
    return codes


def _resolver_failure_code(row: dict[str, Any]) -> str | None:
    resolver = row.get("resolver") if isinstance(row.get("resolver"), dict) else {}
    value = resolver.get("failure_code")
    return str(value) if value else None


def _is_contract_ready(row: dict[str, Any]) -> bool:
    contract = row.get("contract_readiness") if isinstance(row.get("contract_readiness"), dict) else {}
    resolver = row.get("resolver") if isinstance(row.get("resolver"), dict) else {}
    quality = row.get("quality_gate") if isinstance(row.get("quality_gate"), dict) else {}
    return bool(contract.get("ready")) and resolver.get("status") == "success" and quality.get("status") == "passed"


def _dedupe_reason_codes(codes: Iterable[str]) -> list[str]:
    return _sorted_reason_codes({code: 1 for code in codes if code})


def _sorted_reason_codes(counts: dict[str, int]) -> list[str]:
    return sorted(counts, key=lambda code: (REASON_CODE_ORDER.index(code) if code in REASON_CODE_ORDER else len(REASON_CODE_ORDER), code))


def _sort_reason_counts(counts: dict[str, int]) -> dict[str, int]:
    return {code: counts[code] for code in _sorted_reason_codes(counts)}


def _build_family_summaries(rows: list[dict[str, Any]], family_counts: dict[str, int]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for family in FAMILY_PRECEDENCE:
        family_rows = [row for row in rows if row["primary_family"] == family]
        resolver_counts: dict[str, int] = {}
        quality_counts: dict[str, int] = {}
        contract_counts: dict[str, int] = {}
        reason_counts: dict[str, int] = {}
        for row in family_rows:
            _increment(resolver_counts, str(row["resolver"]["status"]))
            _increment(quality_counts, str(row["quality_gate"]["status"]))
            _increment(contract_counts, str(row["contract_readiness"]["status"]))
            for code in row.get("secondary_reason_codes", []):
                _increment(reason_counts, str(code))
        summaries[family] = {
            "count": family_counts[family],
            "paths": [row["path"] for row in family_rows],
            "resolver_status_counts": dict(sorted(resolver_counts.items())),
            "quality_status_counts": dict(sorted(quality_counts.items())),
            "contract_status_counts": dict(sorted(contract_counts.items())),
            "reason_code_counts": _sort_reason_counts(reason_counts),
        }
    return summaries


def _reason_dicts(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = diagnostic.get("reasons") if isinstance(diagnostic.get("reasons"), list) else []
    return [reason for reason in reasons if isinstance(reason, dict)]


def _resolver_diagnostics(exc: SemanticHumanoidResolutionError) -> list[dict[str, Any]]:
    if exc.diagnostics:
        return [dict(item) for item in exc.diagnostics]
    return [_diagnostic_from_exception("resolver", exc)]


def _diagnostic_from_exception(prefix: str, exc: Exception) -> dict[str, Any]:
    return {"code": _exception_code(exc) or prefix, "message": str(exc)}


def _exception_code(exc: Exception) -> str | None:
    text = str(exc)
    if ":" in text:
        candidate = text.split(":", 1)[0].strip()
        if candidate:
            return candidate
    if text:
        candidate = text.split(".", 1)[0].strip().split(" ", 1)[0]
        return candidate or None
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_readable_glb(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise CorpusInputError(f"unreadable .glb input: {path}; check file permissions and rerun") from exc


def _increment(target: dict[str, int], key: str) -> None:
    target[key] = target.get(key, 0) + 1


def _common_parent(paths: list[Path]) -> Path:
    resolved = [path.resolve().parent for path in paths]
    common = Path(str(Path(*Path(resolved[0]).parts))) if len(resolved) == 1 else Path(_common_path_string(resolved))
    return common


def _common_path_string(paths: list[Path]) -> str:
    import os

    return os.path.commonpath([str(path) for path in paths])


def _display_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _path_sort_key(path: Path) -> str:
    return path.resolve().as_posix().casefold()


def _has_glob_magic(value: str) -> bool:
    return any(char in value for char in "*?[")
