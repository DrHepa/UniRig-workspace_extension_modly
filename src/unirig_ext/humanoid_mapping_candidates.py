from __future__ import annotations

import glob
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .gltf_skin_analysis import GltfSkinAnalysisError, read_accessor, read_glb_container, summarize_joint_weights
from .humanoid_contract import REQUIRED_CHAINS, REQUIRED_ROLES, HumanoidContractError, build_contract_from_declared_data
from .humanoid_quality_gate import HumanoidQualityGateError, run_humanoid_quality_gate
from .kimodo_probe import KimodoProbeBackend, ProbeResult, probe_result_to_dict, unavailable_probe_result
from .semantic_body_graph import build_semantic_body_report
from .semantic_humanoid_resolver import SemanticHumanoidResolutionError, extract_joint_graph, resolve_humanoid


CANDIDATE_SCHEMA = "unirig.bone_humanoid_mapping_candidate.v1"
SEMANTIC_CANDIDATES_SCHEMA = "unirig.semantic_candidates.v1"
KIMODO_CONTRACT_SCHEMA = "modly.humanoid.v1"
REAL_CORPUS_ENV_VAR = "UNIRIG_HUMANOID_MAPPING_REAL_EXPORTS"
SEMANTIC_CORE_ROLES = ("hips", "spine", "head")
SEMANTIC_REJECTED_LIMIT = 5
FULL_TOPOLOGY_SUFFICIENT_ASSETS = [
    "export-1777624346995.glb",
    "export-1777624347439.glb",
    "export-1777624348231.glb",
    "export-1777624349386.glb",
    "export-1777624350989.glb",
    "export-1777624351785.glb",
    "export-1777624352711.glb",
    "export-1777624353511.glb",
    "export-1777624354819.glb",
    "export-1777624355568.glb",
    "export-1777624356306.glb",
    "export-1777624357151.glb",
    "export-1777667214327.glb",
    "export-1777667214813.glb",
    "export-1777667215683.glb",
    "export-1777667217263.glb",
    "export-1777667219523.glb",
    "export-1777667221931.glb",
    "export-1777667225380.glb",
    "export-1777667227081.glb",
    "export-1777667229578.glb",
    "export-1777667231792.glb",
    "export-1777667233710.glb",
    "export-1777667235487.glb",
    "export-1777797700051.glb",
    "export-1777797700523.glb",
    "export-1777797701138.glb",
    "export-1777797702012.glb",
    "export-1777797708360.glb",
    "export-1777797709183.glb",
    "export-1777797710061.glb",
    "export-1777797711831.glb",
    "export-1777797712639.glb",
    "export-1777797714326.glb",
]
REPRESENTATIVE_CORPUS_ASSETS = {
    "direct_ok": ["export-1777624350989.glb", "export-1777797710061.glb"],
    "resolver_ok_direct_fail": [
        "export-1777797700523.glb",
        "export-1777797712639.glb",
        "export-1777797701138.glb",
        "export-1777667225380.glb",
    ],
    "ambiguous": ["export-1777797714326.glb", "export-1777667229578.glb", "export-1777624349386.glb"],
    "negative_controls": ["export-1777797713460.glb", "export-1777797711066.glb"],
}
ROLE_ALIASES = {
    "spine": "torso",
    "left_lower_arm": "left_forearm",
    "right_lower_arm": "right_forearm",
}
AMBIGUITY_CODES = {"semantic_symmetry_ambiguous", "semantic_leg_symmetry_ambiguous", "semantic_graph_ambiguous"}


class CandidateInputError(ValueError):
    pass


class CandidateOutputError(ValueError):
    pass


def select_candidate_inputs(inputs: Iterable[str | Path], *, manifest: str | Path | None = None) -> list[Path]:
    raw_inputs = list(inputs)
    if manifest is not None:
        raw_inputs.extend(_read_manifest_paths(Path(manifest)))
    if not raw_inputs:
        raise CandidateInputError("no input paths were provided; pass explicit .glb paths or --manifest")
    selected: list[Path] = []
    for raw in raw_inputs:
        text = str(raw)
        if _has_glob_magic(text):
            matches = [Path(match) for match in glob.glob(text)]
            glbs = [path for path in matches if path.is_file() and path.suffix.lower() == ".glb"]
            if not glbs:
                raise CandidateInputError(f"glob input selected no .glb files: {text}")
            selected.extend(glbs)
            continue
        path = Path(raw)
        if not path.exists():
            raise CandidateInputError(f"input path does not exist: {path}")
        if path.is_dir():
            glbs = [child for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".glb"]
            if not glbs:
                raise CandidateInputError(f"directory input contains no .glb files: {path}")
            selected.extend(glbs)
            continue
        if not path.is_file():
            raise CandidateInputError(f"input is not a regular file: {path}")
        if path.suffix.lower() != ".glb":
            raise CandidateInputError(f"explicit input is not a .glb file: {path}")
        selected.append(path)
    unique = {path.resolve(): path.resolve() for path in selected}
    paths = sorted(unique.values(), key=lambda item: item.as_posix().casefold())
    for path in paths:
        _validate_glb_magic(path)
    return paths


def build_candidate_reports(
    inputs: Iterable[str | Path],
    *,
    manifest: str | Path | None = None,
    source: str = "explicit-input",
    kimodo_backend: Any | None = None,
    probe_retarget: bool = False,
) -> dict[str, Any]:
    paths = select_candidate_inputs(inputs, manifest=manifest)
    candidates = [
        build_candidate_for_glb(path, source=source, kimodo_backend=kimodo_backend, probe_retarget=probe_retarget)
        for path in paths
    ]
    return {"schema": CANDIDATE_SCHEMA, "summary": _summary(candidates), "candidates": candidates}


def build_representative_corpus_manifest(exports_root: str | Path) -> dict[str, Any]:
    root = Path(exports_root).expanduser().resolve()
    if not root.exists():
        raise CandidateInputError(f"representative corpus root does not exist: {root}")
    if not root.is_dir():
        raise CandidateInputError(f"representative corpus root is not a directory: {root}")
    groups: dict[str, list[str]] = {}
    assets: list[str] = []
    for group, names in REPRESENTATIVE_CORPUS_ASSETS.items():
        group_paths: list[str] = []
        for name in names:
            path = root / name
            if not path.exists():
                raise CandidateInputError(f"representative corpus asset is missing: {path}")
            if not path.is_file():
                raise CandidateInputError(f"representative corpus asset is not a file: {path}")
            group_paths.append(str(path.resolve()))
            assets.append(str(path.resolve()))
        groups[group] = sorted(group_paths, key=lambda item: Path(item).name.casefold())
    assets = sorted(assets, key=lambda item: Path(item).name.casefold())
    return {
        "schema": f"{CANDIDATE_SCHEMA}.representative_corpus.v1",
        "env_var": REAL_CORPUS_ENV_VAR,
        "root": str(root),
        "total_assets": len(assets),
        "groups": groups,
        "assets": assets,
        "read_only": True,
    }


def build_full_topology_sufficient_corpus_manifest(exports_root: str | Path) -> dict[str, Any]:
    root = Path(exports_root).expanduser().resolve()
    if not root.exists():
        raise CandidateInputError(f"full topology-sufficient corpus root does not exist: {root}")
    if not root.is_dir():
        raise CandidateInputError(f"full topology-sufficient corpus root is not a directory: {root}")
    assets: list[str] = []
    for name in FULL_TOPOLOGY_SUFFICIENT_ASSETS:
        path = root / name
        if not path.exists():
            raise CandidateInputError(f"full topology-sufficient corpus asset is missing: {path}")
        if not path.is_file():
            raise CandidateInputError(f"full topology-sufficient corpus asset is not a file: {path}")
        assets.append(str(path.resolve()))
    assets = sorted(assets, key=lambda item: Path(item).name.casefold())
    return {
        "schema": f"{CANDIDATE_SCHEMA}.full_topology_sufficient_corpus.v1",
        "env_var": REAL_CORPUS_ENV_VAR,
        "root": str(root),
        "total_assets": len(assets),
        "topology_sufficient": True,
        "assets": assets,
        "read_only": True,
    }


def render_run_suggestions(*, exports_root: str = "$UNIRIG_HUMANOID_MAPPING_REAL_EXPORTS", output_root: str = "/tmp/opencode/unirig-map-candidates") -> str:
    return "\n".join(
        [
            "# UniRig humanoid mapping candidate diagnostics",
            "# Diagnostic only: does not publish humanoid metadata and must not write beside source exports.",
            "",
            "## Focused unit/CLI tests",
            "python3 -m unittest discover -s tests -p test_humanoid_mapping_candidates.py -v",
            "",
            "## Optional env-gated representative corpus test",
            f"{REAL_CORPUS_ENV_VAR}={exports_root} python3 -m unittest discover -s tests -p test_humanoid_mapping_candidates.py -v",
            "",
            "## Optional CLI dry run (write only under /tmp/opencode)",
            f"mkdir -p {output_root}",
            f"python3 -m unirig_ext.humanoid_mapping_candidates_cli {exports_root}/export-1777624350989.glb {exports_root}/export-1777797714326.glb --json-out {output_root}/candidates.json --jsonl-out {output_root}/candidates.jsonl",
            "",
        ]
    )


def build_candidate_for_glb(
    path: str | Path,
    *,
    source: str = "explicit-input",
    kimodo_backend: Any | None = None,
    probe_retarget: bool = False,
) -> dict[str, Any]:
    glb_path = Path(path).resolve()
    _validate_glb_magic(glb_path)
    digest = _sha256_file(glb_path)
    diagnostics: list[dict[str, Any]] = []
    weight_analysis: tuple[dict[str, Any], dict[str, Any]] | None = None
    declared: dict[str, Any] | None = None
    contract_ready = False
    quality_status = "not_run"
    quality_reasons: list[dict[str, Any]] = []

    try:
        container = read_glb_container(glb_path)
        graph = extract_joint_graph(container.json)
    except (GltfSkinAnalysisError, SemanticHumanoidResolutionError) as exc:
        raise CandidateInputError(str(exc)) from exc

    try:
        weight_analysis = summarize_joint_weights(container)
    except Exception as exc:
        diagnostics.append(_diagnostic_from_exception("skin_weight_data_unavailable", exc))

    try:
        declared = resolve_humanoid(container.json)
        try:
            semantic_report = build_semantic_body_report(container, declared, weight_analysis=weight_analysis)
            quality = run_humanoid_quality_gate(container, declared, semantic_report=semantic_report, weight_analysis=weight_analysis)
            quality_status = quality.status
        except HumanoidQualityGateError as exc:
            quality_status = "failed"
            quality_reasons = _reason_dicts(exc.diagnostic)
            diagnostics.append(dict(exc.diagnostic))
        try:
            build_contract_from_declared_data(declared, source_hash=digest, output_hash=digest)
            contract_ready = quality_status in {"passed", "not_run"}
        except HumanoidContractError as exc:
            diagnostics.append(_diagnostic_from_exception("contract_invalid", exc))
    except SemanticHumanoidResolutionError as exc:
        diagnostics.extend(_resolver_diagnostics(exc))

    roles = _build_roles(declared, diagnostics)
    chains = _build_chains(roles)
    topology = _topology(container, graph)
    status = _candidate_status(roles, contract_ready=contract_ready, diagnostics=diagnostics, topology=topology)
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "asset": {"path": str(glb_path), "sha256": digest, "source": source},
        "topology": topology,
        "roles": roles,
        "chains": chains,
        "symmetry": _symmetry(roles, graph),
        "transforms": _transforms(graph),
        "skin_weights": _skin_weights(weight_analysis, quality_status=quality_status, quality_reasons=quality_reasons),
        "kimodo_projection": {
            "contract_schema": KIMODO_CONTRACT_SCHEMA,
            "role_aliases": dict(sorted(ROLE_ALIASES.items())),
            "probe": {"status": "not_run", "primary_failure_layer": None, "code": None, "message": "Kimodo probe was not requested.", "diagnostics": []},
        },
        "status": status,
        "diagnostics": sorted(diagnostics, key=lambda item: (str(item.get("code", "")), str(item.get("message", "")))),
    }
    if kimodo_backend is not None:
        candidate["kimodo_projection"]["probe"] = _run_probe(glb_path, candidate, kimodo_backend, probe_retarget=probe_retarget)
    return candidate


def build_semantic_candidates_sidecar(candidate: dict[str, Any], *, unsafe_flags: Iterable[str] = ()) -> dict[str, Any]:
    """Return compact, untrusted sidecar candidates for best-effort mapping only."""
    roles = candidate.get("roles") if isinstance(candidate.get("roles"), dict) else {}
    topology = candidate.get("topology") if isinstance(candidate.get("topology"), dict) else {}
    transforms = candidate.get("transforms") if isinstance(candidate.get("transforms"), dict) else {}
    transform_nodes = transforms.get("nodes") if isinstance(transforms.get("nodes"), dict) else {}
    unsafe = sorted({str(flag) for flag in unsafe_flags if str(flag)})

    accepted_roles: dict[str, dict[str, Any]] = {}
    for role, role_payload in sorted(roles.items()):
        if not isinstance(role_payload, dict):
            continue
        bone = role_payload.get("bone")
        if not isinstance(bone, str) or not bone:
            continue
        node_payload = transform_nodes.get(bone) if isinstance(transform_nodes.get(bone), dict) else {}
        accepted_roles[str(role)] = {
            "role": str(role),
            "node_id": bone,
            "node_name": bone,
            "node_index": int(node_payload.get("index", 0)),
            "confidence": round(float(role_payload.get("confidence", 0.0)), 3),
            "reasons": _semantic_role_reasons(role_payload),
            "source": "semantic_humanoid_resolver",
            "unsafe_flags": [],
            "rejected": False,
        }

    missing_core = sorted(role for role in SEMANTIC_CORE_ROLES if role not in accepted_roles)
    diagnostics = _semantic_candidate_diagnostics(candidate, missing_core=missing_core, topology=topology)
    status = "candidate" if not missing_core else "blocked"

    return {
        "schema": SEMANTIC_CANDIDATES_SCHEMA,
        "producer": {"source_schema": CANDIDATE_SCHEMA, "resolver": "semantic_humanoid_resolver"},
        "trust": {
            "trusted": False,
            "stabilization_eligible": False,
            "reason": "contract_missing_or_untrusted",
        },
        "status": status,
        "roles": accepted_roles,
        "chains": _semantic_candidate_chains(accepted_roles),
        "rejected_candidates": _semantic_rejected_candidates(roles, unsafe_flags=unsafe),
        "topology": _semantic_topology(topology),
        "transforms": _semantic_transforms(transforms, accepted_roles),
        "diagnostics": diagnostics,
    }


def dumps_candidate_json(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def _semantic_role_reasons(role_payload: dict[str, Any]) -> list[str]:
    evidence = role_payload.get("evidence") if isinstance(role_payload.get("evidence"), list) else []
    reasons = [str(item) for item in evidence if item]
    return sorted(set(reasons)) or ["semantic_candidate"]


def _semantic_candidate_chains(accepted_roles: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    chains: dict[str, list[str]] = {}
    for chain_name, chain_roles in REQUIRED_CHAINS.items():
        present = [role for role in chain_roles if role in accepted_roles]
        if present:
            chains[str(chain_name)] = present
            if chain_name == "spine":
                chains["torso"] = present
    return dict(sorted(chains.items()))


def _semantic_rejected_candidates(roles: dict[str, Any], *, unsafe_flags: list[str]) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for role, role_payload in sorted(roles.items()):
        if not isinstance(role_payload, dict):
            continue
        bone = role_payload.get("bone")
        fail_reasons = role_payload.get("fail_reasons") if isinstance(role_payload.get("fail_reasons"), list) else []
        if bone and not unsafe_flags and not fail_reasons:
            continue
        rejected.append(
            {
                "role": str(role),
                "node_id": str(bone) if bone else None,
                "confidence": round(float(role_payload.get("confidence", 0.0)), 3),
                "reasons": sorted({str(item) for item in fail_reasons if item}) or (["unsafe_context"] if unsafe_flags and bone else ["semantic_candidate_unavailable"]),
                "unsafe_flags": list(unsafe_flags) if bone else [],
                "rejected": True,
            }
        )
    return sorted(rejected, key=lambda item: (0 if item.get("unsafe_flags") else 1, str(item.get("role", ""))))[:SEMANTIC_REJECTED_LIMIT]


def _semantic_topology(topology: dict[str, Any]) -> dict[str, Any]:
    return {
        "joint_count": int(topology.get("joint_count") or 0),
        "root_count": int(topology.get("root_count") or 0),
        "max_depth": int(topology.get("max_depth") or 0),
        "branch_points": list(topology.get("branch_points") or [])[:SEMANTIC_REJECTED_LIMIT],
        "leaf_count": int(topology.get("leaf_count") or 0),
    }


def _semantic_transforms(transforms: dict[str, Any], accepted_roles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    transform_nodes = transforms.get("nodes") if isinstance(transforms.get("nodes"), dict) else {}
    nodes: dict[str, dict[str, Any]] = {}
    for role_payload in accepted_roles.values():
        node_id = role_payload.get("node_id")
        if not isinstance(node_id, str):
            continue
        source = transform_nodes.get(node_id) if isinstance(transform_nodes.get(node_id), dict) else {}
        local = source.get("rest_local") if isinstance(source.get("rest_local"), list) else []
        translation = _matrix_translation(local)
        nodes[node_id] = {
            "index": int(source.get("index", role_payload.get("node_index", 0))),
            "parent": source.get("parent"),
            "rest_local_translation": translation,
        }
    return {
        "status": transforms.get("status", "available"),
        "matrix_order": transforms.get("matrix_order", "row-major"),
        "basis": dict(transforms.get("basis")) if isinstance(transforms.get("basis"), dict) else {"up": "Y", "forward": "Z", "status": "inferred"},
        "nodes": dict(sorted(nodes.items())),
    }


def _matrix_translation(matrix: Any) -> list[float]:
    if isinstance(matrix, list) and len(matrix) >= 3:
        values: list[float] = []
        for row in matrix[:3]:
            if isinstance(row, list) and len(row) >= 4:
                values.append(round(float(row[3]), 6))
            else:
                values.append(0.0)
        return values
    return [0.0, 0.0, 0.0]


def _semantic_candidate_diagnostics(candidate: dict[str, Any], *, missing_core: list[str], topology: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = [dict(item) for item in candidate.get("diagnostics", []) if isinstance(item, dict)]
    if missing_core:
        reasons = ["missing_core_roles"]
        if int(topology.get("joint_count") or 0) < 8:
            reasons.append("too_few_joints")
        diagnostics.append(
            {
                "code": "insufficient_core_candidates",
                "message": "semantic candidates cannot drive animation mapping without hips, spine, and head.",
                "missing_roles": missing_core,
                "reasons": reasons,
            }
        )
    return sorted(diagnostics, key=lambda item: (str(item.get("code", "")), str(item.get("message", ""))))


def write_candidates_json(candidates_or_report: list[dict[str, Any]] | dict[str, Any], path: str | Path) -> Path:
    output = _validate_output_parent(path)
    if isinstance(candidates_or_report, dict) and "candidates" in candidates_or_report:
        report = dict(candidates_or_report)
        report["candidates"] = _sort_candidates(list(report["candidates"]))
        report["summary"] = _summary(report["candidates"])
    else:
        candidates = _sort_candidates(list(candidates_or_report))  # type: ignore[arg-type]
        report = {"schema": CANDIDATE_SCHEMA, "summary": _summary(candidates), "candidates": candidates}
    _atomic_write_text(output, dumps_candidate_json(report))
    return output


def write_candidates_jsonl(candidates: list[dict[str, Any]], path: str | Path) -> Path:
    output = _validate_output_parent(path)
    text = "".join(dumps_candidate_json(candidate).replace("\n", " ").rstrip() + "\n" for candidate in _sort_candidates(candidates))
    _atomic_write_text(output, text)
    return output


def _run_probe(glb_path: Path, candidate: dict[str, Any], backend: Any, *, probe_retarget: bool) -> dict[str, Any]:
    if not backend.available():
        if hasattr(backend, "unavailable_result"):
            return probe_result_to_dict(backend.unavailable_result())
        return probe_result_to_dict(unavailable_probe_result())
    probe_root = getattr(backend, "probe_output_root", None)
    if probe_root is not None:
        temp_dir = tempfile.mkdtemp(prefix="unirig-map-candidates-", dir=Path(probe_root))
        return _run_probe_in_directory(glb_path, candidate, backend, probe_retarget=probe_retarget, temp_dir=Path(temp_dir))
    with tempfile.TemporaryDirectory(prefix="unirig-map-candidates-") as temp_dir:
        return _run_probe_in_directory(glb_path, candidate, backend, probe_retarget=probe_retarget, temp_dir=Path(temp_dir))


def _run_probe_in_directory(glb_path: Path, candidate: dict[str, Any], backend: Any, *, probe_retarget: bool, temp_dir: Path) -> dict[str, Any]:
    copied = Path(temp_dir) / glb_path.name
    shutil.copy2(glb_path, copied)
    sidecar_payload = _sidecar_payload(candidate)
    sidecar_path = copied.with_suffix(".rigmeta.json")
    sidecar_path.write_text(json.dumps(sidecar_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        result = backend.probe(copied, sidecar_payload, probe_retarget=probe_retarget)
    except Exception as exc:
        return probe_result_to_dict(ProbeResult(status="rejected", primary_failure_layer="ambiguous", code="probe_exception", message=str(exc), diagnostics=[_diagnostic_from_exception("probe_exception", exc)]))
    return probe_result_to_dict(result)


def _sidecar_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    roles = candidate.get("roles", {}) if isinstance(candidate.get("roles"), dict) else {}
    required_roles = {role: payload.get("bone") for role, payload in roles.items() if isinstance(payload, dict) and payload.get("bone")}
    transforms = candidate.get("transforms", {}) if isinstance(candidate.get("transforms"), dict) else {}
    transform_nodes = transforms.get("nodes") if isinstance(transforms.get("nodes"), dict) else {}
    contract_nodes = {
        node_id: {
            "name": node_id,
            "index": int(transform_payload.get("index", index)),
            "transforms": dict(transform_payload),
        }
        for index, (node_id, transform_payload) in enumerate(sorted(transform_nodes.items()))
        if isinstance(transform_payload, dict)
    }
    return {
        "humanoid_contract": {
            "schema": KIMODO_CONTRACT_SCHEMA,
            "required_roles": required_roles,
            "chains": candidate.get("chains", {}),
            "nodes": contract_nodes,
            "confidence": {"roles": {role: payload.get("confidence", 0.0) for role, payload in roles.items() if isinstance(payload, dict)}},
            "diagnostics": candidate.get("diagnostics", []),
        }
    }


def _build_roles(declared: dict[str, Any] | None, diagnostics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fail_reasons = _diagnostic_codes(diagnostics)
    declared_roles = declared.get("roles", {}) if isinstance(declared, dict) and isinstance(declared.get("roles"), dict) else {}
    confidence = declared.get("confidence", {}).get("roles", {}) if isinstance(declared, dict) and isinstance(declared.get("confidence"), dict) else {}
    roles: dict[str, dict[str, Any]] = {}
    for role in REQUIRED_ROLES:
        bone = declared_roles.get(role)
        if isinstance(bone, str) and bone:
            roles[role] = {
                "bone": bone,
                "confidence": round(float(confidence.get(role, 0.0)), 3),
                "evidence": ["semantic_humanoid_resolver", "per_asset_topology", "rest_transform_symmetry"],
                "fail_reasons": [],
            }
        else:
            roles[role] = {"bone": None, "confidence": 0.0, "evidence": [], "fail_reasons": fail_reasons or ["semantic_required_roles_missing"]}
    return dict(sorted(roles.items()))


def _build_chains(roles: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    chains: dict[str, list[str]] = {}
    for chain_name, chain_roles in REQUIRED_CHAINS.items():
        bones = [roles[role]["bone"] for role in chain_roles if roles.get(role, {}).get("bone")]
        chains[chain_name] = [str(bone) for bone in bones]
    return dict(sorted(chains.items()))


def _candidate_status(roles: dict[str, dict[str, Any]], *, contract_ready: bool, diagnostics: list[dict[str, Any]], topology: dict[str, Any]) -> str:
    if any(not payload.get("bone") for payload in roles.values()):
        if int(topology.get("joint_count") or 0) < 24:
            return "blocked"
        codes = set(_diagnostic_codes(diagnostics))
        return "ambiguous" if codes & AMBIGUITY_CODES else "blocked"
    return "contract_ready" if contract_ready else "candidate"


def _topology(container: Any, graph: Any) -> dict[str, Any]:
    return {
        "joint_count": len(graph.joints),
        "root_count": len(graph.roots),
        "roots": list(graph.roots),
        "max_depth": max((node.depth for node in graph.nodes.values()), default=0),
        "branch_points": sorted(node.id for node in graph.nodes.values() if len(node.children) > 1),
        "leaf_count": len(graph.leaves),
        "leaves": sorted(graph.leaves),
        "hierarchy": _hierarchy(graph),
        "bbox": _position_bbox(container),
    }


def _hierarchy(graph: Any) -> dict[str, dict[str, Any]]:
    return {
        node.id: {
            "parent": node.parent,
            "children": list(node.children),
            "depth": node.depth,
            "index": node.index,
        }
        for node in sorted(graph.nodes.values(), key=lambda item: item.index)
    }


def _position_bbox(container: Any) -> dict[str, Any]:
    gltf = container.json
    meshes = gltf.get("meshes") if isinstance(gltf.get("meshes"), list) else []
    positions: list[tuple[float, float, float]] = []
    for mesh in meshes:
        primitives = mesh.get("primitives") if isinstance(mesh, dict) and isinstance(mesh.get("primitives"), list) else []
        for primitive in primitives:
            attributes = primitive.get("attributes") if isinstance(primitive, dict) and isinstance(primitive.get("attributes"), dict) else {}
            position_accessor = attributes.get("POSITION")
            if isinstance(position_accessor, int):
                for row in read_accessor(container, position_accessor):
                    if len(row) >= 3:
                        positions.append((float(row[0]), float(row[1]), float(row[2])))
    if positions:
        mins = [round(min(row[axis] for row in positions), 6) for axis in range(3)]
        maxs = [round(max(row[axis] for row in positions), 6) for axis in range(3)]
        return {"status": "available", "min": mins, "max": maxs}
    return {"status": "unavailable", "reason": "POSITION accessor min/max unavailable"}


def _symmetry(roles: dict[str, dict[str, Any]], graph: Any) -> dict[str, Any]:
    pair_scores: dict[str, Any] = {}
    axis_scores = {"x": 0.0, "z": 0.0}
    pairs = [
        ("arms", "left_upper_arm", "right_upper_arm"),
        ("legs", "left_upper_leg", "right_upper_leg"),
        ("hands", "left_hand", "right_hand"),
        ("feet", "left_foot", "right_foot"),
    ]
    for label, left_role, right_role in pairs:
        left = roles.get(left_role, {}).get("bone")
        right = roles.get(right_role, {}).get("bone")
        if left in graph.nodes and right in graph.nodes:
            lx = graph.nodes[left].rest_world[0][3]
            rx = graph.nodes[right].rest_world[0][3]
            lz = graph.nodes[left].rest_world[2][3]
            rz = graph.nodes[right].rest_world[2][3]
            x_sep = abs(float(lx) - float(rx))
            z_sep = abs(float(lz) - float(rz))
            axis_scores["x"] += x_sep
            axis_scores["z"] += z_sep
            pair_scores[label] = {"left": left, "right": right, "x_separation": round(x_sep, 6), "z_separation": round(z_sep, 6)}
    lateral_axis = "x" if axis_scores["x"] >= axis_scores["z"] else "z"
    total = axis_scores["x"] + axis_scores["z"]
    confidence = 0.0 if total == 0.0 else axis_scores[lateral_axis] / total
    return {"lateral_axis": lateral_axis, "left_right_confidence": round(confidence, 3), "pair_scores": dict(sorted(pair_scores.items()))}


def _transforms(graph: Any) -> dict[str, Any]:
    nodes = {
        node.id: {"index": node.index, "parent": node.parent, "rest_local": node.rest_local, "rest_world": node.rest_world}
        for node in sorted(graph.nodes.values(), key=lambda item: item.index)
    }
    return {"status": "available", "matrix_order": "row-major", "basis": {"up": "Y", "forward": "Z", "status": "inferred"}, "nodes": nodes}


def _skin_weights(weight_analysis: tuple[dict[str, Any], dict[str, Any]] | None, *, quality_status: str, quality_reasons: list[dict[str, Any]]) -> dict[str, Any]:
    if weight_analysis is None:
        return {"status": "not_run", "weighted_joint_count": 0, "quality_gate": quality_status, "reasons": quality_reasons, "summary_only": True}
    summaries, summary = weight_analysis
    return {
        "status": "summary",
        "weighted_joint_count": len(summaries),
        "quality_gate": quality_status,
        "reasons": quality_reasons,
        "summary_only": True,
        "mesh_summary": summary,
    }


def _read_manifest_paths(path: Path) -> list[str]:
    if not path.exists():
        raise CandidateInputError(f"manifest path does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateInputError(f"manifest is not valid JSON: {path}") from exc
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and isinstance(payload.get("assets"), list):
        values = payload["assets"]
    else:
        raise CandidateInputError("manifest must be a JSON list or an object with an assets list")
    paths = [str(item) for item in values if isinstance(item, (str, Path))]
    if len(paths) != len(values):
        raise CandidateInputError("manifest assets must all be path strings")
    return paths


def _summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        status = str(candidate.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {"total_assets": len(candidates), "status_counts": dict(sorted(counts.items())), "publication_evidence": False}


def _sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda item: str(item.get("asset", {}).get("path", "")).casefold())


def _validate_output_parent(path: str | Path) -> Path:
    output = Path(path)
    parent = output.parent if output.parent != Path("") else Path(".")
    if not parent.exists():
        raise CandidateOutputError(f"output parent does not exist: {parent}")
    if not parent.is_dir():
        raise CandidateOutputError(f"output parent is not a directory: {parent}")
    return output


def _atomic_write_text(output: Path, text: str) -> None:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False)
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
        temp_path.replace(output)
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _validate_glb_magic(path: Path) -> None:
    if not path.exists():
        raise CandidateInputError(f"input path does not exist: {path}")
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        raise CandidateInputError(f"unreadable .glb input: {path}; check file permissions and rerun") from exc
    if magic != b"glTF":
        raise CandidateInputError(f"input is not an embedded GLB file: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostic_from_exception(prefix: str, exc: Exception) -> dict[str, Any]:
    return {"code": _exception_code(exc) or prefix, "message": str(exc)}


def _resolver_diagnostics(exc: SemanticHumanoidResolutionError) -> list[dict[str, Any]]:
    if exc.diagnostics:
        return [dict(item) for item in exc.diagnostics]
    return [_diagnostic_from_exception("resolver", exc)]


def _exception_code(exc: Exception) -> str | None:
    text = str(exc)
    if ":" in text:
        candidate = text.split(":", 1)[0].strip()
        return candidate or None
    if text:
        return text.split(".", 1)[0].strip().split(" ", 1)[0] or None
    return None


def _reason_dicts(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = diagnostic.get("reasons") if isinstance(diagnostic.get("reasons"), list) else []
    return [reason for reason in reasons if isinstance(reason, dict)]


def _diagnostic_codes(diagnostics: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("code")) for item in diagnostics if item.get("code")})


def _has_glob_magic(value: str) -> bool:
    return any(char in value for char in "*?[")
