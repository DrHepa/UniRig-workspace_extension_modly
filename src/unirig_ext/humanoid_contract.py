from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


HUMANOID_SCHEMA = "modly.humanoid.v1"
MINIMUM_CONFIDENCE = 0.75

REQUIRED_ROLES = (
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
)

REQUIRED_CHAINS = {
    "spine": ("hips", "spine", "chest", "neck", "head"),
    "left_arm": ("left_upper_arm", "left_lower_arm", "left_hand"),
    "right_arm": ("right_upper_arm", "right_lower_arm", "right_hand"),
    "left_leg": ("left_upper_leg", "left_lower_leg", "left_foot"),
    "right_leg": ("right_upper_leg", "right_lower_leg", "right_foot"),
}

OPTIONAL_TOE_ROLES = {
    "left_leg": "left_toe",
    "right_leg": "right_toe",
}

OPTIONAL_SHOULDER_ROLE_ALIASES = {
    "left_shoulder": ("left_shoulder", "left_clavicle"),
    "right_shoulder": ("right_shoulder", "right_clavicle"),
}

OPTIONAL_PROXIMAL_ARM_CHAINS = {
    "left_arm_proximal": ("left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand"),
    "right_arm_proximal": ("right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand"),
}

OPTIONAL_FINGER_CHAINS = {
    "left_thumb": ("left_thumb_1", "left_thumb_2", "left_thumb_3"),
    "left_index": ("left_index_1", "left_index_2", "left_index_3"),
    "left_middle": ("left_middle_1", "left_middle_2", "left_middle_3"),
    "left_ring": ("left_ring_1", "left_ring_2", "left_ring_3"),
    "left_little": ("left_little_1", "left_little_2", "left_little_3"),
    "right_thumb": ("right_thumb_1", "right_thumb_2", "right_thumb_3"),
    "right_index": ("right_index_1", "right_index_2", "right_index_3"),
    "right_middle": ("right_middle_1", "right_middle_2", "right_middle_3"),
    "right_ring": ("right_ring_1", "right_ring_2", "right_ring_3"),
    "right_little": ("right_little_1", "right_little_2", "right_little_3"),
}

WARNING_OPTIONAL_FINGERS_MISSING = "optional_fingers_missing"
WARNING_OPTIONAL_FINGER_CHAIN_PARTIAL = "optional_finger_chain_partial"
WARNING_BASIS_INFERRED = "basis_inferred"
WARNING_INVERSE_BIND_UNAVAILABLE = "inverse_bind_unavailable"


class HumanoidContractError(ValueError):
    pass


def build_humanoid_contract(output_path: Path, *, source_hash: str) -> dict:
    declared = _read_declared_humanoid_source(Path(output_path))
    if declared is None:
        raise HumanoidContractError(
            "No declared humanoid metadata was found for the published output. "
            "Required roles must be provided explicitly; silent bone-name guessing is disabled."
        )
    return build_contract_from_declared_data(declared, source_hash=source_hash, output_hash=_sha256_file(Path(output_path)))


def build_contract_from_declared_data(declared: dict[str, Any], *, source_hash: str, output_hash: str) -> dict:
    roles = _require_mapping(declared, "roles")
    raw_nodes = _require_list(declared, "nodes")
    nodes = _build_nodes(raw_nodes)

    required_roles: dict[str, str] = {}
    for role in REQUIRED_ROLES:
        node_id = roles.get(role)
        if not isinstance(node_id, str) or not node_id.strip():
            raise HumanoidContractError(f"Missing required humanoid role '{role}' in declared metadata.")
        if node_id not in nodes:
            raise HumanoidContractError(f"Required humanoid role '{role}' references unknown node '{node_id}'.")
        required_roles[role] = node_id

    optional_roles = _build_optional_shoulder_roles(roles, nodes)

    chains = _build_required_chains(required_roles)
    _append_optional_proximal_arm_chains(chains, required_roles, optional_roles)
    _append_optional_toe_chains(chains, roles, nodes)
    warnings = _build_optional_finger_chains(chains, roles, nodes)
    basis = _build_basis(declared.get("basis"))
    if basis["status"] == "inferred":
        warnings.append(_warning(WARNING_BASIS_INFERRED, "Coordinate basis is inferred rather than asserted by the source metadata."))

    confidence = _build_confidence(declared.get("confidence"), required_roles=required_roles, optional_roles=optional_roles, chains=chains)
    provenance = _build_provenance(declared.get("provenance"))

    contract = {
        "schema": HUMANOID_SCHEMA,
        "required_roles": required_roles,
        "optional_roles": optional_roles,
        "chains": chains,
        "nodes": nodes,
        "basis": basis,
        "confidence": confidence,
        "provenance": provenance,
        "warnings": sorted(warnings, key=lambda item: (item["code"], item["message"])),
        "hashes": {
            "source_sha256": _validate_hash(source_hash, "source_hash"),
            "output_sha256": _validate_hash(output_hash, "output_hash"),
            "glb_extras": "deferred",
        },
    }
    validate_humanoid_contract(contract)
    return contract


def validate_humanoid_contract(contract: dict) -> None:
    if contract.get("schema") != HUMANOID_SCHEMA:
        raise HumanoidContractError(f"Unsupported humanoid contract schema: {contract.get('schema')!r}.")

    nodes = contract.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise HumanoidContractError("Humanoid contract must include a non-empty nodes mapping.")

    roles = contract.get("required_roles")
    if not isinstance(roles, dict):
        raise HumanoidContractError("Humanoid contract must include required_roles mapping.")
    for role in REQUIRED_ROLES:
        node_id = roles.get(role)
        if not isinstance(node_id, str) or not node_id:
            raise HumanoidContractError(f"Missing required humanoid role '{role}' in contract.")
        if node_id not in nodes:
            raise HumanoidContractError(f"Required humanoid role '{role}' references unknown node '{node_id}'.")

    optional_roles = contract.get("optional_roles", {})
    if not isinstance(optional_roles, dict):
        raise HumanoidContractError("Humanoid contract optional_roles must be a mapping when present.")
    for role, node_id in optional_roles.items():
        if role not in OPTIONAL_SHOULDER_ROLE_ALIASES:
            raise HumanoidContractError(f"Unsupported optional humanoid role '{role}'.")
        if not isinstance(node_id, str) or node_id not in nodes:
            raise HumanoidContractError(f"Optional humanoid role '{role}' references unknown node '{node_id}'.")

    chains = contract.get("chains")
    if not isinstance(chains, dict):
        raise HumanoidContractError("Humanoid contract must include chains mapping.")
    for chain_name, chain_roles in REQUIRED_CHAINS.items():
        expected = [roles[role] for role in chain_roles]
        actual = chains.get(chain_name)
        if isinstance(actual, list):
            for node_id in actual:
                if node_id not in nodes:
                    raise HumanoidContractError(f"Chain '{chain_name}' references unknown node '{node_id}'.")
        if not isinstance(actual, list) or actual[: len(expected)] != expected:
            raise HumanoidContractError(
                f"Required chain '{chain_name}' must start parent-to-child as {expected}; got {actual!r}."
            )
        _validate_chain_order(chain_name, actual, nodes)
    for chain_name, chain in chains.items():
        if not isinstance(chain, list):
            raise HumanoidContractError(f"Chain '{chain_name}' must be a list of node IDs.")
        for node_id in chain:
            if node_id not in nodes:
                raise HumanoidContractError(f"Chain '{chain_name}' references unknown node '{node_id}'.")
        _validate_chain_order(chain_name, chain, nodes)

    for node_id, node in nodes.items():
        transforms = node.get("transforms") if isinstance(node, dict) else None
        if not isinstance(transforms, dict):
            raise HumanoidContractError(f"Node '{node_id}' must include transforms.")
        _validate_matrix(transforms.get("rest_local"), f"Node '{node_id}' rest_local")
        _validate_matrix(transforms.get("rest_world"), f"Node '{node_id}' rest_world")

    confidence = contract.get("confidence")
    if not isinstance(confidence, dict):
        raise HumanoidContractError("Humanoid contract must include confidence mapping.")
    for role in REQUIRED_ROLES:
        value = confidence.get("roles", {}).get(role) if isinstance(confidence.get("roles"), dict) else None
        if not isinstance(value, (int, float)) or float(value) < MINIMUM_CONFIDENCE:
            raise HumanoidContractError(
                f"Required humanoid role '{role}' confidence {value!r} is below minimum {MINIMUM_CONFIDENCE}."
            )


def _build_nodes(raw_nodes: list[Any]) -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise HumanoidContractError(f"Declared node at index {index} must be an object.")
        node_id = raw.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise HumanoidContractError(f"Declared node at index {index} is missing a non-empty id.")
        parent = raw.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise HumanoidContractError(f"Declared node '{node_id}' parent must be a string or null.")
        rest_local = raw.get("rest_local")
        rest_world = raw.get("rest_world")
        _validate_matrix(rest_local, f"Declared node '{node_id}' rest_local")
        _validate_matrix(rest_world, f"Declared node '{node_id}' rest_world")
        nodes[node_id] = {
            "name": str(raw.get("name") or node_id),
            "index": int(raw.get("index", index)),
            "parent": parent,
            "children": [],
            "transforms": {
                "rest_local": rest_local,
                "rest_world": rest_world,
                "matrix_order": "row-major",
                "units": "meters",
            },
            "inverse_bind": raw.get("inverse_bind", "unavailable"),
        }
    for node_id, node in nodes.items():
        parent = node["parent"]
        if parent is not None:
            if parent not in nodes:
                raise HumanoidContractError(f"Declared node '{node_id}' references unknown parent '{parent}'.")
            nodes[parent]["children"].append(node_id)
    for node in nodes.values():
        node["children"].sort()
    return dict(sorted(nodes.items()))


def _build_required_chains(required_roles: dict[str, str]) -> dict[str, list[str]]:
    return {name: [required_roles[role] for role in roles] for name, roles in REQUIRED_CHAINS.items()}


def _build_optional_shoulder_roles(roles: dict[str, Any], nodes: dict[str, dict]) -> dict[str, str]:
    optional_roles: dict[str, str] = {}
    for canonical_role, aliases in OPTIONAL_SHOULDER_ROLE_ALIASES.items():
        node_id = None
        for alias in aliases:
            value = roles.get(alias)
            if value is not None:
                node_id = value
                break
        if node_id is None:
            continue
        if not isinstance(node_id, str) or node_id not in nodes:
            raise HumanoidContractError(f"Optional humanoid role '{canonical_role}' references unknown node '{node_id}'.")
        optional_roles[canonical_role] = node_id
    return optional_roles


def _append_optional_proximal_arm_chains(
    chains: dict[str, list[str]],
    required_roles: dict[str, str],
    optional_roles: dict[str, str],
) -> None:
    for chain_name, role_names in OPTIONAL_PROXIMAL_ARM_CHAINS.items():
        shoulder_role = role_names[0]
        shoulder_node = optional_roles.get(shoulder_role)
        if shoulder_node is None:
            continue
        chains[chain_name] = [shoulder_node, *(required_roles[role] for role in role_names[1:])]


def _append_optional_toe_chains(chains: dict[str, list[str]], roles: dict[str, Any], nodes: dict[str, dict]) -> None:
    for chain_name, role in OPTIONAL_TOE_ROLES.items():
        node_id = roles.get(role)
        if node_id is None:
            continue
        if not isinstance(node_id, str) or node_id not in nodes:
            raise HumanoidContractError(f"Optional humanoid role '{role}' references unknown node '{node_id}'.")
        chains[chain_name].append(node_id)


def _build_optional_finger_chains(chains: dict[str, list[str]], roles: dict[str, Any], nodes: dict[str, dict]) -> list[dict]:
    warnings: list[dict] = []
    any_declared = False
    for chain_name, chain_roles in OPTIONAL_FINGER_CHAINS.items():
        declared_nodes: list[str] = []
        missing_roles: list[str] = []
        for role in chain_roles:
            node_id = roles.get(role)
            if isinstance(node_id, str) and node_id in nodes:
                declared_nodes.append(node_id)
            else:
                missing_roles.append(role)
        chains[chain_name] = declared_nodes
        if declared_nodes:
            any_declared = True
            if missing_roles:
                warnings.append(
                    _warning(
                        WARNING_OPTIONAL_FINGER_CHAIN_PARTIAL,
                        f"Optional finger chain '{chain_name}' is partially declared; missing roles: {', '.join(missing_roles)}.",
                    )
                )
    if not any_declared:
        warnings.append(
            _warning(
                WARNING_OPTIONAL_FINGERS_MISSING,
                "Optional finger chains were not declared; full-body contract remains valid.",
            )
        )
    return warnings


def _build_basis(raw_basis: Any) -> dict:
    source = raw_basis if isinstance(raw_basis, dict) else {}
    status = str(source.get("status") or "unknown").strip().lower()
    if status not in {"asserted", "inferred", "unknown"}:
        status = "unknown"
    return {
        "up": str(source.get("up") or "unknown"),
        "forward": str(source.get("forward") or "unknown"),
        "handedness": str(source.get("handedness") or "unknown"),
        "status": status,
    }


def _build_confidence(
    raw_confidence: Any,
    *,
    required_roles: dict[str, str],
    optional_roles: dict[str, str],
    chains: dict[str, list[str]],
) -> dict:
    source = raw_confidence if isinstance(raw_confidence, dict) else {}
    role_source = source.get("roles") if isinstance(source.get("roles"), dict) else {}
    chain_source = source.get("chains") if isinstance(source.get("chains"), dict) else {}
    all_roles = set(required_roles) | set(optional_roles)
    roles = {role: float(role_source.get(role, 1.0)) for role in sorted(all_roles)}
    chain_confidence = {name: float(chain_source.get(name, 1.0)) for name in chains}
    required_values = [roles[role] for role in required_roles]
    return {"minimum_required": MINIMUM_CONFIDENCE, "roles": roles, "chains": chain_confidence, "overall": min(required_values)}


def _build_provenance(raw_provenance: Any) -> dict:
    source = raw_provenance if isinstance(raw_provenance, dict) else {}
    return {
        "source": str(source.get("source") or "declared-metadata"),
        "method": str(source.get("method") or "explicit"),
    }


def _validate_chain_order(chain_name: str, chain: list[str], nodes: dict[str, dict]) -> None:
    for ancestor_id, descendant_id in zip(chain, chain[1:]):
        if not _is_descendant(descendant_id, ancestor_id, nodes):
            raise HumanoidContractError(
                f"Chain '{chain_name}' is not ordered ancestor-to-descendant: expected '{ancestor_id}' to contain node '{descendant_id}'."
            )


def _is_descendant(descendant_id: str, ancestor_id: str, nodes: dict[str, dict]) -> bool:
    current = nodes.get(descendant_id, {}).get("parent")
    seen: set[str] = set()
    while isinstance(current, str) and current not in seen:
        if current == ancestor_id:
            return True
        seen.add(current)
        current = nodes.get(current, {}).get("parent")
    return False


def _validate_matrix(matrix: Any, label: str) -> None:
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise HumanoidContractError(f"{label} must be a 4x4 matrix.")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 4:
            raise HumanoidContractError(f"{label} must be a 4x4 matrix.")
        for value in row:
            if not isinstance(value, (int, float)):
                raise HumanoidContractError(f"{label} must contain only numeric values.")


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise HumanoidContractError(f"Declared humanoid metadata must include object '{key}'.")
    return value


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise HumanoidContractError(f"Declared humanoid metadata must include list '{key}'.")
    return value


def _warning(code: str, message: str) -> dict:
    return {"code": code, "message": message, "severity": "warning"}


def _validate_hash(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise HumanoidContractError(f"{label} must be a SHA-256 hex digest.")
    return value.lower()


def _read_declared_humanoid_source(output_path: Path) -> dict[str, Any] | None:
    companion = output_path.with_name(f"{output_path.stem}.humanoid.json")
    if companion.exists():
        payload = json.loads(companion.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise HumanoidContractError(f"Declared humanoid metadata file must be a JSON object: {companion}")
        return payload
    glb_payload = _read_glb_json(output_path)
    extras = glb_payload.get("extras") if isinstance(glb_payload, dict) else None
    declared = extras.get("unirig_humanoid") if isinstance(extras, dict) else None
    return declared if isinstance(declared, dict) else None


def _read_glb_json(output_path: Path) -> dict[str, Any]:
    data = output_path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        return {}
    json_length = struct.unpack_from("<I", data, 12)[0]
    chunk_type = data[16:20]
    if chunk_type != b"JSON":
        return {}
    raw_json = data[20 : 20 + json_length].rstrip(b" \t\r\n\x00")
    payload = json.loads(raw_json.decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
