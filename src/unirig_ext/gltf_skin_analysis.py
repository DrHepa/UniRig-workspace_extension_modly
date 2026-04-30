from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


COMPONENT_BYTE_SIZES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COMPONENT_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
STRUCT_FORMATS = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}


class GltfSkinAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class GlbContainer:
    path: Path
    json: dict[str, Any]
    bin_chunk: bytes


@dataclass(frozen=True)
class WeightedVertex:
    position: tuple[float, float, float]
    influences: tuple[tuple[str, float], ...]


@dataclass
class JointWeightSummary:
    count: int = 0
    total_weight: float = 0.0
    min_x: float = float("inf")
    min_y: float = float("inf")
    min_z: float = float("inf")
    max_x: float = float("-inf")
    max_y: float = float("-inf")
    max_z: float = float("-inf")

    def add(self, position: tuple[float, float, float], weight: float) -> None:
        self.count += 1
        self.total_weight += weight
        x, y, z = position
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.min_z = min(self.min_z, z)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)
        self.max_z = max(self.max_z, z)

    def as_diagnostic(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "total_weight": 0.0}
        bbox = [[self.min_x, self.min_y, self.min_z], [self.max_x, self.max_y, self.max_z]]
        center = [(bbox[0][axis] + bbox[1][axis]) / 2.0 for axis in range(3)]
        spread = [bbox[1][axis] - bbox[0][axis] for axis in range(3)]
        return {
            "count": self.count,
            "total_weight": round(self.total_weight, 6),
            "bbox": [[round(value, 6) for value in point] for point in bbox],
            "center": [round(value, 6) for value in center],
            "spread": [round(value, 6) for value in spread],
        }


def read_glb_container(path: Path) -> GlbContainer:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise GltfSkinAnalysisError(f"Unsupported humanoid quality gate input: {path} is not an embedded GLB file.")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2 or total_length > len(data):
        raise GltfSkinAnalysisError(f"Unsupported GLB header in {path}; expected glTF 2.0 with a valid length.")
    offset = 12
    gltf_json: dict[str, Any] | None = None
    bin_chunk = b""
    while offset + 8 <= total_length:
        chunk_length = struct.unpack_from("<I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end > len(data):
            raise GltfSkinAnalysisError(f"Malformed GLB chunk length in {path}.")
        if chunk_type == b"JSON":
            raw_json = data[chunk_start:chunk_end].rstrip(b" \t\r\n\x00")
            parsed = json.loads(raw_json.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise GltfSkinAnalysisError(f"GLB JSON chunk must decode to an object in {path}.")
            gltf_json = parsed
        elif chunk_type == b"BIN\x00":
            bin_chunk = data[chunk_start:chunk_end]
        offset = chunk_end
    if gltf_json is None:
        raise GltfSkinAnalysisError(f"GLB file {path} does not contain a JSON chunk.")
    return GlbContainer(path=path, json=gltf_json, bin_chunk=bin_chunk)


def has_skinned_mesh_primitives(gltf: dict[str, Any]) -> bool:
    for primitive in _iter_primitives(gltf):
        attributes = primitive.get("attributes") if isinstance(primitive.get("attributes"), dict) else {}
        if "POSITION" in attributes or "JOINTS_0" in attributes or "WEIGHTS_0" in attributes:
            return True
    return False


def iter_weighted_vertices(container: GlbContainer) -> Iterable[WeightedVertex]:
    gltf = container.json
    skins = gltf.get("skins") if isinstance(gltf.get("skins"), list) else []
    if len(skins) != 1 or not isinstance(skins[0], dict) or not isinstance(skins[0].get("joints"), list):
        raise GltfSkinAnalysisError("skin_weight_data_unavailable: humanoid quality gate requires exactly one skin with joints.")
    nodes = gltf.get("nodes") if isinstance(gltf.get("nodes"), list) else []
    joint_names = [_node_name(nodes, node_index) for node_index in skins[0]["joints"]]
    emitted = False
    for primitive in _iter_primitives(gltf):
        attributes = primitive.get("attributes") if isinstance(primitive.get("attributes"), dict) else {}
        if not attributes:
            continue
        if "POSITION" not in attributes or "JOINTS_0" not in attributes or "WEIGHTS_0" not in attributes:
            raise GltfSkinAnalysisError("skin_weight_data_unavailable: skinned mesh primitive must provide POSITION, JOINTS_0, and WEIGHTS_0 accessors.")
        positions = read_accessor(container, int(attributes["POSITION"]))
        joints = read_accessor(container, int(attributes["JOINTS_0"]))
        weights = read_accessor(container, int(attributes["WEIGHTS_0"]))
        if len(positions) != len(joints) or len(positions) != len(weights):
            raise GltfSkinAnalysisError("skin_weight_data_unavailable: POSITION, JOINTS_0, and WEIGHTS_0 accessor counts differ.")
        for position, joint_row, weight_row in zip(positions, joints, weights):
            influences: list[tuple[str, float]] = []
            for joint_slot, weight in zip(joint_row, weight_row):
                slot = int(joint_slot)
                if weight > 0.0:
                    if slot < 0 or slot >= len(joint_names):
                        raise GltfSkinAnalysisError("skin_weight_data_unavailable: JOINTS_0 references a joint slot outside the skin.")
                    influences.append((joint_names[slot], float(weight)))
            emitted = True
            yield WeightedVertex(position=(float(position[0]), float(position[1]), float(position[2])), influences=tuple(influences))
    if not emitted:
        raise GltfSkinAnalysisError("skin_weight_data_unavailable: no mesh primitive with skin weight attributes was found.")


def summarize_joint_weights(container: GlbContainer, *, weight_epsilon: float = 0.01) -> tuple[dict[str, JointWeightSummary], dict[str, Any]]:
    summaries: dict[str, JointWeightSummary] = {}
    vertex_count = 0
    influenced_vertices = 0
    min_y = float("inf")
    max_y = float("-inf")
    for vertex in iter_weighted_vertices(container):
        vertex_count += 1
        min_y = min(min_y, vertex.position[1])
        max_y = max(max_y, vertex.position[1])
        if vertex.influences:
            influenced_vertices += 1
        for joint, weight in vertex.influences:
            if weight < weight_epsilon:
                continue
            summaries.setdefault(joint, JointWeightSummary()).add(vertex.position, weight)
    if vertex_count == 0:
        raise GltfSkinAnalysisError("skin_weight_data_unavailable: no weighted vertices were available.")
    return summaries, {
        "vertex_count": vertex_count,
        "influenced_vertices": influenced_vertices,
        "height_min": round(min_y, 6),
        "height_max": round(max_y, 6),
        "height": round(max_y - min_y, 6),
    }


def read_accessor(container: GlbContainer, accessor_index: int) -> list[tuple[float | int, ...]]:
    gltf = container.json
    accessors = gltf.get("accessors") if isinstance(gltf.get("accessors"), list) else []
    buffer_views = gltf.get("bufferViews") if isinstance(gltf.get("bufferViews"), list) else []
    if accessor_index < 0 or accessor_index >= len(accessors) or not isinstance(accessors[accessor_index], dict):
        raise GltfSkinAnalysisError(f"Accessor index {accessor_index} is invalid.")
    accessor = accessors[accessor_index]
    view_index = accessor.get("bufferView")
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(buffer_views) or not isinstance(buffer_views[view_index], dict):
        raise GltfSkinAnalysisError(f"Accessor {accessor_index} does not reference a valid embedded bufferView.")
    view = buffer_views[view_index]
    if view.get("buffer", 0) != 0:
        raise GltfSkinAnalysisError("External or non-zero buffers are unsupported by the phase 1 humanoid quality gate.")
    component_type = accessor.get("componentType")
    type_name = accessor.get("type")
    count = accessor.get("count")
    if component_type not in COMPONENT_BYTE_SIZES or type_name not in TYPE_COMPONENT_COUNTS or not isinstance(count, int):
        raise GltfSkinAnalysisError(f"Accessor {accessor_index} has unsupported component metadata.")
    components = TYPE_COMPONENT_COUNTS[str(type_name)]
    item_size = COMPONENT_BYTE_SIZES[int(component_type)] * components
    stride = int(view.get("byteStride") or item_size)
    if stride < item_size:
        raise GltfSkinAnalysisError(f"Accessor {accessor_index} byteStride is smaller than element size.")
    base_offset = int(view.get("byteOffset") or 0) + int(accessor.get("byteOffset") or 0)
    fmt = "<" + STRUCT_FORMATS[int(component_type)] * components
    rows: list[tuple[float | int, ...]] = []
    for row_index in range(count):
        start = base_offset + row_index * stride
        end = start + item_size
        if end > len(container.bin_chunk):
            raise GltfSkinAnalysisError(f"Accessor {accessor_index} reads beyond the embedded BIN chunk.")
        rows.append(struct.unpack_from(fmt, container.bin_chunk, start))
    return rows


def _iter_primitives(gltf: dict[str, Any]) -> Iterable[dict[str, Any]]:
    meshes = gltf.get("meshes") if isinstance(gltf.get("meshes"), list) else []
    for mesh in meshes:
        primitives = mesh.get("primitives") if isinstance(mesh, dict) and isinstance(mesh.get("primitives"), list) else []
        for primitive in primitives:
            if isinstance(primitive, dict):
                yield primitive


def _node_name(nodes: list[Any], node_index: Any) -> str:
    if not isinstance(node_index, int) or node_index < 0 or node_index >= len(nodes) or not isinstance(nodes[node_index], dict):
        raise GltfSkinAnalysisError("Skin joints must reference valid node objects.")
    name = str(nodes[node_index].get("name") or "").strip()
    return name or f"node_{node_index}"
