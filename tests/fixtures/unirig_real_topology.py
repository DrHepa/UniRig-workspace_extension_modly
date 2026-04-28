from __future__ import annotations


REAL_UNIRIG_52_EDGES = (
    ("bone_0", "bone_1"),
    ("bone_1", "bone_2"),
    ("bone_2", "bone_3"),
    ("bone_3", "bone_4"),
    ("bone_4", "bone_5"),
    ("bone_3", "bone_6"),
    ("bone_6", "bone_7"),
    ("bone_7", "bone_8"),
    ("bone_8", "bone_9"),
    ("bone_9", "bone_10"),
    ("bone_10", "bone_11"),
    ("bone_11", "bone_12"),
    ("bone_9", "bone_13"),
    ("bone_13", "bone_14"),
    ("bone_14", "bone_15"),
    ("bone_9", "bone_16"),
    ("bone_16", "bone_17"),
    ("bone_17", "bone_18"),
    ("bone_9", "bone_19"),
    ("bone_19", "bone_20"),
    ("bone_20", "bone_21"),
    ("bone_9", "bone_22"),
    ("bone_22", "bone_23"),
    ("bone_23", "bone_24"),
    ("bone_3", "bone_25"),
    ("bone_25", "bone_26"),
    ("bone_26", "bone_27"),
    ("bone_27", "bone_28"),
    ("bone_28", "bone_29"),
    ("bone_29", "bone_30"),
    ("bone_30", "bone_31"),
    ("bone_28", "bone_32"),
    ("bone_32", "bone_33"),
    ("bone_33", "bone_34"),
    ("bone_28", "bone_35"),
    ("bone_35", "bone_36"),
    ("bone_36", "bone_37"),
    ("bone_28", "bone_38"),
    ("bone_38", "bone_39"),
    ("bone_39", "bone_40"),
    ("bone_28", "bone_41"),
    ("bone_41", "bone_42"),
    ("bone_42", "bone_43"),
    ("bone_0", "bone_44"),
    ("bone_44", "bone_45"),
    ("bone_45", "bone_46"),
    ("bone_46", "bone_47"),
    ("bone_0", "bone_48"),
    ("bone_48", "bone_49"),
    ("bone_49", "bone_50"),
    ("bone_50", "bone_51"),
)


def real_unirig_52_payload() -> dict:
    """Return a compact glTF JSON fixture for the real anonymous 52-bone UniRig profile."""
    nodes = [
        {"name": "scene_root", "children": [2]},
        {"name": "mesh"},
    ]
    nodes.extend({"name": f"bone_{number}"} for number in range(52))
    nodes.append({"name": "camera_helper"})

    index_by_name = {node["name"]: index for index, node in enumerate(nodes)}
    for parent, child in REAL_UNIRIG_52_EDGES:
        parent_node = nodes[index_by_name[parent]]
        parent_node.setdefault("children", []).append(index_by_name[child])

    nodes[index_by_name["bone_0"]]["translation"] = [0.0, 1.0, 0.0]
    nodes[index_by_name["bone_1"]]["translation"] = [0.0, 2.0, 0.0]
    nodes[index_by_name["bone_3"]]["translation"] = [0.0, 3.0, 0.0]
    nodes[index_by_name["bone_6"]]["translation"] = [-1.0, 0.0, 0.0]
    nodes[index_by_name["bone_25"]]["translation"] = [1.0, 0.0, 0.0]

    joints = [index_by_name[f"bone_{number}"] for number in range(52)]
    return {
        "asset": {"version": "2.0"},
        "nodes": nodes,
        "skins": [{"joints": joints, "inverseBindMatrices": 0}],
        "accessors": [{"count": 52}],
    }
