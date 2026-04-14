# UniRig Process Extension

This repository is a **thin wrapper** around upstream UniRig for Modly's public `process` contract.

## Ownership boundary

The wrapper owns only three things:

- the stable Modly entrypoint (`manifest.json` → `processor.py`)
- deterministic upstream staging and readiness verification in `setup.py`
- protocol adaptation plus sidecar handoff in `src/unirig_ext/`

Everything else is **upstream-first**: the wrapper stages a pinned UniRig checkout, verifies it, then delegates extract/inference work to upstream commands.

## What stays stable

- node id: `rig-mesh`
- stdin payload: `input.filePath`, `input.nodeId`, optional `params.seed`
- stdout event types: `progress`, `log`, `done`, `error`
- success result shape: `{"filePath": "..._unirig.glb"}`

## Install and repair

Modly may run:

```bash
python3 setup.py '{"python_exe":"/path/to/python3","ext_dir":"/path/to/extension","gpu_sm":86,"cuda_version":128}'
```

`setup.py` stages a pinned upstream runtime into deterministic paths under `.unirig-runtime/`, creates `venv/`, writes readiness/preflight artifacts, and reuses an already-matching staged checkout when possible.

## Support posture

This repository makes support claims only when there is **validated evidence**.

| Host | Repo posture | Claim |
| --- | --- | --- |
| Windows x86_64 | Pinned prebuilt path validated with real install, rig, export, and Blender verification. | **Validated** for the current pinned prebuilt workflow in this repo. |
| Linux x86_64 | Pinned prebuilt-first posture is documented. | **Unvalidated end-to-end** in this repo. Do not treat as fully supported. |
| Linux ARM64 | Staged non-Blender bringup documents what can be checked or advanced without claiming a full runtime port. | **Experimental** and **unvalidated**. The current path is `staged-source-build` for non-Blender bringup only, requires real system CUDA and real nvcc, keeps `torch_scatter`/`torch_cluster` on a source-build path, and treats `spconv` as import-ready only if verified by guarded `cumm -> spconv -> spconv.pytorch` import smoke. Any Blender smoke is **external Blender evidence only**, not wrapper readiness. Deferred bpy still blocks full runtime support. |
| Anything else | No repo evidence. | Unsupported until validated evidence exists. |

Validated Windows evidence for this repo means: install from GitHub, run the full Modly workspace flow through `RIG`, export the result, and open the merged output successfully in Blender. If you validate another host/runtime combination with the same level of evidence, update the docs accordingly. Until then, KEEP THE CLAIMS CONSERVATIVE.

Background Blender smoke alone is NOT validation evidence and must not be presented as support for any host.

Windows x86_64 remains validated for the current pinned prebuilt workflow in this repo.

Linux ARM64 currently describes staged non-Blender bringup, not platform support. It expects Python 3.11, an NVIDIA GPU, real system CUDA, and real nvcc before the wrapper can even evaluate the source-build path. `torch_scatter` and `torch_cluster` remain source-build expectations. `spconv` becomes import-ready only if verified by guarded `cumm -> spconv -> spconv.pytorch` import smoke, and even that remains experimental and unvalidated evidence rather than platform support.

For Linux ARM64 `bpy`, the repo records external Blender evidence only. `setup.py` can discover one Blender candidate deterministically, run a background smoke, and report `missing`, `discovered-incompatible`, `external-bpy-smoke-ready`, or `error` without claiming the wrapper runtime is ready. The known Blender Python 3.12 versus wrapper Python 3.11 mismatch is reported explicitly as a mismatch and stays `discovered-incompatible`.

That means Linux ARM64 full runtime remains blocked even when background Blender smoke succeeds. External Blender evidence only is useful for diagnosis, but it does NOT mean wrapper readiness, runtime readiness, runtime validation, validated support, or a completed runtime port.

For extract/merge, `context.venv_python` remains the default/fallback wrapper-owned path. This tranche records qualification evidence only.

Qualification verdict language is conservative by design: `not-ready`, `candidate-with-known-risks`, and `ready-for-separate-defaulting-change` are tranche verdicts only. Even the strongest verdict is qualification evidence only for a separate defaulting decision.

This change also documents an **optional Blender subprocess seam** that is **Linux ARM64-first**, **experimental**, and **non-default**. That seam is a narrow executable-boundary experiment for `extract-prepare`, `extract-skin`, and `merge`; it is distinct from wrapper runtime readiness and it is NOT a platform support claim.

The artifacts are different on purpose:

- `source_build.external_blender` records external Blender evidence such as discovery, incompatibility, or background smoke.
- `source_build.executable_boundary.extract_merge` records separate executable-boundary proof for the optional seam.

Even if executable-boundary proof exists, Linux ARM64 full runtime remains blocked. External Blender evidence and executable-boundary proof are useful diagnostics, but neither one means wrapper runtime readiness, runtime validation, validated support, or a completed runtime port. Seam/default-candidate evidence is only qualification evidence for a separate defaulting decision, not support. Broader runtime redesign stays deferred. This does not claim full runtime support.

## Runtime notes

- runtime state is stored in `.unirig-runtime/bootstrap_state.json`
- preflight artifacts are stored in `.unirig-runtime/logs/`
- the staged upstream checkout lives in `.unirig-runtime/vendor/unirig/`
- heavyweight checkpoints are not committed to this repository

## Scope

Implemented here:

- public Modly process contract
- pinned upstream runtime staging and reuse
- readiness verification with actionable failures
- deterministic output naming and `.rigmeta.json` sidecar
- automated unit coverage for processor protocol, bootstrap/setup, and docs posture

Not claimed here:

- private `workspace_tool` behavior
- blanket platform support
- generic compile fallback promises for hosts without validation evidence
- private-runtime parity on every machine profile

## Running checks

```bash
python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py
python3 -m unittest discover -s tests -v
```

## Architecture

See `docs/architecture.md` for the thin-wrapper/upstream-first design.
