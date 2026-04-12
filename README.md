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
| Linux x86_64 | Pinned prebuilt-first posture is documented. | **Unvalidated** end-to-end in this repo. Do not treat as fully supported. |
| Linux ARM64 | Extra preflight/source-build diagnostics exist. | **Experimental** and **unvalidated** for reproducible end-to-end success. |
| Anything else | No repo evidence. | Unsupported until validated evidence exists. |

Validated Windows evidence for this repo means: install from GitHub, run the full Modly workspace flow through `RIG`, export the result, and open the merged output successfully in Blender. If you validate another host/runtime combination with the same level of evidence, update the docs accordingly. Until then, KEEP THE CLAIMS CONSERVATIVE.

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
