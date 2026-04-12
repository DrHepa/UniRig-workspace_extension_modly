# UniRig Public Process Architecture

## Intent

This repository is a **thin wrapper** that preserves Modly's public process boundary while delegating rigging behavior to upstream UniRig.

## Thin-wrapper responsibilities

1. `manifest.json` and `processor.py` keep the public Modly contract stable.
2. `setup.py` performs **deterministic upstream staging**, venv creation, preflight reporting, and readiness persistence.
3. `bootstrap.ensure_ready()` verifies the staged runtime and returns actionable failures.
4. `pipeline.run()` adapts Modly inputs to one deterministic upstream execution path.
5. `metadata.write_sidecar()` writes the stable `.rigmeta.json` handoff beside the published mesh.

This is an **upstream-first** design: the wrapper does not own a second rigging policy engine.

## Runtime flow

1. Modly validates root `manifest.json` and launches `processor.py`.
2. `processor.py` validates the request and calls `bootstrap.ensure_ready()`.
3. `setup.py`/bootstrap guarantee the pinned upstream runtime is staged under `.unirig-runtime/vendor/unirig/`.
4. `pipeline.run()` prepares deterministic input/output paths.
5. Upstream UniRig commands perform extract → skeleton → skin → merge.
6. The wrapper publishes `*_unirig.glb`, writes the sidecar, and emits Modly `done`.

## Repository layout

- `manifest.json` — public Modly process manifest
- `processor.py` — JSON-line protocol adapter
- `setup.py` — deterministic staging, install, preflight, and readiness writer
- `src/unirig_ext/bootstrap.py` — minimal runtime-state normalization and readiness verification
- `src/unirig_ext/pipeline.py` — deterministic upstream command adapter
- `src/unirig_ext/io.py` — input validation, staging, and output publication
- `src/unirig_ext/metadata.py` — sidecar writer with stable runtime facts only
- `tests/` — contract, bootstrap, pipeline, and docs posture checks

## Support posture

Support claims must be backed by **validated evidence**.

| Host | Evidence in this repo | Public claim |
| --- | --- | --- |
| Windows x86_64 | Pinned prebuilt checks, unit coverage, and a real install → rig → export flow verified in Blender | Validated for the current pinned prebuilt workflow |
| Linux x86_64 | Prebuilt-first docs and unit coverage | Unvalidated end-to-end; not fully supported |
| Linux ARM64 | Additional preflight/source-build diagnostics | Experimental and unvalidated |
| Other hosts | No evidence | Unsupported until validated evidence exists |

The rule is simple: if we do not have validated evidence, we say **unvalidated** or **unsupported**.

For Windows x86_64, validated evidence in this repo currently means the pinned prebuilt workflow completed end-to-end in Modly and the merged output was opened successfully in Blender.

## Deliberate non-goals

- reintroducing private `workspace_tool` behavior
- keeping wrapper-owned hook/scaffold execution branches
- documenting blanket platform support without evidence
- hiding runtime failures behind fallback policy

## Why this design

- **Public contract first**: callers keep the same Modly process surface.
- **Smaller wrapper**: less local policy means less drift from upstream.
- **Deterministic staging**: setup/bootstrap own paths and readiness, not rigging behavior.
- **Conservative support claims**: docs and runtime messaging stay honest about what is actually validated.
