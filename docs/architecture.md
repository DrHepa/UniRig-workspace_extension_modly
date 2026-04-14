# UniRig Public Process Architecture

## Intent

This repository is a **thin wrapper** that preserves Modly's public process boundary while delegating rigging behavior to upstream UniRig.

## Thin-wrapper responsibilities

1. `manifest.json` and `processor.py` keep the public Modly contract stable.
2. `setup.py` acts as the wrapper's **planner/executor** for deterministic upstream staging, host-aware install planning, venv creation, preflight reporting, and readiness persistence.
3. `bootstrap.ensure_ready()` verifies the staged runtime and returns actionable failures.
4. `pipeline.run()` adapts Modly inputs to one deterministic upstream execution path.
5. `metadata.write_sidecar()` writes the stable `.rigmeta.json` handoff beside the published mesh.

This is an **upstream-first** design: the wrapper does not own a second rigging policy engine.

## Planner and runtime-state split

The host-aware policy now has a narrow split so runtime behavior stays explicit and inspectable:

- `setup.py` is the **planner/executor**. It classifies the current host, builds the install plan, runs preflight, executes only the allowed install path, and writes the resulting planner/preflight/install state.
- `src/unirig_ext/bootstrap.py` is the **normalizer/runtime-context source**. It loads persisted state, preserves the planner/preflight/install facts, normalizes them into the runtime context, and turns blocked readiness into actionable errors.

That split keeps the real runtime path primary: planning and staging happen once in `setup.py`, while bootstrap reads the recorded facts instead of inventing a second policy layer.

## Linux ARM64 staged bringup model

Linux ARM64 is still **experimental** and **unvalidated** in this repo. The current model is staged non-Blender bringup, not platform support and not a claim of full runtime support.

The stage graph is:

`baseline → pyg → spconv → bpy-deferred`

- `baseline` checks wrapper-owned prerequisites before any dependency claim is made: Linux ARM64, Python 3.11, NVIDIA GPU visibility, real system CUDA, real nvcc, compiler, Python headers, and recorded CUDA facts.
- `pyg` is the first non-Blender dependency stage. `torch_scatter` and `torch_cluster` stay on a source-build path and only advance when install plus import/smoke verification succeeds.
- `spconv` is a guarded `cumm -> spconv -> spconv.pytorch` stage. It becomes import-ready only if the install flow completes and `spconv.pytorch` import smoke passes; otherwise it stays blocked or deferred with explicit boundary ownership.
- `bpy-deferred` is always recorded as deferred for Linux ARM64 in the current change. In plain terms: deferred bpy is still a blocker.

### External Blender evidence boundary

- `setup.py` now owns deterministic Blender discovery for Linux ARM64, the background Blender probe, and the conservative classification step. That is deterministic Blender discovery for evidence collection, not a runtime handoff.
- The only allowed Linux ARM64 `bpy` results in this change are `missing`, `discovered-incompatible`, `external-bpy-smoke-ready`, or `error`.
- Those results are **external Blender evidence only**. A successful background smoke does not mean the wrapper can import `bpy`, does not mean support is validated, and does not mean full runtime is ready.
- The known Blender Python 3.12 versus wrapper Python 3.11 mismatch is reported explicitly as a mismatch and remains `discovered-incompatible`.
- `src/unirig_ext/bootstrap.py` normalizes the persisted Blender evidence into runtime-facing state, but full runtime remains blocked for Linux ARM64 in every one of those evidence classes and the evidence does not imply wrapper runtime readiness.

### Extract/merge executable-boundary seam

- `context.venv_python` remains the default/fallback wrapper-owned path for extract/merge.
- A separate **optional Blender subprocess seam** may be evaluated only as a **Linux ARM64-first**, **experimental**, **non-default** executable-boundary check for `extract-prepare`, `extract-skin`, and `merge`.
- Qualification verdicts for this seam are limited to `not-ready`, `candidate-with-known-risks`, and `ready-for-separate-defaulting-change`.
- The persisted artifact for that seam is `source_build.executable_boundary.extract_merge`, which records executable-boundary proof instead of external Blender discovery evidence.
- This tranche records qualification evidence only. The executable-boundary proof can show that the wrapper reached Blender across a subprocess boundary without proving the broader wrapper runtime is ready, and it remains distinct from wrapper runtime readiness.
- Full runtime readiness (still blocked) remains a different outcome. Even with successful `source_build.executable_boundary.extract_merge` proof, Linux ARM64 stays blocked for full runtime use in this change.
- Seam/default-candidate evidence is only input to a separate defaulting decision; it is not support, not validation, and not a runtime redesign.
- Windows x86_64 remains a non-regression invariant: the validated pinned prebuilt workflow is unchanged and does not depend on this seam.

### Boundary ownership

- **Wrapper-owned boundary**: host classification, staged plan selection, prerequisite detection, persisted readiness facts, and actionable failure reporting.
- **Environment boundary**: machine-specific conditions the wrapper cannot manufacture, such as NVIDIA GPU presence, real system CUDA, real nvcc, compiler toolchain, Python headers, and other host readiness facts.
- **Upstream/package boundary**: package viability and source-build behavior for upstream-sensitive dependencies such as `torch_scatter`, `torch_cluster`, `spconv`, and Blender-facing `bpy` portability.

The wrapper may explain which boundary failed, but it must not hide an environment boundary or upstream/package boundary behind a fake success state.

### Why `bpy` stays deferred

`bpy` remains out of scope for this staged bringup because the repo does not yet have validated Linux ARM64 wrapper-runtime evidence. Non-Blender progress is useful, and external Blender evidence only is still worth recording, but it is NOT the same thing as runtime readiness or runtime validation for the full UniRig workflow. Until Linux ARM64 has validated wrapper-runtime evidence, deferred `bpy` remains a blocker and the repo does not claim full runtime support.

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
| Linux x86_64 | Prebuilt-first docs and unit coverage | Conservative existing posture: unvalidated end-to-end; not fully supported |
| Linux ARM64 | Planner-backed staged non-Blender bringup, blocker reporting, persisted readiness metadata, guarded `spconv` import-smoke evidence when verified, and external Blender evidence only when background smoke is available | Experimental, staged non-Blender bringup only, import-ready if verified, and unvalidated |
| Other hosts | No evidence | Unsupported until validated evidence exists |

The rule is simple: if we do not have validated evidence, we say **unvalidated** or **unsupported**.

For Windows x86_64, validated evidence in this repo currently means the pinned prebuilt workflow completed end-to-end in Modly and the merged output was opened successfully in Blender.

Background Blender smoke alone is NOT validation evidence and must not be presented as support for any host.

Windows x86_64 non-regression invariant: Linux ARM64 work MUST NOT weaken or replace the validated Windows x86_64 pinned prebuilt workflow. Windows stays on its existing branch and remains the reference path with validated evidence in this repo.

Linux ARM64 now goes farther than prep-only diagnostics, but it still stops short of a support claim BY DESIGN. The current evidence supports staged non-Blender bringup only: baseline gating, PyG source-build verification, guarded `spconv` import-ready reporting only when `cumm -> spconv -> spconv.pytorch` import smoke succeeds, and external Blender evidence only when the background Blender probe reports `external-bpy-smoke-ready`. If Blender reports Python 3.12 while the wrapper still targets Python 3.11, the repo reports that mismatch as `discovered-incompatible` instead of hiding it behind a success claim. It does not eliminate environment boundary risk, it does not solve upstream/package boundary gaps, and it does not resolve deferred `bpy` portability.

That ARM64 staged non-Blender bringup is still bounded work only. Any future progress must remain conservative until real wrapper-runtime evidence exists, and it still does not claim runtime validation or completed dependency porting in this wrapper. Even with external Blender evidence only or successful `source_build.executable_boundary.extract_merge` proof, full runtime remains blocked. External Blender evidence, executable-boundary proof, qualification verdicts, and full runtime readiness are different artifacts and MUST stay separate in both docs and persisted state. Any positive verdict is only readiness evidence for a separate defaulting decision. Broader runtime redesign stays deferred.

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
