# UniRig Public Process Architecture

## Intent

This repository is a **thin wrapper** that preserves Modly's public process boundary while delegating rigging behavior to upstream UniRig.

## Thin-wrapper responsibilities

1. `manifest.json` and `processor.py` keep the public Modly contract stable.
2. `setup.py` acts as the wrapper's **planner/executor** for deterministic upstream staging, host-aware install planning, venv creation, preflight reporting, and readiness persistence.
3. `bootstrap.ensure_ready()` verifies the staged runtime and returns actionable failures.
4. `pipeline.run()` adapts Modly inputs to one deterministic upstream execution path.
5. `metadata.write_sidecar()` writes the stable `.rigmeta.json` handoff beside the published mesh and appends the sidecar-first `humanoid_contract` when declared humanoid metadata is available.

This is an **upstream-first** design: the wrapper does not own a second rigging policy engine.

At the process boundary, Modly may send top-level `workspaceDir` / `tempDir` fields to public `process` extensions. This wrapper currently uses `workspaceDir` as the canonical publication target when it is present, non-empty, and exists on disk; `tempDir` remains host/runtime context, not the published workflow destination.

## Planner and runtime-state split

The host-aware policy now has a narrow split so runtime behavior stays explicit and inspectable:

- `setup.py` is the **planner/executor**. It classifies the current host, builds the install plan, runs preflight, executes only the allowed install path, and writes the resulting planner/preflight/install state.
- `src/unirig_ext/bootstrap.py` is the **normalizer/runtime-context source**. It loads persisted state, preserves the planner/preflight/install facts, normalizes them into the runtime context, and turns blocked readiness into actionable errors.

That split keeps the real runtime path primary: planning and staging happen once in `setup.py`, while bootstrap reads the recorded facts instead of inventing a second policy layer.

## Linux ARM64 staged bringup model

Linux ARM64 is still **experimental** in this repo, but the current code now has one tested-host clean-install + real-workflow validation on Ubuntu 24.04 / aarch64. That is evidence about the tested host/workflow, not a blanket platform support claim.

The stage graph is:

`baseline → pyg → spconv → bpy-deferred`

- `baseline` checks wrapper-owned prerequisites before any dependency claim is made: Linux ARM64, the selected bootstrap Python, NVIDIA GPU visibility, real system CUDA, real nvcc, compiler, Python headers, and recorded CUDA facts.
- `pyg` is the first non-Blender dependency stage. `torch_scatter` and `torch_cluster` stay on a source-build path and only advance when install plus import/smoke verification succeeds.
- `spconv` is a guarded `cumm -> spconv -> spconv.pytorch` stage. It becomes import-ready only if the install flow completes and `spconv.pytorch` import smoke passes; otherwise it stays blocked or deferred with explicit boundary ownership.
- `bpy-deferred` still marks the boundary between external Blender evidence and full wrapper-runtime support. In plain terms: the tested host can recover a useful partial state for a seam-backed subset, but Linux ARM64 is not yet advertised as blanket full-runtime support. Deferred `bpy` still marks the unsupported side of that boundary.

### External Blender evidence boundary

- `setup.py` now owns deterministic Blender discovery for Linux ARM64, the background Blender probe, and the conservative classification step. That is deterministic Blender discovery for evidence collection, not a runtime handoff.
- The only allowed Linux ARM64 `bpy` results in this change are `missing`, `discovered-incompatible`, `external-bpy-smoke-ready`, or `error`.
- Those results are **external Blender evidence only**. A successful background smoke is not the same thing as blanket wrapper support, and it must stay distinct from executable-boundary proof and from any recovered partial runtime state.
- `src/unirig_ext/bootstrap.py` normalizes the persisted Blender evidence into runtime-facing state and can recover a tested-host `partial` mode only when separate seam/backing proof exists. Even then, that is still not the same thing as blanket wrapper runtime readiness for Linux ARM64.

### Extract/merge executable-boundary seam

- `context.venv_python` remains the default/fallback wrapper-owned path for extract/merge.
- A separate **optional Blender subprocess seam** may be evaluated only as a **Linux ARM64-first**, **experimental**, **non-default** executable-boundary check for `extract-prepare`, `extract-skin`, and `merge`.
- Qualification verdicts for this seam are limited to `not-ready`, `candidate-with-known-risks`, and `ready-for-separate-defaulting-change`.
- The persisted artifact for that seam is `source_build.executable_boundary.extract_merge`, which records executable-boundary proof instead of external Blender discovery evidence.
- This tranche records qualification evidence only. The executable-boundary proof can show that the wrapper reached Blender across a subprocess boundary without proving blanket wrapper support, and it remains distinct from wrapper runtime readiness.
- On the tested host, bootstrap can now recover a conservative `partial` state for the proven subset `extract-prepare`, `skeleton`, `extract-skin`, `skin`, and `merge`. That recovered `partial` state is still narrower than blanket `ready` and still does not authorize broader platform support claims.
- Seam/default-candidate evidence is only input to a separate defaulting decision; it is not support, not blanket validation, and not a runtime redesign.
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
2. `processor.py` validates the request, reads `input.filePath` plus top-level `workspaceDir` / `tempDir`, and calls `bootstrap.ensure_ready()`.
3. `setup.py`/bootstrap guarantee the pinned upstream runtime is staged under `.unirig-runtime/vendor/unirig/`.
4. `pipeline.run()` prepares deterministic input/output paths.
5. Upstream UniRig commands perform extract → skeleton → skin → merge.
6. The wrapper publishes the canonical `*_unirig.glb` to `workspaceDir/Workflows/` when `workspaceDir` is usable, writes the sidecar beside that canonical file, and emits Modly `done` with `done.result.filePath` pointing at the workspace artifact so `Add to Scene` recognizes it as workflow output.
7. If `workspaceDir` is missing, empty, or not present on disk, publication falls back to the deterministic source-derived path from `input.filePath` for compatibility.

Canonical publication rules:

- `workspaceDir/Workflows/<input-stem>_unirig.glb` is the primary published artifact when `workspaceDir` is available.
- The sidecar lives beside the canonical published output, not beside the temporary run directory.
- `done.result.filePath` must report the canonical published artifact, not an implementation-internal temp path.
- On Linux ARM64, any mirror back onto the original input path is secondary compatibility behavior only; it does not change the canonical workspace artifact or the reported `done.result.filePath`.

Humanoid contract rules:

- `humanoid_contract.schema` is `modly.humanoid.v1` and lives under the adjacent `.rigmeta.json` sidecar.
- The contract is sidecar-first and authoritative; `done.result.filePath` remains the unchanged public process result shape.
- Required core full-body roles and chains must be explicit and validated before claiming humanoid metadata. Toes are optional: declared toe nodes extend the corresponding leg chains, while missing toes keep leg chains valid through the foot nodes. Optional fingers degrade through deterministic warnings when they are absent or partially declared.
- GLB extras mirroring is deferred so published GLB bytes and hashes remain explicit and the sidecar stays the source of truth.
- `metadata_mode` is the public output contract switch. `auto` attempts bounded resolution and writes a legacy-compatible fallback warning when no valid source exists; `legacy` omits all humanoid fields; `humanoid` must fail closed before `done` unless a valid contract can be built.
- Humanoid resolution priority is deterministic: companion `<output-stem>.humanoid.json`, then read-only GLB extras (`extras.unirig_humanoid`), then semantic resolver evidence from the published output GLB when the output skin/rest/weight evidence is strong enough. Any topology profile compatibility remains narrow and evidence-backed; unknown, ambiguous, contaminated, or contract-insufficient output topology is rejected instead of inferred.
- The mode switch never mutates GLB bytes: no GLB mutation is part of the contract, and the adjacent sidecar remains authoritative.

## Humanoid corpus profiling boundary

`humanoid-corpus-profiling` is a read-only diagnostic layer around the existing parser, semantic resolver, humanoid contract builder, semantic body graph, and quality gate. It exists to make batches of rigged GLBs reproducible as corpus evidence before changing resolver or publication behavior.

The profiler is exposed as:

```bash
python3 -m unirig_ext.humanoid_corpus_cli <directory|glob|file...> --json-out /tmp/report.json [--markdown-out /tmp/report.md] [--hash]
```

Architectural boundaries:

- It does not add a `manifest.json` node and does not change `processor.py`.
- It does not write `.rigmeta.json`, mutate GLB bytes, emit Modly `done`, or participate in runtime publication.
- Its JSON report is authoritative for diagnostics; Markdown is generated only from that JSON.
- It assigns one primary evidence/failure family per asset and preserves overlapping evidence as secondary reason codes.
- It can guide future resolver, quality-gate, verified-transfer, or upstream-rigging work, but the report itself is never humanoid publication evidence.

This boundary is deliberate. Corpus reports help prevent GLB-by-GLB patching, but they are not a loophole around strict humanoid metadata validation.

## Repository layout

- `manifest.json` — public Modly process manifest
- `processor.py` — JSON-line protocol adapter
- `setup.py` — deterministic staging, install, preflight, and readiness writer
- `src/unirig_ext/bootstrap.py` — minimal runtime-state normalization and readiness verification
- `src/unirig_ext/pipeline.py` — deterministic upstream command adapter
- `src/unirig_ext/io.py` — input validation, staging, and output publication
- `src/unirig_ext/metadata.py` — sidecar writer with stable runtime facts and optional humanoid contract enrichment
- `src/unirig_ext/humanoid_contract.py` — deterministic `modly.humanoid.v1` payload builder and validator
- `src/unirig_ext/humanoid_corpus_profiler.py` — read-only corpus profiler and deterministic JSON/Markdown report model
- `src/unirig_ext/humanoid_corpus_cli.py` — diagnostic CLI entrypoint for corpus profiling outside runtime publication
- `tests/` — contract, bootstrap, pipeline, and docs posture checks

## Support posture

Support claims must be backed by **validated evidence**.

| Host | Evidence in this repo | Public claim |
| --- | --- | --- |
| Windows x86_64 | Pinned prebuilt checks, unit coverage, and a real install → rig → export flow verified in Blender | Validated for the current pinned prebuilt workflow |
| Linux x86_64 | Prebuilt-first docs and unit coverage | Conservative existing posture: unvalidated end-to-end; not fully supported |
| Linux ARM64 | Clean-install + real-workflow validation on the tested host, planner-backed staged bringup, guarded `spconv` import-smoke evidence when verified, external Blender evidence, and a recovered partial subset for `extract-prepare`, `skeleton`, `extract-skin`, `skin`, and `merge` | Validated on the tested host/workflow only; still experimental as a general ARM64 platform claim |
| Other hosts | No evidence | Unsupported until validated evidence exists |

The rule is simple: if we do not have validated evidence, we say **unvalidated** or **unsupported**.

For Windows x86_64, validated evidence in this repo currently means the pinned prebuilt workflow completed end-to-end in Modly and the merged output was opened successfully in Blender.

Background Blender smoke alone is NOT validation evidence and must not be presented as support for any host.

Windows x86_64 non-regression invariant: Linux ARM64 work MUST NOT weaken or replace the validated Windows x86_64 pinned prebuilt workflow. Windows stays on its existing branch and remains the reference path with validated evidence in this repo.

Linux ARM64 now goes farther than prep-only diagnostics: on the tested host, a clean install plus real rig workflow can recover a useful `partial` state and produce a rigged final GLB. But that result is still bounded evidence only. It does not eliminate environment boundary risk, it does not prove every Linux ARM64 host behaves the same way, and it does not authorize a blanket support claim.

The architectural rule stays the same: external Blender evidence, executable-boundary proof, qualification verdicts, recovered `partial` runtime, and full `ready` runtime are different artifacts and MUST stay separate in docs and persisted state. Any positive qualification verdict or tested-host `partial` recovery is still only evidence for a later separate defaulting decision. Broader runtime redesign stays deferred. This keeps deferred `bpy`, partial state, and separate defaulting distinct from blanket wrapper runtime readiness.

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
