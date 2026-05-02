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
- stdin payload: `input.filePath`, `input.nodeId`, optional `params.seed`, plus top-level host workspace fields such as `workspaceDir` / `tempDir`
- stdout event types: `progress`, `log`, `done`, `error`
- success result shape: `{"filePath": "..._unirig.glb"}`

## Workspace publication contract

For Modly `process` extensions, the host may provide top-level `workspaceDir` / `tempDir` fields. UniRig treats `workspaceDir` as the canonical publication target when it is present, non-empty, and already exists on disk.

- The canonical workflow output is published to `workspaceDir/Workflows/<input-stem>_unirig.glb`.
- The `.rigmeta.json` sidecar is written beside that canonical file.
- `done.result.filePath` points to that canonical workspace artifact so Modly `Add to Scene` treats it as workflow output.
- If `workspaceDir` is missing, empty, or does not exist, the wrapper keeps the compatible fallback and derives the output path from `input.filePath`.
- On Linux ARM64, any compatibility mirror back onto the original input path is secondary only; it is NOT the canonical output and does not replace the workspace `done.result.filePath`.

## Humanoid metadata contract

UniRig writes retargeting metadata sidecar-first under the adjacent `.rigmeta.json` file. When declared humanoid source metadata is available, the sidecar includes `humanoid_contract.schema = "modly.humanoid.v1"` with required core full-body roles, ordered parent-to-child chains, rest transforms, basis hints, confidence, provenance, warnings, and mesh hashes. Toes are optional: when declared, toe nodes extend the leg chains; when absent, the core leg chains remain valid through the foot nodes. Optional fingers may be absent or partial, but incomplete finger chains must produce deterministic warnings instead of pretending the hand mapping is complete.

The public Modly process result is unchanged: `done.result.filePath` still contains only the canonical GLB path. Consumers that do not understand `humanoid_contract` can keep reading the legacy sidecar fields. GLB extras mirroring is deferred; the `.rigmeta.json` sidecar remains the authoritative contract so output byte and hash semantics stay explicit.

### `metadata_mode`

The public node parameter `metadata_mode` controls humanoid sidecar emission without changing `done.result.filePath` and with **no GLB mutation**:

- `auto` is the default. UniRig tries deterministic humanoid resolution and, if no valid source exists, still writes the legacy-compatible sidecar with a deterministic fallback warning and provenance.
- `legacy` suppresses humanoid output completely. Even if a companion file or GLB extras exist, the `.rigmeta.json` sidecar omits `humanoid_contract`, humanoid provenance, and humanoid warnings.
- `humanoid` is fail closed. A valid humanoid source is required; if resolution or validation fails, processing emits `error` before `done` and explains how to provide metadata.

Resolution priority is bounded and deterministic: adjacent companion `<output-stem>.humanoid.json` first, read-only GLB extras (`extras.unirig_humanoid`) second, and semantic resolver evidence from the published output GLB when the output skin/rest/weight evidence is strong enough. Any topology profile compatibility remains narrow and evidence-backed; unknown, ambiguous, or asset-specific topology is rejected instead of guessed. Strict humanoid publication remains fail-closed: resolver success alone is not enough; quality-gate checks can still reject unsafe high-region weights, passive/accessory contamination, non-local spread, ambiguity, or contract-insufficient output skeletons.

## Humanoid corpus profiling

UniRig also exposes a diagnostic-only corpus profiler for analyzing batches of rigged GLBs by stable evidence/failure families instead of patching individual assets one by one.

```bash
python3 -m unirig_ext.humanoid_corpus_cli \
  "/path/to/exports/*.glb" \
  --json-out /tmp/unirig-corpus-profile.json \
  --markdown-out /tmp/unirig-corpus-profile.md \
  --limit 24 \
  --hash
```

For long corpora, `--limit N` profiles the first `N` selected GLBs after deterministic path sorting and marks the report as limited. Progress is emitted only on stderr as tab-separated lines:

```text
{index}/{total}\t{path}\tSTARTED|OK|FAILED
```

stdout is reserved for a deterministic final summary, for example `status=limited selected=2 completed=2 failed=0 partial=false limited=true`. JSON reports include `report_status`, `is_partial`, `is_limited`, `assets_selected`, `assets_completed`, and `assets_failed`. During row-oriented profiling, the JSON output is refreshed atomically after each completed asset so partial results remain valid JSON; Markdown is still generated from JSON only and visibly warns when a report is partial or limited.

The profiler is intentionally **read-only**:

- it does not call `processor.py`;
- it does not mutate GLB files;
- it does not write `.rigmeta.json` sidecars;
- it does not relax `metadata_mode=humanoid`;
- its reports are diagnostic evidence only and MUST NOT be used as humanoid publication evidence.

Generated reports should stay outside the repository by default, for example under `/tmp`. Large corpora can be slow because the profiler reuses the same parser, semantic resolver, contract, and quality-gate code paths that make runtime publication conservative.

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
| Linux ARM64 | Clean GitHub install plus a real rig workflow has now been validated on the tested Ubuntu 24.04 / aarch64 host, using a conservative `partial` runtime mode and the proven seam-backed subset. | **Validated on the tested host/workflow only.** This is still **not** a blanket platform support claim for other Linux ARM64 hosts. Other ARM64 hosts remain experimental until they are revalidated with the same level of evidence. |
| Anything else | No repo evidence. | Unsupported until validated evidence exists. |

Validated Windows evidence for this repo means: install from GitHub, run the full Modly workspace flow through `RIG`, export the result, and open the merged output successfully in Blender. If you validate another host/runtime combination with the same level of evidence, update the docs accordingly. Until then, KEEP THE CLAIMS CONSERVATIVE.

Background Blender smoke alone is NOT validation evidence and must not be presented as support for any host.

Windows x86_64 remains validated for the current pinned prebuilt workflow in this repo.

For the tested Linux ARM64 host, the current clean-install path can now recover a practical `partial` runtime mode instead of stopping in setup-only diagnostics. The proven subset is:

- `extract-prepare`
- `skeleton`
- `extract-skin`
- `skin`
- `merge`

That tested-host success still keeps the language fine-grained:

- `context.venv_python` remains the default/fallback wrapper-owned path in general.
- The Blender subprocess seam stays **optional**, **Linux ARM64-first**, **experimental**, and **non-default**.
- The clean-install proof only shows that the tested host/workflow can recover the same already-proven partial runtime subset without manual backup restoration.
- It does **not** mean blanket Linux ARM64 support, default-seam adoption, or a completed platform port.

Deferred bpy still marks the boundary between this recovered partial runtime mode and any broader full-runtime support claim.
That recovered **partial runtime recovery** is still distinct from wrapper runtime readiness and from any later separate defaulting decision.

The repo therefore still documents Linux ARM64 as a **staged non-Blender bringup** that now has one tested-host clean-install + real-workflow success. It still depends on real system CUDA, real nvcc, `torch_scatter`, `torch_cluster`, and guarded `spconv` import-ready only if verified by `cumm -> spconv -> spconv.pytorch` import smoke. That wording is intentionally conservative and does not claim full runtime support.

For Linux ARM64 `bpy`, the repo still records **external Blender evidence only**. The evidence classes still include `discovered-incompatible` and `external-bpy-smoke-ready`, and even the tested-host partial runtime mode is not the same thing as blanket wrapper support or wrapper runtime readiness.

For `extract/merge`, `context.venv_python` remains the default/fallback wrapper-owned path in general. The **optional Blender subprocess seam** is qualification evidence only for a **separate defaulting** decision, remains distinct from wrapper runtime readiness, and does not override the staged-source-build / source-build expectations for the rest of Linux ARM64.

The evidence lanes are deliberately separate:

- `source_build.external_blender` records external Blender evidence such as discovery, incompatibility, or background smoke.
- `source_build.executable_boundary.extract_merge` records separate executable-boundary proof for the optional seam.
- `partial` runtime recovery records that the tested host has enough proof to run the proven subset, not that every Linux ARM64 host is ready.

Qualification verdict language also stays conservative by design: `not-ready`, `candidate-with-known-risks`, and `ready-for-separate-defaulting-change` are qualification evidence only. Even the strongest verdict is evidence for a later separate defaulting decision, not a support claim. Broader runtime redesign stays deferred.

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
- sidecar-first humanoid metadata modes with fail-closed strict validation
- diagnostic-only humanoid corpus profiling for family-level evidence reports
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
