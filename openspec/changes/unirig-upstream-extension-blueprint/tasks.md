# Tasks: UniRig Upstream Extension Blueprint

## Phase 1: Repository Skeleton

- [x] 1.1 Create root `manifest.json` with stable extension `id`, `type: "process"`, `entry: "processor.py"`, and one `rig-mesh` node (`input: "mesh"`, `output: "mesh"`); verify composite ID stays `ext_id/rig-mesh`.
- [x] 1.2 Create `processor.py` and the `src/unirig_ext/` package skeleton (`bootstrap.py`, `io.py`, `pipeline.py`, `metadata.py`) so the repo tarball contains every manifest-referenced runtime artifact.
- [x] 1.3 Create `setup.py` for isolated dependency/runtime preparation, keeping heavy checkpoints out of the repo tree and documenting cache/install paths in code comments.

## Phase 2: Runtime Extraction and Core Pipeline

- [x] 2.1 Port only portable bootstrap/runtime-prep logic from `extensions/unirig-workspace-v1/generator.py` into `src/unirig_ext/bootstrap.py`; explicitly reject `workspace_tool`, private routes, and private UI hooks.
- [x] 2.2 Implement `src/unirig_ext/io.py` to validate one mesh input, stage working files, derive deterministic output paths, and raise deterministic errors for missing/invalid mesh input.
- [x] 2.3 Implement `src/unirig_ext/pipeline.py` to orchestrate prepare → skeleton → skin → merge/export with MVP params only, producing one rigged mesh output and no extra public nodes.

## Phase 3: Processor Contract and Metadata Output

- [x] 3.1 Implement `processor.py` as the upstream process adapter: parse JSON stdin, dispatch only `rig-mesh`, call bootstrap/pipeline, and emit progress/log/error/done events through upstream-supported JSON-line channels.
- [x] 3.2 Implement `src/unirig_ext/metadata.py` so each successful run writes an adjacent `.rigmeta.json` sidecar with deterministic fields derived from the output GLB.
- [x] 3.3 Wire success/failure contracts so stdout returns `{"type":"done","result":{"filePath":"..."}}` on success and actionable repair/retry signals on setup or runtime failure.

## Phase 4: Automated Verification

- [x] 4.1 Create `tests/test_manifest.py` to verify root artifact presence, valid `process` manifest shape, stable node IDs, and rejection cases for missing root files or non-mesh/private manifest shapes.
- [x] 4.2 Create `tests/test_processor_protocol.py` to verify stdin/stdout protocol, `rig-mesh` dispatch, deterministic invalid-input failures, and that progress/error messages stay within upstream process-runner channels.
- [x] 4.3 Create `tests/test_metadata.py` and `tests/test_setup_bootstrap.py` to verify sidecar schema, deterministic output naming, bootstrap success on a fresh temp environment, and actionable bootstrap failure behavior.

## Phase 5: Upstream Install/Run and Private-Bank Validation

- [ ] 5.1 Validate the repo from the real GitHub `HEAD` tarball against upstream Modly install flow: confirm install succeeds with root artifacts present and fails before registration when `manifest.json` or `processor.py` is missing.
- [ ] 5.2 Run upstream Modly execution for `ext_id/rig-mesh` with representative meshes; verify one mesh-in/one mesh-out behavior, output GLB + `.rigmeta.json`, and no dependence on private patches or private runtime services.
- [ ] 5.3 Compare runtime outputs, logs, and failure signatures against `modly-private` UniRig behavior as a reference bank; record accepted MVP parity gaps that remain explicitly out of scope.

## Phase 6: Documentation

- [x] 6.1 Create `README.md` with install/repair steps, runtime footprint, GitHub-install assumptions, and explicit MVP scope boundaries.
- [x] 6.2 Create `docs/architecture.md` describing the public process-extension architecture, deferred multi-node roadmap, and the upstream-vs-private validation checklist used in Phase 5.
