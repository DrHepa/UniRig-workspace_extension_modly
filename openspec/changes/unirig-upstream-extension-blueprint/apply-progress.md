# Apply Progress: UniRig Upstream Extension Blueprint

## Mode

Standard

## Completed In This Pass

- Tightened `src/unirig_ext/bootstrap.py` so the real extension root derived from the checked-in runtime stays primary by default, while `UNIRIG_EXTENSION_ROOT` is now an explicit opt-in override path instead of ambient process state.
- Reduced runtime subprocess host leakage by replacing `os.environ.copy()` in `runtime_environment()` with a small OS-critical passthrough allowlist plus explicit UniRig runtime variables.
- Updated `tests/test_setup_bootstrap.py` to lock the new root-resolution and runtime-environment contracts, and updated `tests/test_processor_protocol.py` to execute a copied temporary processor tree so subprocess tests no longer rely on `UNIRIG_EXTENSION_ROOT`.
- Restored the public processor baseline by validating mesh inputs before bootstrap, honoring `UNIRIG_EXTENSION_ROOT` during processor runs, and fixing the deterministic upstream staging token in `src/unirig_ext/pipeline.py` so processor protocol tests use the intended `run-processor` runtime filenames.
- Extended the generated Windows managed-venv `sitecustomize.py` shim so it still silently registers DLL directories first, then defensively calls `torch.serialization.add_safe_globals([Box])` when both `torch` and `box.box.Box` are importable, without patching upstream `run.py`.
- Expanded `tests/test_setup_bootstrap.py` to verify the generated `sitecustomize.py` now includes the `box.box.Box` safe-globals registration logic while remaining silent/non-fatal.
- Added a narrow Windows-only runtime staging shim in `setup.py` that rewrites the staged upstream config `configs/model/unirig_ar_350m_1024_81920_float32.yaml` from `_attn_implementation: flash_attention_2` to `_attn_implementation: eager`, matching the observed Windows evidence that FA2 fails on non-Ampere GPUs and on torch float32.
- Made the shim explicit and observable without reintroducing broad policy logic: every Windows staging pass now writes `.unirig-runtime/logs/runtime-stage-patches.json` and logs whether the config patch was applied, already present, missing, or no longer needed.
- Expanded `tests/test_setup_bootstrap.py` to verify the Windows config patch is applied during runtime staging, remains idempotent on repeated staging/reuse, and does not change Linux staging behavior.
- Added a Windows-only managed-venv `sitecustomize.py` install step in `setup.py` so every venv Python subprocess auto-registers discovered DLL folders via `os.add_dll_directory()` before imports, without patching upstream UniRig sources.
- Centralized Windows DLL discovery in `src/unirig_ext/bootstrap.py` with shared `torch/lib` and `nvidia/*/bin` glob rules, keeping PATH augmentation as a secondary aid while reusing the same discovery contract for the generated shim.
- Expanded `tests/test_setup_bootstrap.py` and `tests/test_processor_protocol.py` to cover generated `sitecustomize.py` content/installation and to assert runtime subprocess PATH ordering still follows the centralized discovery helper.
- Kept the thin-wrapper/upstream-first bootstrap intact in `setup.py`, but restored the narrow Windows-only dependency shim: upstream `requirements.txt` now filters out generic `flash_attn` entries only on Windows, then setup explicitly installs pinned `triton-windows==3.3.1.post19` and the pinned Hugging Face `flash_attn` wheel.
- Added Windows-only post-setup smoke checks in `setup.py` for `flash_attn`, `triton`, `flash_attn.layers.rotary`, and `spconv.pytorch`, so Windows bootstrap fails before writing a ready state if the pinned shim still is not importable.
- Tightened the Windows prebuilt dependency path in `setup.py` to keep torch `2.7.0`/`cu128` pinned while explicitly installing the explored-compatible pair `cumm-cu126==0.7.11` + `spconv-cu126==2.3.8`, and expanded smoke checks to validate both `cumm.core_cc` and `spconv.pytorch` with actionable repair guidance.
- Expanded `tests/test_setup_bootstrap.py` to cover Windows upstream-requirements filtering, explicit Triton + flash-attn install commands, the full Windows smoke-check sequence, and early failure when Triton import breaks, while keeping Linux requirements behavior unchanged.
- Restored the old Windows Triton self-heal behavior in `setup.py` by pinning and installing `triton-windows==3.3.1.post19` during Windows bootstrap before the flash-attn wheel path.
- Expanded Windows post-setup smoke checks to fail fast on `flash_attn`, `triton`, `flash_attn.layers.rotary`, and `spconv.pytorch`, and persisted the smoke-check status into `.unirig-runtime/bootstrap_state.json` for diagnostics.
- Updated `tests/test_setup_bootstrap.py` to cover the Windows Triton install path, actionable Triton install failures, multi-import smoke-check execution, and early blocking when Triton import still fails.
- Made Linux ARM64 source-build activation explicit in `setup.py` by recording per-plan activation diagnostics/state, keeping the experimental/source-build path distinct from Windows x64 and Linux x64 prebuilt defaults.
- Added an actionable guard that rejects source-build overrides on unsupported hosts, so Windows x64 and Linux x64 policy changes cannot silently route through ARM64-only compile expectations.
- Sharpened the preflight checklist text in `src/unirig_ext/bootstrap.py` so Linux ARM64 hosts explicitly report active toolchain/source-build enforcement while non-ARM64 hosts report ARM64 diagnostics as reference-only.
- Expanded `tests/test_setup_bootstrap.py` coverage for Linux ARM64 activation state and for rejecting unsupported Windows x64 source-build overrides while preserving Linux x64 prebuilt-reference behavior.
- Replaced the scaffold-first runtime bootstrap with a real-runtime-first setup flow in `setup.py`, including isolated runtime roots, upstream UniRig source staging, dependency-install hooks, asset-prefetch hooks, and honest ready/error state writing.
- Ported the real portable runtime context from the private UniRig implementation into `src/unirig_ext/bootstrap.py`, including runtime layout validation, bootstrap version gating, runtime vendor paths, and explicit rejection of stale scaffold state.
- Rebuilt `src/unirig_ext/pipeline.py` and `src/unirig_ext/io.py` around the real prepare → extract → skeleton → skin → merge flow, while keeping stage-command overrides only as secondary developer hooks.
- Expanded metadata/logging so `processor.py` and `.rigmeta.json` report the runtime mode clearly, and synced the same runtime code into the locally installed extension for immediate Modly-side repair/retest.
- Added stronger automated coverage for the real runtime branch: setup state creation, stale/incomplete runtime rejection, and processor execution through a fake UniRig runtime fixture that proves the real branch produces a non-identical output mesh.
- Ran the REAL local installed-extension provisioning flow with the exact Modly payload shape on Linux ARM64 (`python_exe`, `ext_dir`, `gpu_sm=121`, `cuda_version=130`) instead of stopping at offline tests.
- Patched `setup.py` in both the source repo and installed extension to skip `open3d` on Linux ARM64, because no compatible wheel exists here and upstream `src/inference/merge.py` only imported it unnecessarily.
- Added a runtime-source compatibility patch so the staged upstream `src/inference/merge.py` tolerates missing `open3d`, and added a unit test proving filtered requirements now drop `open3d` on Linux ARM64.
- Added a Linux ARM64 fallback that retries `torch_scatter` and `torch_cluster` from source without build isolation when PyG CUDA wheels are unavailable, pushing local provisioning one step further before the next real blocker surfaced.
- Added a local CUDA 12.8 toolchain provisioning branch in `setup.py` that installs `nvidia-cuda-runtime-cu12`, `nvidia-cuda-cupti-cu12`, and `nvidia-cuda-nvcc-cu12` into the extension venv when torch expects CUDA 12.8 but the host `nvcc` does not match.
- Materialized a synthetic `.unirig-runtime/cuda-toolchain/` tree from the pip-installed NVIDIA packages and prepared explicit native-build env vars (`CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `PATH`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `CPATH`, `CPLUS_INCLUDE_PATH`) so source builds stop reading the system CUDA 13.0 layout.
- Re-ran the real installed-extension provisioning flow and proved the NEXT blocker honestly: on Linux ARM64, `nvidia-cuda-nvcc-cu12==12.8.93` installs headers, NVVM bits, and `ptxas`, but does NOT provide `bin/nvcc`, so PyTorch CUDA extension builds still cannot satisfy `CUDA_HOME/bin/nvcc`.
- Re-ran real provisioning after the user installed the side-by-side system toolkit and confirmed the setup now selects `/usr/local/cuda-12.8/bin/nvcc` instead of falling back to the incomplete pip CUDA toolchain.
- Added a Linux ARM64 preflight guard in `setup.py` that detects missing Python development headers before retrying PyG source builds, turning the next failure into an actionable environment blocker instead of a long compiler crash.
- Continued the installed-extension provisioning pass after `python3.12-dev` was installed, confirmed the flow now gets past the Python headers blocker, and captured the next honest blocker: `pip install spconv-cu120` has no matching distribution for this Linux ARM64 / Python 3.12 environment.
- Fixed developer stage-hook input propagation in `src/unirig_ext/pipeline.py` so skeleton/skin hooks now receive absolute paths resolved inside `context.unirig_dir` even though the hook process still runs from `context.extension_root`.
- Expanded `tests/test_processor_protocol.py` and the hook fixture to verify the hook path/env contract end-to-end and to prove hook failures surface back through the processor protocol with actionable exit-code errors.
- Added an explicit Windows bootstrap interpreter resolver in `setup.py` before preflight/venv creation, verifying candidates by execution and preferring Python 3.11 before documented fallbacks 3.12 and 3.10.
- Persisted requested-vs-resolved Windows bootstrap interpreter details into preflight/state artifacts and added checklist output so host Python mismatches can be diagnosed after bootstrap.
- Expanded `tests/test_setup_bootstrap.py` with resolver preference coverage, unsupported-version rejection coverage, and a main-path assertion that the same resolved Windows interpreter is used for both preflight and venv creation.
- Added deterministic per-run stage subprocess logs under `.unirig-runtime/logs/<run-id>/` in `src/unirig_ext/pipeline.py`, covering extract/skeleton/skin/merge without changing Linux/Windows execution semantics.
- Updated `PipelineError` stage failures to point at the persisted stage log and include a short stdout/stderr tail summary, while still saving logs for tolerated Windows native-crash-success cases.
- Expanded `tests/test_processor_protocol.py` to assert stage-log persistence, log-path error messaging, and persisted logging for tolerated Windows native crash return codes.

## Verification

- `python3 -m unittest discover -s tests -v` ✅ (61 tests)
- `python3 -m unittest tests.test_processor_protocol -v` ✅ (17 tests)
- `python3 -m unittest discover -s tests -v` ✅ (58 tests)
- `python3 -m unittest tests.test_setup_bootstrap -v` ✅ (24 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ⚠️ 1 unrelated pre-existing failure remains: `tests.test_processor_protocol.PipelineGuardrailTests.test_build_execution_plan_uses_runtime_staging_names_for_upstream_inputs` still expects `.modly_stage_input_run-processor.obj`, while current code produces `.modly_stage_input_run-fixed.obj`.
- `python3 -m unittest tests.test_setup_bootstrap -v` ✅
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ⚠️ 1 unrelated pre-existing failure remains: `tests.test_processor_protocol.PipelineGuardrailTests.test_build_execution_plan_uses_runtime_staging_names_for_upstream_inputs` expects `.modly_stage_input_run-processor.obj`, while current code produces `.modly_stage_input_run-fixed.obj` from `run_dir.name`.
- `python3 -m py_compile processor.py setup.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (18 tests)
- Direct processor validation through `tests/test_processor_protocol.py` real-runtime fixture ✅ output GLB written, `.rigmeta.json` written, and output bytes differ from input bytes.
- Local installed extension sync ✅ runtime code and setup were copied into `/home/drhepa/Documentos/Modly/extensions/unirig-process-extension`.
- Real local provisioning attempt ✅ reached actual dependency installation in the installed extension and moved past the initial `open3d` blocker.
- `python3 -m py_compile setup.py src/unirig_ext/bootstrap.py src/unirig_ext/io.py src/unirig_ext/pipeline.py src/unirig_ext/metadata.py tests/test_setup_bootstrap.py` ✅
- `python3 -m unittest tests.test_setup_bootstrap -v` ✅ (7 tests, including local CUDA toolchain env coverage)
- `venv/bin/python -m py_compile /home/drhepa/Documentos/Modly/extensions/unirig-process-extension/setup.py` ✅
- Real installed-extension provisioning rerun ✅ local CUDA pip packages were installed into the extension venv and `.unirig-runtime/cuda-toolchain/` was materialized before failing on the missing `nvcc` binary instead of the old CUDA 13.0 mismatch.
- `python3 -m py_compile setup.py tests/test_setup_bootstrap.py` ✅
- `python3 -m unittest tests.test_setup_bootstrap -v` ✅ (10 tests, including the missing-`Python.h` preflight coverage)
- Real installed-extension provisioning rerun after the CUDA 12.8 system install ✅ logs now show `using version-matched system CUDA toolkit for torch CUDA 12.8: CUDA_HOME=/usr/local/cuda-12.8, nvcc=/usr/local/cuda-12.8/bin/nvcc` before failing on missing `/usr/include/python3.12/Python.h`.
- Real installed-extension provisioning rerun after installing `python3.12-dev` ✅ moved past the missing-`Python.h` blocker, retried Linux ARM64 PyG source builds, and then failed at `pip install spconv-cu120` with `No matching distribution found for spconv-cu120`.
- `venv/bin/python -m pip index versions spconv-cu120` ❌ `No matching distribution found for spconv-cu120`
- `venv/bin/python -m pip index versions spconv` ❌ `No matching distribution found for spconv`
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (48 tests)
- `python3 -m unittest tests.test_setup_bootstrap tests.test_processor_protocol -v` ✅ (38 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (52 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (61 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (64 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (65 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (71 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (48 tests)

## Deferred Explicitly

- Phase 5 remains open: real GitHub `HEAD` install validation, upstream Modly execution with representative meshes, and private-bank parity comparison are still not completed in this pass.
- Full on-machine bootstrap of the actual upstream UniRig dependency stack is still blocked on Linux ARM64 / Blackwell native-extension compatibility: after patching past `open3d` and the system CUDA 13.0 mismatch, the pip-distributed CUDA 12.8 toolchain is still incomplete on this platform because `nvidia-cuda-nvcc-cu12` does not ship `bin/nvcc`.
- Full on-machine bootstrap is STILL blocked after the CUDA fix because Linux ARM64 source builds for `torch_scatter` / `torch_cluster` now require the missing system Python development headers for the `/usr/bin/python3.12` base interpreter used by the extension venv.
- Full on-machine bootstrap is STILL blocked after fixing Python headers because the next hard dependency, `spconv`, has no pip-discoverable wheel/distribution in this environment, and the upstream UniRig runtime imports `spconv.pytorch` directly inside core point-cloud model modules.

## Risks

- The public extension now has a real runtime path, but actual end-to-end rigging on this machine still depends on a successful `setup.py` repair/bootstrap against the target runtime environment.
- Linux ARM64 dependency provisioning for the complete UniRig stack still needs live validation in Modly/Phase 5, especially around whether a TRUE CUDA 12.8 `nvcc` binary can be supplied for PyG native builds on ARM64.
- Even with the correct CUDA toolkit selected, Modly cannot perform a real rigging retest until the machine has matching Python 3.12 development headers installed (for example `python3.12-dev`) so PyG native extensions can compile.
- Even after fixing Python headers, Modly still cannot perform a real rigging retest until a compatible `spconv` package/source-install path exists for Linux ARM64 + Python 3.12 + this CUDA/Torch stack.
