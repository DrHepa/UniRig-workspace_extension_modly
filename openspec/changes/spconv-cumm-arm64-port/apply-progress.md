# Apply Progress: spconv-cumm-arm64-port

## Mode

Standard

## Completed In This Pass

- Moved CUDA/toolchain activation behind real compile gates in `setup.py`: Windows x64 and Linux x64 now keep the wheel-first path by default and do not resolve matching system CUDA or provision the local pip CUDA toolchain unless source-build is actually active or `prepare_cuda_toolchain` / `MODLY_UNIRIG_PREPARE_CUDA_TOOLCHAIN=1` explicitly requests it.
- Kept Linux ARM64/source-build behavior intact by reusing the same compile environment for PyG fallback and `cumm`/`spconv` source installs only when the source-build plan is active.
- Added regression tests proving Windows x64 and Linux x64 prebuilt-default installs skip compile/toolchain setup, while Linux ARM64/source-build still enters the compile environment path.
- Added an explicit ARM64 source-build manifest for `cumm` and `spconv` with repo/ref selection, build-mode selection, and experimental defaults that keep x86_64 on the prebuilt path.
- Added source-build orchestration plumbing in `setup.py`: runtime/vendor staging, repo/ref resolution, source-build env flags (`CUMM_CUDA_VERSION`, `CUMM_DISABLE_JIT`, `SPCONV_DISABLE_JIT`, arch flags), and clear failures when staged sources are incomplete.
- Persisted source-build intent/results into `bootstrap_state.json` with a stable `source_build` shape and mirrored that data into the runtime sidecar metadata.
- Updated tests to validate the manifest, source-build plan resolution, source-build env flags, local source staging, and failure behavior for incomplete source trees.
- Synchronized the updated bootstrap runtime files into the locally installed extension for immediate diagnostic benefit.
- Added a conservative Windows x64 prebuilt manifest plus preflight validation that blocks unsupported host Python minors before later `pip install` noise, explicitly anchoring the repo's pinned Windows path to the validated Python 3.10/3.11/3.12 prebuilt stack.
- Extended the preflight checklist/report to describe the Windows x64 pinned prebuilt artifact path (torch 2.7.0/cu128, PyG torch-2.7.0+cu128, explicit `cumm-cu126==0.7.11` + `spconv-cu126==2.3.8`) without weakening Linux x64 or Linux ARM64 behavior.
- Added regression coverage for the new Windows x64 manifest, blocked unsupported Python diagnostics, supported-Python success behavior, and the updated preflight checklist wording.
- Added a thin Windows-only runtime DLL-path shim in `src/unirig_ext/bootstrap.py` that prepends `venv/Lib/site-packages/torch/lib` plus discovered `venv/Lib/site-packages/nvidia/*/bin` directories to subprocess `PATH`, while keeping Linux behavior unchanged.
- Reused that shared runtime environment in `setup.py` Windows smoke checks and in runtime-stage subprocess execution so `spconv`/`cumm` imports see the same DLL search paths after setup and during real processing.
- Added regression tests covering DLL-path discovery, Linux no-op behavior, Windows smoke-check PATH injection, and runtime stage subprocess environment usage.

## Verification

- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (46 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (27 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (51 tests)
- `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py` ✅
- `python3 -m unittest discover -s tests -v` ✅ (51 tests)

## Notes

- ARM64 remains experimental/source-build-only; this pass wires the source-build path and makes failures honest, but it does **not** claim a full compiled `spconv`/`cumm` success on this machine.
