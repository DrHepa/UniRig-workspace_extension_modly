# Code Review Rules

## Scope
- This repository is a public Modly `process` extension, not a private `workspace_tool`.
- Keep the root contract coherent: `manifest.json`, `processor.py`, `setup.py`, `src/unirig_ext/`, and `tests/` must agree.

## Python
- Prefer explicit, readable control flow over clever shortcuts.
- Fail loudly with actionable messages instead of silently falling back.
- Keep filesystem and subprocess behavior deterministic.
- Avoid hidden runtime dependencies on local machine state unless clearly documented.

## Runtime / Bootstrap
- Treat the real runtime path as the primary path.
- Developer hooks are secondary debugging aids, not the default execution model.
- Runtime bootstrap must record enough state/logging to diagnose failures.
- Platform-specific behavior must be explicit in code and conservative in docs.

## Platform Claims
- Do not claim Windows or Linux ARM64 support unless the repo has real validation evidence.
- Prefer wording such as supported, experimental, partially validated, or unvalidated.
- Keep README and architecture docs aligned on support posture.

## Tests
- Keep or improve unit coverage for manifest validation, bootstrap behavior, processor protocol, and metadata outputs.
- New behavior should ship with tests when practical.
- Do not break `python3 -m py_compile setup.py processor.py src/unirig_ext/*.py tests/*.py`.
- Do not break `python3 -m unittest discover -s tests -v`.

## Repository Hygiene
- Do not reintroduce heavyweight vendored runtime trees into the repository without an explicit decision.
- Keep generated runtime state, caches, and virtual environments out of git.
- Prefer small, reviewable commits with honest documentation.
