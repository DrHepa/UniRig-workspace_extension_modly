from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture.md"


class DocsContractTests(unittest.TestCase):
    def test_docs_describe_stable_event_types_with_optional_liveness_metadata(self) -> None:
        readme = README.read_text(encoding="utf-8").lower()
        architecture = ARCHITECTURE.read_text(encoding="utf-8").lower()

        for document in (readme, architecture):
            self.assertIn("progress", document)
            self.assertIn("log", document)
            self.assertIn("done", document)
            self.assertIn("error", document)
            self.assertIn("optional metadata", document)
            self.assertIn("stage", document)
            self.assertIn("kind", document)
            self.assertIn("status", document)
            self.assertIn("elapsedseconds", document)
            self.assertIn("no stdout/stderr streaming", document)

    def test_architecture_doc_names_the_contract_tests_that_lock_liveness_behavior(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8").lower()

        self.assertIn("tests/test_processor_protocol.py", architecture)
        self.assertIn("tests/test_docs_contract.py", architecture)
        self.assertIn("public protocol", architecture)
        self.assertIn("docs contract", architecture)

    def test_readme_documents_workspace_publication_contract(self) -> None:
        readme = README.read_text(encoding="utf-8").lower()

        self.assertIn("workspacedir", readme)
        self.assertIn("tempdir", readme)
        self.assertIn("workflows", readme)
        self.assertIn("canonical", readme)
        self.assertIn("add to scene", readme)
        self.assertIn("fallback", readme)
        self.assertIn("input", readme)
        self.assertIn("linux arm64", readme)
        self.assertIn("secondary", readme)

    def test_architecture_documents_workspace_publication_flow(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8").lower()

        self.assertIn("workspacedir", architecture)
        self.assertIn("tempdir", architecture)
        self.assertIn("workflows", architecture)
        self.assertIn("canonical", architecture)
        self.assertIn("done.result.filepath", architecture)
        self.assertIn("add to scene", architecture)
        self.assertIn("fallback", architecture)
        self.assertIn("input", architecture)
        self.assertIn("linux arm64", architecture)
        self.assertIn("secondary", architecture)

    def test_docs_frame_linux_arm64_extract_merge_as_qualification_only(self) -> None:
        readme = README.read_text(encoding="utf-8")
        architecture = ARCHITECTURE.read_text(encoding="utf-8")

        for document in (readme.lower(), architecture.lower()):
            self.assertIn("qualification", document)
            self.assertIn("experimental", document)
            self.assertIn("context.venv_python", document)
            self.assertIn("default/fallback", document)
            self.assertIn("separate defaulting", document)
            self.assertIn("runtime redesign stays deferred", document)

    def test_readme_describes_thin_wrapper_and_validated_only_support_posture(self) -> None:
        readme = README.read_text(encoding="utf-8")
        normalized = readme.lower()

        self.assertIn("thin wrapper", normalized)
        self.assertIn("upstream-first", normalized)
        self.assertIn("validated evidence", normalized)
        self.assertIn("validated", normalized)
        self.assertIn("windows x86_64", normalized)
        self.assertIn("linux x86_64", normalized)
        self.assertIn("linux arm64", normalized)
        self.assertIn("blender", normalized)
        self.assertIn("validated on the tested host/workflow only", normalized)
        self.assertIn("partial runtime mode", normalized)
        self.assertIn("real system cuda", normalized)
        self.assertIn("real nvcc", normalized)
        self.assertIn("torch_scatter", normalized)
        self.assertIn("torch_cluster", normalized)
        self.assertIn("spconv", normalized)
        self.assertIn("deferred bpy", normalized)
        self.assertIn("experimental", normalized)
        self.assertIn("runtime readiness", normalized)
        self.assertIn("validated evidence", normalized)
        self.assertIn("full runtime support", normalized)
        self.assertNotIn("linux arm64 is supported", normalized)
        self.assertNotIn("linux arm64 validated", normalized)
        self.assertNotIn("unirig-upstream-extension-blueprint", normalized)
        self.assertNotIn("phase 5", normalized)
        self.assertNotIn("developer stage hooks", normalized)
        self.assertNotIn("scaffold mode", normalized)

    def test_readme_keeps_windows_validated_and_linux_arm64_non_blender_only_wording(self) -> None:
        readme = README.read_text(encoding="utf-8")
        normalized = readme.lower()

        self.assertIn("windows x86_64", normalized)
        self.assertIn("validated for the current pinned prebuilt workflow", normalized)
        self.assertIn("linux x86_64", normalized)
        self.assertIn("unvalidated end-to-end", normalized)
        self.assertIn("linux arm64", normalized)
        self.assertIn("experimental", normalized)
        self.assertIn("validated on the tested host/workflow only", normalized)
        self.assertIn("partial", normalized)
        self.assertIn("staged", normalized)
        self.assertIn("source-build", normalized)
        self.assertIn("real system cuda", normalized)
        self.assertIn("real nvcc", normalized)
        self.assertIn("torch_scatter", normalized)
        self.assertIn("torch_cluster", normalized)
        self.assertIn("spconv", normalized)
        self.assertIn("import-ready only if verified", normalized)
        self.assertIn("spconv.pytorch", normalized)
        self.assertIn("import smoke", normalized)
        self.assertIn("deferred bpy", normalized)
        self.assertIn("does not claim full runtime support", normalized)
        self.assertNotIn("prep/probe", normalized)
        self.assertNotIn("prep-ready", normalized)
        self.assertNotIn("linux arm64 is supported", normalized)
        self.assertNotIn("linux arm64 validated", normalized)

    def test_readme_keeps_windows_validation_distinct_from_blender_smoke(self) -> None:
        readme = README.read_text(encoding="utf-8")
        normalized = readme.lower()

        self.assertIn("windows x86_64 remains validated for the current pinned prebuilt workflow", normalized)
        self.assertIn("background blender smoke alone is not validation evidence", normalized)
        self.assertNotIn("blender smoke support", normalized)
        self.assertNotIn("blender smoke-ready means supported", normalized)

    def test_readme_describes_linux_arm64_blender_evidence_as_external_and_blocked(self) -> None:
        readme = README.read_text(encoding="utf-8")
        normalized = readme.lower()

        self.assertIn("external blender evidence only", normalized)
        self.assertIn("discovered-incompatible", normalized)
        self.assertIn("external-bpy-smoke-ready", normalized)
        self.assertIn("external blender evidence", normalized)
        self.assertIn("partial runtime", normalized)
        self.assertIn("not the same thing as blanket wrapper support", normalized)
        self.assertNotIn("external blender evidence means ready", normalized)

    def test_readme_keeps_extract_merge_on_wrapper_runtime_despite_blender_smoke(self) -> None:
        readme = README.read_text(encoding="utf-8")
        normalized = readme.lower()

        self.assertIn("extract/merge", normalized)
        self.assertIn("context.venv_python", readme)
        self.assertIn("linux arm64-first", normalized)
        self.assertIn("optional blender subprocess seam", normalized)
        self.assertIn("wrapper runtime readiness", normalized)
        self.assertIn("platform support claim", normalized)
        self.assertIn("default/fallback", normalized)
        self.assertNotIn("default blender subprocess path", normalized)

    def test_readme_keeps_qualification_verdicts_and_evidence_classes_separate(self) -> None:
        readme = README.read_text(encoding="utf-8")
        normalized = readme.lower()

        self.assertIn("not-ready", normalized)
        self.assertIn("candidate-with-known-risks", normalized)
        self.assertIn("ready-for-separate-defaulting-change", normalized)
        self.assertIn("qualification evidence only", normalized)
        self.assertIn("windows x86_64 remains validated for the current pinned prebuilt workflow", normalized)
        self.assertIn("external blender evidence only", normalized)
        self.assertIn("executable-boundary proof", normalized)
        self.assertIn("partial runtime recovery", normalized)
        self.assertIn("separate defaulting decision", normalized)

    def test_readme_describes_any_blender_subprocess_seam_as_optional_experimental_and_non_default(self) -> None:
        readme = README.read_text(encoding="utf-8")
        normalized = readme.lower()

        self.assertIn("optional blender subprocess seam", normalized)
        self.assertIn("linux arm64-first", normalized)
        self.assertIn("experimental", normalized)
        self.assertIn("non-default", normalized)
        self.assertIn("distinct from wrapper runtime readiness", normalized)
        self.assertIn("external blender evidence", normalized)
        self.assertIn("executable-boundary proof", normalized)
        self.assertNotIn("default blender subprocess path", normalized)
        self.assertNotIn("linux arm64 is supported", normalized)

    def test_architecture_doc_focuses_on_wrapper_boundary_not_phase_checklists(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8")
        normalized = architecture.lower()

        self.assertIn("thin wrapper", normalized)
        self.assertIn("deterministic upstream staging", normalized)
        self.assertIn("validated evidence", normalized)
        self.assertIn("validated for the current pinned prebuilt workflow", normalized)
        self.assertIn("blender", normalized)
        self.assertIn("planner/executor", normalized)
        self.assertIn("normalizer/runtime-context source", normalized)
        self.assertIn("baseline → pyg → spconv → bpy-deferred", architecture)
        self.assertIn("wrapper-owned boundary", normalized)
        self.assertIn("environment boundary", normalized)
        self.assertIn("upstream/package boundary", normalized)
        self.assertIn("windows x86_64 non-regression invariant", normalized)
        self.assertIn("out of scope", normalized)
        self.assertIn("bpy", normalized)
        self.assertIn("linux x86_64", normalized)
        self.assertIn("unvalidated end-to-end", normalized)
        self.assertIn("linux arm64", normalized)
        self.assertIn("validated on the tested host/workflow only", normalized)
        self.assertIn("experimental", normalized)
        self.assertIn("real system cuda", normalized)
        self.assertIn("real nvcc", normalized)
        self.assertIn("torch_scatter", normalized)
        self.assertIn("torch_cluster", normalized)
        self.assertIn("spconv", normalized)
        self.assertIn("cumm", normalized)
        self.assertIn("guarded", normalized)
        self.assertIn("import smoke", normalized)
        self.assertIn("import-ready", normalized)
        self.assertIn("runtime readiness", normalized)
        self.assertIn("validated evidence", normalized)
        self.assertIn("full runtime support", normalized)
        self.assertNotIn("prep/probe", normalized)
        self.assertNotIn("prep-ready", normalized)
        self.assertNotIn("linux arm64 is supported", normalized)
        self.assertNotIn("linux arm64 validated", normalized)
        self.assertNotIn("diagnostic-only", normalized)
        self.assertNotIn("unirig-upstream-extension-blueprint", normalized)
        self.assertNotIn("phase 5", normalized)
        self.assertNotIn("phase 5 validation checklist", normalized)
        self.assertNotIn("secondary developer-hook overrides", normalized)

    def test_architecture_doc_keeps_windows_validated_and_linux_arm64_non_blender_only_wording(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8")
        normalized = architecture.lower()

        self.assertIn("windows x86_64", normalized)
        self.assertIn("validated for the current pinned prebuilt workflow", normalized)
        self.assertIn("windows x86_64 non-regression invariant", normalized)
        self.assertIn("linux arm64", normalized)
        self.assertIn("experimental", normalized)
        self.assertIn("validated on the tested host/workflow only", normalized)
        self.assertIn("partial", normalized)
        self.assertIn("staged", normalized)
        self.assertIn("import-ready", normalized)
        self.assertIn("import smoke", normalized)
        self.assertIn("cumm", normalized)
        if "deferred bpy" not in normalized and "bpy-deferred" not in normalized:
            self.fail("Architecture docs must mention deferred bpy (or bpy-deferred) to keep the ARM64 support boundary explicit.")
        self.assertIn("out of scope", normalized)
        self.assertNotIn("prep/probe", normalized)
        self.assertNotIn("linux arm64 is supported", normalized)
        self.assertNotIn("linux arm64 validated", normalized)

    def test_architecture_doc_keeps_windows_validation_distinct_from_blender_smoke(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8")
        normalized = architecture.lower()

        self.assertIn("windows x86_64 non-regression invariant", normalized)
        self.assertIn("background blender smoke alone is not validation evidence", normalized)
        self.assertNotIn("blender smoke support", normalized)
        self.assertNotIn("blender smoke-ready means supported", normalized)

    def test_architecture_doc_describes_external_blender_evidence_boundary(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8")
        normalized = architecture.lower()

        self.assertIn("deterministic blender discovery", normalized)
        self.assertIn("external blender evidence only", normalized)
        self.assertIn("executable_boundary.extract_merge", architecture)
        self.assertIn("executable-boundary proof", normalized)
        self.assertIn("discovered-incompatible", normalized)
        self.assertIn("external-bpy-smoke-ready", normalized)
        self.assertIn("tested-host `partial` mode", architecture)
        self.assertIn("not the same thing as blanket wrapper support", normalized)
        self.assertIn("wrapper runtime readiness", normalized)

    def test_architecture_doc_keeps_reroute_scope_deferred_and_windows_distinct(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8")
        normalized = architecture.lower()

        self.assertIn("windows x86_64 non-regression invariant", normalized)
        self.assertIn("validated windows x86_64 pinned prebuilt workflow", normalized)
        self.assertIn("external blender evidence only", normalized)
        self.assertIn("executable_boundary.extract_merge", architecture)
        self.assertIn("linux arm64-first", normalized)
        self.assertIn("optional blender subprocess seam", normalized)
        self.assertIn("partial", normalized)
        self.assertIn("blanket support claim", normalized)
        self.assertNotIn("linux arm64 validated support", normalized)

    def test_architecture_doc_keeps_qualification_verdicts_and_evidence_classes_separate(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8")
        normalized = architecture.lower()

        self.assertIn("not-ready", normalized)
        self.assertIn("candidate-with-known-risks", normalized)
        self.assertIn("ready-for-separate-defaulting-change", normalized)
        self.assertIn("qualification evidence only", normalized)
        self.assertIn("windows x86_64 non-regression invariant", normalized)
        self.assertIn("external blender evidence only", normalized)
        self.assertIn("executable-boundary proof", normalized)
        self.assertIn("partial runtime", normalized)
        self.assertIn("must stay separate", normalized)

    def test_architecture_doc_describes_any_blender_subprocess_seam_as_optional_experimental_and_non_default(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8")
        normalized = architecture.lower()

        self.assertIn("optional blender subprocess seam", normalized)
        self.assertIn("linux arm64-first", normalized)
        self.assertIn("experimental", normalized)
        self.assertIn("non-default", normalized)
        self.assertIn("distinct from wrapper runtime readiness", normalized)
        self.assertIn("external blender evidence", normalized)
        self.assertIn("executable-boundary proof", normalized)
        self.assertIn("windows x86_64 non-regression invariant", normalized)
        self.assertNotIn("linux arm64 is supported", normalized)


if __name__ == "__main__":
    unittest.main()
