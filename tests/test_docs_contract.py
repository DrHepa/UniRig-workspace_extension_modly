from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture.md"


class DocsContractTests(unittest.TestCase):
    def test_readme_describes_thin_wrapper_and_validated_only_support_posture(self) -> None:
        readme = README.read_text(encoding="utf-8")

        self.assertIn("thin wrapper", readme.lower())
        self.assertIn("upstream-first", readme.lower())
        self.assertIn("validated evidence", readme.lower())
        self.assertIn("validated", readme.lower())
        self.assertIn("windows x86_64", readme.lower())
        self.assertIn("blender", readme.lower())
        self.assertIn("unvalidated", readme.lower())
        self.assertNotIn("Developer stage hooks", readme)
        self.assertNotIn("scaffold mode", readme)

    def test_architecture_doc_focuses_on_wrapper_boundary_not_phase_checklists(self) -> None:
        architecture = ARCHITECTURE.read_text(encoding="utf-8")

        self.assertIn("thin wrapper", architecture.lower())
        self.assertIn("deterministic upstream staging", architecture.lower())
        self.assertIn("validated evidence", architecture.lower())
        self.assertIn("validated for the current pinned prebuilt workflow", architecture.lower())
        self.assertIn("blender", architecture.lower())
        self.assertNotIn("Phase 5 validation checklist", architecture)
        self.assertNotIn("secondary developer-hook overrides", architecture)


if __name__ == "__main__":
    unittest.main()
