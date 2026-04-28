from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext.metadata_mode import MetadataModeError, normalize_metadata_mode


class MetadataModeTests(unittest.TestCase):
    def test_missing_mode_defaults_to_auto(self) -> None:
        self.assertEqual(normalize_metadata_mode({}), "auto")

    def test_accepted_modes_are_normalized_case_insensitively(self) -> None:
        self.assertEqual(normalize_metadata_mode({"metadata_mode": " legacy "}), "legacy")
        self.assertEqual(normalize_metadata_mode({"metadata_mode": "HUMANOID"}), "humanoid")

    def test_empty_mode_is_rejected_with_valid_modes(self) -> None:
        with self.assertRaisesRegex(MetadataModeError, "metadata_mode.*auto, legacy, humanoid"):
            normalize_metadata_mode({"metadata_mode": "   "})

    def test_non_string_mode_is_rejected_with_actionable_message(self) -> None:
        with self.assertRaisesRegex(MetadataModeError, "metadata_mode must be a string"):
            normalize_metadata_mode({"metadata_mode": 7})

    def test_unknown_mode_is_rejected_with_requested_value_and_valid_modes(self) -> None:
        with self.assertRaisesRegex(MetadataModeError, "unsupported.*bones.*auto, legacy, humanoid"):
            normalize_metadata_mode({"metadata_mode": "bones"})


if __name__ == "__main__":
    unittest.main()
