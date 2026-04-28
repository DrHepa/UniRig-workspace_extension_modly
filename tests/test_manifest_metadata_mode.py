from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManifestMetadataModeTests(unittest.TestCase):
    def _params_by_id(self) -> dict[str, dict]:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        params = manifest["nodes"][0]["params_schema"]
        return {param["id"]: param for param in params}

    def test_metadata_mode_is_public_enum_defaulting_to_auto(self) -> None:
        params = self._params_by_id()

        self.assertEqual(params["metadata_mode"]["type"], "select")
        self.assertEqual(params["metadata_mode"]["default"], "auto")
        self.assertEqual(
            params["metadata_mode"]["options"],
            [
                {"value": "auto", "label": "Auto"},
                {"value": "legacy", "label": "Legacy Bones"},
                {"value": "humanoid", "label": "Humanoid Contract"},
            ],
        )

    def test_seed_param_is_preserved_unchanged(self) -> None:
        params = self._params_by_id()

        self.assertEqual(
            params["seed"],
            {
                "id": "seed",
                "label": "Seed",
                "type": "int",
                "default": 12345,
                "min": 0,
                "max": 2147483647,
                "tooltip": "Deterministic seed forwarded to the runtime pipeline.",
            },
        )


if __name__ == "__main__":
    unittest.main()
