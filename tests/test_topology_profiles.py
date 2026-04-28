from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirig_ext.topology_profiles import TopologyProfileError, build_declared_data_from_known_profile


class TopologyProfileTests(unittest.TestCase):
    def test_exact_known_profile_generates_stable_declared_data(self) -> None:
        payload = json.loads((ROOT / "tests" / "fixtures" / "unirig_topology_profile_minimal.json").read_text(encoding="utf-8"))

        first = build_declared_data_from_known_profile(payload)
        second = build_declared_data_from_known_profile(payload)

        self.assertEqual(first, second)
        self.assertEqual(first["roles"]["hips"], "hips")
        self.assertEqual(first["roles"]["right_foot"], "right_foot")
        self.assertEqual(first["provenance"]["source"], "known-unirig-topology-profile")
        self.assertEqual(first["confidence"]["roles"]["hips"], 1.0)

    def test_unknown_topology_is_rejected_fail_closed(self) -> None:
        payload = {"asset": {"version": "2.0"}, "nodes": [{"name": "mystery"}], "skins": []}

        with self.assertRaisesRegex(TopologyProfileError, "Unknown or ambiguous UniRig topology"):
            build_declared_data_from_known_profile(payload)


if __name__ == "__main__":
    unittest.main()
