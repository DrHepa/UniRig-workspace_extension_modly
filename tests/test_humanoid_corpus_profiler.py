from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_humanoid_quality_gate import _write_glb, write_embedded_skin_glb
from unirig_ext.humanoid_corpus_profiler import (
    FAMILY_PRECEDENCE,
    CorpusInputError,
    CorpusOutputError,
    build_corpus_report,
    classify_asset_family,
    dumps_canonical_json,
    render_markdown_from_report_json,
    select_glb_inputs,
    validate_output_parent,
    write_json_report,
)


class HumanoidCorpusSelectionTests(unittest.TestCase):
    def test_selector_accepts_directory_glob_and_explicit_list_with_stable_sort(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-select-") as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            zed = root / "Zed.glb"
            alpha = root / "alpha.glb"
            beta = nested / "beta.glb"
            ignored = root / "notes.txt"
            for path in (zed, alpha, beta, ignored):
                path.write_bytes(b"placeholder")

            self.assertEqual([path.name for path in select_glb_inputs([root])], ["alpha.glb", "Zed.glb"])
            self.assertEqual([path.name for path in select_glb_inputs([str(root / "*.glb")])], ["alpha.glb", "Zed.glb"])
            selected = select_glb_inputs([zed, beta, alpha])

            self.assertEqual([path.relative_to(root).as_posix() for path in selected], ["alpha.glb", "nested/beta.glb", "Zed.glb"])

    def test_selector_fails_loudly_for_unsafe_missing_empty_and_bad_output_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-errors-") as temp_dir:
            root = Path(temp_dir)
            text = root / "not-glb.txt"
            text.write_text("not a GLB", encoding="utf-8")
            empty_dir = root / "empty"
            empty_dir.mkdir()
            unreadable = root / "unreadable.glb"
            unreadable.write_bytes(b"not readable")
            unreadable.chmod(0)

            try:
                with self.assertRaisesRegex(CorpusInputError, "does not exist"):
                    select_glb_inputs([root / "missing.glb"])
                with self.assertRaisesRegex(CorpusInputError, "explicit input is not a .glb file"):
                    select_glb_inputs([text])
                with self.assertRaisesRegex(CorpusInputError, "no .glb files"):
                    select_glb_inputs([empty_dir])
                with self.assertRaisesRegex(CorpusInputError, "unreadable .glb input"):
                    select_glb_inputs([unreadable])
                with self.assertRaisesRegex(CorpusInputError, "unreadable .glb input"):
                    build_corpus_report([unreadable])
                with self.assertRaisesRegex(CorpusOutputError, "output parent does not exist"):
                    validate_output_parent(root / "missing" / "report.json")
            finally:
                unreadable.chmod(0o600)


class HumanoidCorpusClassifierTests(unittest.TestCase):
    def _row(self, **overrides: object) -> dict:
        row = {
            "parse_status": "parsed",
            "skin_count": 1,
            "root_count": 1,
            "skin_joint_count": 19,
            "weighted_joint_count": 19,
            "resolver": {"status": "success", "source_kind": "semantic_humanoid_resolver", "failure_code": None},
            "quality_gate": {"status": "passed", "reasons": []},
            "contract_readiness": {"status": "ready", "ready": True, "failure_code": None},
            "diagnostics": [],
        }
        row.update(overrides)
        return row

    def test_classifier_precedence_and_exactly_one_primary_family(self) -> None:
        cases = [
            ("malformed_unparseable", self._row(parse_status="unparseable", skin_count=0)),
            ("missing_invalid_skin_evidence", self._row(skin_count=0, resolver={"status": "failed", "source_kind": None, "failure_code": "semantic_skin_missing"})),
            ("resolver_ambiguity", self._row(resolver={"status": "failed", "source_kind": None, "failure_code": "semantic_symmetry_ambiguous"})),
            ("shallow_minimal_rig_output", self._row(skin_joint_count=4, weighted_joint_count=4, resolver={"status": "failed", "source_kind": None, "failure_code": "semantic_spine_missing"})),
            ("high_region_contamination", self._row(quality_gate={"status": "failed", "reasons": [{"code": "high_region_weighted_by_torso_or_arm"}, {"code": "non_local_weight_spread"}]})),
            ("passive_accessory_sleeve_contamination", self._row(quality_gate={"status": "failed", "reasons": [{"code": "semantic_passive_noncontract_subtree"}]})),
            ("non_local_weight_spread", self._row(quality_gate={"status": "failed", "reasons": [{"code": "non_local_weight_spread"}]})),
            ("contract_ready", self._row()),
            ("other_unknown", self._row(contract_readiness={"status": "failed", "ready": False, "failure_code": "contract_warning"})),
        ]

        self.assertEqual(FAMILY_PRECEDENCE, [family for family, _row in cases])
        for expected, row in cases:
            classified = classify_asset_family(row)
            self.assertEqual(classified["primary_family"], expected)
            self.assertIsInstance(classified["primary_family"], str)

    def test_secondary_reason_codes_preserve_overlapping_evidence_in_stable_order(self) -> None:
        row = self._row(
            quality_gate={
                "status": "failed",
                "reasons": [
                    {"code": "non_local_weight_spread"},
                    {"code": "sleeve_branch_under_arm"},
                    {"code": "high_region_weighted_by_torso_or_arm"},
                ],
            }
        )

        classified = classify_asset_family(row)

        self.assertEqual(classified["primary_family"], "high_region_contamination")
        self.assertEqual(
            classified["secondary_reason_codes"],
            ["high_region_weighted_by_torso_or_arm", "sleeve_branch_under_arm", "non_local_weight_spread"],
        )


class HumanoidCorpusReportTests(unittest.TestCase):
    def test_json_schema_rows_and_malformed_assets_are_stable_without_publication_side_effects(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-report-") as temp_dir:
            root = Path(temp_dir)
            clean = write_resolver_ready_glb(root / "clean.glb")
            malformed = root / "broken.glb"
            malformed.write_bytes(b"not a glb")

            report = build_corpus_report([malformed, clean], include_hash=True)

            self.assertEqual(list(report), ["schema_version", "corpus_summary", "per_asset_rows", "family_summaries", "reason_codes"])
            self.assertEqual([row["path"] for row in report["per_asset_rows"]], ["broken.glb", "clean.glb"])
            malformed_row = report["per_asset_rows"][0]
            clean_row = report["per_asset_rows"][1]
            required_row_keys = {
                "path", "name", "size_bytes", "sha256", "node_count", "skin_count", "root_count",
                "skin_joint_count", "weighted_joint_count", "resolver", "quality_gate", "contract_readiness",
                "primary_family", "secondary_reason_codes", "diagnostics", "publication_evidence",
            }
            self.assertTrue(required_row_keys.issubset(malformed_row))
            self.assertEqual(malformed_row["primary_family"], "malformed_unparseable")
            self.assertEqual(malformed_row["resolver"]["status"], "not_run")
            self.assertEqual(clean_row["primary_family"], "contract_ready")
            self.assertEqual(clean_row["publication_evidence"], False)
            self.assertFalse((root / "clean.rigmeta.json").exists())
            self.assertEqual(report["corpus_summary"]["total_assets"], 2)
            self.assertEqual(report["family_summaries"]["contract_ready"]["count"], 1)
            self.assertEqual(report["family_summaries"]["contract_ready"]["paths"], ["clean.glb"])
            self.assertEqual(report["family_summaries"]["contract_ready"]["resolver_status_counts"], {"success": 1})
            self.assertEqual(report["family_summaries"]["contract_ready"]["quality_status_counts"], {"passed": 1})
            self.assertEqual(report["family_summaries"]["contract_ready"]["contract_status_counts"], {"ready": 1})
            self.assertEqual(report["family_summaries"]["contract_ready"]["reason_code_counts"], {"contract_ready": 1})
            self.assertEqual(report["family_summaries"]["malformed_unparseable"]["paths"], ["broken.glb"])
            self.assertEqual(report["family_summaries"]["malformed_unparseable"]["resolver_status_counts"], {"not_run": 1})
            self.assertEqual(report["family_summaries"]["malformed_unparseable"]["quality_status_counts"], {"not_run": 1})
            self.assertEqual(report["family_summaries"]["malformed_unparseable"]["contract_status_counts"], {"not_ready": 1})
            self.assertEqual(
                report["family_summaries"]["malformed_unparseable"]["reason_code_counts"],
                {"malformed_unparseable": 1, "Unsupported humanoid quality gate input": 1},
            )

    def test_markdown_is_rendered_from_json_only_and_canonical_output_is_stable(self) -> None:
        report = {
            "schema_version": "unirig.humanoid_corpus.v1",
            "corpus_summary": {"total_assets": 1, "parseable_assets": 1, "family_counts": {"contract_ready": 1}, "reason_code_counts": {}},
            "per_asset_rows": [
                {
                    "path": "clean.glb",
                    "name": "clean.glb",
                    "primary_family": "contract_ready",
                    "secondary_reason_codes": [],
                    "quality_gate": {"status": "passed", "reasons": []},
                    "resolver": {"status": "success", "source_kind": "semantic_humanoid_resolver", "failure_code": None},
                    "contract_readiness": {"status": "ready", "ready": True, "failure_code": None},
                    "publication_evidence": False,
                }
            ],
            "family_summaries": {family: {"count": 1 if family == "contract_ready" else 0, "paths": ["clean.glb"] if family == "contract_ready" else []} for family in FAMILY_PRECEDENCE},
            "reason_codes": [],
        }
        canonical = dumps_canonical_json(report)

        with patch("unirig_ext.humanoid_corpus_profiler.read_glb_container") as reader:
            markdown = render_markdown_from_report_json(json.loads(canonical))
            markdown_again = render_markdown_from_report_json(json.loads(canonical))

        reader.assert_not_called()
        self.assertEqual(markdown, markdown_again)
        self.assertIn("Diagnostic only", markdown)
        self.assertIn("contract_ready", markdown)
        self.assertEqual(canonical, dumps_canonical_json(json.loads(canonical)))


class HumanoidCorpusCliTests(unittest.TestCase):
    def test_cli_writes_json_and_optional_markdown_reports_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-cli-") as temp_dir:
            root = Path(temp_dir)
            glb = write_embedded_skin_glb(root / "clean.glb")
            json_out = root / "report.json"
            markdown_out = root / "report.md"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unirig_ext.humanoid_corpus_cli",
                    str(glb),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(SRC)},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(json_out.read_text(encoding="utf-8"))["corpus_summary"]["total_assets"], 1)
            self.assertIn("Diagnostic only", markdown_out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()


def write_resolver_ready_glb(target: Path) -> Path:
    nodes = [
        {"name": "hips", "translation": [0.0, 0.0, 0.0], "children": [1, 5, 8]},
        {"name": "spine", "translation": [0.0, 1.0, 0.0], "children": [2]},
        {"name": "chest", "translation": [0.0, 0.7, 0.0], "children": [3, 11, 14]},
        {"name": "neck", "translation": [0.0, 0.45, 0.0], "children": [4]},
        {"name": "head", "translation": [0.0, 0.3, 0.0]},
        {"name": "left_upper_leg", "translation": [-0.25, -0.1, 0.0], "children": [6]},
        {"name": "left_lower_leg", "translation": [0.0, -0.9, 0.0], "children": [7]},
        {"name": "left_foot", "translation": [0.0, -0.8, 0.2]},
        {"name": "right_upper_leg", "translation": [0.25, -0.1, 0.0], "children": [9]},
        {"name": "right_lower_leg", "translation": [0.0, -0.9, 0.0], "children": [10]},
        {"name": "right_foot", "translation": [0.0, -0.8, 0.2]},
        {"name": "left_upper_arm", "translation": [-0.65, 0.25, 0.0], "children": [12]},
        {"name": "left_lower_arm", "translation": [-0.6, -0.15, 0.0], "children": [13]},
        {"name": "left_hand", "translation": [-0.45, -0.15, 0.0]},
        {"name": "right_upper_arm", "translation": [0.65, 0.25, 0.0], "children": [15]},
        {"name": "right_lower_arm", "translation": [0.6, -0.15, 0.0], "children": [16]},
        {"name": "right_hand", "translation": [0.45, -0.15, 0.0]},
    ]
    positions = [
        (-0.95, 1.45, -0.05),
        (-1.85, 1.25, 0.05),
        (0.95, 1.45, -0.05),
        (1.85, 1.25, 0.05),
        (0.0, 2.25, 0.0),
        (0.0, 0.3, 0.0),
        (-0.25, -1.65, 0.2),
        (0.25, -1.65, 0.2),
    ]
    joint_rows = [(13, 0, 0, 0), (13, 0, 0, 0), (16, 0, 0, 0), (16, 0, 0, 0), (4, 0, 0, 0), (0, 0, 0, 0), (7, 0, 0, 0), (10, 0, 0, 0)]
    weight_rows = [(1.0, 0.0, 0.0, 0.0) for _ in positions]
    return _write_glb(target, nodes=nodes, joints=list(range(len(nodes))), positions=positions, joint_rows=joint_rows, weight_rows=weight_rows)
