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
    build_corpus_report_from_rows,
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

    def test_selector_limit_applies_after_deterministic_sort(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-limit-select-") as temp_dir:
            root = Path(temp_dir)
            for name in ("zulu.glb", "Alpha.glb", "middle.glb"):
                (root / name).write_bytes(b"placeholder")

            selected = select_glb_inputs([root], limit=2)

            self.assertEqual([path.name for path in selected], ["Alpha.glb", "middle.glb"])

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

            self.assertEqual(
                list(report),
                [
                    "schema_version",
                    "report_status",
                    "is_partial",
                    "is_limited",
                    "assets_selected",
                    "assets_completed",
                    "assets_failed",
                    "corpus_summary",
                    "per_asset_rows",
                    "family_summaries",
                    "reason_codes",
                ],
            )
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
            self.assertEqual(report["report_status"], "complete_with_failures")
            self.assertEqual(report["assets_selected"], 2)
            self.assertEqual(report["assets_completed"], 1)
            self.assertEqual(report["assets_failed"], 1)
            self.assertFalse(report["is_partial"])
            self.assertFalse(report["is_limited"])

    def test_report_metadata_status_covers_limited_failures_and_partial(self) -> None:
        ok = self._synthetic_row("ok.glb", profile_status="OK", primary_family="contract_ready")
        failed = self._synthetic_row("bad.glb", profile_status="FAILED", primary_family="malformed_unparseable")

        complete = build_corpus_report_from_rows([ok], assets_selected=1, is_limited=False)
        limited = build_corpus_report_from_rows([ok], assets_selected=1, is_limited=True)
        complete_with_failures = build_corpus_report_from_rows([ok, failed], assets_selected=2, is_limited=False)
        limited_with_failures = build_corpus_report_from_rows([ok, failed], assets_selected=2, is_limited=True)
        partial = build_corpus_report_from_rows([ok], assets_selected=3, is_limited=False)

        self.assertEqual(complete["report_status"], "complete")
        self.assertEqual(limited["report_status"], "limited")
        self.assertEqual(complete_with_failures["report_status"], "complete_with_failures")
        self.assertEqual(limited_with_failures["report_status"], "limited_with_failures")
        self.assertEqual(partial["report_status"], "partial")
        self.assertEqual(partial["assets_selected"], 3)
        self.assertEqual(partial["assets_completed"], 1)
        self.assertEqual(partial["assets_failed"], 0)
        self.assertTrue(partial["is_partial"])

    def test_atomic_json_refresh_preserves_prior_rows_after_later_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-atomic-") as temp_dir:
            root = Path(temp_dir)
            paths = [root / "a.glb", root / "b.glb", root / "c.glb"]
            for path in paths:
                path.write_bytes(b"placeholder")
            json_out = root / "report.json"
            calls = []

            def fake_profile(path: Path, *, base: Path, include_hash: bool) -> dict:
                calls.append(path.name)
                if path.name == "b.glb":
                    raise RuntimeError("boom while profiling")
                return self._synthetic_row(path.name, profile_status="OK", primary_family="contract_ready")

            with patch("unirig_ext.humanoid_corpus_profiler._profile_asset", side_effect=fake_profile):
                report = build_corpus_report(paths, json_refresh_path=json_out)

            written = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(calls, ["a.glb", "b.glb", "c.glb"])
            self.assertEqual([row["path"] for row in written["per_asset_rows"]], ["a.glb", "b.glb", "c.glb"])
            self.assertEqual(written["per_asset_rows"][1]["profile_status"], "FAILED")
            self.assertEqual(written["report_status"], "complete_with_failures")
            self.assertEqual(report, written)

    def test_row_failure_isolated_as_diagnostic_row_and_remaining_rows_continue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-row-failure-") as temp_dir:
            root = Path(temp_dir)
            paths = [root / "a.glb", root / "b.glb", root / "c.glb"]
            for path in paths:
                path.write_bytes(b"placeholder")

            def fake_profile(path: Path, *, base: Path, include_hash: bool) -> dict:
                if path.name == "b.glb":
                    raise ValueError("cannot decode skin")
                return self._synthetic_row(path.name, profile_status="OK", primary_family="contract_ready")

            with patch("unirig_ext.humanoid_corpus_profiler._profile_asset", side_effect=fake_profile):
                report = build_corpus_report(paths)

            failed = report["per_asset_rows"][1]
            self.assertEqual(report["assets_completed"], 2)
            self.assertEqual(report["assets_failed"], 1)
            self.assertEqual(failed["profile_status"], "FAILED")
            self.assertEqual(failed["failure"]["code"], "profile_failed")
            self.assertEqual(failed["failure"]["category"], "row_profile_error")
            self.assertIn("cannot decode skin", failed["failure"]["message"])
            self.assertEqual(report["per_asset_rows"][2]["profile_status"], "OK")

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

    def test_markdown_warns_for_partial_and_limited_reports_from_json_only(self) -> None:
        report = build_corpus_report_from_rows(
            [self._synthetic_row("a.glb", profile_status="OK", primary_family="contract_ready")],
            assets_selected=3,
            is_limited=True,
        )

        with patch("unirig_ext.humanoid_corpus_profiler.read_glb_container") as reader:
            markdown = render_markdown_from_report_json(json.loads(dumps_canonical_json(report)))

        reader.assert_not_called()
        self.assertIn("WARNING: report_status=partial", markdown)
        self.assertIn("limited selection", markdown)

    def test_weight_summary_is_reused_once_and_preserves_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-reuse-") as temp_dir:
            root = Path(temp_dir)
            glb = write_resolver_ready_glb(root / "clean.glb")
            baseline = build_corpus_report([glb])

            with patch("unirig_ext.humanoid_corpus_profiler.summarize_joint_weights", wraps=__import__("unirig_ext.gltf_skin_analysis", fromlist=["summarize_joint_weights"]).summarize_joint_weights) as summary:
                reused = build_corpus_report([glb])

        self.assertEqual(summary.call_count, 1)
        baseline_row = baseline["per_asset_rows"][0]
        reused_row = reused["per_asset_rows"][0]
        self.assertEqual(reused_row["primary_family"], baseline_row["primary_family"])
        self.assertEqual(reused_row["secondary_reason_codes"], baseline_row["secondary_reason_codes"])
        self.assertEqual(reused_row["quality_gate"]["status"], baseline_row["quality_gate"]["status"])
        self.assertEqual(reused_row["quality_gate"].get("diagnostic"), baseline_row["quality_gate"].get("diagnostic"))

    def _synthetic_row(self, path: str, *, profile_status: str, primary_family: str) -> dict:
        return {
            "path": path,
            "name": Path(path).name,
            "size_bytes": 1,
            "sha256": None,
            "profile_status": profile_status,
            "parse_status": "parsed" if profile_status == "OK" else "unparseable",
            "node_count": 1,
            "skin_count": 1 if profile_status == "OK" else 0,
            "root_count": 1 if profile_status == "OK" else 0,
            "skin_joint_count": 1 if profile_status == "OK" else 0,
            "weighted_joint_count": 1 if profile_status == "OK" else 0,
            "resolver": {"status": "success" if profile_status == "OK" else "not_run", "source_kind": None, "failure_code": None},
            "quality_gate": {"status": "passed" if profile_status == "OK" else "not_run", "reasons": []},
            "contract_readiness": {"status": "ready" if profile_status == "OK" else "not_ready", "ready": profile_status == "OK", "failure_code": None},
            "primary_family": primary_family,
            "secondary_reason_codes": ["contract_ready"] if primary_family == "contract_ready" else ["malformed_unparseable"],
            "diagnostics": [],
            "publication_evidence": False,
        }


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

    def test_cli_limit_accepts_positive_integer_and_rejects_invalid_without_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-cli-limit-") as temp_dir:
            root = Path(temp_dir)
            for name in ("b.glb", "a.glb"):
                write_embedded_skin_glb(root / name)
            json_out = root / "report.json"

            ok = self._run_cli([str(root), "--json-out", str(json_out), "--limit", "1"])
            self.assertEqual(ok.returncode, 0, ok.stderr)
            report = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(report["assets_selected"], 1)
            self.assertEqual(report["report_status"], "limited")

            for invalid in ("0", "-1", "abc"):
                bad_out = root / f"bad-{invalid.replace('-', 'neg')}.json"
                bad = self._run_cli([str(root), "--json-out", str(bad_out), "--limit", invalid])
                self.assertEqual(bad.returncode, 2, bad.stderr)
                self.assertFalse(bad_out.exists())

    def test_cli_progress_goes_to_stderr_and_stdout_has_final_summary_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unirig-corpus-cli-progress-") as temp_dir:
            root = Path(temp_dir)
            for name in ("b.glb", "a.glb"):
                write_embedded_skin_glb(root / name)
            json_out = root / "report.json"

            result = self._run_cli([str(root), "--json-out", str(json_out), "--limit", "2"])

            self.assertEqual(result.returncode, 0, result.stderr)
            stdout_lines = result.stdout.strip().splitlines()
            stderr_lines = [line for line in result.stderr.strip().splitlines() if line]
            self.assertEqual(stdout_lines, ["status=limited selected=2 completed=2 failed=0 partial=false limited=true"])
            self.assertEqual(
                stderr_lines,
                [
                    "1/2\ta.glb\tSTARTED",
                    "1/2\ta.glb\tOK",
                    "2/2\tb.glb\tSTARTED",
                    "2/2\tb.glb\tOK",
                ],
            )
            self.assertNotIn("STARTED", result.stdout)

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "unirig_ext.humanoid_corpus_cli", *args],
            cwd=ROOT,
            env={"PYTHONPATH": str(SRC)},
            text=True,
            capture_output=True,
            check=False,
        )


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
