from __future__ import annotations

import argparse
import sys

from .humanoid_mapping_candidates import (
    CandidateInputError,
    CandidateOutputError,
    build_candidate_reports,
    write_candidates_json,
    write_candidates_jsonl,
)
from .kimodo_probe import KimodoProbeBackend, KimodoProbeConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m unirig_ext.humanoid_mapping_candidates_cli",
        description="Emit deterministic diagnostic UniRig bone_N humanoid mapping candidates. Diagnostic only; not publication evidence.",
    )
    parser.add_argument("inputs", nargs="*", help="Explicit .glb path(s), directories, or glob patterns.")
    parser.add_argument("--manifest", help="Optional JSON manifest: list of paths or object with assets list.")
    parser.add_argument("--json-out", required=True, help="Output JSON report path. Parent directory must already exist.")
    parser.add_argument("--jsonl-out", help="Optional JSONL path with one candidate object per line. Parent directory must already exist.")
    parser.add_argument("--kimodo-root", help="Optional Kimodo checkout hint. Missing/unavailable Kimodo is non-fatal.")
    parser.add_argument("--source-bvh", help="Optional complete source BVH for calibrated Kimodo candidate probing.")
    parser.add_argument("--probe-retarget", action="store_true", help="Ask the optional Kimodo backend to run retarget checks on disposable copies only.")
    parser.add_argument("--probe-output-root", help="Optional existing directory for disposable Kimodo probe copies and sidecars.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        backend = KimodoProbeBackend(args.kimodo_root, source_bvh=args.source_bvh, probe_output_root=args.probe_output_root) if args.kimodo_root or args.probe_retarget or args.source_bvh or args.probe_output_root else None
        report = build_candidate_reports(
            args.inputs,
            manifest=args.manifest,
            source="manifest" if args.manifest else "explicit-input",
            kimodo_backend=backend,
            probe_retarget=args.probe_retarget,
        )
        write_candidates_json(report, args.json_out)
        if args.jsonl_out:
            write_candidates_jsonl(report["candidates"], args.jsonl_out)
        summary = report["summary"]
        print(
            "status=complete selected={selected} publication_evidence={publication}".format(
                selected=summary["total_assets"],
                publication=str(summary["publication_evidence"]).lower(),
            )
        )
        return 0
    except (CandidateInputError, CandidateOutputError, KimodoProbeConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: failed to build humanoid mapping candidates: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
