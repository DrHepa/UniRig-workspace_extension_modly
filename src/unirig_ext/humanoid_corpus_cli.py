from __future__ import annotations

import argparse
import sys

from .humanoid_corpus_profiler import (
    CorpusInputError,
    CorpusOutputError,
    build_corpus_report,
    render_markdown_from_report_json,
    write_markdown_report_from_json,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--limit must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m unirig_ext.humanoid_corpus_cli",
        description="Profile GLB assets into deterministic UniRig humanoid evidence/failure families. Diagnostic only; not publication evidence.",
    )
    parser.add_argument("inputs", nargs="+", help="Directory, glob, or explicit .glb input path(s).")
    parser.add_argument("--json-out", required=True, help="Output JSON report path. Parent directory must already exist.")
    parser.add_argument("--markdown-out", help="Optional Markdown report path rendered strictly from the JSON report data.")
    parser.add_argument("--hash", action="store_true", help="Include per-asset SHA-256 digest for report identity only.")
    parser.add_argument("--limit", type=_positive_int, help="Profile only the first N assets after deterministic sorting. N must be positive.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        def progress(index: int, total: int, path: str, status: str) -> None:
            print(f"{index}/{total}\t{path}\t{status}", file=sys.stderr)

        report = build_corpus_report(
            args.inputs,
            include_hash=args.hash,
            limit=args.limit,
            progress_callback=progress,
            json_refresh_path=args.json_out,
        )
        markdown_path = None
        if args.markdown_out:
            markdown_path = write_markdown_report_from_json(report, args.markdown_out)
        else:
            # Keep stdout deterministic and generated from the JSON-shaped report only.
            render_markdown_from_report_json(report)
        _ = markdown_path
        print(
            "status={status} selected={selected} completed={completed} failed={failed} partial={partial} limited={limited}".format(
                status=report["report_status"],
                selected=report["assets_selected"],
                completed=report["assets_completed"],
                failed=report["assets_failed"],
                partial=str(report["is_partial"]).lower(),
                limited=str(report["is_limited"]).lower(),
            )
        )
        return 0
    except (CorpusInputError, CorpusOutputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: failed to profile corpus: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
