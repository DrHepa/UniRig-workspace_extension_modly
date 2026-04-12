from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def main() -> int:
    input_path = Path(os.environ["UNIRIG_STAGE_INPUT"])
    output_path = Path(os.environ["UNIRIG_STAGE_OUTPUT"])
    trace_path = Path(os.environ["UNIRIG_HOOK_TRACE_FILE"])
    stage_name = os.environ["UNIRIG_STAGE_NAME"]

    trace_payload = {
        "stage": stage_name,
        "cwd": str(Path.cwd()),
        "input": str(input_path),
        "input_exists": input_path.exists(),
        "input_is_absolute": input_path.is_absolute(),
        "output": str(output_path),
        "run_dir": os.environ["UNIRIG_STAGE_RUN_DIR"],
    }

    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace_payload, sort_keys=True) + "\n")

    if os.environ.get("UNIRIG_HOOK_FAIL_STAGE") == stage_name:
        return int(os.environ.get("UNIRIG_HOOK_FAIL_CODE", "17"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
