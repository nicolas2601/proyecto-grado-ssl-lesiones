from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class V10ReportError(RuntimeError):
    """The generated V10 research report is missing or incompatible."""


def build_v10_report(
    report_path: str | Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    path = Path(report_path)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise V10ReportError("V10 generated report is missing or invalid") from exc
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise V10ReportError("V10 generated report SHA256 mismatch")
    try:
        report = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V10ReportError("V10 generated report is missing or invalid") from exc
    if not isinstance(report, dict) or report.get("release") != "v10.0.0":
        raise V10ReportError("report is not generated for V10.0.0")
    return {
        **report,
        "api_scope": "MoCo V10 classification development evidence; research use only.",
        "disclaimer": (
            "Predictions are not calibrated clinical probabilities and do not establish "
            "clinical validity."
        ),
    }
