"""
HTML/Markdown report generation.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Mapping

import numpy as np


def write_html(
    out_dir: str,
    manifest: dict,
    qa_summary: Mapping[str, object] | None = None,
    artifact_paths: Mapping[str, str] | None = None,
) -> str:
    """Persist a lightweight QA report."""

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.html")

    def _format_value(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.3f}"
        return str(value)

    with open(path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>DTM QA Report</title>")
        f.write("<style>body{font-family:sans-serif;margin:2rem;}table{border-collapse:collapse;}"
                "td,th{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}th{background:#f0f0f0;}"
                "pre{background:#f7f7f7;padding:1rem;border:1px solid #ccc;overflow:auto;}</style>")
        f.write("</head><body>")
        f.write("<h1>DTM from Mapillary — QA Report</h1>")
        f.write(f"<p>Generated: {datetime.datetime.utcnow().isoformat()}Z</p>")

        f.write("<h2>Manifest</h2><pre>")
        f.write(json.dumps(manifest, indent=2, default=_json_serializer))
        f.write("</pre>")

        if qa_summary:
            f.write("<h2>QA Metrics</h2><table>")
            f.write("<tr><th>Metric</th><th>Value</th></tr>")
            for key, value in qa_summary.items():
                if isinstance(value, Mapping):
                    for sub_key, sub_val in value.items():
                        f.write(f"<tr><td>{key}.{sub_key}</td><td>{_format_value(sub_val)}</td></tr>")
                else:
                    f.write(f"<tr><td>{key}</td><td>{_format_value(value)}</td></tr>")
            f.write("</table>")

        if artifact_paths:
            f.write("<h2>Artifacts</h2><ul>")
            for label, artefact in artifact_paths.items():
                f.write(f"<li>{label}: <code>{artefact}</code></li>")
            f.write("</ul>")

        f.write("</body></html>")
    return path


def _json_serializer(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
