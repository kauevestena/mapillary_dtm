"""
HTML/Markdown report generation.
"""
from __future__ import annotations
import json, os, datetime

def write_html(out_dir: str, manifest: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<html><body>")
        f.write("<h1>DTM from Mapillary — Report</h1>")
        f.write(f"<p>Generated: {datetime.datetime.utcnow().isoformat()}Z</p>")
        f.write("<pre>")
        f.write(json.dumps(manifest, indent=2))
        f.write("</pre>")
        f.write("</body></html>")
    return path
