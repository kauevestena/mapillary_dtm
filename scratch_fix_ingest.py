from pathlib import Path
p = Path("tests/test_ingest_cache.py")
content = p.read_text()

header = """from __future__ import annotations
import sys
import types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
"""

if "import sys" not in content:
    content = content.replace("from __future__ import annotations", header)
    p.write_text(content)

