from pathlib import Path

p = Path("constants.py")
content = p.read_text()

if "MIN_ROAD_MASK_RATIO" not in content:
    content += "\n# Semantics\nMIN_ROAD_MASK_RATIO = 0.05\n"
    p.write_text(content)
