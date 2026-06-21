import re

with open("cli/pipeline.py", "r") as f:
    text = f.read()

text = re.sub(r'allow_synthetic\s*:\s*bool\n', '', text)
text = re.sub(r'if strict_production and allow_synthetic:\n\s*raise ValueError\("Cannot run strict production with synthetic fallbacks enabled"\)\n', '', text)
text = re.sub(r'allow_synthetic:\s*bool\s*=\s*typer\.Option\(.*?\),\n', '', text, flags=re.DOTALL)

with open("cli/pipeline.py", "w") as f:
    f.write(text)

with open("geom/anchors.py", "r") as f:
    text = f.read()

text = re.sub(r'only when ``allow_synthetic`` is true.\n\s*', '', text)

with open("geom/anchors.py", "w") as f:
    f.write(text)
