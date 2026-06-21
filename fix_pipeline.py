import re

with open("cli/pipeline.py", "r") as f:
    text = f.read()

# Remove allow_synthetic from args and usages
text = re.sub(r'allow_synthetic:\s*bool\s*=\s*False,?\s*\n', '', text)
text = re.sub(r'allow_synthetic:\s*bool,?\s*\n', '', text)
text = re.sub(r'allow_synthetic=allow_synthetic,?\s*\n', '', text)
text = re.sub(r'\"allow_synthetic\":\s*allow_synthetic,?\s*\n', '', text)
text = re.sub(r'allow_heuristic=allow_synthetic,?\s*\n', '', text)
text = re.sub(r'\"allow_heuristic\":\s*allow_synthetic,?\s*\n', '', text)
text = re.sub(r'strict=not allow_synthetic', 'strict=True', text)

# In line 542: if not allow_synthetic and not reconB: -> if not reconB:
text = re.sub(r'if not allow_synthetic and not reconB:', 'if not reconB:', text)

# In find_anchors(seqs, token=token, allow_synthetic=allow_synthetic)
text = re.sub(r'allow_synthetic=allow_synthetic', '', text)
text = re.sub(r',\s*\)', ')', text)

# allow_synthetic_depth=allow_synthetic
text = re.sub(r'allow_synthetic_depth=allow_synthetic,?\s*\n', '', text)
text = re.sub(r'include_plane_sweep=allow_synthetic if name != "vo" else False,?\s*\n', 'include_plane_sweep=False,\n', text)

with open("cli/pipeline.py", "w") as f:
    f.write(text)
