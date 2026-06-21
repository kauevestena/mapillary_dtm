import re

with open("geom/anchors.py", "r") as f:
    text = f.read()

# Remove allow_synthetic from args and usages
text = re.sub(r'allow_synthetic:\s*bool\s*=\s*False,?\s*\n', '', text)
text = re.sub(r'allow_synthetic:\s*bool,?\s*\n', '', text)

# Remove the check in anchors.py:
# if allow_synthetic: ...
text = re.sub(r'\n\s*if allow_synthetic:.*?(?=\n\s*[a-zA-Z#])', '\n', text, flags=re.DOTALL)

with open("geom/anchors.py", "w") as f:
    f.write(text)

with open("ground/ground_extract_3d.py", "r") as f:
    text2 = f.read()

text2 = re.sub(r'allow_synthetic_depth:\s*bool\s*=\s*True,?\s*\n', '', text2)
text2 = re.sub(r'allow_synthetic=allow_synthetic_depth,?\s*\n', '', text2)

with open("ground/ground_extract_3d.py", "w") as f:
    f.write(text2)
