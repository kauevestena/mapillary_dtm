import re

with open("cli/pipeline.py", "r") as f:
    text = f.read()

# remove `if any(os.environ.get(name) == "1" for name in ("OPEN_SFM_FORCE_SYNTHETIC", "COLMAP_FORCE_SYNTHETIC")):`
# and its block in pipeline.py
text = re.sub(r'\s*if any\(os.environ.get.*?COLMAP_FORCE_SYNTHETIC.*?\):\n\s*log.warning.*?\n', '\n', text, flags=re.DOTALL)

with open("cli/pipeline.py", "w") as f:
    f.write(text)

with open("scripts/check_env.py", "r") as f:
    text = f.read()

text = re.sub(r'def check_no_forced_synthetic.*?\n\n\n', '', text, flags=re.DOTALL)
text = re.sub(r'    check_no_forced_synthetic\(\)\n', '', text)

with open("scripts/check_env.py", "w") as f:
    f.write(text)

with open("scripts/download_sample_data_impl.py", "r") as f:
    text = f.read()

text = re.sub(r'OPEN_SFM_FORCE_SYNTHETIC=1 COLMAP_FORCE_SYNTHETIC=1 \\\\\n', '', text)
text = re.sub(r'     OPEN_SFM_FORCE_SYNTHETIC=1 COLMAP_FORCE_SYNTHETIC=1 \\\\\n', '', text)

# also remove --allow-synthetic
text = re.sub(r'--allow-synthetic \\\\\n', '', text)
text = re.sub(r'     --allow-synthetic \\\\\n', '', text)

with open("scripts/download_sample_data_impl.py", "w") as f:
    f.write(text)

with open("scripts/generate_sample_geometry.py", "r") as f:
    text = f.read()

text = re.sub(r'os.environ\["COLMAP_FORCE_SYNTHETIC"\] = "0"\n', '', text)
text = re.sub(r'os.environ\["OPEN_SFM_FORCE_SYNTHETIC"\] = "0"\n', '', text)

with open("scripts/generate_sample_geometry.py", "w") as f:
    f.write(text)

