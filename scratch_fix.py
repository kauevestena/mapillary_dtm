import re

with open('geom/anchors.py', 'r') as f:
    c = f.read()
c = re.sub(r'\s*\*,', '', c)
c = re.sub(r'def _synthesize_anchors\(.*?(?=def |class |$)', '', c, flags=re.DOTALL)
with open('geom/anchors.py', 'w') as f:
    f.write(c)

with open('tests/test_breakline_integration.py', 'r') as f:
    c = f.read()
c = c.replace('from ground.breakline_integration', 'from dtm_from_mapillary.ground.breakline_integration')
with open('tests/test_breakline_integration.py', 'w') as f:
    f.write(c)
