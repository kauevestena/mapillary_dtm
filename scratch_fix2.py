with open('tests/test_breakline_integration.py', 'r') as f:
    c = f.read()
c = c.replace('from dtm_from_mapillary.ground.breakline_integration', 'from ground.breakline_integration')
with open('tests/test_breakline_integration.py', 'w') as f:
    f.write(c)
