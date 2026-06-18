import os
import re

tests_dir = 'tests'
for root, _, files in os.walk(tests_dir):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path, 'r') as file:
                content = file.read()

            original = content
            # Remove allow_synthetic
            content = re.sub(r',\s*allow_synthetic=(True|False)', '', content)
            content = re.sub(r'allow_synthetic=(True|False),\s*', '', content)
            # Remove allow_heuristic
            content = re.sub(r',\s*allow_heuristic=(True|False)', '', content)
            content = re.sub(r'allow_heuristic=(True|False),\s*', '', content)
            # Remove force_synthetic
            content = re.sub(r',\s*force_synthetic=(True|False)', '', content)
            content = re.sub(r'force_synthetic=(True|False),\s*', '', content)
            
            if original != content:
                with open(path, 'w') as file:
                    file.write(content)
                print(f"Fixed {path}")

