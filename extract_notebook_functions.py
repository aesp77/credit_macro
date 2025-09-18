# extract_notebook_functions.py
import json
import os
from pathlib import Path

def extract_functions_from_notebook(notebook_path):
    """Extract function definitions from a notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    functions = []
    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def ' in source or 'class ' in source:
                lines = source.split('\n')
                for line in lines:
                    if line.strip().startswith('def '):
                        func_name = line.split('def ')[1].split('(')[0]
                        functions.append(func_name)
                    elif line.strip().startswith('class '):
                        class_name = line.split('class ')[1].split('(')[0].split(':')[0]
                        functions.append(class_name)
    return functions

# Extract from all notebooks
print("Functions found in notebooks:\n")
notebooks_dir = Path('notebooks')

for notebook in notebooks_dir.glob('*.ipynb'):
    functions = extract_functions_from_notebook(notebook)
    if functions:
        print(f"{notebook.name}:")
        for name in functions:
            print(f"  - {name}")
        print()