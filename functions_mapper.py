# function_mapper.py
import ast
import os
from pathlib import Path

def get_functions_from_py_file(filepath):
    """Extract function and class names from a .py file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(('function', node.name))
            elif isinstance(node, ast.ClassDef):
                functions.append(('class', node.name))
        return functions
    except:
        return []

# Map all existing .py files
print("EXISTING FUNCTIONS IN .PY FILES:\n")
src_dir = Path('src')

py_functions = {}
for py_file in src_dir.rglob('*.py'):
    if '__pycache__' not in str(py_file):
        funcs = get_functions_from_py_file(py_file)
        if funcs:
            rel_path = py_file.relative_to(src_dir)
            py_functions[str(rel_path)] = funcs
            print(f"{rel_path}:")
            for ftype, fname in funcs:
                print(f"  - {fname} ({ftype})")
            print()

# Now identify conflicts with notebook functions
print("\nPOTENTIAL CONFLICTS/DUPLICATES:")
print("-" * 40)

notebook_functions = {
    'total_return.ipynb': [
        'calculate_cds_total_return',  # appears twice in same notebook!
        'calculate_corrected_strategy_metrics',
        'calculate_generic_strategy_metrics', 
        'TRSDatabaseBuilder',
        'calculate_trs',
        'build_trs_database',
        'get_trs_data',
        'test_database'
    ]
}

# Check for duplicates
for nb_name, nb_funcs in notebook_functions.items():
    for func in nb_funcs:
        # Check if function exists in any .py file
        for py_file, py_funcs in py_functions.items():
            if func in [f[1] for f in py_funcs]:
                print(f"'{func}' exists in both {nb_name} and {py_file}")

# Check for duplicate function names within notebooks
print("\nDUPLICATES WITHIN NOTEBOOKS:")
print(f"- 'calculate_cds_total_return' appears twice in total_return.ipynb")