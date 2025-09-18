# migrate_notebooks_to_modules.py
"""
Complete migration script: Extract, create modules, and analyze redundancies
"""

import json
import ast
from pathlib import Path
import nbformat

# Configuration
MIGRATION_PLAN = {
    'src/models/trs.py': {
        'source': 'notebooks/total_return.ipynb',
        'functions': [
            'calculate_cds_total_return',
            'interpolate_curve_rolldown',
            'TRSDatabaseBuilder',
            'calculate_trs',
            'build_trs_database', 
            'get_trs_data',
            'test_database'
        ],
        'description': 'Total Return Series calculations and database'
    },
    
    'src/models/strategy_calc.py': {
        'source': 'notebooks/total_return.ipynb',
        'functions': [
            'calculate_corrected_strategy_metrics',
            'calculate_generic_strategy_metrics',
            'test_generic_strategy_calculator',
            'calculate_steepener_pnl',
            'visualize_steepener_strategy'
        ],
        'description': 'Strategy P&L calculations'
    },
    
    'src/data/curve_loader.py': {
        'source': 'notebooks/01_data_exploration.ipynb',
        'functions': [
            'get_current_series',
            'get_current_curve',
            'get_historical_curves',
            'populate_database_with_curves'
        ],
        'description': 'Curve data loading utilities'
    },
    
    'src/utils/data_cleaning.py': {
        'source': 'notebooks/04_pnl_reconstruction.ipynb',
        'functions': [
            'clean_spread_data'
        ],
        'description': 'Data cleaning utilities'
    }
}

def extract_function_source(notebook_path, function_name):
    """Extract the actual source code of a function from notebook"""
    try:
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                if f'def {function_name}(' in source or f'class {function_name}' in source:
                    return source
    except Exception as e:
        print(f"    Error reading {notebook_path}: {e}")
    return None

def create_new_modules():
    """Create new module files with functions from notebooks"""
    
    print("STEP 1: CREATING NEW MODULES")
    print("="*50)
    
    created_files = []
    
    for module_path, config in MIGRATION_PLAN.items():
        print(f"\nCreating {module_path}")
        print(f"  Source: {config['source']}")
        print(f"  Description: {config['description']}")
        
        # Create module header
        module_content = f'''"""
{config['description']}
Migrated from {config['source']}
TO BE REVIEWED FOR REDUNDANCIES
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('../')

'''
        
        # Extract each function
        functions_found = 0
        for func_name in config['functions']:
            print(f"  Extracting {func_name}...")
            source = extract_function_source(config['source'], func_name)
            if source:
                module_content += f"\n\n{source}\n"
                functions_found += 1
            else:
                print(f"    WARNING: Could not find {func_name}")
        
        # Save the module
        if functions_found > 0:
            path = Path(module_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(module_content)
            print(f"  ✓ Saved {functions_found} functions to {module_path}")
            created_files.append(module_path)
        else:
            print(f"  ✗ No functions found, skipping {module_path}")
    
    return created_files

def get_functions_from_py_file(filepath):
    """Extract function names from a .py file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                functions.append(node.name)
        return functions
    except:
        return []

def analyze_redundancies(created_files):
    """Find redundancies between new and existing modules"""
    
    print("\n\nSTEP 2: REDUNDANCY ANALYSIS")
    print("="*50)
    
    # Define comparison pairs
    comparisons = [
        ('src/models/trs.py', 'src/data/data_manager.py'),
        ('src/models/strategy_calc.py', 'src/data/data_manager.py'),
        ('src/models/trs.py', 'src/models/database.py'),
        ('src/models/strategy_calc.py', 'src/models/position.py')
    ]
    
    redundancies = []
    
    for new_file, existing_file in comparisons:
        if new_file in created_files and Path(existing_file).exists():
            print(f"\nComparing {new_file} vs {existing_file}:")
            
            new_funcs = set(get_functions_from_py_file(new_file))
            existing_funcs = set(get_functions_from_py_file(existing_file))
            
            # Look for similar named functions
            similar = []
            for nf in new_funcs:
                for ef in existing_funcs:
                    # Check for similar names
                    if nf.lower() in ef.lower() or ef.lower() in nf.lower():
                        similar.append((nf, ef))
            
            if similar:
                print("  Potential redundancies found:")
                for nf, ef in similar:
                    print(f"    - {nf} <-> {ef}")
                    redundancies.append((new_file, existing_file, nf, ef))
            else:
                print("  No obvious redundancies")
    
    return redundancies

def create_test_notebook():
    """Create a test notebook to verify imports"""
    
    print("\n\nSTEP 3: CREATING TEST NOTEBOOK")
    print("="*50)
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Test Imports for Migrated Functions\n", 
                         "This notebook tests that all migrated functions can be imported"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import sys\n",
                    "sys.path.append('..')\n",
                    "\n",
                    "# Test imports\n",
                    "try:\n",
                    "    from src.models.trs import calculate_cds_total_return, TRSDatabaseBuilder\n",
                    "    print('✓ TRS imports successful')\n",
                    "except ImportError as e:\n",
                    "    print(f'✗ TRS import failed: {e}')\n",
                    "\n",
                    "try:\n",
                    "    from src.models.strategy_calc import calculate_generic_strategy_metrics\n",
                    "    print('✓ Strategy imports successful')\n",
                    "except ImportError as e:\n",
                    "    print(f'✗ Strategy import failed: {e}')\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save test notebook
    test_path = Path('notebooks/test_migrated_functions.ipynb')
    with open(test_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"Created test notebook: {test_path}")

def main():
    """Run the complete migration process"""
    
    print("\nNOTEBOOK TO MODULE MIGRATION")
    print("="*50)
    
    # Save migration plan
    with open('migration_plan.json', 'w') as f:
        json.dump(MIGRATION_PLAN, f, indent=2)
    print("Migration plan saved to migration_plan.json")
    
    # Step 1: Create new modules
    created_files = create_new_modules()
    
    # Step 2: Analyze redundancies
    redundancies = analyze_redundancies(created_files)
    
    # Step 3: Create test notebook
    create_test_notebook()
    
    # Summary
    print("\n\nMIGRATION SUMMARY")
    print("="*50)
    print(f"Created {len(created_files)} new module files")
    print(f"Found {len(redundancies)} potential redundancies")
    print("\nNext steps:")
    print("1. Run notebooks/test_migrated_functions.ipynb to verify imports")
    print("2. Review redundancies and decide which versions to keep")
    print("3. Update imports in streamlit_monitor.py to use new modules")
    
if __name__ == "__main__":
    main()