# fix_imports.py
from pathlib import Path
import re

def fix_imports():
    """Fix import statements in migrated files to use proper module paths"""
    
    files_to_fix = [
        'src/models/trs.py',
        'src/models/strategy_calc.py',
        'src/utils/data_cleaning.py'
    ]
    
    # These are the import fixes needed
    import_fixes = [
        # Fix database imports
        (r'from models\.database import', 'from src.models.database import'),
        (r'from models\.enums import', 'from src.models.enums import'),
        # Remove sys.path hacks
        (r'import sys\nsys\.path\.append\([\'"].+[\'"]\)\n', ''),
        # Fix relative imports if any
        (r'from database import', 'from src.models.database import'),
        (r'from enums import', 'from src.models.enums import'),
    ]
    
    for filepath in files_to_fix:
        path = Path(filepath)
        if path.exists():
            content = path.read_text()
            original = content
            
            for pattern, replacement in import_fixes:
                content = re.sub(pattern, replacement, content)
            
            if content != original:
                path.write_text(content)
                print(f"Fixed imports in {filepath}")
            else:
                print(f"No import changes needed in {filepath}")

if __name__ == "__main__":
    fix_imports()