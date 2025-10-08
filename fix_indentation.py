#!/usr/bin/env python
"""Fix indentation in 5_option_pricing.py"""

filepath = 'src/apps/pages/5_option_pricing.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines 521-895 need to be dedented by 4 spaces (1 indent level)
fixed_lines = []
for i, line in enumerate(lines):
    line_num = i + 1  # 1-indexed
    
    if 522 <= line_num <= 895:
        # Remove 4 spaces from the beginning if present
        if line.startswith('            '):  # 12 spaces
            fixed_lines.append('        ' + line[12:])  # Replace with 8 spaces
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write back
with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print(f"Fixed {len([l for i, l in enumerate(lines) if 521 <= i < 895])} lines")
