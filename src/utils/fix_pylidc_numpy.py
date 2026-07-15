import argparse
import os
import re

def patch_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    for old, new in replacements.items():
        content = re.sub(old, new, content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Patched: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Directory containing the pylidc library')
    
    args = parser.parse_args()

    # Replacement rules
    replacements = {
        r'\bnp\.int\b': 'int',
        r'\bnp\.bool\b': 'bool',
        r'\bnp\.float\b': 'float',
    }

    # Walk through the pylidc directory
    for root, _, files in os.walk(args.root_dir):
        for file in files:
            if file.endswith(".py"):
                patch_file(os.path.join(root, file), replacements)