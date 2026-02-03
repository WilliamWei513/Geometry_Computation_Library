import os
import re
import shutil
import ast
from datetime import datetime

def detect_imports_end(lines):
    """Detect where import section ends (including comments and blank lines between imports)."""
    last_import_line = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')) or stripped.startswith('#') or not stripped:
            if stripped.startswith(('import ', 'from ')):
                last_import_line = i + 1
        elif stripped and not stripped.startswith('#'):
            break
    return last_import_line

def detect_all_functions(lines):
    """Detect all function definitions with their line ranges."""
    functions = {}
    current_func = None
    current_start = None
    current_indent = None
    
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        
        indent = len(line) - len(line.lstrip())
        match = re.match(r'^def\s+(\w+)\s*\(', line.strip())
        
        if match and indent == 0:
            if current_func:
                functions[current_func] = (current_start, i - 1)
            current_func = match.group(1)
            current_start = i
            current_indent = 0
        elif current_func and indent == 0 and line.strip() and not line.strip().startswith('#'):
            if not re.match(r'^def\s+\w+\s*\(', line.strip()):
                functions[current_func] = (current_start, i - 1)
                current_func = None
    
    if current_func:
        for i in range(len(lines), current_start - 1, -1):
            if lines[i-1].strip() and not lines[i-1].strip().startswith('#'):
                functions[current_func] = (current_start, i)
                break
    
    return functions

def detect_main_code_range(lines, import_end, all_functions):
    """Detect main code range (top-level code after all functions)."""
    total = len(lines)
    main_start = None
    
    for i, line in enumerate(lines, 1):
        if re.match(r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", line.strip()):
            main_start = i
            break
    
    if main_start is None:
        func_end = import_end
        if all_functions:
            func_end = max(end for _, end in all_functions.values())
        
        for i in range(func_end, total):
            line = lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            is_top_level = len(line) - len(line.lstrip()) == 0
            if is_top_level and not re.match(r'^(def|class)\s+\w+', stripped):
                main_start = i + 1
                break
    
    if main_start is None:
        return None
    
    main_end = total
    while main_end > main_start:
        stripped = lines[main_end - 1].strip()
        if stripped and not stripped.startswith('#'):
            break
        main_end -= 1
    
    return (main_start, main_end) if main_start <= main_end else None

def find_used_functions(lines, main_range, all_functions):
    """Find all functions used in main code and their dependencies recursively."""
    main_start, main_end = main_range
    main_code = ''.join(lines[main_start-1:main_end])
    
    used = set()
    to_check = set()
    
    for func_name in all_functions.keys():
        if re.search(r'\b' + re.escape(func_name) + r'\s*\(', main_code):
            to_check.add(func_name)
    
    while to_check:
        func_name = to_check.pop()
        if func_name in used:
            continue
        used.add(func_name)
        
        if func_name in all_functions:
            start, end = all_functions[func_name]
            func_code = ''.join(lines[start-1:end])
            
            for other_func in all_functions.keys():
                if other_func != func_name and other_func not in used:
                    if re.search(r'\b' + re.escape(other_func) + r'\s*\(', func_code):
                        to_check.add(other_func)
    
    return used

def cleanup_script(input_file, create_backup=True):
    """
    Clean up a Python script by keeping only imports, used functions, and main code.
    Automatically detects everything.
    """
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(input_file, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    total = len(lines)
    if total == 0:
        print("Error: File is empty.")
        return False
    
    print(f"File: {input_file} ({total} lines)")
    
    import_end = detect_imports_end(lines)
    print(f"Imports: lines 1-{import_end}")
    
    all_functions = detect_all_functions(lines)
    print(f"Functions found: {len(all_functions)}")
    for name, (s, e) in sorted(all_functions.items(), key=lambda x: x[1][0]):
        print(f"  - {name}: lines {s}-{e}")
    
    main_range = detect_main_code_range(lines, import_end, all_functions)
    if not main_range:
        print("Error: Could not detect main code.")
        return False
    print(f"Main code: lines {main_range[0]}-{main_range[1]}")
    
    used_functions = find_used_functions(lines, main_range, all_functions)
    unused_functions = set(all_functions.keys()) - used_functions
    
    print(f"\nUsed functions: {len(used_functions)}")
    for name in sorted(used_functions):
        s, e = all_functions[name]
        print(f"  - {name}: lines {s}-{e}")
    
    if unused_functions:
        print(f"\nUnused functions (will be removed): {len(unused_functions)}")
        for name in sorted(unused_functions):
            s, e = all_functions[name]
            print(f"  - {name}: lines {s}-{e}")
    
    if not used_functions and not main_range:
        print("Nothing to keep. Aborting.")
        return False
    
    if input("\nProceed? (y/n): ").lower() != 'y':
        print("Cancelled.")
        return False
    
    if create_backup:
        try:
            backup = f"{input_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(input_file, backup)
            print(f"Backup: {backup}")
        except Exception as e:
            print(f"Warning: Backup failed: {e}")
            if input("Continue without backup? (y/n): ").lower() != 'y':
                return False
    
    result = lines[:import_end]
    
    for name in sorted(used_functions, key=lambda n: all_functions[n][0]):
        s, e = all_functions[name]
        result.extend(lines[s-1:e])
        if not result[-1].endswith('\n'):
            result.append('\n')
    
    result.extend(lines[main_range[0]-1:main_range[1]])
    
    try:
        with open(input_file, 'w', encoding='utf-8') as f:
            f.writelines(result)
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

    print(f"\nDone: {len(result)} lines (was {total})")
    return True

def interactive_cleanup():
    """Interactive CLI for cleanup script."""
    print("=" * 50)
    print("Python Script Auto Cleanup Tool")
    print("=" * 50)
    
    input_file = None
    while not input_file:
        path = input("\nFile path: ").strip()
        if path and os.path.isfile(path):
            input_file = path
        else:
            print("File not found.")
    
    print()
    return cleanup_script(input_file)

if __name__ == '__main__':
    interactive_cleanup()
