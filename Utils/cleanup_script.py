import os
import re
import shutil
from datetime import datetime

def cleanup_script(input_file, func_ranges, main_code_range, import_lines=9, create_backup=True):
    """
    Clean up a Python script by keeping only specified functions and main code.
    
    Parameters:
        input_file: Path to the Python file to clean
        func_ranges: Dict mapping function names to (start, end) tuples (1-indexed)
        main_code_range: Tuple (start, end) for main code to keep (1-indexed)
        import_lines: Number of lines to keep from the beginning (default 9)
        create_backup: Whether to create a backup file (default True)
    """
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found or is not a file.")
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
    
    import_lines = min(max(0, import_lines), total)
    main_start, main_end = main_code_range
    
    errors = []
    for name, (s, e) in func_ranges.items():
        if s < 1 or e > total or s > e:
            errors.append(f"Function '{name}': invalid range ({s}, {e})")
    if main_start < 1 or main_end > total or main_start > main_end:
        errors.append(f"Main code: invalid range ({main_start}, {main_end})")
    
    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
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
    
    result = lines[:import_lines]
    for _, (s, e) in sorted(func_ranges.items(), key=lambda x: x[1][0]):
        result.extend(lines[s-1:e])
    result.extend(lines[main_start-1:main_end])
    
    try:
        with open(input_file, 'w', encoding='utf-8') as f:
            f.writelines(result)
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

    print(f"Done: {len(result)} lines (was {total})")
    if func_ranges:
        print(f"Kept: {', '.join(sorted(func_ranges.keys()))}")
    return True

def get_input(prompt, default=None):
    """Get user input with optional default."""
    suffix = f" [{default}]: " if default is not None else ": "
    val = input(prompt + suffix).strip()
    return default if not val and default is not None else val

def get_int(prompt, default=None):
    """Get integer input."""
    val = get_input(prompt, default)
    try:
        return int(val) if val else default
    except ValueError:
        print("Invalid number.")
        return default

def get_range(prompt, max_line=None):
    """Get a line range (start, end) from user."""
    while True:
        val = get_input(prompt)
        if not val:
            return None
        if ',' not in val:
            print("Format: start,end (e.g., 10,20)")
            continue
        try:
            parts = val.split(',')
            s, e = int(parts[0].strip()), int(parts[1].strip())
            if s < 1 or e < 1 or s > e:
                print("Invalid range: start must be >= 1 and <= end")
                continue
            if max_line and e > max_line:
                print(f"End ({e}) exceeds file length ({max_line})")
                continue
            return (s, e)
        except (ValueError, IndexError):
            print("Format: start,end (e.g., 10,20)")

def detect_main_code(input_file, func_ranges, import_lines):
    """Auto-detect main code range by finding top-level code (0 indentation)."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        return None
    
    if not lines:
        return None
    
    total = len(lines)
    main_start = None
    
    for i, line in enumerate(lines, 1):
        if re.match(r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", line.strip()):
            main_start = i + 1
            break
    
    if main_start is None:
        last_def_end = import_lines
        in_def = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            is_top_level = len(line) - len(line.lstrip()) == 0
            
            if is_top_level:
                if re.match(r'^(def|class)\s+\w+', stripped):
                    in_def = True
                    last_def_end = i
                elif in_def:
                    last_def_end = i - 1
                    in_def = False
        
        for i in range(last_def_end, total):
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

def interactive_cleanup():
    """Interactive CLI for cleanup script."""
    print("=" * 50)
    print("Python Script Cleanup Tool")
    print("=" * 50)
    
    input_file = None
    while not input_file:
        path = get_input("\nFile path")
        if path and os.path.isfile(path):
            input_file = path
        else:
            print("File not found.")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = len(f.readlines())
        print(f"File has {total_lines} lines.")
    except:
        total_lines = None
    
    import_lines = get_int("Import lines to keep", 9)
    if import_lines is None or import_lines < 0:
        import_lines = 9
    
    auto_detect = get_input("\nAuto-detect main code range? (y/n)", "y")
    main_range = None
    
    if auto_detect.lower() == 'y':
        print("Detecting main code...")
        main_range = detect_main_code(input_file, {}, import_lines)
        if main_range:
            print(f"Detected: lines {main_range[0]}-{main_range[1]}")
            if get_input("Use this range? (y/n)", "y").lower() != 'y':
                main_range = None
        else:
            print("Could not auto-detect.")
    
    print("\nEnter functions (name,start,end). Empty to finish.")
    func_ranges = {}
    while True:
        val = get_input("Function")
        if not val:
            break
        parts = val.split(',')
        if len(parts) != 3:
            print("Format: name,start,end")
            continue
        try:
            name = parts[0].strip()
            s, e = int(parts[1].strip()), int(parts[2].strip())
            if not name or s < 1 or e < s:
                print("Invalid input.")
                continue
            if total_lines and e > total_lines:
                print(f"End ({e}) exceeds file length.")
                continue
            func_ranges[name] = (s, e)
            print(f"  Added: {name} ({s}-{e})")
        except ValueError:
            print("Format: name,start,end")
    
    if not main_range:
        if auto_detect.lower() == 'y' and func_ranges:
            print("\nRe-detecting main code with function info...")
            main_range = detect_main_code(input_file, func_ranges, import_lines)
            if main_range:
                print(f"Detected: lines {main_range[0]}-{main_range[1]}")
                if get_input("Use this range? (y/n)", "y").lower() != 'y':
                    main_range = None
        
        if not main_range:
            print("\nEnter main code range manually.")
            while not main_range:
                main_range = get_range("Main code (start,end)", total_lines)
    
    print(f"\nSummary:")
    print(f"  File: {input_file}")
    print(f"  Import lines: {import_lines}")
    print(f"  Functions: {len(func_ranges)}")
    for name, (s, e) in func_ranges.items():
        print(f"    - {name}: {s}-{e}")
    print(f"  Main code: {main_range[0]}-{main_range[1]}")
    
    if get_input("\nProceed? (y/n)", "n").lower() == 'y':
        if cleanup_script(input_file, func_ranges, main_range, import_lines):
            print("\nDone!")
            return True
    else:
        print("\nCancelled.")
    return False

if __name__ == '__main__':
    interactive_cleanup()
